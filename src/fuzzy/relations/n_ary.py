"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Tuple, Union

import igraph
import numpy as np
import scipy.sparse as sps
import torch

# from line_profiler import profile
from torch import Size, Tensor

from fuzzy.sets.membership import Membership
from fuzzy.utils import TorchJitModule, check_path_to_save_torch_module
from fuzzy.utils.options.abstract.primitive import GroupedOptions

from ..utils.classes import Loggable

# , log_classmethod, log_func, log_method
from ..utils.functions import exp_sum_log
from .linkage import BinaryLinks, GroupedLinks


class NAryMaskMethods(str, Enum):
    """
    The available methods for completing n-ary fuzzy relations.
    """

    PROD = "prod"  # the default implementation
    EXP_SUM_LOG = "exp_sum_log"  # an alternative implementation
    # a more efficient calculation if sum is to be applied immediately after
    LINEAR_SUM = "linear"


class NAryRelation(TorchJitModule, Loggable):
    """
    This class represents an n-ary fuzzy relation. An n-ary fuzzy relation is a relation that takes
    n arguments and returns a (float) value. This class is useful for representing fuzzy relations
    that take multiple arguments, such as a t-norm that takes two or more arguments and returns a
    truth value.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *indices: Union[Tuple[int, int], List[Tuple[int, int]]],
        device: torch.device,
        grouped_links: Union[None, GroupedLinks] = None,
        nan_replacement: float = 0.0,
        method: NAryMaskMethods = NAryMaskMethods.PROD,
        **kwargs,
    ):
        """
        Apply an n-ary relation to the indices (i.e., relation's matrix) on the provided device.

        Args:
            items: The 2-tuple indices to apply the n-ary relation to (e.g., (0, 1), (1, 0)).
            device: The device to use for the relation.
            grouped_links: The end-user can provide the links to use for the relation; this is
                useful for when the links are already created and the user wants to use them, or
                for a relation that requires more complex setup. Default is None.
            nan_replacement: The value to use when a value is missing in the relation (i.e., nan);
                this is useful for when input to the relation is not complete. Default is 0.0
                (penalize), a value of 1.0 would ignore missing values (i.e., do not penalize).
            method: The selected method to apply the mask to the membership degrees. Some
                implementations are more efficient if we can incorporate subsequent calculations
                (e.g., it is possible to use torch.nn.functional.linear if we know a .sum is applied
                immediately afterward).
        """
        super().__init__(**kwargs)
        self.device: torch.device = device
        if nan_replacement not in [0.0, 1.0]:
            raise ValueError("The nan_replacement must be either 0.0 or 1.0.")
        self.nan_replacement: float = nan_replacement
        self.method: NAryMaskMethods = method
        # cache the selected method; very important for performance to avoid
        # branching
        self._apply_mask_func: Callable[[Membership], torch.Tensor] = (
            self._cache_apply_mask_func()
        )
        self.matrix = None  # created later (via self._rebuild)
        self.grouped_links: Union[None, GroupedLinks] = (
            None  # created later (via self._rebuild)
        )
        self.graph = None  # will be created later (via self._rebuild)

        # variables used for when the indices are given
        self.indices: List[List[Tuple[int, int]]] = []
        self._coo_matrix: List[sps._coo.coo_matrix] = []
        self._original_shape: List[Tuple[int, int]] = []

        if not indices:  # indices are not given
            if grouped_links is None:
                raise ValueError(
                    "At least one set of indices must be provided, or GroupedLinks must be given."
                )
            # note that many features are not available when using
            # grouped_links
            self.grouped_links = grouped_links
        else:  # indices are given
            if not isinstance(indices[0], list):
                indices = [indices]

            # this scenario is for when we have multiple compound indices that use the same relation
            # this is useful for computational efficiency (i.e., not having to
            # use a for loop)
            for relation_indices in indices:
                if len(set(relation_indices)) < len(relation_indices):
                    raise ValueError(
                        "The indices must be unique for the relation to be well-defined."
                    )
                coo_matrix = self.convert_indices_to_matrix(relation_indices)
                self._original_shape.append(coo_matrix.shape)
                self._coo_matrix.append(coo_matrix)
            # now convert to a list of matrices
            max_var = max(t[0] for t in self._original_shape)
            max_term = max(t[1] for t in self._original_shape)
            self.indices.extend(indices)
            self._rebuild(*(max_var, max_term))

        # # test if the relation is well-defined & build it
        # # the last index, -1, is the relation index; first 2 are (variable, term) indices
        # membership_shape: torch.Size = self.grouped_links.shape[:-1]
        # # but we also need to include a dummy batch dimension (32) for the grouped_links
        # membership_shape: torch.Size = torch.Size([32] + list(membership_shape))
        # self.applied_mask = self.grouped_links(
        #     Membership(
        #         # elements=torch.empty(membership_shape, device=self.device),
        #         degrees=torch.zeros(membership_shape, device=self.device),
        #         # mask=torch.empty(membership_shape, device=self.device),
        #     )
        # )
        self.applied_mask: Union[None, torch.Tensor] = (
            None  # created later (via self.apply_mask)
        )

    # @log_method
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.indices})"

    # @log_method
    def __hash__(self) -> int:
        return hash(self.nan_replacement) + hash(self.device)

    # @log_method
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NAryRelation) or not isinstance(self, type(other)):
            return False
        applied_mask, other_applied_mask = self.get_mask(), other.get_mask()
        return (
            applied_mask.shape == other_applied_mask.shape
            and (applied_mask - other_applied_mask).sum() == 0
            and self.nan_replacement == other.nan_replacement
        )

    @property
    def shape(self) -> torch.Size:
        """
        Get the shape of the relation's matrix.

        Returns:
            The shape of the relation's matrix.
        """
        return self.grouped_links.shape

    @staticmethod
    # @log_func
    def convert_indices_to_matrix(indices) -> sps._coo.coo_matrix:
        """
        Convert the given indices to a COO matrix.

        Args:
            indices: The indices where a '1' will be placed at each index.

        Returns:
            The COO matrix with a '1' at each index.
        """
        data = np.ones(len(indices))  # a '1' indicates a relation exists
        row, col = zip(*indices)
        return sps.coo_matrix((data, (row, col)), dtype=np.int8)

    # @log_method
    def get_mask(self) -> torch.Tensor:
        """
        Get the applied mask.

        Returns:
            The applied mask.
        """
        # test if the relation is well-defined & build it
        # the last index, -1, is the relation index; first 2 are (variable,
        # term) indices
        membership_shape: torch.Size = self.grouped_links.shape[:-1]
        # but we also need to include a dummy batch dimension (32) for the
        # grouped_links
        batched_membership_shape: torch.Size = torch.Size([32] + list(membership_shape))
        with torch.no_grad():  # disable grad checking
            dummy_membership: Membership = Membership(
                # elements=torch.empty(membership_shape, device=self.device),
                degrees=torch.ones(batched_membership_shape, device=self.device),
                mask=torch.ones(membership_shape, device=self.device),
            )
            mask = self.grouped_links(dummy_membership)
            if mask.is_sparse and not mask.is_coalesced():
                mask = mask.coalesce()
        return mask

    # @log_method
    def to(self, device: torch.device, *args, **kwargs) -> "NAryRelation":
        """
        Move the n-ary relation to the specified device.

        Args:
            device: The device to move the n-ary relation to.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The n-ary relation on the specified device.
        """
        super().to(device, *args, **kwargs)
        self.device = device
        if self.grouped_links is not None:
            self.grouped_links.to(device)
        return self

    # @log_method
    def _state_dict(self, path: Path) -> MutableMapping[str, Any]:
        """
        An internal method to get the state dictionary for the n-ary relation. A path is required
        to save the grouped_links, as it is not saved in the state dictionary.

        Allows subclasses to override the save method without having to repeat the code for saving
        the state dictionary of the general n-ary relation.

        Note: THIS WILL SAVE THE GROUPED_LINKS TO THE GIVEN PATH (AFTER SOME MODIFICATION).

        Args:
            path: The path to save the grouped_links.

        Returns:
            The state dictionary for the n-ary relation.
        """
        state_dict: MutableMapping[str, Any] = self.state_dict()
        state_dict["nan_replacement"] = self.nan_replacement
        state_dict["class_name"] = self.__class__.__name__

        if len(self.indices) == 0:
            # we will rebuild from the grouped_links, so we do not need to save
            # the indices
            grouped_links_dir: Path = path / "grouped_links"
            self.grouped_links.save(path=grouped_links_dir)
            state_dict["grouped_links"] = (
                grouped_links_dir  # save the path to the grouped_links
            )
        else:
            # we will rebuild from the indices, so we do not need to save the
            # grouped_links
            state_dict["indices"] = (
                self.indices if len(self.indices) > 1 else self.indices[0]
            )
        return state_dict

    # @log_method
    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the n-ary relation to a dictionary given a path.

        Args:
            path: The (requested) path to save the n-ary relation. This may be modified to ensure
            all necessary files are saved (e.g., it may be turned into a directory instead).

        Returns:
            The dictionary representation of the n-ary relation.
        """
        check_path_to_save_torch_module(path)
        dir_path: Path = path.parent / path.name.split(".")[0]
        state_dict: MutableMapping[str, Any] = self._state_dict(path=dir_path)

        # where to save the state_dict depends on whether the indices are given
        # or not
        save_location: Path = (
            dir_path / "state_dict.pt" if len(self.indices) == 0 else path
        )

        torch.save(state_dict, save_location)

        return state_dict

    @classmethod
    # @log_classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
        t_norm_callback: Union[None, Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> "NAryRelation":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Args:
            path: The path to load the t-norm relation.
            device: The physical device to load the t-norm relation onto.
            t_norm_callback: A function to call to dynamically modify the keyword arguments
                passed onto the TNorm subclass after it has been identified.

        Returns:
            The n-ary relation.
        """
        if path.is_file() and path.suffix == ".pt":
            # load from indices
            state_dict: MutableMapping = torch.load(path, weights_only=False)
        else:
            # load from grouped_links, path is a directory
            state_dict: MutableMapping = torch.load(
                path / "state_dict.pt", weights_only=False
            )
        nan_replacement = state_dict.pop("nan_replacement")
        kwargs: Dict[str, Any] = {"nan_replacement": nan_replacement, "device": device}

        class_name = state_dict.pop("class_name")
        fuzzy_cls = cls.get_subclass(class_name)

        if "indices" in state_dict:
            indices = state_dict.pop("indices")
            return fuzzy_cls(*indices, **kwargs)
        grouped_links: Path = state_dict.pop("grouped_links")
        grouped_links_kwargs: Dict[str, Any] = {"device": device}
        if (grouped_links / "configuration").exists():
            # order matters here; GroupedLinks cannot load before
            # GumbelSoftmaxOptions.load
            configuration: GroupedOptions = GroupedOptions.load(
                grouped_links / "configuration"
            )
            # this directory cannot exist when calling GroupedLinks.load
            shutil.rmtree(grouped_links / "configuration")
            grouped_links_kwargs["configuration"] = configuration
        kwargs["grouped_links"] = GroupedLinks.load(
            grouped_links, **grouped_links_kwargs
        )

        if t_norm_callback is not None:
            kwargs = t_norm_callback(kwargs)

        obj = fuzzy_cls(**kwargs)
        # add other attributes that may be specific to the subclass
        obj.load_state_dict(state_dict, strict=False)
        return obj

    # @log_method
    def create_ndarray(self, max_var: int, max_term: int) -> None:
        """
        Make (or update) the numpy matrix from the COO matrices.

        Args:
            max_var: The maximum number of variables.
            max_term: The maximum number of terms.

        Returns:
            None
        """
        matrices = []
        for coo_matrix in self._coo_matrix:
            # first resize
            coo_matrix.resize(max_var, max_term)
            matrices.append(coo_matrix.toarray())
        if len(matrices) > 0:  # need at least one array to stack
            # make a new axis and stack along that axis
            self.matrix: np.ndarray = np.stack(matrices).swapaxes(0, 1).swapaxes(1, 2)

    # @log_method
    def create_igraph(self) -> None:
        """
        Create the graph representation of the relation(s).

        Returns:
            None
        """
        graphs: List[igraph.Graph] = []
        for relation in self.indices:
            # create a directed (mode="in") star graph with the relation as the
            # center (vertex 0)
            graphs.append(igraph.Graph.Star(n=len(relation) + 1, mode="in", center=0))
            # relation vertices are the first vertices in the graph
            # located at index 0
            relation_vertex: igraph.Vertex = graphs[-1].vs.find(0)
            # set item and tags for the relation vertex for easy retrieval;
            # name is for graph union
            (
                relation_vertex["name"],
                relation_vertex["item"],
                relation_vertex["tags"],
            ) = (hash(self) + hash(tuple(relation)), self, {"relation"})
            # anchor vertices are the var-term pairs that are involved in the
            # relation vertex
            anchor_vertices: List[igraph.Vertex] = relation_vertex.predecessors()
            # set anchor vertices' item and tags for easy retrieval; name is
            # for graph union
            for anchor_vertex, index_pair in zip(anchor_vertices, relation):
                anchor_vertex["name"], anchor_vertex["item"], anchor_vertex["tags"] = (
                    index_pair,
                    index_pair,
                    {"anchor"},
                )
        if len(graphs) > 0:  # need at least one graph to union
            self.graph = igraph.union(graphs, byname=True)

    # @log_method
    def _rebuild(self, *shape) -> None:
        """
        Rebuild the relation's matrix and graph.

        Args:
            shape: The new shape of the n-ary fuzzy relation; assuming shape is (max_var, max_term).

        Returns:
            None
        """
        # re-create the self.matrix
        self.create_ndarray(shape[0], shape[1])
        # update the self.grouped_links to reflect the new shape
        # these links are used to zero out the values that are not part of the
        # relation
        self.grouped_links = GroupedLinks(
            modules_list=[BinaryLinks(links=self.matrix, device=self.device)]
        )
        # re-create the self.graph (has to happen after self.grouped_links is
        # created)
        self.create_igraph()

    # @log_method
    def resize(self, *shape) -> None:
        """
        Resize the matrix in-place to the given shape, and then rebuild the relations' members.

        Args:
            shape: The new shape of the matrix.

        Returns:
            None
        """
        for coo_matrix in self._coo_matrix:
            coo_matrix.resize(*shape)
        self._rebuild(*shape)

    def _apply_mask(
        self, membership: Membership, inplace: bool = False
    ) -> torch.Tensor:
        """
        Get the applied mask from the GroupedLinks using the given Membership object. Caution
        should be used with this method as it may or may not update the cached applied mask
        depending on if inplace is True (default is False).

        Args:
            membership: The Membership object to use in determining the applied mask.
            inplace: Whether to modify the self.applied_mask with this obtained applied mask.

        Returns:
            The applied mask based on the given membership.
        """
        applied_mask: torch.Tensor = self.grouped_links(membership=membership)
        # if applied_mask.is_sparse:
        #     applied_mask: torch.Tensor = self.applied_mask.to_dense()
        if inplace:
            self.applied_mask = applied_mask
        after_mask = membership.degrees.unsqueeze(-1) * applied_mask
        return after_mask + (1 - applied_mask)

    # @log_method
    # @profile
    def apply_mask(self, membership: Membership) -> torch.Tensor:
        """
        Apply the n-ary relation's mask to the given memberships.

        Args:
            membership: The membership values to apply the minimum n-ary relation to.

        Returns:
            The masked membership values (zero may or may not be a valid degree of truth).
        """
        membership_shape: torch.Size = membership.degrees.shape
        if self.grouped_links.shape[:-1] != membership_shape[1:]:
            # if len(membership_shape) > 2:
            # this is for the case where masks have been stacked due to
            # compound relations
            # get the last two dimensions
            membership_shape = membership_shape[1:]
            self.resize(*membership_shape)
        del membership_shape  # free up memory

        # the below is VALID but NOT compatible w/ autograd
        # indices = self.applied_mask.to(torch.int64)
        # indices = indices.unsqueeze(0).expand(membership.degrees.size(0), -1, -1)
        # after_mask = torch.gather(membership.degrees, -1, indices)
        # return after_mask.nan_to_num(self.nan_replacement)

        # select memberships that are not zeroed out (i.e., involved in the relation)
        # with torch.autograd.graph.save_on_cpu():  # save the graph on the CPU
        # (for memory)
        # applied_mask = self.grouped_links.grouped_links.modules_list[0].logits
        return self._apply_mask_func(membership)

    def _cache_apply_mask_func(self) -> Callable[[Membership], torch.Tensor]:
        if self.method == NAryMaskMethods.PROD:
            return self._prod_apply_mask
        if self.method == NAryMaskMethods.LINEAR_SUM:
            return self._linear_sum_apply_mask
        if self.method == NAryMaskMethods.EXP_SUM_LOG:
            return self._exp_sum_log_apply_mask
        raise NotImplementedError(
            f"The given method '{self.method}' does not have an implemented behavior within "
            f"{type(self)}."
        )

    def _prod_apply_mask(self, membership: Membership) -> Tensor:
        """
        The default resolution strategy for applying the mask to the given fuzzy relation.

        Args:
            membership: The membership values.

        Returns:
            The fuzzy relation values.
        """
        # torch.prod on large dims can be slow and non-fusible
        after_mask: torch.Tensor = self._apply_mask(
            membership=membership, inplace=True
        )  # update self.applied_mask
        prod_result = after_mask.prod(dim=2, keepdim=False).nan_to_num(
            self.nan_replacement
        )
        return prod_result

    def _exp_sum_log_apply_mask(self, membership: Membership) -> Tensor:
        """
        A possibly more efficient resolution strategy for applying the mask to the given
        fuzzy relation; it is mathematically equivalent to the product technique.

        Args:
            membership: The membership values.

        Returns:
            The fuzzy relation values.
        """
        # "exp-sum-log trick" for stable product via log-domain
        # more accurately: a numerically stable log-space product
        # so use the below version to be much faster and more numerically stable, GPU-friendly
        # exp_sum_log_impl_result = torch.exp(
        #     torch.sum(torch.log(vals + 1e-12), dim=2)
        # ).nan_to_num(self.nan_replacement)
        after_mask: torch.Tensor = self._apply_mask(
            membership=membership, inplace=True
        )  # update self.applied_mask
        exp_sum_log_func_result = exp_sum_log(after_mask, dim=2).nan_to_num(
            self.nan_replacement
        )
        # assert torch.allclose(exp_sum_log_impl_result, exp_sum_log_func_result)
        return exp_sum_log_func_result

    def _linear_sum_apply_mask(self, membership: Membership) -> Tensor:
        """
        A very efficient resolution strategy for applying the mask to the given fuzzy relation if
        the summation is taken immediately afterward; it is *NOT* mathematically equivalent to the
        product/exp-sum-log technique unless they also are followed by sum(dim=1). This is
        particularly helpful if a TSK product inference engine is utilized, and the fuzzy inference
        is exploiting the softmax that occurs within the calculations.

        Args:
            membership: The membership values.

        Returns:
            The fuzzy relation values.
        """
        # WARNING: this will not work for information with missing data
        # (notice that there is no use of self.nan_replacement)
        membership_shape: Size = membership.degrees.shape
        batch_size, var_count, term_count = (
            membership_shape[0],
            membership_shape[1],
            membership_shape[2],
        )
        applied_mask: torch.Tensor = self.grouped_links(membership=membership)
        # avoiding the above call can lead to significant performance increases
        # applied_mask = self.grouped_links.grouped_links.modules_list[0].logits
        n_rules = applied_mask.shape[-1]
        linear_result = torch.nn.functional.linear(  # pylint: disable=not-callable
            membership.degrees.view(batch_size, var_count * term_count),
            applied_mask.view(var_count * term_count, n_rules).T,
        )
        # the above is equivalent if you apply .sum(dim=1) to PROD result
        # assert torch.allclose(result.sum(dim=1).half(), linear_result.half())
        return linear_result

    # @log_method
    def forward(self, membership: Membership) -> torch.Tensor:
        """
        Apply the n-ary relation to the given memberships.

        Args:
            membership: The membership values to apply the minimum n-ary relation to.

        Returns:
            The minimum membership value, according to the n-ary relation (i.e., which truth values
            to actually consider).
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__} has no defined forward function. Please create a class "
            f"and inherit from {self.__class__.__name__}, or use a predefined class."
        )
