"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

from pathlib import Path
from typing import Any, List, MutableMapping, Tuple, Union

import igraph
import numpy as np
import scipy.sparse as sps
import torch

from fuzzy.sets.membership import Membership
from fuzzy.utils import TorchJitModule, check_path_to_save_torch_module

from .linkage import BinaryLinks, GroupedLinks


class NAryRelation(TorchJitModule):
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
        """
        super().__init__(**kwargs)
        self.device: torch.device = device
        if nan_replacement not in [0.0, 1.0]:
            raise ValueError("The nan_replacement must be either 0.0 or 1.0.")
        self.nan_replacement: float = nan_replacement

        self.matrix = None  # created later (via self._rebuild)
        self.grouped_links: Union[None, GroupedLinks] = (
            None  # created later (via self._rebuild)
        )
        # self.applied_mask: Union[None, torch.Tensor] = (
        #     None  # created later (at the end of the constructor)
        # )
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.indices})"

    def __hash__(self) -> int:
        return hash(self.nan_replacement) + hash(self.device)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NAryRelation) or not isinstance(self, type(other)):
            return False
        applied_mask, other_applied_mask = self.get_mask(), other.get_mask()
        return (
            applied_mask.shape == other_applied_mask.shape
            and torch.allclose(applied_mask, other_applied_mask)
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
    def load(cls, path: Path, device: torch.device) -> "NAryRelation":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Returns:
            None
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
        class_name = state_dict.pop("class_name")

        if "indices" in state_dict:
            indices = state_dict.pop("indices")
            return cls.get_subclass(class_name)(
                *indices,
                device=device,
                nan_replacement=nan_replacement,
            )
        grouped_links: Path = state_dict.pop("grouped_links")
        obj = cls.get_subclass(class_name)(
            device=device,
            grouped_links=GroupedLinks.load(grouped_links, device=device),
            nan_replacement=nan_replacement,
        )
        # add other attributes that may be specific to the subclass
        obj.load_state_dict(state_dict, strict=False)
        return obj

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
        self.applied_mask: torch.Tensor = self.grouped_links(
            membership=membership
        ).to_dense()
        if not self.applied_mask.is_contiguous():
            self.applied_mask = self.applied_mask.contiguous()

        # ORIGINAL ELEMENT-WISE MULTIPLICATION
        # after_mask = membership.degrees.unsqueeze(dim=-1) * self.applied_mask.unsqueeze(
        #     0
        # )
        # MEMORY-EFFICIENT ELEMENT-WISE MULTIPLICATION
        # after_mask = torch.einsum("...i,...ij->...ij", membership.degrees, self.applied_mask)
        result = []
        # split the degrees into chunks to avoid memory issues if the number of variables is large
        # this then splits the batch to individual observation's degree of
        # memberships
        n_chunks: int = (
            membership.degrees.size(0) if membership.degrees.size(1) > 1000 else 1
        )
        for chunk in torch.chunk(membership.degrees, chunks=n_chunks, dim=0):
            after_mask = torch.einsum("...i,...ij->...ij", chunk, self.applied_mask)

            # complement mask adds zeros where the mask is zero, these are not part of the relation
            # nan_to_num replaces nan values with the nan_replacement value
            # (often not needed)
            result.append(
                (
                    after_mask + (1 - self.applied_mask)
                )  # resulting shape is same as after_mask.shape
                # torch.einsum("...ijk,ijk->...ijk", after_mask,
                # 1 - self.applied_mask)  # resulting shape is same as
                # after_mask.shape
                .prod(dim=2, keepdim=False).nan_to_num(self.nan_replacement)
            )
            del after_mask
        return torch.concat(result)

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
