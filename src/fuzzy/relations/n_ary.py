"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Tuple, Union

import igraph
import numpy as np
import scipy.sparse as sps
import torch

from fuzzy.sets.membership import Membership
from fuzzy.utils import TorchJitModule

from ..utils.classes import Loggable

# , log_classmethod, log_func, log_method
from ..utils.functions import exp_sum_log, module_class
from .linkage import BinaryLinks, GroupedLinks

# from line_profiler import profile


# Deciding "is there any NaN in this tensor at all?" on a CUDA tensor requires either
# a device sync (to pull the reduced boolean back to Python) or paying for the
# NaN-handling unconditionally. Calibration (see benchmarks/) showed a CUDA
# tensor.any()-then-bool() sync costs roughly 400-750us in practice, regardless of the
# tensor's size - far more than a bare torch.cuda.synchronize() (~3us) - almost
# certainly the cost of the reduction kernel plus the blocking scalar readback
# together, not the sync primitive itself. Meanwhile, unconditionally doing the (cheap)
# NaN-handling scales linearly with tensor size. Below this many elements, paying for
# NaN-handling unconditionally is cheaper than syncing to find out whether it is
# needed; above it, the sync-and-maybe-skip approach wins. selected.numel() is known
# from tensor shape metadata alone, so checking it costs nothing on the GPU. This
# threshold is an empirically-calibrated heuristic (measured on one GPU) rather than a
# theoretically derived constant, so it may not sit exactly at the true crossover on
# different hardware - but the crossover exists on any GPU where a sync has a
# meaningfully higher fixed cost than issuing a same-sized elementwise kernel, which is
# generally true.
GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL: int = 4_000_000


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
        self._apply_mask_func: Callable[
            [Membership], Tuple[torch.Tensor, torch.Tensor]
        ] = self._cache_apply_mask_func()
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

        self._cached_links: Union[None, torch.Tensor] = None
        self._cached_links_complement: Union[None, torch.Tensor] = None
        self._cached_links_bool: Union[None, torch.Tensor] = None

        self._use_gather: bool = False
        self._gather_indices: Union[None, torch.Tensor] = None
        self._gather_active: Union[None, torch.Tensor] = None
        self._all_active: bool = False
        self._cached_mask: Union[None, torch.Tensor] = None

        self._precompute_gather_indices()

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
        self.invalidate_links_cache()
        if self._use_gather:
            self._gather_indices = self._gather_indices.to(device)
            self._gather_active = self._gather_active.to(device)
            self._cached_mask = self._cached_mask.to(device)
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
            grouped_links_dir: Path = (
                path / f"grouped_links:{module_class(self.grouped_links)}"
            )
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
        path.mkdir(parents=True, exist_ok=True)
        state_dict: MutableMapping[str, Any] = self._state_dict(path=path)
        torch.save(state_dict, path / "state_dict.pt")
        return state_dict

    @classmethod
    # @log_classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
    ) -> "NAryRelation":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Args:
            path: The path to load the t-norm relation.
            device: The physical device to load the t-norm relation onto.

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
        kwargs: Dict[str, Any] = {
            "nan_replacement": state_dict.pop("nan_replacement"),
            "device": device,
        }

        fuzzy_cls = cls.get_subclass(class_name=state_dict.pop("class_name"))

        if "indices" in state_dict:
            indices = state_dict.pop("indices")
            return fuzzy_cls(*indices, **kwargs)

        grouped_links: Path = state_dict.pop("grouped_links")
        kwargs["grouped_links"] = GroupedLinks.load(grouped_links, device=device)

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
        # re-create self.graph (has to happen after self.grouped_links is
        # created)
        self.create_igraph()
        self._precompute_gather_indices()

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
        self.invalidate_links_cache()

    def invalidate_links_cache(self) -> None:
        """
        Invalidate the cached links tensors. Must be called after any change to
        the underlying links (e.g., resize, device move, Gumbel-Softmax resample).
        """
        self._cached_links = None
        self._cached_links_complement = None
        self._cached_links_bool = None

    def _links_are_cacheable(self) -> bool:
        """
        Whether this relation's links are safe to cache indefinitely (until an explicit
        invalidate_links_cache() call).

        Only BinaryLinks are safe: its forward() ignores its argument entirely and always
        returns the same fixed tensor, so "compute once, reuse forever" is correct. Any
        other module type in grouped_links.modules_list - such as a stochastic,
        Gumbel-Softmax-resampled logits module (see GroupedLinks' own docstring) - could
        legitimately produce a *different* result on every call, and this cache has no way
        to detect that on its own; it can only be told to forget via
        invalidate_links_cache(). Recomputing every call for anything other than
        BinaryLinks keeps the cache safe by construction rather than relying on every
        future caller remembering to invalidate it.

        Returns:
            True if every module backing this relation's links is a BinaryLinks.
        """
        if self.grouped_links is None or not hasattr(
            self.grouped_links, "modules_list"
        ):
            return False
        return all(
            isinstance(module, BinaryLinks)
            for module in self.grouped_links.modules_list
        )

    def _ensure_links_cache(self, membership: Membership) -> None:
        """
        Lazily compute and cache the links tensor and its derived forms.
        """
        if self._cached_links is not None and self._links_are_cacheable():
            return
        links = self.grouped_links(membership=membership)
        self._cached_links = links
        self._cached_links_bool = links.bool()
        self._cached_links_complement = 1 - links

    def _apply_mask(self, membership: Membership) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the after-mask tensor and the raw applied mask from the GroupedLinks using
        the given Membership object.

        Args:
            membership: The Membership object to use in determining the applied mask.

        Returns:
            The after-mask tensor, and the applied mask based on the given membership.
        """
        self._ensure_links_cache(membership)
        applied_mask = self._cached_links
        after_mask = membership.degrees.unsqueeze(-1) * applied_mask
        return after_mask + self._cached_links_complement, applied_mask

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
        return self._apply_mask_with_mask(membership)[0]

    def _apply_mask_with_mask(
        self, membership: Membership
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The implementation behind apply_mask(), also returning the raw mask that was
        applied. TNorm.forward() implementations (see relations/t_norm.py) need both
        values to build their returned Membership - they used to get the second one by
        reading self.applied_mask right after calling apply_mask(), but mutating a
        module attribute mid-forward is incompatible with torch.compile(fullgraph=True)
        whenever that forward() runs inside torch.utils.checkpoint (Dynamo forbids
        in-place attribute mutation inside a checkpoint's traced subgraph), so it is
        returned directly instead.

        Args:
            membership: The membership values to apply the n-ary relation to.

        Returns:
            The masked membership values, and the raw applied mask.
        """
        membership_shape: torch.Size = membership.degrees.shape
        if self.grouped_links.shape[:-1] != membership_shape[1:]:
            self.resize(*membership_shape[1:])

        return self._apply_mask_func(membership)

    def _cache_apply_mask_func(
        self,
    ) -> Callable[[Membership], Tuple[torch.Tensor, torch.Tensor]]:
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

    def _precompute_gather_indices(self) -> None:
        """
        Precompute gather indices for an optimized mask application that avoids
        the O(batch * vars * terms * rules) intermediate tensor. Only applicable
        when all links are BinaryLinks and each (variable, rule) pair has at most
        one active term.
        """
        self._use_gather = False
        if self.grouped_links is None or not hasattr(
            self.grouped_links, "modules_list"
        ):
            return
        if self.method not in (NAryMaskMethods.PROD, NAryMaskMethods.EXP_SUM_LOG):
            return

        all_binary = all(
            isinstance(m, BinaryLinks) for m in self.grouped_links.modules_list
        )
        if not all_binary:
            return

        mask = self.grouped_links(membership=None)
        terms_per_var_rule = mask.sum(dim=1)
        if terms_per_var_rule.max().item() > 1:
            return

        self._use_gather = True
        self._gather_indices = mask.to(torch.long).argmax(dim=1)
        self._gather_active = terms_per_var_rule.bool()
        self._all_active = bool(self._gather_active.all().item())
        self._cached_mask = mask
        self._apply_mask_func = self._gather_apply_mask

    def _gather_apply_mask(
        self, membership: Membership
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized mask application using torch.gather. Produces a (batch, vars, rules)
        tensor directly instead of materializing the full (batch, vars, terms, rules)
        intermediate, reducing memory by a factor of n_terms.

        This must reproduce _prod_apply_mask's NaN behaviour exactly, not just its
        non-NaN values, since which of the two runs is an invisible implementation
        detail (chosen automatically by _precompute_gather_indices). _prod_apply_mask's
        formula - degrees * mask + (1 - mask), then a product over the term dimension -
        means IEEE754's NaN * 0 = NaN causes a NaN in *any* term of a variable to poison
        *every* rule that touches that variable at all, including rules that select a
        different term of it, and even rules that do not use that variable's terms in
        their antecedent at all (every term, active or not, participates in the product).
        A plain torch.gather of only the selected term would miss all of that
        propagation, so it is reconstructed explicitly below.
        """
        applied_mask = self._cached_mask
        degrees = membership.degrees
        batch_size = degrees.shape[0]
        idx = self._gather_indices.unsqueeze(0).expand(batch_size, -1, -1)
        selected = torch.gather(degrees, dim=2, index=idx)

        # a NaN anywhere among a variable's terms must poison every rule that variable
        # participates in (structurally active or not) - see docstring above
        any_nan_per_variable = degrees.isnan().any(dim=2, keepdim=True)
        if (
            not torch.compiler.is_compiling()
            and selected.numel() > GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL
        ):
            # large tensor: sync once to find out whether there is anything to do,
            # and skip the (comparatively expensive at this size) NaN-handling
            # entirely when there isn't - see
            # GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL. Guarded off entirely while
            # compiling: is_compiling() is a compile-time constant, so Dynamo prunes
            # this branch rather than tracing into the graph-breaking bool(...any())
            # sync below, always taking the small-tensor (unconditional) path
            # instead.
            has_nan = bool(any_nan_per_variable.any())
            if has_nan:
                selected = torch.where(
                    any_nan_per_variable.expand_as(selected),
                    torch.full_like(selected, float("nan")),
                    selected,
                )
        else:
            # small tensor: skip the sync entirely and always do the (cheap at this
            # size) NaN-handling unconditionally
            has_nan = True
            selected = torch.where(
                any_nan_per_variable.expand_as(selected),
                torch.full_like(selected, float("nan")),
                selected,
            )

        if not self._all_active:
            # Preserve IEEE NaN propagation: inactive entries where the gathered
            # degree is NaN must stay NaN (matching NaN * 0 + 1 = NaN behavior),
            # while inactive entries with valid degrees become 1.0 (product
            # identity).
            selected = torch.where(
                self._gather_active.unsqueeze(0) | selected.isnan(),
                selected,
                torch.ones(1, device=degrees.device, dtype=degrees.dtype),
            )

        if not has_nan:
            # only reachable from the large-tensor branch above; nan_to_num is a
            # no-op away from NaN/inf, and has_nan already proves `selected` is
            # NaN-free (it is only ever assembled from `degrees` entries and the
            # constant 1.0 above, neither of which introduces a NaN when has_nan is
            # False) - skipping it avoids a full elementwise pass over the tensor,
            # which profiling showed was the single largest cost in the rule engine
            # for FLCs with many input variables.
            return selected, applied_mask
        return selected.nan_to_num(self.nan_replacement), applied_mask

    def _prod_apply_mask(
        self, membership: Membership
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The default resolution strategy for applying the mask to the given fuzzy relation.

        Args:
            membership: The membership values.

        Returns:
            The fuzzy relation values.
        """
        # torch.prod on large dims can be slow and non-fusible
        after_mask, applied_mask = self._apply_mask(membership=membership)
        prod_result = after_mask.prod(dim=2, keepdim=False).nan_to_num(
            self.nan_replacement
        )
        return prod_result, applied_mask

    def _exp_sum_log_apply_mask(
        self, membership: Membership
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        after_mask, applied_mask = self._apply_mask(membership=membership)
        exp_sum_log_func_result = exp_sum_log(after_mask, dim=2).nan_to_num(
            self.nan_replacement
        )
        # assert torch.allclose(exp_sum_log_impl_result, exp_sum_log_func_result)
        return exp_sum_log_func_result, applied_mask

    def _linear_sum_apply_mask(
        self, membership: Membership
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A very efficient resolution strategy for applying the mask to the given fuzzy relation if
        the summation is taken immediately afterward; it is *NOT* mathematically equivalent to the
        product/exp-sum-log technique unless they also are followed by sum(dim=1). This is
        particularly helpful if a TSK product inference engine is utilized, and the fuzzy inference
        is exploiting the softmax that occurs within the calculations.

        Args:
            membership: The membership values.

        Returns:
            The fuzzy relation values, and the applied mask reshaped to
            (var_count * term_count, n_rules) - the same 2D view used internally
            for the linear combination below, rather than the raw
            (var_count, term_count, n_rules) shape PROD/EXP_SUM_LOG/gather return.
        """
        membership_shape: torch.Size = membership.degrees.shape
        batch_size, var_count, term_count = (
            membership_shape[0],
            membership_shape[1],
            membership_shape[2],
        )
        self._ensure_links_cache(membership)
        # BinaryLinks stores its links as int8; torch.nn.functional.linear requires both
        # operands to share a dtype, so without this cast this method raises
        # "expected mat1 and mat2 to have the same dtype" for any real (non-float) links -
        # a bug that had gone unnoticed because no test exercised this method
        # before
        applied_mask = self._cached_links.to(dtype=membership.degrees.dtype)
        n_rules = applied_mask.shape[-1]
        reshaped_mask = applied_mask.view(var_count * term_count, n_rules)
        linear_result = torch.nn.functional.linear(  # pylint: disable=not-callable
            membership.degrees.view(batch_size, var_count * term_count),
            reshaped_mask.T,
        )
        return linear_result, reshaped_mask

    # @log_method
    def forward(self, membership: Membership) -> Membership:
        """
        Apply the n-ary relation to the given memberships.

        Args:
            membership: The membership values to apply the n-ary relation to.

        Returns:
            The resulting membership values, according to the n-ary relation (i.e., which truth
            values to actually consider).
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__} has no defined forward function. Please create a class "
            f"and inherit from {self.__class__.__name__}, or use a predefined class."
        )
