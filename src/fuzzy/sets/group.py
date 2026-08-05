"""
This module contains the FuzzySetGroup class, which is a generic and abstract torch.nn.Module
class that contains a torch.nn.ModuleList of FuzzySet objects. The expectation here is
that each FuzzySet may define fuzzy sets of different conventions, such as Gaussian,
Triangular, Trapezoidal, etc. Then, subsequent inference engines can handle these heterogeneously
defined fuzzy sets with no difficulty. Further, this class was specifically designed to incorporate
dynamic addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..utils import NestedTorchJitModule
from ..utils.classes import Loggable
from ..utils.functions import ParameterSignature, signature_of
from .cache import MembershipCache

# from ..utils.functions import log_method
from .membership import Membership


class FuzzySetGroup(NestedTorchJitModule, Loggable):
    """
    A generic and abstract torch.nn.Module class that contains a torch.nn.ModuleList
    of FuzzySet objects. The expectation here is that each FuzzySet may define fuzzy sets of
    different conventions, such as Gaussian, Triangular, Trapezoidal, etc.
    Then, subsequent inference engines can handle these heterogeneously defined fuzzy sets
    with no difficulty. Further, this class was specifically designed to incorporate dynamic
    addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.

    However, this class does *not* carry out any functionality that is necessarily tied to fuzzy
    sets, it is simply named so as this was its intended purpose - grouping fuzzy sets. In other
    words, the same "trick" of using a torch.nn.ModuleList of torch.nn.Module objects applies to
    any kind of torch.nn.Module object.
    """

    def __init__(
        self,
        *args,
        modules_list: Union[None, List[torch.nn.Module]] = None,
        device: torch.device = None,
        cache_membership: bool = True,
        membership_cache_size: int = 2,
        **kwargs,
    ):
        """
        Initialize the FuzzySetGroup object.

        Args:
            *args: Optional positional arguments.
            modules_list: A list of torch.nn.Module objects.
            device: The device to move the FuzzySetGroup object to; if None, the device is
            inferred from the modules_list.
            cache_membership: Whether to memoize this group's forward pass; see
            fuzzy.sets.cache for the conditions under which a memoized result is re-used.
            membership_cache_size: How many distinct observation tensors to memoize at once.
            **kwargs: Optional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if modules_list is None:
            modules_list = []
        # self.create_logger(self.__class__.__name__, debug=False)
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.device = device
        # stored as plain (non-underscore-prefixed) attributes, rather than being
        # consumed only by the cache below, so that get_object_attributes() picks
        # them up and NestedTorchJitModule.save()/load() round-trips them (they are
        # also constructor parameter names, so load() passes them back into __init__
        # instead of silently reverting to the defaults above)
        self.cache_membership = cache_membership
        self.membership_cache_size = membership_cache_size
        # memoizes this group's forward pass (the concatenation of every module's response);
        # see fuzzy.sets.cache for when a result is re-used
        self._membership_cache = MembershipCache(
            maxsize=membership_cache_size, enabled=cache_membership
        )
        # memoizes the centers/widths/mask concatenation built in __getattribute__ below,
        # keyed per attribute on the (id, version) signature of the tensors it was built from;
        # left untyped (rather than Dict[str, Tuple[...]]) since torch.jit.script warns about
        # instance-level generic annotations on an empty container assigned in
        # __init__
        self._attribute_cache = {}

    @torch.jit.ignore
    def clear_membership_cache(self) -> None:
        """
        Discard any memoized forward-pass result and any memoized centers/widths/mask
        concatenation, forcing the next access of either to be recalculated.

        Returns:
            None
        """
        cache: Union[None, MembershipCache] = getattr(
            self, "_membership_cache", None)
        if cache is not None:
            cache.clear()
        attribute_cache: Union[None, Dict] = getattr(
            self, "_attribute_cache", None)
        if attribute_cache is not None:
            attribute_cache.clear()

    def _group_parameter_signature(self) -> Optional[ParameterSignature]:
        """
        Summarize the parameters that every module in this group depends upon, so the
        membership cache can tell when a memoized result has been outdated by a parameter
        changing.

        Each module's own parameter_signature() is preferred over hardcoding
        get_centers()/get_widths()/get_mask() here, since a module may depend on additional
        parameters of its own (e.g. Trapezoidal's plateaus). Hardcoding the trio would let
        such a parameter change go unnoticed, silently serving a stale group-level result even
        though the module's own individual cache would have correctly recomputed it.

        Returns:
            A signature of every module's parameters, or None if some module exposes neither
            parameter_signature() nor the get_centers/get_widths/get_mask trio that fuzzy sets
            are expected to (in which case there is nothing to safely key a memoized result on,
            and caching is skipped).
        """
        signature: ParameterSignature = []
        for module in self.__dict__["_modules"]["modules_list"]:
            get_own_signature = getattr(module, "parameter_signature", None)
            if get_own_signature is not None:
                signature.extend(get_own_signature())
                continue
            get_centers = getattr(module, "get_centers", None)
            get_widths = getattr(module, "get_widths", None)
            get_mask = getattr(module, "get_mask", None)
            if get_centers is None or get_widths is None or get_mask is None:
                return None
            signature.extend(signature_of(
                [get_centers(), get_widths(), get_mask()]))
        return signature

    @torch.jit.ignore
    def _lookup_group_membership(
        self, observations: torch.Tensor
    ) -> Optional[Membership]:
        """
        Retrieve a memoized group-level membership for the given observations, if valid.

        Looked up defensively so a torch.jit.script'ed copy of this module - which does not
        carry the cache over - simply calculates the memberships instead of failing.

        Args:
            observations: The observations that membership degrees are wanted for.

        Returns:
            The memoized Membership, or None if it has to be (re)calculated.
        """
        cache: Union[None, MembershipCache] = getattr(
            self, "_membership_cache", None)
        if cache is None:
            return None
        signature: Optional[ParameterSignature] = self._group_parameter_signature(
        )
        if signature is None:
            return None
        return cache.lookup(observations, signature)

    @torch.jit.ignore
    def _store_group_membership(
        self, observations: torch.Tensor, membership: Membership
    ) -> None:
        """
        Memoize the group-level membership calculated for the given observations.

        Args:
            observations: The observations the membership degrees were calculated for.
            membership: The calculated (concatenated) membership degrees and mask.

        Returns:
            None
        """
        cache: Union[None, MembershipCache] = getattr(
            self, "_membership_cache", None)
        if cache is None:
            return
        signature: Optional[ParameterSignature] = self._group_parameter_signature(
        )
        if signature is None:
            return
        cache.store(observations, signature, membership)

    def _concatenated_module_attribute(self, item: str) -> torch.Tensor:
        """
        Concatenate the given attribute across every module in this group, memoizing
        the result until the underlying tensors change. Backs the centers/widths/mask
        properties below, which are implemented as ordinary @property methods (rather
        than a __getattribute__ override, as this used to be) because torch.compile's
        Dynamo tracer refuses to trace into any torch.nn.Module that defines a custom
        __getattribute__ at all, unconditionally graph-breaking on it - a plain
        @property is traceable.

        Args:
            item: One of "centers", "widths", or "mask".

        Returns:
            The single module's tensor directly if there is only one (nothing to
            concatenate or memoize); otherwise the memoized concatenation.
        """
        modules_list = self.__dict__["_modules"]["modules_list"]
        if len(modules_list) == 0:
            raise ValueError(
                "The torch.nn.ModuleList of FuzzySetGroup is empty.")
        module_attributes: List[torch.Tensor] = [
            getattr(module, f"get_{item}")() for module in modules_list
        ]
        if len(module_attributes) == 1:
            return module_attributes[0]

        signature: ParameterSignature = signature_of(module_attributes)
        attribute_cache: Union[None, Dict[str, Tuple]] = self.__dict__.get(
            "_attribute_cache"
        )
        if attribute_cache is not None:
            cached = attribute_cache.get(item)
            if cached is not None and cached[0] == signature:
                return cached[1]

        concatenated = torch.cat(module_attributes, dim=-1)
        if attribute_cache is not None:
            attribute_cache[item] = (signature, concatenated)
        return concatenated

    @property
    @torch.jit.unused
    def centers(self) -> torch.Tensor:
        """
        Returns:
            Every module's centers, concatenated along the last dimension.
        """
        return self._concatenated_module_attribute("centers")

    @property
    @torch.jit.unused
    def widths(self) -> torch.Tensor:
        """
        Returns:
            Every module's widths, concatenated along the last dimension.
        """
        return self._concatenated_module_attribute("widths")

    @property
    @torch.jit.unused
    def mask(self) -> torch.Tensor:
        """
        Returns:
            Every module's mask, concatenated along the last dimension.
        """
        return self._concatenated_module_attribute("mask")

    # @log_method
    def __hash__(self) -> int:
        _hash: str = ""
        for module in self.modules_list:
            _hash += str(hash(module))
        return hash(_hash)

    # @log_method
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FuzzySetGroup):
            return False
        if len(self.modules_list) != len(other.modules_list):
            return False
        for self_module, other_module in zip(
                self.modules_list, other.modules_list):
            if not self_module == other_module:
                return False
        return True

    # @log_method
    def to(self, device: torch.device, *args, **kwargs) -> "FuzzySetGroup":
        """
        Move the FuzzySetGroup to a different device.

        Args:
            device: The device to move the FuzzySetGroup object to.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The FuzzySetGroup object.
        """
        super().to(device, *args, **kwargs)
        # .to() also accepts a dtype-only call (e.g. .to(torch.float64)); in that case
        # 'device' here is actually a dtype, and self.device must be left alone rather than
        # corrupted with it. Deliberately not resolved through a probe tensor here (unlike
        # DynamicParameterList.to()): that would canonicalize an unindexed device like
        # torch.device("cuda") into an indexed torch.device("cuda", 0), which no longer
        # compares equal to what the caller actually passed.
        if not isinstance(device, torch.dtype):
            self.device = device
        for module in self.modules_list:
            module.to(device)
        # each module already cleared its own membership cache, but this group's own caches
        # (keyed on those modules' tensor identities/versions) were not, and moving a module
        # can replace its parameters' data without bumping a version counter
        self.clear_membership_cache()
        return self

    # @log_method
    def forward(self, observations) -> Membership:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of FuzzySetGroup.
        Expand the FuzzySetGroup if necessary.
        """
        if len(self.modules_list) == 0:
            raise ValueError(
                "The torch.nn.ModuleList of FuzzySetGroup is empty.")

        # modules' responses are membership degrees when modules are FuzzySet

        if len(self.modules_list) == 1:
            # for computational efficiency, return the response from the only
            # module
            return self.modules_list[0](observations)

        # the group-level concatenation is memoized the same way a single fuzzy set's
        # membership is: a hit requires the same observations object and every module's
        # parameters to be unchanged, so it is safe across an optimizer step (see
        # fuzzy.sets.cache). A miss still lets each module serve its own cache.
        cached: Optional[Membership] = self._lookup_group_membership(
            observations)
        if cached is not None:
            return cached

        # this can be computationally expensive, but it is necessary to calculate the responses
        # from all the modules in the torch.nn.ModuleList of FuzzySetGroup
        # ideally this should be done in parallel, but it is not possible with the current
        # implementation; only use this if the torch.nn.Module objects are different
        # module_elements: List[torch.Tensor] = []
        module_memberships: List[torch.Tensor] = (
            []
        )  # the primary response from the module
        module_masks: List[torch.Tensor] = (
            []
        )  # the secondary response denoting module filter
        for module in self.modules_list:
            membership: Membership = module(observations)
            # module_elements.append(membership.elements)
            module_memberships.append(membership.degrees)
            module_masks.append(membership.mask)

        if any(degrees.is_sparse for degrees in module_memberships) and not all(
                degrees.is_sparse for degrees in module_memberships):
            # torch.cat cannot mix sparse and dense layouts; this happens whenever the
            # group holds fuzzy sets with different use_sparse_tensor settings (a legitimate
            # per-variable memory choice, e.g. a high-cardinality variable set sparse next to
            # a low-cardinality one left dense). Densify only the sparse ones so every module
            # keeps its own choice up until this point, and only pay the conversion cost when
            # a mix actually occurs.
            module_memberships = [
                degrees.to_dense() if degrees.is_sparse else degrees
                for degrees in module_memberships
            ]

        result = Membership(
            # elements=torch.cat(module_elements, dim=-1),
            degrees=torch.cat(module_memberships, dim=-1),
            mask=torch.cat(module_masks, dim=-1),
        )
        self._store_group_membership(observations, result)
        return result
