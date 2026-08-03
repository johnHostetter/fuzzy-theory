"""
Implements a memoization layer for membership calculations.

Calculating membership degrees is the dominant cost of fuzzification, and it is common for the
same observations to be evaluated more than once against unchanged parameters (e.g., a held-out
tensor evaluated repeatedly, several fuzzy logic controllers sharing one granulation layer,
plotting or area approximations that re-query the same grid). The MembershipCache within stores
the result of such a calculation and returns it verbatim - the very same torch.Tensor, and
therefore the very same autograd graph - whenever it can prove the result is still valid.

The cache is keyed on *identity and version*, never on tensor contents; hashing the contents of
the observations would cost as much as the membership calculation it is meant to avoid. As a
consequence, a cache hit requires the caller to pass the same tensor object again. A freshly
created tensor (e.g., the next mini-batch) always misses, which is the correct answer rather
than a shortcoming.

An entry is only served when all the following hold:

    1. The observations are the same object (compared by identity through a weak reference) and
       have not been mutated in place since (compared by version counter).
    2. Every parameter the calculation depends on is the same object, and has not been mutated
       in place since. This is what makes the cache safe across an optimizer step.
    3. The autograd context matches; a result computed with gradients disabled has no graph
       attached and must never be handed back to a caller that needs gradients.
    4. The entry's autograd graph has not already been consumed by a backward pass. PyTorch
       frees the buffers of a graph once it has been traversed, so re-using such a result
       raises "Trying to backward through the graph a second time". A backward hook evicts the
       entry the moment its graph is used, which is what makes the cache safe within the usual
       forward -> backward -> step training loop.
"""

import weakref
from typing import Any, List, Optional, Tuple

import torch

from .membership import Membership

# A signature identifying the tensors a calculation depended upon; see signature_of().
# A plain List rather than a variadic Tuple[..., ...], since the latter's Ellipsis is not a
# type TorchScript's annotation resolver supports, and this is only ever compared with '==' /
# '!=' (never hashed or used as a dict key), so a List loses nothing here. The third element
# of each tuple is the tensor's requires_grad flag at signature time - see
# signature_of().
ParameterSignature = List[Tuple[int, int, bool]]


def version_of(tensor: torch.Tensor) -> int:
    """
    Read the version counter of a tensor, which PyTorch increments whenever the tensor is
    mutated in place (as an optimizer does when it applies an update).

    Inference tensors do not track a version counter at all, so -1 is reported for them; this
    never compares equal to a real version, meaning entries involving inference tensors are
    simply not re-used.

    Args:
        tensor: The tensor to read the version counter of.

    Returns:
        The version counter of the tensor, or -1 if it does not track one.
    """
    try:
        return tensor._version  # pylint: disable=protected-access
    except RuntimeError:
        # "Inference tensors do not track version counter."
        return -1


def signature_of(tensors: List[torch.Tensor]) -> ParameterSignature:
    """
    Summarize the tensors that a calculation depended upon, such that the summary changes if any
    of those tensors is replaced by another object, is mutated in place, or has its
    requires_grad flag toggled.

    The identity of a tensor is safe to use here (rather than a weak reference) because the
    caller - a torch.nn.Module - holds a strong reference to its own parameters for as long as
    the cache entry can be looked up, so an identifier cannot be recycled behind our back.

    requires_grad is included because it changes the autograd graph a calculation produces
    without changing the tensor's identity or bumping its version counter: freezing a
    parameter (requires_grad_(False)), computing with it, then unfreezing it and reusing the
    same observations would otherwise hand back a cached result whose graph was built with
    that parameter detached - so its gradient would silently stay None forever afterward,
    even though it is trainable again.

    Args:
        tensors: The tensors that a calculation depended upon (e.g., centers and widths).

    Returns:
        A hashable and comparable signature of those tensors.
    """
    return [
        (id(tensor), version_of(tensor), tensor.requires_grad) for tensor in tensors
    ]


class MembershipCacheEntry:  # pylint: disable=too-few-public-methods
    """
    A single memoized membership calculation, along with everything needed to later decide
    whether it is still valid.

    The observations are referenced *weakly* so that a cache entry can never be the reason a
    (potentially large) batch of observations is kept alive. Should the observations be
    collected, the entry can no longer match anything and is discarded.
    """

    __slots__ = (
        "observations_ref",
        "observations_version",
        "observations_requires_grad",
        "parameter_signature",
        "grad_enabled",
        "membership",
        "valid",
    )

    def __init__(
        self,
        observations: torch.Tensor,
        parameter_signature: ParameterSignature,
        membership: Membership,
    ):
        self.observations_ref: "weakref.ref[torch.Tensor]" = weakref.ref(observations)
        self.observations_version: int = version_of(observations)
        # see signature_of() for why requires_grad must be tracked alongside
        # identity/version
        self.observations_requires_grad: bool = observations.requires_grad
        self.parameter_signature: ParameterSignature = parameter_signature
        self.grad_enabled: bool = torch.is_grad_enabled()
        self.membership: Membership = membership
        self.valid: bool = True

    def matches(
        self, observations: torch.Tensor, parameter_signature: ParameterSignature
    ) -> bool:
        """
        Determine whether this entry may be served for the given observations and parameters.

        Args:
            observations: The observations that membership degrees are wanted for.
            parameter_signature: The current signature of the parameters (see signature_of).

        Returns:
            True if the memoized membership is still valid for these arguments.
        """
        return (
            self.valid
            and self.grad_enabled == torch.is_grad_enabled()
            and self.observations_ref() is observations
            and self.observations_version == version_of(observations)
            and self.observations_requires_grad == observations.requires_grad
            and self.parameter_signature == parameter_signature
        )

    def invalidate(self, *_: Any) -> None:
        """
        Mark this entry as no longer usable, and release the memoized membership (and with it
        the autograd graph it holds onto).

        This accepts and ignores arbitrary arguments so that it can be registered directly as a
        backward hook, which is called with the incoming gradient.

        Returns:
            None
        """
        self.valid = False
        self.membership = None


class MembershipCache:
    """
    A small, bounded memo of membership calculations for a single module.

    The cache holds at most 'maxsize' entries and evicts the oldest first. A size of one is the
    classic "remember the last call"; the default of two additionally covers the common pattern
    of alternating between two tensors (e.g., a training batch and a fixed evaluation tensor)
    without the two thrashing each other.

    Nothing is cached while torch.inference_mode is enabled. Tensors created there carry
    restrictions that make them unsafe to hand out later, and inference mode already avoids the
    bookkeeping that caching is meant to save.
    """

    def __init__(self, maxsize: int = 2, enabled: bool = True):
        if maxsize < 1:
            raise ValueError(f"The cache size must be at least 1, but got {maxsize}.")
        self.maxsize: int = maxsize
        self.enabled: bool = enabled
        self._entries: List[MembershipCacheEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(
        self, observations: torch.Tensor, parameter_signature: ParameterSignature
    ) -> Optional[Membership]:
        """
        Retrieve a memoized membership for the given observations, if a valid one is held.

        Args:
            observations: The observations that membership degrees are wanted for.
            parameter_signature: The current signature of the parameters (see signature_of).

        Returns:
            The memoized Membership, or None if the calculation has to be (re)done.
        """
        if not self.enabled or torch.is_inference_mode_enabled():
            return None

        self._discard_unusable()
        for entry in self._entries:
            if entry.matches(observations, parameter_signature):
                return entry.membership
        return None

    def store(
        self,
        observations: torch.Tensor,
        parameter_signature: ParameterSignature,
        membership: Membership,
    ) -> None:
        """
        Memoize a membership calculation.

        If the result carries an autograd graph, a backward hook is attached that evicts this
        entry as soon as that graph is traversed, since the graph cannot be traversed twice.

        Args:
            observations: The observations the membership degrees were calculated for.
            parameter_signature: The signature of the parameters used (see signature_of).
            membership: The resulting membership degrees and mask to memoize.

        Returns:
            None
        """
        if not self.enabled or torch.is_inference_mode_enabled():
            return

        self._discard_unusable()
        entry = MembershipCacheEntry(observations, parameter_signature, membership)

        degrees: torch.Tensor = membership.degrees
        if degrees.requires_grad and degrees.grad_fn is not None:
            # the memoized result is part of an autograd graph, whose buffers are freed once
            # backward traverses it; drop the entry at that point so it is
            # never re-used
            degrees.register_hook(entry.invalidate)

        self._entries.append(entry)
        while len(self._entries) > self.maxsize:
            self._entries.pop(0)  # evict the oldest entry first

    def clear(self) -> None:
        """
        Discard every memoized membership.

        Returns:
            None
        """
        for entry in self._entries:
            entry.invalidate()
        self._entries.clear()

    def _discard_unusable(self) -> None:
        """
        Drop entries that can no longer be served, either because a backward pass consumed them
        or because the observations they were keyed on have been collected.

        Returns:
            None
        """
        self._entries = [
            entry
            for entry in self._entries
            if entry.valid and entry.observations_ref() is not None
        ]
