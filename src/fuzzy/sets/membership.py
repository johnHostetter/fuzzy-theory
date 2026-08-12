"""
The Membership class contains information describing both membership *degrees* and membership *mask*
for some given *elements*. The membership degrees are often the degree of membership, truth,
activation, applicability, etc. of a fuzzy set, or more generally, a concept. The membership mask is
shaped such that it helps filter or 'mask' out membership degrees that belong to fuzzy sets or
concepts that are not actually real. The distinction between the two is made as applying the mask
will zero out membership degrees that are not real, but this might be incorrectly interpreted as
having zero degree of membership to the fuzzy set. By including the elements' information with the
membership degrees and mask, it is possible to keep track of the original elements that were used to
calculate the membership degrees. This is useful for debugging purposes, and it is also useful for
understanding the membership degrees and mask in the context of the original elements. Also, it can
be used in conjunction with the mask to filter out membership degrees that are not real, as well as
assist in performing advanced operations.
"""

# required instead of dataclass for torch.jit.script
from collections import namedtuple
from typing import List, Tuple, Union

import torch

from fuzzy.utils.options.impl.impl_enums import DimensionEnum


class NamedTensor(namedtuple(typename="NamedTensor", field_names=("data", "names"))):
    """
    A tensor paired with a name for each of its dimensions (e.g., "batch", "variable"),
    validated at construction to actually match the tensor's number of dimensions.
    """

    def __new__(
        cls,
        data: torch.Tensor,
        names: Union[List[str], Tuple[str, ...], Tuple[DimensionEnum, ...]],
    ):
        assert isinstance(data, torch.Tensor), "The data must be a torch.Tensor"
        assert data.ndim == len(names), "The data must have the same shape as names"
        return super().__new__(cls, data, names)


# A plain namedtuple, deliberately *not* a subclass with a custom __new__: a
# subclassed namedtuple is exotic enough that torch.compile's Dynamo tracer cannot
# see through it (treats both construction and cross-function attribute access as an
# opaque, untraceable object, forcing a graph break at both points), while a plain
# namedtuple is fully transparent to it. The previous __new__ override's validation
# logic had already been fully commented out, so this is a pure simplification with
# no behavior change - see fuzzy.sets.abstract.FuzzySet.forward for where instances
# are constructed and fuzzy.logic.control.defuzzification.TSK.forward for where an
# incoming instance's fields are accessed across what can become a graph
# boundary.
Membership = namedtuple("Membership", ["degrees", "mask", "formula"], defaults=[""])
# Plain namedtuple() fields carry no type information at all, and torch.jit.script's
# NamedTuple support falls back to assuming every field is a Tensor when it cannot
# determine otherwise (matching degrees/mask, both real Tensors) - without this,
# constructing a Membership with formula=<a str> inside a scripted method (e.g.
# FuzzySet.forward) fails to compile with "Expected a value of type 'Tensor' ...
# for argument 'formula'". Setting __annotations__ directly (rather than switching
# to typing.NamedTuple) keeps Membership a plain namedtuple - see the comment above
# for why that specifically matters for torch.compile's Dynamo tracer.
Membership.__annotations__ = {
    "degrees": torch.Tensor,
    "mask": torch.Tensor,
    "formula": str,
}
Membership.__doc__ = """
The Membership class contains information describing membership *degrees* and membership *mask*
for some given *elements* as well as information regarding how the membership was calculated. The
membership degrees are often the degree of membership, truth, activation, applicability, etc. of a
fuzzy set, or more generally, a concept. The membership mask is shaped such that it helps filter
or 'mask' out membership degrees that belong to fuzzy sets or concepts that are not actually real.

The distinction between the two is made as applying the mask will zero out membership degrees
that are not real, but this might be incorrectly interpreted as having zero degree of
membership to the fuzzy set.

By including the elements' information with the membership degrees and mask, it is possible to
keep track of the original elements that were used to calculate the membership degrees. This
is useful for debugging purposes, and it is also useful for understanding the membership
degrees and mask in the context of the original elements. Also, it can be used in conjunction
with the mask to filter out membership degrees that are not real, as well as assist in
performing advanced operations.

Each membership instance also tracks the *name* of the class (e.g., "Gaussian", "Minimum") that
produced its degrees, via the "formula" field - not the specific instance that computed them, and
not even the class object itself. Storing the producing instance would pin its entire module
(parameters, buffers, membership cache, etc.) alive for as long as any Membership built from it is
reachable; storing the class name (a plain str) is enough to answer "what kind of formula produced
this and what range should its degrees be in" - look it up via TorchJitModule.get_subclass(name) to
get back the actual class and its degree_range (see FuzzySet.degree_range) - while remaining
memory-trivial, and is the only representation torch.jit.script can infer a type for on this
namedtuple's untyped fields (see the __annotations__ assignment above) and torch.compile's Dynamo
tracer can see through (unlike an arbitrary object carrying tensors, or a raw class/type value).
"""
