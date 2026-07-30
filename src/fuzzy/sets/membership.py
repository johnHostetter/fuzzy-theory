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

from collections import namedtuple  # required instead of dataclass for torch.jit.script
from typing import Union, List, Tuple

import torch

from fuzzy.utils.options.impl.impl_enums import DimensionEnum


class NamedTensor(namedtuple(typename="NamedTensor", field_names=("data", "names"))):
    def __new__(
        cls,
        data: torch.Tensor,
        names: Union[List[str], Tuple[str, ...], Tuple[DimensionEnum, ...]],
    ):
        assert isinstance(data, torch.Tensor), "The data must be a torch.Tensor"
        assert data.ndim == len(names), "The data must have the same shape as names"
        return super().__new__(cls, data, names)


class Membership(namedtuple(typename="Membership", field_names=("degrees", "mask"))):
    """
    The Membership class contains information describing both membership *degrees* and
    membership *mask* for some given *elements*. The membership degrees are often the degree of
    membership, truth, activation, applicability, etc. of a fuzzy set, or more generally, a concept.
    The membership  mask is shaped such that it helps filter or 'mask' out membership degrees that
    belong to fuzzy sets or concepts that are not actually real.

    The distinction between the two is made as applying the mask will zero out membership degrees
    that are not real, but this might be incorrectly interpreted as having zero degree of
    membership to the fuzzy set.

    By including the elements' information with the membership degrees and mask, it is possible to
    keep track of the original elements that were used to calculate the membership degrees. This
    is useful for debugging purposes, and it is also useful for understanding the membership
    degrees and mask in the context of the original elements. Also, it can be used in conjunction
    with the mask to filter out membership degrees that are not real, as well as assist in
    performing advanced operations.
    """

    def __new__(cls, degrees: torch.Tensor, mask: torch.Tensor):
        # assert isinstance(
        #     degrees, torch.Tensor
        # ), "The membership degrees must be a torch.Tensor"
        # assert isinstance(
        #     mask, torch.Tensor
        # ), "The membership mask must be a torch.Tensor"
        # assert all(
        #     name is not None for name in degrees.names
        # ), f"All dimensions of the membership degree tensor must be named: {degrees.names}"
        # assert all(
        #     name is not None for name in mask.names
        # ), f"All dimensions of the mask tensor must be named: {mask.names}"
        return super().__new__(cls, degrees, mask)
