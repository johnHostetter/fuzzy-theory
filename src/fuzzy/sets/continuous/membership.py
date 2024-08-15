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

from collections import namedtuple


class Membership(
    namedtuple(typename="Membership", field_names=("elements", "degrees", "mask"))
):
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
