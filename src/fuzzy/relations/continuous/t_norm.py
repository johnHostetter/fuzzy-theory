"""
This module contains the implementation of the n-ary t-norm fuzzy relations. These relations
are used to combine multiple membership values into a single value. The minimum and product
relations are implemented here.
"""

from fuzzy.sets.continuous.membership import Membership
from fuzzy.relations.continuous.n_ary import NAryRelation


class Minimum(NAryRelation):
    """
    This class represents the minimum n-ary fuzzy relation. This is a special case of
    the n-ary fuzzy relation where the minimum value is returned.
    """

    def forward(self, membership: Membership) -> Membership:
        """
        Apply the minimum n-ary relation to the given memberships.

        Args:
            membership: The membership values to apply the minimum n-ary relation to.

        Returns:
            The minimum membership, according to the n-ary relation (i.e., which truth values
            to actually consider).
        """
        # first filter out the values that are not part of the relation
        # then take the minimum value of those that remain in the last dimension
        return Membership(
            elements=membership.elements,
            degrees=self.apply_mask(membership=membership)
            .min(dim=-2, keepdim=True)
            .values,
            mask=self.mask,
        )


class Product(NAryRelation):
    """
    This class represents the algebraic product n-ary fuzzy relation. This is a special case of
    the n-ary fuzzy relation where the product value is returned.
    """

    def forward(self, membership: Membership) -> Membership:
        """
        Apply the algebraic product n-ary relation to the given memberships.

        Args:
            membership: The membership values to apply the algebraic product n-ary relation to.

        Returns:
            The algebraic product membership value, according to the n-ary relation
            (i.e., which truth values to actually consider).
        """
        # first filter out the values that are not part of the relation
        # then take the minimum value of those that remain in the last dimension
        return Membership(
            elements=membership.elements,
            degrees=self.apply_mask(membership=membership).prod(dim=-2, keepdim=True),
            mask=self.mask,
        )
