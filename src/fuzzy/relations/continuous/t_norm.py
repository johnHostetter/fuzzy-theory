"""
This module contains the implementation of the n-ary t-norm fuzzy relations. These relations
are used to combine multiple membership values into a single value. The minimum and product
relations are implemented here.
"""

import torch

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
            .min(dim=-2, keepdim=False)
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
            degrees=self.apply_mask(membership=membership).prod(dim=-2, keepdim=False),
            mask=self.mask,
        )


class SoftmaxSum(NAryRelation):
    """
    This class represents the softmax sum n-ary fuzzy relation. This is a special case when dealing
    with high-dimensional TSK systems, where the softmax sum is used to leverage Gaussians'
    defuzzification relationship to the softmax function.
    """

    def forward(self, membership: Membership) -> Membership:
        """
        Calculates the fuzzy compound's applicability using the softmax sum inference engine.
        This is particularly useful for when dealing with high-dimensional data, and is considered
        a traditional variant of TSK fuzzy stems on high-dimensional datasets.

        Args:
            membership: The memberships.

        Returns:
            The applicability of the fuzzy compounds (e.g., fuzzy logic rules).
        """
        # intermediate_values = self.calc_intermediate_input(antecedents_memberships)
        firing_strengths = membership.sum(dim=1)
        max_values, _ = firing_strengths.max(dim=-1, keepdim=True)
        return torch.nn.functional.softmax(firing_strengths - max_values, dim=-1)
