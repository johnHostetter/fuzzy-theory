"""
This module contains the implementation of the n-ary t-norm fuzzy relations. These relations
are used to combine multiple membership values into a single value. The minimum and product
relations are implemented here.
"""

from abc import ABC

import torch

from fuzzy.sets.membership import Membership
from fuzzy.relations.n_ary import NAryRelation


class TNorm(NAryRelation, ABC):
    """
    This class represents the abstract n-ary fuzzy t-norm relation. This is a special case of the
    n-ary fuzzy relation where the t-norm operation is assumed. This class is abstract and should
    not be instantiated directly, but all fuzzy t-norm relations should inherit from this class.
    """

    def __str__(self) -> str:
        if len(self.indices) == 1:
            return " AND ".join([f"({i}, {j})" for i, j in self.indices[0]])
        return super().__str__()


class Minimum(TNorm):
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
            mask=self.applied_mask,
        )


class Product(TNorm):
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
            mask=self.applied_mask,
        )


class SoftmaxSum(TNorm):
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
        intermediate_values: torch.Tensor = self.apply_mask(membership=membership)
        # pylint: disable=fixme
        # TODO: these dimensions are possibly not correct, need to be fixed/tested
        firing_strengths = intermediate_values.sum(dim=1)
        max_values, _ = firing_strengths.max(dim=-1, keepdim=True)
        return Membership(
            elements=membership.elements,
            degrees=torch.nn.functional.softmax(firing_strengths - max_values, dim=-1),
            mask=self.applied_mask,
        )


class GeneralizedLukasiewicz(TNorm):
    """
    This class represents the generalized Lukasiewicz n-ary fuzzy relation. This is a special case
    of the n-ary fuzzy relation where the generalized Lukasiewicz value is returned.
    """

    def forward(self, membership: Membership) -> Membership:
        intermediate_values: torch.Tensor = self.apply_mask(membership=membership)
        # pylint: disable=fixme
        # TODO: these dimensions are possibly not correct, need to be fixed/tested
        firing_strengths = intermediate_values.sum(dim=1)
        return Membership(
            elements=membership.elements,
            degrees=torch.nn.functional.relu(
                firing_strengths
                - (membership.elements.shape[-1] - 1)  # subtract # of inputs - 1
            ),
            mask=self.applied_mask,
        )


class SoftmaxMean(TNorm):
    """
    This class represents the softmax mean n-ary fuzzy relation. This is a special case when dealing
    with high-dimensional TSK systems, where the softmax mean is used to leverage Gaussians'
    defuzzification relationship to the softmax function.

    This is particularly useful for when dealing with high-dimensional data, and is considered
    a traditional variant of TSK fuzzy systems on high-dimensional datasets.

    This technique is also known as the "HTSK" method, proposed by Yuqi Cui,
    Dongrui Wu, and Yifan Xu, in 2021. The benefit of this method is that it is able to
    overcome the curse of dimensionality, as the scale of the rule activation no longer depends
    upon the dimensionality; "even in very high dimensional space, assuming the input feature
    vectors are properly pre-processed (z-score or zero-one normalization, etc.), we can still
    guarantee the stability of HTSK." To read more, the paper is titled:

        "Curse of Dimensionality for TSK Fuzzy Neural Networks: Explanation and Solution".
    """

    def forward(self, membership: Membership) -> Membership:
        """
        Calculates the fuzzy compound's applicability using the softmax mean inference engine.
        This is particularly useful for when dealing with high-dimensional data, and is considered
        a traditional variant of TSK fuzzy systems on high-dimensional datasets.

        Args:
            membership: The memberships.

        Returns:
            The applicability of the fuzzy compounds (e.g., fuzzy logic rules).
        """
        intermediate_values: torch.Tensor = self.apply_mask(membership=membership)
        # pylint: disable=fixme
        # TODO: these dimensions are possibly not correct, need to be fixed/tested
        firing_strengths = intermediate_values.mean(
            dim=1
        )  # we take the mean instead of the sum
        max_values, _ = firing_strengths.max(
            dim=-1, keepdim=True
        )  # add this to prevent overflow
        return Membership(
            elements=membership.elements,
            degrees=torch.nn.functional.softmax(firing_strengths - max_values, dim=-1),
            mask=self.applied_mask,
        )
