"""
Test the SoftmaxSum and SoftmaxMean n-ary relations calculate as expected. Neither had
any test coverage at all before this - SoftmaxSum was only ever constructed (never
forward()-called) in test_rule.py, and SoftmaxMean was not referenced anywhere.
"""

import torch

from fuzzy.relations.t_norm import SoftmaxMean, SoftmaxSum
from fuzzy.sets import Membership
from tests.test_relations.test_n_ary import AVAILABLE_DEVICE, TestNAryRelation


class TestSoftmaxSum(TestNAryRelation):
    """
    Test the SoftmaxSum n-ary relation.
    """

    def test_forward_matches_formula(self) -> None:
        """
        Returns:
            None
        """
        n_ary = SoftmaxSum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        result: Membership = n_ary.forward(membership)

        intermediate_values = n_ary.apply_mask(membership=membership)
        firing_strengths = intermediate_values.sum(dim=1)
        max_values = firing_strengths.amax(dim=-1, keepdim=True)
        expected_degrees = torch.nn.functional.softmax(
            firing_strengths - max_values, dim=-1
        )
        self.assertTrue(torch.allclose(result.degrees, expected_degrees))
        self.assertFalse(bool(result.degrees.isnan().any()))
        # softmax output must sum to 1 along the rule dimension
        self.assertTrue(
            torch.allclose(
                result.degrees.sum(dim=-1),
                torch.ones(result.degrees.shape[0], device=AVAILABLE_DEVICE),
            )
        )


class TestSoftmaxMean(TestNAryRelation):
    """
    Test the SoftmaxMean (a.k.a. "HTSK") n-ary relation.
    """

    def test_forward_matches_formula(self) -> None:
        """
        Returns:
            None
        """
        n_ary = SoftmaxMean((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        result: Membership = n_ary.forward(membership)

        intermediate_values = n_ary.apply_mask(membership=membership)
        firing_strengths = intermediate_values.mean(dim=1)
        max_values = firing_strengths.amax(dim=-1, keepdim=True)
        expected_degrees = torch.nn.functional.softmax(
            firing_strengths - max_values, dim=-1
        )
        self.assertTrue(torch.allclose(result.degrees, expected_degrees))
        self.assertFalse(bool(result.degrees.isnan().any()))
        self.assertTrue(
            torch.allclose(
                result.degrees.sum(dim=-1),
                torch.ones(result.degrees.shape[0], device=AVAILABLE_DEVICE),
            )
        )

    def test_scale_invariance_to_number_of_variables(self) -> None:
        """
        Regression/design guard: SoftmaxMean's whole documented purpose (the "HTSK"
        method) is that using mean rather than sum keeps the rule activation scale
        stable regardless of input dimensionality - unlike SoftmaxSum, whose
        firing_strengths grow with the number of variables. Confirms the mean
        actually decouples the pre-softmax scale from variable count by comparing
        against a relation over twice as many (duplicated) variables.

        Returns:
            None
        """
        n_ary = SoftmaxMean((0, 0), (1, 0), device=AVAILABLE_DEVICE)
        degrees = torch.rand(4, 2, 2, device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )
        intermediate_values = n_ary.apply_mask(membership=membership)
        firing_strengths = intermediate_values.mean(dim=1)

        # duplicate every variable's degrees (so the "same information" appears
        # twice); the per-rule MEAN firing strength must be unchanged
        doubled_n_ary = SoftmaxMean(
            (0, 0), (1, 0), (2, 0), (3, 0), device=AVAILABLE_DEVICE
        )
        doubled_degrees = torch.cat([degrees, degrees], dim=1)
        doubled_membership = Membership(
            degrees=doubled_degrees, mask=torch.ones(
                4, 2, device=AVAILABLE_DEVICE))
        doubled_intermediate_values = doubled_n_ary.apply_mask(
            membership=doubled_membership
        )
        doubled_firing_strengths = doubled_intermediate_values.mean(dim=1)

        self.assertTrue(
            torch.allclose(
                firing_strengths,
                doubled_firing_strengths,
                atol=1e-5))
