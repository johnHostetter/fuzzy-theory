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

    def test_forward_matches_hand_computed_golden_values(self) -> None:
        """
        Golden-value/drift-detection test: unlike test_forward_matches_formula
        (which rebuilds "expected" from the same sum+softmax steps forward() itself
        performs, so it only catches a divergence between forward() and this test's
        own copy of that logic, not a defect common to both), this uses a small,
        hand-designed input where the two rules' firing strengths and softmax
        outputs are computed independently (by hand, not by calling any of
        forward()'s own building blocks) and pinned as literal expected values.

        Two variables, two terms each, two rules: rule A uses (var0, term0) and
        (var1, term0); rule B uses (var0, term1) and (var1, term1). degrees:
        var0 = [0.6, 0.4], var1 = [0.3, 0.7]. Since each rule uses exactly one term
        per variable, the per-variable/per-rule product collapses to that term's
        degree directly (e.g. rule A's var0 contribution is just 0.6), so the sum
        per rule is: rule A = 0.6 + 0.3 = 0.9, rule B = 0.4 + 0.7 = 1.1. softmax
        (shifted by the max, 1.1) of [0.9, 1.1] is [0.45016600, 0.54983400].

        Returns:
            None
        """
        n_ary = SoftmaxSum([(0, 0), (1, 0)], [
                           (0, 1), (1, 1)], device=AVAILABLE_DEVICE)
        degrees = torch.tensor(
            [[[0.6, 0.4], [0.3, 0.7]]], device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )
        result: Membership = n_ary.forward(membership)
        expected_degrees = torch.tensor(
            [[0.45016600, 0.54983400]], device=AVAILABLE_DEVICE
        )
        self.assertTrue(
            torch.allclose(
                result.degrees,
                expected_degrees,
                atol=1e-6))

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

    def test_forward_matches_hand_computed_golden_values(self) -> None:
        """
        Golden-value/drift-detection test: see TestSoftmaxSum's equivalent test for
        why this is a meaningfully independent check versus test_forward_matches_
        formula's tautological one. Same hand-designed input (see that docstring for
        the derivation) - the only difference is the per-rule reduction is a mean
        rather than a sum: rule A = (0.6 + 0.3) / 2 = 0.45, rule B =
        (0.4 + 0.7) / 2 = 0.55. softmax (shifted by the max, 0.55) of [0.45, 0.55]
        is [0.47502081, 0.52497919].

        Returns:
            None
        """
        n_ary = SoftmaxMean([(0, 0), (1, 0)], [
                            (0, 1), (1, 1)], device=AVAILABLE_DEVICE)
        degrees = torch.tensor(
            [[[0.6, 0.4], [0.3, 0.7]]], device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )
        result: Membership = n_ary.forward(membership)
        expected_degrees = torch.tensor(
            [[0.47502081, 0.52497919]], device=AVAILABLE_DEVICE
        )
        self.assertTrue(
            torch.allclose(
                result.degrees,
                expected_degrees,
                atol=1e-6))

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
