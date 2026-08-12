"""
Test the n-ary relation implementing the generalized Lukasiewicz calculates as expected.
"""

import torch

from fuzzy.relations.t_norm import GeneralizedLukasiewicz
from fuzzy.sets import Membership
from tests.test_relations.test_n_ary import AVAILABLE_DEVICE, TestNAryRelation


class TestGeneralizedLukasiewicz(TestNAryRelation):
    """
    Test the GeneralizedLukasiewicz n-ary relation.
    """

    def test_forward_does_not_crash_and_matches_formula(self) -> None:
        """
        Regression guard: forward() used to reference membership.elements, a field
        that was dropped from Membership (see fuzzy.sets.membership) - every call
        crashed with AttributeError, and this class had no test coverage at all.

        Returns:
            None
        """
        n_ary = GeneralizedLukasiewicz((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        result: Membership = n_ary.forward(membership)

        intermediate_values = n_ary.apply_mask(membership=membership)
        expected_firing_strengths = intermediate_values.sum(dim=1)
        expected_degrees = torch.nn.functional.relu(
            expected_firing_strengths - (membership.degrees.shape[1] - 1)
        )
        self.assertTrue(torch.allclose(result.degrees, expected_degrees))
        self.assertFalse(bool(result.degrees.isnan().any()))

    def test_gradient_flows_above_threshold_and_vanishes_below(self) -> None:
        """
        Golden-value/drift-detection test: no gradient test existed for
        GeneralizedLukasiewicz at all. forward()'s relu(sum(degrees) - (n_vars -
        1)) has a hard zero-gradient region below the threshold - confirms
        degrees receive a real, non-zero gradient when the sum exceeds it
        (n_vars=2, so threshold=1), and exactly (not NaN) zero when it doesn't.

        Returns:
            None
        """
        n_ary = GeneralizedLukasiewicz([(0, 0), (1, 0)], device=AVAILABLE_DEVICE)

        above_threshold = torch.tensor(
            [[[0.9], [0.9]]], device=AVAILABLE_DEVICE, requires_grad=True
        )
        membership = Membership(
            degrees=above_threshold, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE),
            formula="test",
        )
        result = n_ary.forward(membership)
        self.assertFalse(bool((result.degrees == 0).all()))
        result.degrees.sum().backward()
        self.assertFalse(bool(above_threshold.grad.isnan().any()))
        self.assertFalse(bool((above_threshold.grad == 0).all()))

        below_threshold = torch.tensor(
            [[[0.1], [0.1]]], device=AVAILABLE_DEVICE, requires_grad=True
        )
        membership = Membership(
            degrees=below_threshold, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE),
            formula="test",
        )
        result = n_ary.forward(membership)
        self.assertTrue(bool((result.degrees == 0).all()))
        result.degrees.sum().backward()
        self.assertFalse(bool(below_threshold.grad.isnan().any()))
        self.assertTrue(bool((below_threshold.grad == 0).all()))
