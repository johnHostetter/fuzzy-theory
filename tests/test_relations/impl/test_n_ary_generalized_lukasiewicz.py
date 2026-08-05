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
