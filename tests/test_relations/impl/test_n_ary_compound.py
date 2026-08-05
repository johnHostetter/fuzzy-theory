"""
Test that we can create a compound n-ary relation to aggregate multiple n-ary relations together.
"""

import torch

from fuzzy.relations.compound import Compound
from fuzzy.relations.t_norm import Minimum, Product
from fuzzy.sets import Membership
from tests.test_relations.impl.common import assert_matches_expected
from tests.test_relations.test_n_ary import AVAILABLE_DEVICE, TestNAryRelation


class TestCompound(TestNAryRelation):
    """
    Test the Compound n-ary relation, which allows the user to compound/aggregate multiple n-ary
    relations together.
    """

    def test_combination_of_t_norms(self) -> None:
        """
        Test we can create a combination of t-norms to reflect more complex compound propositions.

        Returns:
            None
        """
        n_ary_min = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        n_ary_prod = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        t_norm = Compound(n_ary_min, n_ary_prod)
        compound_values = t_norm(membership=membership)
        expected_compound_values = torch.cat(
            [
                n_ary_min(membership=membership).degrees,
                n_ary_prod(membership=membership).degrees,
            ],
            dim=-1,
        ).unsqueeze(dim=-1)
        self.assertTrue(
            torch.allclose(compound_values.degrees, expected_compound_values)
        )

        # we can then follow it up with another t-norm

        n_ary_next_min = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        min_membership: Membership = n_ary_next_min(compound_values)
        assert_matches_expected(
            min_membership.degrees,
            [
                [7.4245834e-01 * 2.5514542e-04],
                [8.4526926e-01 * 9.6005607e-01],
                [9.9679035e-01 * 5.7408627e-04],
            ],
            AVAILABLE_DEVICE,
        )
