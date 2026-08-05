"""
Test the n-ary relation implementing the product t-norm calculates as expected.
"""

import unittest
from unittest import mock

import torch

from fuzzy.relations.t_norm import Product
from fuzzy.relations.t_norm import gather_prod as t_norm_gather_prod
from fuzzy.sets import Membership
from tests.test_relations.impl.common import (
    assert_apply_mask_matches_expected,
    assert_matches_expected,
)
from tests.test_relations.test_n_ary import (
    AVAILABLE_DEVICE,
    N_COMPOUNDS,
    N_OBSERVATIONS,
    TestNAryRelation,
)


class TestProduct(TestNAryRelation):
    """
    Test the Product n-ary relation.
    """

    def test_algebraic_product(self) -> None:
        """
        Test the n-ary product operation given a single relation.

        Returns:

        """
        n_ary = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        # test the mask application
        assert_apply_mask_matches_expected(n_ary, membership, AVAILABLE_DEVICE)

        # test the forward pass
        prod_membership: Membership = n_ary.forward(membership)
        assert_matches_expected(
            prod_membership.degrees,
            [
                [7.4245834e-01 * 2.5514542e-04],
                [8.4526926e-01 * 9.6005607e-01],
                [9.9679035e-01 * 5.7408627e-04],
            ],
            AVAILABLE_DEVICE,
        )

        # check that it is torch.jit scriptable (currently not working)
        # n_ary_script = torch.jit.script(n_ary)
        #
        # after_mask_script = n_ary_script.apply_mask(membership=membership)
        # self.assertTrue(torch.allclose(after_mask_script, expected_after_mask))
        #
        # min_values_script = n_ary_script.forward(membership)
        # self.assertTrue(torch.allclose(min_values_script, expected_min_values))

    def test_multiple_indices_passed_as_list(self) -> None:
        """
        Test the Product operation given multiple relations, where some variables are never used
        by those relations. This is a test to ensure that the Product operation can handle
        relations that do not use all variables (i.e., does not wrongly output zeros).

        Returns:
            None
        """
        n_ary = Product(
            [(0, 1), (1, 0)],
            [(1, 1), (2, 1)],
            [(2, 1), (2, 0)],
            [(0, 1), (2, 0)],
            [(1, 1), (0, 1)],
            device=AVAILABLE_DEVICE,
        )
        membership = self.test_gaussian_membership()
        prod_membership: Membership = n_ary(membership)
        expected_prod_values = torch.tensor(
            [
                [
                    membership.degrees[0][0][1].item()
                    * membership.degrees[0][1][0].item(),
                    membership.degrees[0][1][1].item()
                    * membership.degrees[0][2][1].item(),
                    membership.degrees[0][2][1].item()
                    * membership.degrees[0][2][0].item(),
                    membership.degrees[0][0][1].item()
                    * membership.degrees[0][2][0].item(),
                    membership.degrees[0][1][1].item()
                    * membership.degrees[0][0][1].item(),
                ],
                [
                    membership.degrees[1][0][1].item()
                    * membership.degrees[1][1][0].item(),
                    membership.degrees[1][1][1].item()
                    * membership.degrees[1][2][1].item(),
                    membership.degrees[1][2][1].item()
                    * membership.degrees[1][2][0].item(),
                    membership.degrees[1][0][1].item()
                    * membership.degrees[1][2][0].item(),
                    membership.degrees[1][1][1].item()
                    * membership.degrees[1][0][1].item(),
                ],
                [
                    membership.degrees[2][0][1].item()
                    * membership.degrees[2][1][0].item(),
                    membership.degrees[2][1][1].item()
                    * membership.degrees[2][2][1].item(),
                    membership.degrees[2][2][1].item()
                    * membership.degrees[2][2][0].item(),
                    membership.degrees[2][0][1].item()
                    * membership.degrees[2][2][0].item(),
                    membership.degrees[2][1][1].item()
                    * membership.degrees[2][0][1].item(),
                ],
            ],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )
        self.assertEqual(prod_membership.degrees.shape[0], N_OBSERVATIONS)
        self.assertEqual(prod_membership.degrees.shape[1], N_COMPOUNDS)
        self.assertEqual(prod_membership.degrees.shape, expected_prod_values.shape)
        self.assertTrue(
            torch.allclose(prod_membership.degrees.to_dense(), expected_prod_values)
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "the fused Triton path is CUDA-only"
    )
    def test_forward_uses_fused_triton_kernel_when_eligible(self) -> None:
        """
        The fused kernel path (gather-eligible, every variable structurally active,
        no NaN present) must actually be taken when all its preconditions hold, not
        just happen to be available - and its result must still match the general
        apply_mask()+prod() path.

        Returns:
            None
        """
        n_ary = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        self.assertTrue(n_ary._use_gather)  # pylint: disable=protected-access
        self.assertTrue(n_ary._all_active)  # pylint: disable=protected-access
        # exactly 2 variables, matching the relation's own shape - using data with
        # MORE variables than the relation references (e.g. test_gaussian_membership(),
        # 4 variables) would silently trigger an implicit resize() to cover the extra
        # variables, which would make them "inactive" and turn _all_active False,
        # defeating the point of this test
        degrees = torch.tensor(
            [[[0.2, 0.8], [0.6, 0.4]], [[0.9, 0.1], [0.3, 0.7]]],
            device=AVAILABLE_DEVICE,
        )
        mask = torch.ones(2, 2, device=AVAILABLE_DEVICE)
        membership = Membership(degrees=degrees, mask=mask)

        with mock.patch(
            "fuzzy.relations.t_norm.gather_prod", wraps=t_norm_gather_prod
        ) as mocked_gather_prod:
            result = n_ary(membership)
        mocked_gather_prod.assert_called_once()

        # rule is (var0, term1) AND (var1, term0)
        expected_prod_values = torch.tensor(
            [[0.8 * 0.6], [0.1 * 0.3]],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(result.degrees, expected_prod_values))

    @unittest.skipUnless(
        torch.cuda.is_available(), "the fused Triton path is CUDA-only"
    )
    def test_forward_falls_back_when_nan_present(self) -> None:
        """
        A NaN observation must route around the fused kernel entirely (it only
        implements the no-NaN case - see relations/triton_kernels.py), landing on
        the general path, which still produces the documented NaN-poisoning
        behavior.

        Returns:
            None
        """
        n_ary = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        self.assertTrue(n_ary._use_gather)  # pylint: disable=protected-access
        self.assertTrue(n_ary._all_active)  # pylint: disable=protected-access
        degrees = torch.tensor(
            [[[0.2, 0.8], [0.6, 0.4]], [[0.9, 0.1], [0.3, 0.7]]],
            device=AVAILABLE_DEVICE,
        )
        # var 0's term 0 (not the term the rule actually selects, term 1) is NaN -
        # must still poison every rule that touches var 0, per the documented
        # any-term-NaN-poisons-the-variable contract
        degrees[0, 0, 0] = float("nan")
        mask = torch.ones(2, 2, device=AVAILABLE_DEVICE)
        membership_with_nan = Membership(degrees=degrees, mask=mask)

        with mock.patch("fuzzy.relations.t_norm.gather_prod") as mocked_gather_prod:
            result = n_ary(membership_with_nan)
        mocked_gather_prod.assert_not_called()
        # the internal NaN-poisoning is replaced by nan_replacement (default 0.0)
        # before this method returns, so the observable symptom of poisoning having
        # happened is the default-0.0 result, not a literal NaN in the output - had
        # poisoning NOT occurred, this would be the un-poisoned product 0.8 *
        # 0.6 = 0.48
        self.assertAlmostEqual(result.degrees[0].item(), 0.0, places=5)

    def test_forward_falls_back_when_not_all_active(self) -> None:
        """
        A relation where not every variable is used by every rule (_all_active is
        False) must route around the fused kernel, which only implements the
        every-variable-always-active case.

        Returns:
            None
        """
        n_ary = Product([(0, 0)], [(1, 0)], device=AVAILABLE_DEVICE)
        self.assertFalse(n_ary._all_active)  # pylint: disable=protected-access
        degrees = torch.rand(4, 2, 1, device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE)
        )

        with mock.patch("fuzzy.relations.t_norm.gather_prod") as mocked_gather_prod:
            n_ary(membership)
        mocked_gather_prod.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available(), "comparing the fused path against its own fallback"
    )
    def test_fused_and_fallback_paths_agree(self) -> None:
        """
        With TRITON_AVAILABLE forced off, Product.forward() must take the general
        path and produce the same result (forward and gradient) as the fused path
        does when left enabled, on a relation eligible for both.

        Returns:
            None
        """
        torch.manual_seed(0)
        n_vars, n_terms, n_rules, batch_size = 12, 4, 6, 8
        indices = [
            [(v, torch.randint(0, n_terms, (1,)).item()) for v in range(n_vars)]
            for _ in range(n_rules)
        ]
        degrees = (
            torch.rand(batch_size, n_vars, n_terms, device=AVAILABLE_DEVICE) * 0.9
            + 0.05
        )
        mask = torch.ones(n_vars, n_terms, device=AVAILABLE_DEVICE)

        n_ary_fused = Product(*indices, device=AVAILABLE_DEVICE)
        degrees_fused = degrees.clone().requires_grad_(True)
        result_fused = n_ary_fused(Membership(degrees=degrees_fused, mask=mask))
        result_fused.degrees.sum().backward()

        n_ary_fallback = Product(*indices, device=AVAILABLE_DEVICE)
        degrees_fallback = degrees.clone().requires_grad_(True)
        with mock.patch("fuzzy.relations.t_norm.TRITON_AVAILABLE", False):
            result_fallback = n_ary_fallback(
                Membership(degrees=degrees_fallback, mask=mask)
            )
            result_fallback.degrees.sum().backward()

        self.assertTrue(
            torch.allclose(result_fused.degrees, result_fallback.degrees, atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(degrees_fused.grad, degrees_fallback.grad, atol=1e-4)
        )
