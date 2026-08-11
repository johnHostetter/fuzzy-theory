"""
Test the n-ary relation implementing the minimum t-norm calculates as expected.
"""

import numpy as np
import torch

from fuzzy.relations.t_norm import Minimum
from fuzzy.sets import FuzzySet, FuzzySetGroup, Gaussian, Membership
from tests.test_relations.impl.common import (
    assert_apply_mask_matches_expected,
    assert_matches_expected,
)
from tests.test_relations.test_n_ary import AVAILABLE_DEVICE, TestNAryRelation


class TestMinimum(TestNAryRelation):
    """
    Test the Minimum n-ary relation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypercube = FuzzySetGroup(
            modules_list=[
                FuzzySet.stack(
                    [
                        Gaussian(
                            centers=np.array([-1, 0.0, 1.0]),
                            widths=np.array([1.0, 1.0, 1.0]),
                            device=AVAILABLE_DEVICE,
                        ),
                        Gaussian(
                            centers=np.array([-1.0, 0.0, 1.0]),
                            widths=np.array([1.0, 1.0, 1.0]),
                            device=AVAILABLE_DEVICE,
                        ),
                    ]
                )
            ]
        )

    def test_minimum(self) -> None:
        """
        Test the n-ary minimum operation given a single relation.

        Returns:
            None
        """
        n_ary = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        # test the mask application
        assert_apply_mask_matches_expected(n_ary, membership, AVAILABLE_DEVICE)

        # test the forward pass
        min_membership: Membership = n_ary.forward(membership)
        assert_matches_expected(
            min_membership.degrees,
            [[2.5514542e-04], [8.4526926e-01], [5.7408627e-04]],
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

    def test_str_single_relation_and_compound(self) -> None:
        """
        Coverage/regression test: TNorm.__str__() has two branches - a single
        relation is rendered as "(var, term) AND (var, term) ...", while a compound
        (multiple relations passed to one instance) falls back to the generic
        NAryRelation.__str__() - neither had direct test coverage.

        Returns:
            None
        """
        single = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        self.assertEqual("(0, 1) AND (1, 0)", str(single))

        compound = Minimum([(0, 0), (1, 0)], [(0, 1), (1, 1)], device=AVAILABLE_DEVICE)
        self.assertEqual(f"Minimum({compound.indices})", str(compound))

    def test_multiple_indices_passed_as_list(self) -> None:
        """
        Test the Minimum operation given multiple relations, where some variables are never used
        by those relations. This is a test to ensure that the Minimum operation can handle
        relations that do not use all variables (i.e., does not wrongly output zeros).

        Returns:
            None
        """
        input_data: torch.Tensor = torch.tensor(
            [
                [0.27, -0.75],
                [3.0, -0.1],
                [-0.567, -1.87],
                [0.334, 0.996],
            ],
            device=AVAILABLE_DEVICE,
        )

        minimum = Minimum(
            [(0, 0), (1, 0)],
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
            [(0, 1), (1, 1)],
            [(0, 1), (1, 2)],
            device=AVAILABLE_DEVICE,
        )

        membership: Membership = self.hypercube(input_data)
        min_membership: Membership = minimum(membership)
        expected_degrees = torch.tensor(
            [
                [
                    1.99308798e-01,
                    1.99308798e-01,
                    9.29693758e-01,
                    5.69782794e-01,
                    4.67706248e-02,
                ],
                [
                    1.12535176e-07,
                    1.12535176e-07,
                    1.23409802e-04,
                    1.23409802e-04,
                    1.23409802e-04,
                ],
                [
                    4.69118446e-01,
                    3.02911401e-02,
                    4.69118446e-01,
                    3.02911401e-02,
                    2.64703733e-04,
                ],
                [
                    1.86107438e-02,
                    1.68713033e-01,
                    1.86107438e-02,
                    3.70828360e-01,
                    8.94441307e-01,
                ],
            ],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )

        self.assertTrue(
            torch.allclose(min_membership.degrees.to_dense(), expected_degrees)
        )

    def test_gradient_reaches_real_input_when_tied_with_phantom_variable(
        self,
    ) -> None:
        """
        Regression guard: a variable not referenced at all by a given rule (e.g.
        rule 1 below never mentions var0) is forced to the shared masking
        machinery's identity constant, 1.0 - correct for the forward value, but
        whenever a rule's real input also equals exactly 1.0 (a fully-satisfied
        rule - routine as training converges), torch.min()'s tie-breaking used to
        be free to pick that constant as the argmin instead of the real input,
        silently killing the gradient. Confirmed to be index-order-dependent: the
        phantom variable (var0) is placed *before* the real one (var1) here
        specifically because that ordering used to trigger the bug (the other
        ordering happened to work by coincidence, since ties go to the first
        occurrence).

        Returns:
            None
        """
        # rule 0 uses var0 and var1; rule 1 uses ONLY var1 - var0 is a phantom
        # (unreferenced) variable for rule 1, and comes before var1 (the real
        # input) along the vars dimension
        n_ary = Minimum([(0, 0), (1, 0)], [(1, 1)], device=AVAILABLE_DEVICE)
        degrees = torch.tensor(
            [[[0.4, 0.4], [1.0, 1.0]]], device=AVAILABLE_DEVICE, requires_grad=True
        )
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )

        result = n_ary.forward(membership)
        self.assertTrue(
            torch.allclose(
                result.degrees,
                torch.tensor([[0.4, 1.0]], device=AVAILABLE_DEVICE),
            )
        )

        result.degrees.sum().backward()
        # var1 (index 1), term1 - rule 1's only real input, tied at exactly 1.0
        # with var0's (index 0) phantom-forced constant - must receive the
        # gradient, not the constant
        self.assertEqual(1.0, degrees.grad[0, 1, 1].item())
