import unittest

import torch
import numpy as np

from fuzzy.sets.continuous.impl import Gaussian
from fuzzy.sets.continuous.membership import Membership
from fuzzy.relations.continuous.t_norm import Minimum, Product
from fuzzy.relations.continuous.n_ary import NAryRelation, Compound

N_TERMS: int = 2
N_VARIABLES: int = 4
N_OBSERVATIONS: int = 3
AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestNAryRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_mf = Gaussian(
            centers=np.array([[i, i + 1] for i in range(N_VARIABLES)]),
            widths=np.array([[(i + 1) / 2, (i + 1) / 3] for i in range(N_VARIABLES)]),
            device=AVAILABLE_DEVICE,
        )
        self.data: np.ndarray = np.array(
            [
                [0.0412, 0.4543, 0.1980, 0.3821],
                [0.9327, 0.5900, 0.1569, 0.6902],
                [0.0894, 0.9433, 0.9903, 0.5800],
            ]
        )

    def test_gaussian_membership(self) -> Membership:
        """
        Although this test is not directly related to the NAryRelation class, and is possibly
        redundant due to Gaussian's unit testing, it is used to double-check that the Gaussian
        membership function is working as intended as these unit tests rely on correct values
        from the Gaussian membership function to work.

        Returns:
            The membership values for the Gaussian membership function.
        """
        membership: Membership = self.gaussian_mf(
            torch.tensor(self.data, dtype=torch.float32, device=AVAILABLE_DEVICE)
        )

        self.assertEqual(membership.degrees.shape[0], N_OBSERVATIONS)
        self.assertEqual(membership.degrees.shape[1], N_VARIABLES)
        self.assertEqual(membership.degrees.shape[2], N_TERMS)

        # check that the membership is correct
        expected_membership_degrees: torch.Tensor = torch.tensor(
            [
                [
                    [9.9323326e-01, 2.5514542e-04],
                    [7.4245834e-01, 4.6277978e-03],
                    [2.3617040e-01, 3.8928282e-04],
                    [1.8026091e-01, 6.3449936e-04],
                ],
                [
                    [3.0816132e-02, 9.6005607e-01],
                    [8.4526926e-01, 1.1410457e-02],
                    [2.2095737e-01, 3.0867624e-04],
                    [2.6347569e-01, 2.1079029e-03],
                ],
                [
                    [9.6853620e-01, 5.7408627e-04],
                    [9.9679035e-01, 8.1074789e-02],
                    [6.3564914e-01, 1.7616944e-02],
                    [2.3128603e-01, 1.3889252e-03],
                ],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(membership.degrees, expected_membership_degrees))
        return membership

    def test_n_ary_relation(self) -> None:
        """
        Test the abstract n-ary relation.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        self.assertRaises(NotImplementedError, n_ary.forward, None)
        # check that the matrix shape is correct
        self.assertEqual(n_ary._coo_matrix.shape, (2, 2))
        # check that the original shape is stored
        self.assertEqual(n_ary._original_shape, (2, 2))
        # matrix size can increase (in-place) for more potential rows (vars) and columns (terms)
        n_ary._coo_matrix.resize(3, 3)
        self.assertEqual(n_ary._coo_matrix.shape, (3, 3))
        # check that the original shape is still kept after resizing
        self.assertEqual(n_ary._original_shape, (2, 2))

    def test_minimum(self) -> None:
        """
        Test the n-ary minimum relation.

        Returns:
            None
        """
        n_ary = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        # test the mask application
        after_mask = n_ary.apply_mask(membership=membership)
        expected_after_mask = torch.tensor(
            [
                [[7.4245834e-01, 2.5514542e-04]],
                [[8.4526926e-01, 9.6005607e-01]],
                [[9.9679035e-01, 5.7408627e-04]],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(after_mask, expected_after_mask))

        # test the forward pass
        min_membership: Membership = n_ary.forward(membership)
        expected_min_values = torch.tensor(
            [[[2.5514542e-04]], [[8.4526926e-01]], [[5.7408627e-04]]],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(min_membership.degrees, expected_min_values))

        # check that it is torch.jit scriptable (currently not working)
        # n_ary_script = torch.jit.script(n_ary)
        #
        # after_mask_script = n_ary_script.apply_mask(membership=membership)
        # self.assertTrue(torch.allclose(after_mask_script, expected_after_mask))
        #
        # min_values_script = n_ary_script.forward(membership)
        # self.assertTrue(torch.allclose(min_values_script, expected_min_values))

    def test_algebraic_product(self) -> None:
        n_ary = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        # test the mask application
        after_mask = n_ary.apply_mask(membership=membership)
        expected_after_mask = torch.tensor(
            [
                [[7.4245834e-01, 2.5514542e-04]],
                [[8.4526926e-01, 9.6005607e-01]],
                [[9.9679035e-01, 5.7408627e-04]],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(after_mask, expected_after_mask))

        # test the forward pass
        prod_membership: Membership = n_ary.forward(membership)
        expected_prod_values = torch.tensor(
            [
                [[7.4245834e-01 * 2.5514542e-04]],
                [[8.4526926e-01 * 9.6005607e-01]],
                [[9.9679035e-01 * 5.7408627e-04]],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(prod_membership.degrees, expected_prod_values))

        # check that it is torch.jit scriptable (currently not working)
        # n_ary_script = torch.jit.script(n_ary)
        #
        # after_mask_script = n_ary_script.apply_mask(membership=membership)
        # self.assertTrue(torch.allclose(after_mask_script, expected_after_mask))
        #
        # min_values_script = n_ary_script.forward(membership)
        # self.assertTrue(torch.allclose(min_values_script, expected_min_values))

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
        )
        self.assertTrue(
            torch.allclose(compound_values.degrees, expected_compound_values)
        )

        # we can then follow it up with another t-norm

        n_ary_next_min = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        min_membership: Membership = n_ary_next_min(compound_values)
        expected_min_values = torch.tensor(
            [
                [[7.4245834e-01 * 2.5514542e-04]],
                [[8.4526926e-01 * 9.6005607e-01]],
                [[9.9679035e-01 * 5.7408627e-04]],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(min_membership.degrees, expected_min_values))
