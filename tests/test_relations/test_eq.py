"""
Check that the equality operator works as expected for t-norms.
"""

import unittest

import torch

from fuzzy.relations.t_norm import Minimum, Product


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEqualityOfTNorms(unittest.TestCase):
    """
    Checks that the equality operator works as expected for t-norms.
    """

    def test_unequal_t_norms(self) -> None:
        """
        Test that two different t-norms are not equal, despite having the same elements involved.

        Returns:
            None
        """
        minimum_t_norm = Minimum((0, 0), (1, 1), device=AVAILABLE_DEVICE)
        product_t_norm = Product((0, 0), (1, 1), device=AVAILABLE_DEVICE)
        self.assertNotEqual(minimum_t_norm, product_t_norm)

    def test_equal_t_norms(self) -> None:
        """
        Test that two t-norms are equal when they have the same elements involved.

        Returns:
            None
        """
        minimum_t_norm = Minimum((0, 0), (1, 1), device=AVAILABLE_DEVICE)
        another_minimum_t_norm = Minimum((0, 0), (1, 1), device=AVAILABLE_DEVICE)
        self.assertEqual(minimum_t_norm, another_minimum_t_norm)

    def test_other_not_n_ary(self) -> None:
        """
        Test that a t-norm is not equal to an object that is not an n-ary relation.

        Returns:
            None
        """
        minimum_t_norm = Minimum((0, 0), (1, 1), device=AVAILABLE_DEVICE)
        self.assertNotEqual(minimum_t_norm, 1)
