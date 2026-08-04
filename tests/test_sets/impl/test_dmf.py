"""
Additional code to test and validate the Dimension-Dependent fuzzy sets
(GaussianDMF, GaussianNoExpDMF) work as expected.
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.impl.gauss_variants.dmf import GaussianDMF, GaussianNoExpDMF

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDimensionDependent(unittest.TestCase):
    """
    Test and validate the Dimension-Dependent fuzzy sets work as expected, especially
    at the high input dimensionality this class exists for.
    """

    def test_n_inputs_does_not_overflow_at_high_dimensionality(self) -> None:
        """
        Regression guard: n_inputs used to be stored as torch.int8 (max magnitude 127),
        which silently wraps to a negative value at 128+ input dimensions - exactly the
        "high-dimensional" scenario this class exists for (see module docstring) -
        corrupting rho, and with it every membership degree, into NaN.

        Returns:
            None
        """
        n_inputs = 128
        centers = np.zeros((n_inputs, 1))
        widths = np.ones((n_inputs, 1))
        fuzzy_set = GaussianDMF(
            centers=centers,
            widths=widths,
            device=AVAILABLE_DEVICE)

        self.assertEqual(fuzzy_set.n_inputs.item(), n_inputs)
        self.assertFalse(bool(fuzzy_set.rho.isnan().any()))

        # membership at the exact center should be (close to) 1.0, not NaN
        observations = torch.zeros(1, n_inputs, 1, device=AVAILABLE_DEVICE)
        degrees = fuzzy_set.calculate_membership(observations)
        self.assertFalse(bool(degrees.isnan().any()))
        self.assertTrue(
            torch.allclose(
                degrees,
                torch.ones_like(degrees),
                atol=1e-4))

    def test_rho_calculation_matches_below_previous_overflow_threshold(
            self) -> None:
        """
        127 dimensions is the largest value that fit in the previous (buggy) int8
        dtype; confirm the fix does not change behavior for values that already
        worked, and that GaussianNoExpDMF (the sibling subclass) is unaffected too.

        Returns:
            None
        """
        n_inputs = 127
        centers = np.zeros((n_inputs, 1))
        widths = np.ones((n_inputs, 1))
        fuzzy_set = GaussianNoExpDMF(
            centers=centers, widths=widths, device=AVAILABLE_DEVICE
        )
        expected_rho = 1.0 - (
            torch.tensor([745.0]).log() / torch.tensor([float(n_inputs)]).log()
        )
        self.assertTrue(
            torch.allclose(
                fuzzy_set.rho.cpu(),
                expected_rho,
                atol=1e-6))


if __name__ == "__main__":
    unittest.main()
