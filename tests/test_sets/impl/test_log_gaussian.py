"""
Additional code to test and validate the LogGaussian class works as expected.
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.impl import LogGaussian
from fuzzy.sets.impl.gauss_variants.cmf import GaussianKernel
from tests import AVAILABLE_DEVICE


def gaussian_numpy_with_width_multiplier(
    element: np.ndarray, center: np.ndarray, sigma: np.ndarray, width_multiplier: float
):
    """
        Gaussian membership function parameterized by an explicit width_multiplier
        (LogGaussian's "width_multiplier * sigma^2" convention, as opposed to
        test_gaussian.py's gaussian_numpy, which is hardcoded to the width_multiplier
        = 1.0 case). Implemented in NumPy, independently of the PyTorch formula, and
        used to verify LogGaussian.calculate_membership() (which returns the raw
        log-space value, i.e. this function's result before its own exp()) via
        exp(LogGaussian's output).

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Gaussian fuzzy set.
        sigma: The width of the Gaussian fuzzy set.
        width_multiplier: The width multiplier applied to sigma^2 in the
            denominator.

    Returns:
        The membership degree of 'element'.
    """
    return np.exp(
        -1.0 * np.power(element - center, 2) / (width_multiplier * np.power(sigma, 2))
    )


class TestLogGaussian(unittest.TestCase):
    """
    Test and validate the LogGaussian class works as expected.
    """

    def test_bad_width_multiplier(self) -> None:
        """
        Test that the width multiplier must be either 1.0 or 2.0.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            LogGaussian,
            centers=np.ones(1),
            widths=np.ones(1),
            device=AVAILABLE_DEVICE,
            gaussian_kernel=GaussianKernel(
                width_multiplier=0.0,
                slope_multiplier=1.0,
            ),
        )
        self.assertRaises(
            ValueError,
            LogGaussian,
            centers=np.ones(1),
            widths=np.ones(1),
            device=AVAILABLE_DEVICE,
            gaussian_kernel=GaussianKernel(
                width_multiplier=3.0,
                slope_multiplier=1.0,
            ),
        )
        self.assertRaises(
            ValueError,
            LogGaussian,
            centers=np.ones(1),
            widths=np.ones(1),
            device=AVAILABLE_DEVICE,
            gaussian_kernel=GaussianKernel(
                width_multiplier=-1.0,
                slope_multiplier=1.0,
            ),
        )

    def test_calculate_membership_matches_independent_numpy_reference(self) -> None:
        """
        Golden-value/drift-detection test: LogGaussian.internal_calculate_membership()
        returns the raw log-space value (no exp()), so exp() of its output is
        cross-checked against an independent NumPy Gaussian reimplementation
        (gaussian_numpy_with_width_multiplier, not the same code checking itself) -
        the same pattern used for Gaussian/Triangular/Trapezoidal/Lorentzian.
        LogGaussian was previously only ever constructed to check its
        width_multiplier validation, never actually used to compute a membership
        degree.

        Returns:
            None
        """
        centers = np.array([0.0, 1.0])
        widths = np.array([0.5, 0.5])
        width_multiplier = 2.0
        log_gaussian = LogGaussian(
            centers=centers,
            widths=widths,
            device=AVAILABLE_DEVICE,
            gaussian_kernel=GaussianKernel(width_multiplier=width_multiplier),
        )
        observations = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], device=AVAILABLE_DEVICE
        ).unsqueeze(-1)
        degrees = log_gaussian.calculate_membership(observations)

        expected = gaussian_numpy_with_width_multiplier(
            observations.cpu().detach().numpy(),
            centers,
            widths,
            width_multiplier,
        )
        self.assertTrue(
            np.allclose(torch.exp(degrees).cpu().detach().numpy(), expected, atol=1e-5)
        )
        # every degree here is at or below the diagonal's exact-center-match case
        # (degree 0, i.e. log(1)), so none should have needed clamping to see this
        # test actually exercise the intended (unclamped) formula
        self.assertFalse(bool((degrees == -10).any()))
