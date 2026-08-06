"""
Additional code to test and validate the LogGaussian class works as expected.
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.impl import LogGaussian
from fuzzy.sets.impl.gauss_variants.cmf import GaussianKernel
from tests import AVAILABLE_DEVICE


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

    def test_calculate_membership_matches_formula(self) -> None:
        """
        Coverage/regression test: LogGaussian.internal_calculate_membership()'s own
        body (as opposed to Gaussian's override of the same method, which is what
        every other test exercises) had no test coverage - LogGaussian was only
        ever constructed to check its width_multiplier validation above, never
        actually used to compute a membership degree.

        Returns:
            None
        """
        centers = np.array([0.0, 1.0])
        widths = np.array([0.5, 0.5])
        log_gaussian = LogGaussian(
            centers=centers,
            widths=widths,
            device=AVAILABLE_DEVICE,
            gaussian_kernel=GaussianKernel(width_multiplier=2.0),
        )
        observations = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], device=AVAILABLE_DEVICE
        ).unsqueeze(-1)
        degrees = log_gaussian.calculate_membership(observations)

        expected = (
            -1.0
            * torch.pow(observations - log_gaussian.get_centers(), 2)
            / (2.0 * torch.pow(log_gaussian.get_widths(), 2) + 1e-32)
        ).clamp(min=-10, max=0)
        self.assertTrue(torch.allclose(degrees, expected, atol=1e-5))
        # every degree here is at or below the diagonal's exact-center-match case
        # (degree 0, i.e. log(1)), so none should have needed clamping to see this
        # test actually exercise the intended (unclamped) formula
        self.assertFalse(bool((degrees == -10).any()))
