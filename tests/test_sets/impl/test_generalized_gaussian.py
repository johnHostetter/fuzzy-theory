"""
Additional code to test and validate the GeneralizedGuassian class works as expected.
General FuzzySet behavior (construction, gradients, device moves, plotting) is already
exercised generically by test_impl.py's all_subclasses(FuzzySet) sweeps.
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.impl.gauss_variants.cmf import (GaussianKernel,
                                                GeneralizedGuassian)
from tests import AVAILABLE_DEVICE


class TestGeneralizedGuassian(unittest.TestCase):
    """
    Test and validate the GeneralizedGuassian class works as expected.
    """

    def test_negative_width_multiplier_raises(self) -> None:
        """
        Coverage/regression test: the width_multiplier < 0.0 guard had no test
        coverage at all - no existing test constructs a GeneralizedGuassian with a
        non-default (and specifically, invalid) GaussianKernel.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            GeneralizedGuassian,
            centers=np.ones(1),
            widths=np.ones(1),
            device=AVAILABLE_DEVICE,
            gaussian_kernel=GaussianKernel(width_multiplier=-1.0),
        )

    def test_gradient_flows_to_centers_and_multipliers(self) -> None:
        """
        Golden-value/drift-detection test: only grad_fn-is-not-None was ever
        checked for GeneralizedGuassian (generically, across every FuzzySet
        subclass in test_impl.py) - never an actual backward() call inspecting
        real gradient values. Confirms centers, width_multiplier, and
        slope_multiplier - the parameters internal_calculate_membership() actually
        uses - receive a real, non-zero, NaN-free gradient.

        Deliberately does NOT assert anything about the base FuzzySet widths
        parameter: internal_calculate_membership() never reads self.get_widths()
        at all (it uses its own, separately-initialized width_multiplier
        instead), so widths.grad is None here - this is the same open question
        already flagged separately (whether that's intentional or a bug, and
        whether this class is meant to apply exp() like every other Gaussian
        variant), under manual review; asserting anything about it here would
        prejudge that review.

        Returns:
            None
        """
        generalized_gaussian = GeneralizedGuassian(
            centers=np.array([0.0]), widths=np.array([1.0]), device=AVAILABLE_DEVICE
        )
        x = torch.tensor([[0.5]], device=AVAILABLE_DEVICE)

        generalized_gaussian(x).degrees.sum().backward()

        # get_width_multiplier()/get_slope_multiplier() return a freshly
        # concatenated (non-leaf) view each call, so .grad is never populated on
        # their result directly - check the underlying leaf Parameters instead
        self.assertFalse(bool(generalized_gaussian.get_centers().grad.isnan().any()))
        self.assertFalse(bool((generalized_gaussian.get_centers().grad == 0).all()))
        for name, param in generalized_gaussian.named_parameters():
            if "_width_multiplier" in name or "_slope_multiplier" in name:
                self.assertIsNotNone(param.grad)
                self.assertFalse(bool(param.grad.isnan().any()))
                self.assertFalse(bool((param.grad == 0).all()))


if __name__ == "__main__":
    unittest.main()
