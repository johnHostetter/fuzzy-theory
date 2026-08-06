"""
Additional code to test and validate the GeneralizedGuassian class works as expected.
General FuzzySet behavior (construction, gradients, device moves, plotting) is already
exercised generically by test_impl.py's all_subclasses(FuzzySet) sweeps.
"""

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
