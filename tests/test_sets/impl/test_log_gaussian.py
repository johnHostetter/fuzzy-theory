import unittest

import torch
import numpy as np

from fuzzy.sets.impl import LogGaussian

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLogGaussian(unittest.TestCase):
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
            width_multiplier=0.0,
        )
        self.assertRaises(
            ValueError,
            LogGaussian,
            centers=np.ones(1),
            widths=np.ones(1),
            device=AVAILABLE_DEVICE,
            width_multiplier=3.0,
        )
        self.assertRaises(
            ValueError,
            LogGaussian,
            centers=np.ones(1),
            widths=np.ones(1),
            device=AVAILABLE_DEVICE,
            width_multiplier=-1.0,
        )
