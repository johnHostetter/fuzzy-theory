"""
Test the CertaintyFactors save and load behavior.
"""

import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.relations.confidence import CertaintyFactors

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCertaintyFactors(unittest.TestCase):
    """
    Test the CertaintyFactors class.
    """

    def test_save_and_load(self) -> None:
        """
        Test that we can save and load a CertaintyFactors object using the same path.

        Returns:
            None
        """
        original = CertaintyFactors.create_default(
            n_features=5, device=AVAILABLE_DEVICE
        )
        path = Path("certainty_factors")
        state_dict = original.save(path)
        self.assertTrue(path.exists() and path.is_dir())
        self.assertTrue((path / "state_dict.pt").is_file())
        self.assertIsInstance(state_dict, dict)
        self.assertIn("weights", state_dict)

        loaded = CertaintyFactors.load(path, device=AVAILABLE_DEVICE)
        self.assertIsInstance(loaded, CertaintyFactors)
        self.assertEqual(
            AVAILABLE_DEVICE.type, loaded.weights.device.type
        )
        self.assertTrue(
            torch.allclose(original.weights, loaded.weights)
        )

        x = torch.randn(3, 5, device=AVAILABLE_DEVICE)
        with torch.no_grad():
            self.assertTrue(torch.allclose(original(x), loaded(x)))

        shutil.rmtree(path)
        self.assertFalse(path.exists())
