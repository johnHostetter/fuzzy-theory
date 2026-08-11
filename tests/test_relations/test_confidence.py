"""
Test the CertaintyFactors save and load behavior.
"""

import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.relations.confidence import CertaintyFactors
from tests import AVAILABLE_DEVICE


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
        self.assertEqual(AVAILABLE_DEVICE.type, loaded.weights.device.type)
        self.assertTrue(torch.allclose(original.weights, loaded.weights))

        x = torch.randn(3, 5, device=AVAILABLE_DEVICE)
        with torch.no_grad():
            self.assertTrue(torch.allclose(original(x), loaded(x)))

        shutil.rmtree(path)
        self.assertFalse(path.exists())

    def test_device_argument_is_respected(self) -> None:
        """
        Regression test: __init__ used to call weights.to(device) without assigning the
        result (Tensor.to() is not in-place), so the device argument was silently ignored
        whenever the given weights weren't already on that device.

        Returns:
            None
        """
        cpu_weights = torch.ones([4], dtype=torch.float32, device="cpu")
        certainty_factors = CertaintyFactors(
            weights=cpu_weights, device=AVAILABLE_DEVICE
        )
        self.assertEqual(certainty_factors.weights.device.type, AVAILABLE_DEVICE.type)

    def test_load_invalid_path_raises(self) -> None:
        """
        load() must raise a ValueError when given a path that does not contain a saved
        CertaintyFactors.

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            CertaintyFactors.load(
                Path("does_not_exist_certainty_factors_dir"), device=AVAILABLE_DEVICE
            )

    def test_gradient_flows_to_weights(self) -> None:
        """
        Golden-value/drift-detection test: no gradient test existed for
        CertaintyFactors at all. Confirms weights receive a real, non-zero,
        NaN-free gradient.

        Returns:
            None
        """
        certainty_factors = CertaintyFactors.create_default(
            n_features=3, device=AVAILABLE_DEVICE
        )
        x = torch.rand(4, 3, device=AVAILABLE_DEVICE)
        certainty_factors(x).sum().backward()
        self.assertFalse(bool(certainty_factors.weights.grad.isnan().any()))
        self.assertFalse(bool((certainty_factors.weights.grad == 0).all()))
