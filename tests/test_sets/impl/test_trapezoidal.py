"""
Test the Trapezoidal fuzzy set (i.e., membership function).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fuzzy.sets.impl import Trapezoidal, Triangular

from .common import get_test_elements

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trapezoidal_numpy(
    element: np.ndarray, center: np.ndarray, width: np.ndarray, plateau: np.ndarray
):
    """
    Trapezoidal membership function implemented in Numpy for testing.

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Trapezoidal fuzzy set.
        width: The half-width from center to outer foot.
        plateau: The half-width of the flat top region.

    Returns:
        The membership degree of 'element'.
    """
    values = (width - np.abs(element - center)) / (width - plateau)
    values = np.clip(values, 0.0, 1.0)
    return values


class TestTrapezoidal(unittest.TestCase):
    """
    Test the Trapezoidal fuzzy set (i.e., membership function).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = get_test_elements(device=AVAILABLE_DEVICE)

    def test_single_input(self) -> None:
        element = np.array([0.0], dtype=np.float32)
        trapezoidal_mf = Trapezoidal(
            centers=np.array([0.5]),
            widths=np.array([1.0]),
            plateaus=np.array([0.3]),
            device=AVAILABLE_DEVICE,
        )
        center = trapezoidal_mf.get_centers().cpu().detach().numpy()
        width = trapezoidal_mf.get_widths().cpu().detach().numpy()
        plateau = trapezoidal_mf.get_plateaus().cpu().detach().numpy()
        mu_pytorch = trapezoidal_mf(
            torch.tensor(element, device=AVAILABLE_DEVICE)
        ).degrees.to_dense()
        mu_numpy = trapezoidal_numpy(element, center, width, plateau)

        assert torch.allclose(
            trapezoidal_mf.get_centers(),
            torch.tensor(center, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            trapezoidal_mf.get_widths(),
            torch.tensor(width, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            trapezoidal_mf.get_plateaus(),
            torch.tensor(plateau, device=AVAILABLE_DEVICE),
        )
        assert np.allclose(mu_pytorch.cpu().detach().numpy(), mu_numpy, atol=1e-2)

        trapezoidal_mf_scripted = torch.jit.script(trapezoidal_mf)
        assert torch.allclose(
            trapezoidal_mf_scripted(
                torch.tensor(element, device=AVAILABLE_DEVICE)
            ).degrees.to_dense(),
            mu_pytorch,
        )

    def test_multi_input(self) -> None:
        trapezoidal_mf = Trapezoidal(
            centers=np.array([0.5]),
            widths=np.array([0.8]),
            plateaus=np.array([0.2]),
            device=AVAILABLE_DEVICE,
        )
        centers = trapezoidal_mf.get_centers().cpu().detach().numpy()
        widths = trapezoidal_mf.get_widths().cpu().detach().numpy()
        plateaus = trapezoidal_mf.get_plateaus().cpu().detach().numpy()
        mu_pytorch = trapezoidal_mf(self.elements).degrees.to_dense()
        mu_numpy = trapezoidal_numpy(
            self.elements.cpu().detach().numpy(), centers, widths, plateaus
        )

        assert torch.allclose(
            trapezoidal_mf.get_centers(),
            torch.tensor(centers, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            trapezoidal_mf.get_widths(),
            torch.tensor(widths, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            trapezoidal_mf.get_plateaus(),
            torch.tensor(plateaus, device=AVAILABLE_DEVICE),
        )
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
        )

        trapezoidal_mf_scripted = torch.jit.script(trapezoidal_mf)
        assert torch.allclose(
            trapezoidal_mf_scripted(self.elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_multiple_sets(self) -> None:
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        widths = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        plateaus = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        trapezoidal_mf = Trapezoidal(
            centers=centers, widths=widths, plateaus=plateaus, device=AVAILABLE_DEVICE
        )
        mu_pytorch = trapezoidal_mf(self.elements).degrees.to_dense()
        mu_numpy = trapezoidal_numpy(
            self.elements.cpu().detach().numpy(), centers, widths, plateaus
        )

        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
        )

        trapezoidal_mf_scripted = torch.jit.script(trapezoidal_mf)
        assert torch.allclose(
            trapezoidal_mf_scripted(self.elements).degrees.to_dense(), mu_pytorch
        )

    def test_degenerate_to_triangular(self) -> None:
        """When plateaus = 0, the Trapezoidal MF should match Triangular."""
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        widths = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        plateaus = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        trapezoidal_mf = Trapezoidal(
            centers=centers, widths=widths, plateaus=plateaus, device=AVAILABLE_DEVICE
        )
        triangular_mf = Triangular(
            centers=centers, widths=widths, device=AVAILABLE_DEVICE
        )

        mu_trap = trapezoidal_mf(self.elements).degrees.to_dense()
        mu_tri = triangular_mf(self.elements).degrees.to_dense()

        assert torch.allclose(mu_trap, mu_tri, atol=1e-6)

    def test_flat_top_region(self) -> None:
        """Points within the plateau should have membership = 1.0."""
        trapezoidal_mf = Trapezoidal(
            centers=np.array([0.5]),
            widths=np.array([1.0]),
            plateaus=np.array([0.3]),
            device=AVAILABLE_DEVICE,
        )
        points_on_plateau = torch.tensor(
            [0.3, 0.4, 0.5, 0.6, 0.7], device=AVAILABLE_DEVICE
        )
        mu = trapezoidal_mf(points_on_plateau).degrees.to_dense()
        assert torch.allclose(mu, torch.ones_like(mu), atol=1e-6)

    def test_save_and_load(self) -> None:
        centers = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        widths = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        plateaus = np.array([0.1, 0.2, 0.15], dtype=np.float32)
        trapezoidal_mf = Trapezoidal(
            centers=centers, widths=widths, plateaus=plateaus, device=AVAILABLE_DEVICE
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        trapezoidal_mf.save(path)
        loaded_mf = Trapezoidal.load(path, device=AVAILABLE_DEVICE)

        assert torch.allclose(trapezoidal_mf.get_centers(), loaded_mf.get_centers())
        assert torch.allclose(trapezoidal_mf.get_widths(), loaded_mf.get_widths())
        assert torch.allclose(trapezoidal_mf.get_plateaus(), loaded_mf.get_plateaus())

        mu_original = trapezoidal_mf(self.elements).degrees.to_dense()
        mu_loaded = loaded_mf(self.elements).degrees.to_dense()
        assert torch.allclose(mu_original, mu_loaded)

        path.unlink()

    def test_plateaus_must_be_numpy(self) -> None:
        with self.assertRaises(ValueError):
            Trapezoidal(
                centers=np.array([0.5]),
                widths=np.array([1.0]),
                plateaus=[0.3],
                device=AVAILABLE_DEVICE,
            )
