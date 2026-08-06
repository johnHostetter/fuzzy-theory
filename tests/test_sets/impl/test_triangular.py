"""
Test the Triangular fuzzy set (i.e., membership function).
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.abstract import FuzzySetInitMethod, FuzzySetShape
from fuzzy.sets.impl import Triangular
from tests import AVAILABLE_DEVICE

from .common import (
    assert_jit_script_matches_eager,
    assert_membership_matches_numpy,
    get_test_elements,
)


def triangular_numpy(element: np.ndarray, center: np.ndarray, width: np.ndarray):
    """
        Triangular membership function that receives an 'element' value, and uses
        the 'center' and 'width' to determine a degree of membership for 'element'.
        Implemented in Numpy and used in testing.

        https://www.mathworks.com/help/fuzzy/trimf.html

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Triangular fuzzy set.
        width: The width of the Triangular fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    values = 1.0 - (1.0 / width) * np.abs(element - center)
    values[(values < 0)] = 0
    return values


class TestTriangular(unittest.TestCase):
    """
    Test the Triangular fuzzy set (i.e., membership function).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = get_test_elements(device=AVAILABLE_DEVICE)

    def test_single_input(self) -> None:
        """
        Test that single input works for the Triangular membership function.

        Returns:
            None
        """
        element = np.array([0.0], dtype=np.float32)
        triangular_mf = Triangular(
            centers=np.array([1.5409961]),
            widths=np.array([0.30742282]),
            device=AVAILABLE_DEVICE,
        )
        center = triangular_mf.get_centers().cpu().detach().numpy()
        width = triangular_mf.get_widths().cpu().detach().numpy()
        mu_pytorch = triangular_mf(
            torch.tensor(element, device=AVAILABLE_DEVICE)
        ).degrees.to_dense()
        mu_numpy = triangular_numpy(element, center, width)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers(),
            torch.tensor(center, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            triangular_mf.get_widths(),
            torch.tensor(width, device=AVAILABLE_DEVICE),
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert_membership_matches_numpy(mu_pytorch, mu_numpy, atol=1e-2)

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(
            triangular_mf, torch.tensor(element, device=AVAILABLE_DEVICE), mu_pytorch
        )

    def test_multi_input(self) -> None:
        """
        Test that multiple input works for the Triangular membership function.

        Returns:
            None
        """
        triangular_mf = Triangular(
            centers=np.array([1.5410]),
            widths=np.array([0.3074]),
            device=AVAILABLE_DEVICE,
        )
        centers, widths = (
            triangular_mf.get_centers().cpu().detach().numpy(),
            triangular_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = triangular_mf(self.elements).degrees.to_dense()
        mu_numpy = triangular_numpy(
            self.elements.cpu().detach().numpy(), centers, widths
        )

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers(),
            torch.tensor(centers, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            triangular_mf.get_widths(),
            torch.tensor(widths, device=AVAILABLE_DEVICE),
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert_membership_matches_numpy(mu_pytorch, mu_numpy, squeeze_dim=1, atol=1e-2)

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(triangular_mf, self.elements, mu_pytorch)

    def test_multi_input_with_centers_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when centers are
        specified for the fuzzy sets.

        Returns:
            None
        """
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        triangular_mf = Triangular(
            centers=centers, widths=np.array([0.4962566]), device=AVAILABLE_DEVICE
        )
        widths = triangular_mf.get_widths().cpu().detach().numpy()
        mu_pytorch = triangular_mf(self.elements).degrees.to_dense()
        mu_numpy = triangular_numpy(
            self.elements.cpu().detach().numpy(), centers, widths
        )

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers(),
            torch.tensor(centers, device=AVAILABLE_DEVICE),
        )
        assert torch.allclose(
            triangular_mf.get_widths(),
            torch.tensor(widths, device=AVAILABLE_DEVICE),
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert_membership_matches_numpy(mu_pytorch, mu_numpy, squeeze_dim=1, atol=1e-2)

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(triangular_mf, self.elements, mu_pytorch)

    def test_multi_input_with_widths_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when widths are
        specified for the fuzzy sets.

        Returns:
            None
        """
        widths = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0], dtype=np.float32
        )  # negative widths are missing sets
        triangular_mf = Triangular(
            centers=np.array([1.5409961]), widths=widths, device=AVAILABLE_DEVICE
        )
        centers = triangular_mf.get_centers().cpu().detach().numpy()
        mu_pytorch = triangular_mf(self.elements).degrees.to_dense()
        mu_numpy = triangular_numpy(
            self.elements.cpu().detach().numpy(), centers, widths
        )

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers(), torch.tensor(centers, device=AVAILABLE_DEVICE)
        )
        assert torch.allclose(
            triangular_mf.get_widths(), torch.tensor(widths, device=AVAILABLE_DEVICE)
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert_membership_matches_numpy(mu_pytorch, mu_numpy, squeeze_dim=1, atol=1e-2)

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(triangular_mf, self.elements, mu_pytorch)

    def test_multi_input_with_both_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when centers
        and widths are specified for the fuzzy sets.

        Returns:
            None
        """
        centers = np.array([-0.5, -0.25, 0.25, 0.5, 0.75])
        widths = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        triangular_mf = Triangular(
            centers=centers, widths=widths, device=AVAILABLE_DEVICE
        )
        mu_pytorch = triangular_mf(self.elements).degrees.to_dense()
        mu_numpy = triangular_numpy(
            self.elements.cpu().detach().numpy(), centers, widths
        )

        # make sure the Triangular parameters are still identical afterward
        assert np.allclose(triangular_mf.get_centers().cpu().detach().numpy(), centers)
        assert np.allclose(triangular_mf.get_widths().cpu().detach().numpy(), widths)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert_membership_matches_numpy(mu_pytorch, mu_numpy, squeeze_dim=1, atol=1e-2)

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(triangular_mf, self.elements, mu_pytorch)

    def test_create_random(self) -> None:
        """
        Test that a random fuzzy set of this type can be created and that the results are consistent
        with the expected membership degrees.

        Returns:
            None
        """
        triangular_mf = Triangular.create(
            shape=FuzzySetShape(
                n_variables=4,
                n_terms=4,
            ),
            device=AVAILABLE_DEVICE,
            method=FuzzySetInitMethod.RANDOM,
        )
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = triangular_numpy(
            element,
            triangular_mf.get_centers().cpu().detach().numpy(),
            triangular_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = triangular_mf(
            torch.tensor(element[0], device=AVAILABLE_DEVICE)
        ).degrees.to_dense()
        assert_membership_matches_numpy(
            mu_pytorch, target_membership_degrees, rtol=1e-1, atol=1e-1
        )

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(
            triangular_mf,
            torch.tensor(element[0], device=AVAILABLE_DEVICE),
            mu_pytorch,
        )

    def test_gradient_flows_inside_support_and_vanishes_outside(self) -> None:
        """
        Golden-value/drift-detection test: only grad_fn-is-not-None was ever
        checked for Triangular (generically, across every FuzzySet subclass in
        test_impl.py) - never an actual backward() call inspecting real gradient
        values. triangular_numpy's clamp (values[(values < 0)] = 0, implementing
        the membership function's flat-zero region outside its support) is exactly
        the kind of boolean-indexed assignment that could silently break autograd
        or zero out the wrong region if it were ever changed - this confirms both
        that centers/widths receive a real, non-zero gradient for an observation
        inside the triangle's support, and that the gradient is correctly exactly
        zero (not NaN, not accidentally non-zero) for an observation clamped to 0
        outside it.

        Returns:
            None
        """
        triangular_mf = Triangular(
            centers=np.array(
                [1.0]), widths=np.array(
                [2.0]), device=AVAILABLE_DEVICE)
        # center=1.0, width=2.0 -> support is (0.0, 2.0); 1.5 is inside, 5.0 is
        # far outside (clamped to exactly 0)
        inside = torch.tensor([[1.5]], device=AVAILABLE_DEVICE)
        outside = torch.tensor([[5.0]], device=AVAILABLE_DEVICE)

        triangular_mf(inside).degrees.sum().backward()
        centers_grad_inside = triangular_mf.get_centers().grad.clone()
        widths_grad_inside = triangular_mf.get_widths().grad.clone()
        self.assertFalse(bool(centers_grad_inside.isnan().any()))
        self.assertFalse(bool(widths_grad_inside.isnan().any()))
        self.assertFalse(bool((centers_grad_inside == 0).all()))
        self.assertFalse(bool((widths_grad_inside == 0).all()))

        triangular_mf.get_centers().grad = None
        triangular_mf.get_widths().grad = None
        triangular_mf(outside).degrees.sum().backward()
        self.assertFalse(bool(triangular_mf.get_centers().grad.isnan().any()))
        self.assertTrue(bool((triangular_mf.get_centers().grad == 0).all()))
        self.assertTrue(bool((triangular_mf.get_widths().grad == 0).all()))
