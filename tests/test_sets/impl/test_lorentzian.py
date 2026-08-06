"""
Test the Lorentzian fuzzy set's own additions beyond the generic FuzzySet interface
already exercised by test_impl.py - namely its "sigmas" alias for widths, and its
membership formula itself (cross-checked against an independent NumPy
reimplementation, the same pattern used for Gaussian/Triangular/Trapezoidal).
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.impl.basic import Lorentzian
from tests import AVAILABLE_DEVICE

from .common import assert_jit_script_matches_eager, get_test_elements


def lorentzian_numpy(
        element: np.ndarray,
        center: np.ndarray,
        sigma: np.ndarray):
    """
        Lorentzian (Cauchy) membership function that receives an 'element' value, and
        uses the 'center' and 'sigma' to determine a degree of membership for
        'element'. Implemented in Numpy and used in testing.

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Lorentzian fuzzy set.
        sigma: The width of the Lorentzian fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    return 1.0 / (1.0 + np.power((center - element) / (0.5 * sigma), 2))


class TestLorentzian(unittest.TestCase):
    """
    Test the Lorentzian fuzzy set's sigmas property and membership formula.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = get_test_elements(device=AVAILABLE_DEVICE)

    def test_membership_matches_independent_numpy_reference(self) -> None:
        """
        Golden-value/drift-detection test: cross-check Lorentzian's PyTorch
        membership formula against an independently written NumPy
        reimplementation (not the same code checking itself), so a future formula
        regression (e.g. a sign error, or an off-by-factor in the 0.5 * widths
        term) is caught rather than silently passing shape/gradient-only checks.

        Returns:
            None
        """
        lorentzian_mf = Lorentzian(
            centers=np.array([1.5409961]),
            widths=np.array([0.30742282]),
            device=AVAILABLE_DEVICE,
        )
        center = lorentzian_mf.get_centers().cpu().detach().numpy()
        sigma = lorentzian_mf.get_widths().cpu().detach().numpy()
        mu_pytorch = lorentzian_mf(self.elements).degrees.to_dense()
        mu_numpy = lorentzian_numpy(
            self.elements.cpu().detach().numpy(), center, sigma)

        # a single center/width against multiple elements carries an extra
        # dimension on the PyTorch side (4, 1, 1) that plain NumPy broadcasting
        # does not (4, 1) - flatten both before comparing, matching Gaussian's
        # own test_multi_input (assert_membership_matches_numpy's squeeze_dim
        # only handles the multi-center case)
        assert np.allclose(
            mu_pytorch.cpu().detach().numpy().flatten(),
            mu_numpy.flatten(),
            atol=1e-6)

        # test that this is compatible with torch.jit.script
        assert_jit_script_matches_eager(
            lorentzian_mf, self.elements, mu_pytorch)

    def test_gradient_flows_to_centers_and_widths(self) -> None:
        """
        Golden-value/drift-detection test: only grad_fn-is-not-None was ever
        checked for Lorentzian (generically, across every FuzzySet subclass in
        test_impl.py) - never an actual backward() call inspecting real gradient
        values. Confirms centers/widths receive a real, non-zero, NaN-free
        gradient.

        Returns:
            None
        """
        lorentzian_mf = Lorentzian(
            centers=np.array([1.5409961]),
            widths=np.array([0.30742282]),
            device=AVAILABLE_DEVICE,
        )
        lorentzian_mf(self.elements).degrees.sum().backward()
        centers_grad = lorentzian_mf.get_centers().grad
        widths_grad = lorentzian_mf.get_widths().grad
        self.assertFalse(bool(centers_grad.isnan().any()))
        self.assertFalse(bool(widths_grad.isnan().any()))
        self.assertFalse(bool((centers_grad == 0).all()))
        self.assertFalse(bool((widths_grad == 0).all()))

    def test_sigmas_is_an_alias_for_widths(self) -> None:
        """
        Coverage/regression test: sigmas' getter and setter (an alias for
        get_widths()/widths, named to match the Lorentzian/Cauchy distribution's
        own terminology) had no test coverage at all.

        Returns:
            None
        """
        lorentzian = Lorentzian(
            centers=np.array([0.0, 1.0]),
            widths=np.array([0.5, 0.5]),
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(
            torch.equal(
                lorentzian.sigmas,
                lorentzian.get_widths()))

        new_sigmas = torch.tensor([[1.5, 2.5]], device=AVAILABLE_DEVICE)
        lorentzian.sigmas = new_sigmas
        self.assertTrue(torch.equal(lorentzian.get_widths(), new_sigmas))
        self.assertTrue(torch.equal(lorentzian.sigmas, new_sigmas))


if __name__ == "__main__":
    unittest.main()
