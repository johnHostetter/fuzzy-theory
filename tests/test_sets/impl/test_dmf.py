"""
Additional code to test and validate the Dimension-Dependent fuzzy sets
(GaussianDMF, GaussianNoExpDMF) work as expected.
"""

import unittest

import numpy as np
import sympy
import torch

from fuzzy.sets.impl.gauss_variants.dmf import GaussianDMF, GaussianNoExpDMF
from tests import AVAILABLE_DEVICE


class TestDimensionDependent(unittest.TestCase):
    """
    Test and validate the Dimension-Dependent fuzzy sets work as expected, especially
    at the high input dimensionality this class exists for.
    """

    def test_n_inputs_does_not_overflow_at_high_dimensionality(self) -> None:
        """
        Regression guard: n_inputs used to be stored as torch.int8 (max magnitude 127),
        which silently wraps to a negative value at 128+ input dimensions - exactly the
        "high-dimensional" scenario this class exists for (see module docstring) -
        corrupting rho, and with it every membership degree, into NaN.

        Returns:
            None
        """
        n_inputs = 128
        centers = np.zeros((n_inputs, 1))
        widths = np.ones((n_inputs, 1))
        fuzzy_set = GaussianDMF(centers=centers, widths=widths, device=AVAILABLE_DEVICE)

        self.assertEqual(fuzzy_set.n_inputs.item(), n_inputs)
        self.assertFalse(bool(fuzzy_set.rho.isnan().any()))

        # membership at the exact center should be (close to) 1.0, not NaN
        observations = torch.zeros(1, n_inputs, 1, device=AVAILABLE_DEVICE)
        degrees = fuzzy_set.calculate_membership(observations)
        self.assertFalse(bool(degrees.isnan().any()))
        self.assertTrue(torch.allclose(degrees, torch.ones_like(degrees), atol=1e-4))

    def test_rho_calculation_matches_below_previous_overflow_threshold(self) -> None:
        """
        127 dimensions is the largest value that fit in the previous (buggy) int8
        dtype; confirm the fix does not change behavior for values that already
        worked, and that GaussianNoExpDMF (the sibling subclass) is unaffected too.

        Returns:
            None
        """
        n_inputs = 127
        centers = np.zeros((n_inputs, 1))
        widths = np.ones((n_inputs, 1))
        fuzzy_set = GaussianNoExpDMF(
            centers=centers, widths=widths, device=AVAILABLE_DEVICE
        )
        expected_rho = 1.0 - (
            torch.tensor([745.0]).log() / torch.tensor([float(n_inputs)]).log()
        )
        self.assertTrue(torch.allclose(fuzzy_set.rho.cpu(), expected_rho, atol=1e-6))

    def test_gradient_flows_to_centers_and_widths(self) -> None:
        """
        Golden-value/drift-detection test: only grad_fn-is-not-None was ever
        checked for GaussianDMF/GaussianNoExpDMF (generically, across every
        FuzzySet subclass in test_impl.py) - never an actual backward() call
        inspecting real gradient values. Confirms centers/widths receive a real,
        non-zero, NaN-free gradient for both classes.

        Returns:
            None
        """
        for cls in (GaussianDMF, GaussianNoExpDMF):
            fuzzy_set = cls(
                centers=np.array([[0.0], [1.0]]),
                widths=np.array([[1.0], [1.0]]),
                device=AVAILABLE_DEVICE,
            )
            observations = torch.tensor([[[0.001], [0.999]]], device=AVAILABLE_DEVICE)
            fuzzy_set.calculate_membership(observations).sum().backward()
            centers_grad = fuzzy_set.get_centers().grad
            widths_grad = fuzzy_set.get_widths().grad
            self.assertFalse(bool(centers_grad.isnan().any()), cls.__name__)
            self.assertFalse(bool(widths_grad.isnan().any()), cls.__name__)
            self.assertFalse(bool((centers_grad == 0).all()), cls.__name__)
            self.assertFalse(bool((widths_grad == 0).all()), cls.__name__)

    def test_matches_the_additive_formula_away_from_the_center(self) -> None:
        """
        Regression test for a real bug: GaussianDMF/GaussianNoExpDMF used to delegate
        to Gaussian/LogGaussian's internal_calculate_membership via a width_multiplier
        parameter that MULTIPLIES width**2 into the denominator - but the cited
        paper's own formula (already correctly encoded in this class' own
        sympy_formula(), just never checked against what calculate_membership()
        actually computed) and its authors' reference implementation
        (Eandon/HDFIS/lib/membership_functions.py's gauss_dmf_sig) ADD n_inputs**rho
        to width**2 instead.

        Every other test in this file only checks membership AT the exact center,
        where (x-c)**2=0 makes the denominator's value irrelevant to the result -
        this is the only test that would have caught the bug (confirmed via
        revert-and-confirm-failure: temporarily restoring the old delegation-based
        implementation makes this fail, producing 0.024 instead of ~0.782).

        Golden value hand-computed from the authors' own formula: 50 input
        dimensions, width=2.0, an observation exactly 1.0 away from the center.
        """
        n_inputs = 50
        centers = np.zeros((n_inputs, 1))
        widths = np.full((n_inputs, 1), 2.0)
        observations = torch.ones(1, n_inputs, 1, device=AVAILABLE_DEVICE)

        no_exp_fuzzy_set = GaussianNoExpDMF(
            centers=centers, widths=widths, device=AVAILABLE_DEVICE
        )
        exp_fuzzy_set = GaussianDMF(
            centers=centers, widths=widths, device=AVAILABLE_DEVICE
        )

        # hand-computed from gauss_dmf_sig's own formula: (x-c)**2 / (n**rho + w**2)
        rho = 1.0 - torch.tensor([745.0]).log() / torch.tensor([50.0]).log()
        width_multiplier_term = torch.pow(torch.tensor([50.0]), rho)
        expected_no_exp = -1.0 / (width_multiplier_term + 2.0**2)
        expected_exp = torch.exp(expected_no_exp)

        no_exp_degrees = no_exp_fuzzy_set.calculate_membership(observations)
        exp_degrees = exp_fuzzy_set.calculate_membership(observations)

        self.assertTrue(
            torch.allclose(no_exp_degrees.cpu(), expected_no_exp.expand_as(no_exp_degrees.cpu()), atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(exp_degrees.cpu(), expected_exp.expand_as(exp_degrees.cpu()), atol=1e-4)
        )
        # sanity-check the golden value itself against what the buggy (multiplicative)
        # implementation would have produced, so a future reader doesn't need to
        # re-derive why this specific scenario is a meaningful regression guard
        self.assertAlmostEqual(expected_exp.item(), 0.7820, places=3)

    def test_sympy_formulas(self) -> None:
        """
        Coverage/regression test: neither class' sympy_formula() (used for
        formula rendering/plotting - see FuzzySetPlot.render_formula) had any test
        coverage. GaussianDMF's formula is documented as exp() applied to
        GaussianNoExpDMF's.

        Returns:
            None
        """
        no_exp_formula = GaussianNoExpDMF.sympy_formula()
        self.assertIsInstance(no_exp_formula, sympy.Expr)

        exp_formula = GaussianDMF.sympy_formula()
        self.assertIsInstance(exp_formula, sympy.Expr)
        self.assertEqual(exp_formula, sympy.exp(no_exp_formula))


if __name__ == "__main__":
    unittest.main()
