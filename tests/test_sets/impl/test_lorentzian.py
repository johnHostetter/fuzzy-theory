"""
Test the Lorentzian fuzzy set's own additions beyond the generic FuzzySet interface
already exercised by test_impl.py - namely its "sigmas" alias for widths.
"""

import unittest

import numpy as np
import torch

from fuzzy.sets.impl.basic import Lorentzian
from tests import AVAILABLE_DEVICE


class TestLorentzian(unittest.TestCase):
    """
    Test the Lorentzian fuzzy set's sigmas property.
    """

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
