"""
Tests for the linguistic variable class.
"""

import unittest

import torch
import numpy as np

from fuzzy.sets.continuous.impl import Gaussian
from fuzzy.logic.variables import LinguisticVariables


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLinguisticVariable(unittest.TestCase):
    """
    Test the LinguisticVariables class.
    """

    def test_create_linguistic_variable(self) -> None:
        """
        Test that linguistic variables can be created with a set of terms for input and output
        fuzzy sets.

        Returns:
            None
        """
        input_terms = [
            Gaussian(np.zeros(1), np.ones(1), device=AVAILABLE_DEVICE),
            Gaussian(np.zeros(2), np.ones(2), device=AVAILABLE_DEVICE),
        ]
        output_terms = [
            Gaussian(np.zeros(3), np.ones(3), device=AVAILABLE_DEVICE),
            Gaussian(np.zeros(4), np.ones(4), device=AVAILABLE_DEVICE),
        ]
        lv = LinguisticVariables(input_terms, output_terms)
        self.assertEqual(lv.inputs, input_terms)
        self.assertEqual(lv.targets, output_terms)
