"""
Test that various continuous fuzzy set implementations are working as intended, such as the
Gaussian fuzzy set (i.e., membership function), and the Triangular fuzzy set (i.e., membership
function).
"""

import os
import unittest
from pathlib import Path
from typing import MutableMapping
from collections import OrderedDict

import torch

from fuzzy.sets.impl import Gaussian
from fuzzy.sets.abstract import FuzzySet


AVAILABLE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class TestFuzzySet(unittest.TestCase):
    """
    Test the abstract FuzzySet class.
    """

    def test_illegal_attempt_to_create(self) -> None:
        """
        Test that an illegal attempt to create a FuzzySet raises an error.

        Returns:
            None
        """
        with self.assertRaises(NotImplementedError):
            FuzzySet.create(n_variables=4, n_terms=2, device=torch.device("cpu"))

    def test_save_and_load(self) -> None:
        """
        Test that saving and loading a FuzzySet works as intended.

        Returns:
            None
        """
        for subclass in FuzzySet.__subclasses__():
            membership_func = subclass.create(
                n_variables=4, n_terms=4, device=AVAILABLE_DEVICE
            )
            state_dict: MutableMapping = membership_func.state_dict()

            # test that the path must be valid
            with self.assertRaises(ValueError):
                membership_func.save(Path(""))
            with self.assertRaises(ValueError):
                membership_func.save(Path("test"))
            with self.assertRaises(ValueError):
                membership_func.save(
                    Path("test.pth")
                )  # this file extension is not supported; see error message to learn why

            # test that saving the state dict works
            saved_state_dict: OrderedDict = membership_func.save(
                Path("membership_func.pt")
            )

            # check that the saved state dict is the same as the original state dict
            for key in state_dict.keys():
                assert key in saved_state_dict and torch.allclose(
                    state_dict[key], saved_state_dict[key]
                )
            # except the saved state dict includes additional information not captured by
            # the original state dict, such as the class name and the labels
            assert "class_name" in saved_state_dict.keys() and saved_state_dict[
                "class_name"
            ] in (subclass.__name__ for subclass in FuzzySet.__subclasses__())

            loaded_membership_func = FuzzySet.load(
                Path("membership_func.pt"), device=AVAILABLE_DEVICE
            )
            # check that the parameters and members are the same
            assert membership_func == loaded_membership_func
            assert torch.allclose(
                membership_func.get_centers(), loaded_membership_func.get_centers()
            )
            assert torch.allclose(
                membership_func.get_widths(), loaded_membership_func.get_widths()
            )
            if isinstance(
                subclass, Gaussian
            ):  # Gaussian has an additional parameter (alias for widths)
                assert torch.allclose(
                    membership_func.sigmas, loaded_membership_func.sigmas
                )
            # check some functionality that it is still working
            assert torch.allclose(membership_func.area(), loaded_membership_func.area())
            assert torch.allclose(
                membership_func(
                    torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=AVAILABLE_DEVICE)
                ).degrees.to_dense(),
                loaded_membership_func(
                    torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=AVAILABLE_DEVICE)
                ).degrees.to_dense(),
            )
            # delete the file
            os.remove("membership_func.pt")
