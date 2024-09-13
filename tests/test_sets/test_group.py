"""
Test functionality relating to FuzzySetGroup.
"""

import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.sets import Membership
from fuzzy.sets.impl import Gaussian
from fuzzy.sets.group import FuzzySetGroup
from fuzzy.utils.functions import get_object_attributes


AVAILABLE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class TestFuzzySetGroup(unittest.TestCase):
    """
    Test the FuzzySetGroup class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grouped_fuzzy_sets: FuzzySetGroup = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    n_variables=2, n_terms=3, device=AVAILABLE_DEVICE, method="linear"
                ),
                Gaussian.create(
                    n_variables=2, n_terms=3, device=AVAILABLE_DEVICE, method="linear"
                ),
            ]
        )

    def test_grad_fn_is_not_none(self):
        """
        Test that the grad_fn attribute is not None.
        """
        # test individual modules
        input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=AVAILABLE_DEVICE)
        for module in self.grouped_fuzzy_sets.modules_list:
            self.assertIsInstance(module, Gaussian)
            output: Membership = module(input_data)
            self.assertIsNotNone(output.degrees.grad_fn)

        # test grouped fuzzy sets
        output: Membership = self.grouped_fuzzy_sets(input_data)
        self.assertIsNotNone(output.degrees.grad_fn)

    def test_save_grouped_fuzzy_sets(self):
        """
        Test saving grouped fuzzy sets.
        """

        # test compatibility with torch.jit.script
        torch.jit.script(self.grouped_fuzzy_sets)

        # test that FuzzySetGroup can be saved and loaded
        self.grouped_fuzzy_sets.save(Path("test_grouped_fuzzy_sets"))
        loaded_grouped_fuzzy_sets: FuzzySetGroup = self.grouped_fuzzy_sets.load(
            Path("test_grouped_fuzzy_sets"), device=AVAILABLE_DEVICE
        )

        for idx, module in enumerate(self.grouped_fuzzy_sets.modules_list):
            assert torch.equal(
                module.get_centers(),
                loaded_grouped_fuzzy_sets.modules_list[idx].get_centers(),
            )
            assert torch.equal(
                module.get_widths(),
                loaded_grouped_fuzzy_sets.modules_list[idx].get_widths(),
            )

        # check the remaining attributes are the same
        for attribute in get_object_attributes(self.grouped_fuzzy_sets):
            value = getattr(self.grouped_fuzzy_sets, attribute)
            if isinstance(value, torch.nn.ModuleList):
                continue  # already checked above
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(loaded_grouped_fuzzy_sets, attribute))
            else:  # for non-tensors
                assert value == getattr(loaded_grouped_fuzzy_sets, attribute)

        # delete the temporary directory using shutil, ignore errors if there are any
        # read-only files

        shutil.rmtree(Path("test_grouped_fuzzy_sets"), ignore_errors=True)
