"""
Test functionality relating to GroupedFuzzySets.
"""

import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.sets.impl import Gaussian
from fuzzy.sets.group import GroupedFuzzySets
from fuzzy.sets.utils import get_object_attributes


AVAILABLE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class TestGroupedFuzzySets(unittest.TestCase):
    """
    Test the GroupedFuzzySets class.
    """

    def test_save_grouped_fuzzy_sets(self):
        """
        Test saving grouped fuzzy sets.
        """
        grouped_fuzzy_sets: GroupedFuzzySets = GroupedFuzzySets(
            modules_list=[
                Gaussian.create(n_variables=2, n_terms=3, device=AVAILABLE_DEVICE),
                Gaussian.create(n_variables=2, n_terms=3, device=AVAILABLE_DEVICE),
            ]
        )

        # test compatibility with torch.jit.script
        torch.jit.script(grouped_fuzzy_sets)

        # test that GroupedFuzzySets can be saved and loaded
        grouped_fuzzy_sets.save(Path("test_grouped_fuzzy_sets"))
        loaded_grouped_fuzzy_sets: GroupedFuzzySets = grouped_fuzzy_sets.load(
            Path("test_grouped_fuzzy_sets"), device=AVAILABLE_DEVICE
        )

        for idx, module in enumerate(grouped_fuzzy_sets.modules_list):
            assert torch.equal(
                module.get_centers(),
                loaded_grouped_fuzzy_sets.modules_list[idx].get_centers(),
            )
            assert torch.equal(
                module.get_widths(),
                loaded_grouped_fuzzy_sets.modules_list[idx].get_widths(),
            )

        # check the remaining attributes are the same
        for attribute in get_object_attributes(grouped_fuzzy_sets):
            value = getattr(grouped_fuzzy_sets, attribute)
            if isinstance(value, torch.nn.ModuleList):
                continue  # already checked above
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(loaded_grouped_fuzzy_sets, attribute))
            else:  # for non-tensors
                assert value == getattr(loaded_grouped_fuzzy_sets, attribute)

        # delete the temporary directory using shutil, ignore errors if there are any
        # read-only files

        shutil.rmtree(Path("test_grouped_fuzzy_sets"), ignore_errors=True)
