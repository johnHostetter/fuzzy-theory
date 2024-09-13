import shutil
import inspect
import unittest
from pathlib import Path

import numpy as np
import torch

from fuzzy.utils import all_subclasses
from fuzzy.sets.abstract import FuzzySet
import fuzzy.sets.impl  # to make all subclasses available via all_subclasses


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFuzzySetImpl(unittest.TestCase):
    """
    Test various implementations of FuzzySet class about general properties.
    """

    def test_numpy_args_are_required(self) -> None:
        """
        Test that numpy args are required for the FuzzySet class.

        Returns:
            None
        """
        for impl in all_subclasses(FuzzySet):
            kwargs = {"device": AVAILABLE_DEVICE}
            if inspect.isabstract(impl):  # skip abstract classes (e.g., FuzzySet)
                continue
            # centers must be a numpy array
            print(impl)
            self.assertRaises(ValueError, impl, None, None, **kwargs)
            # widths must be a numpy array
            self.assertRaises(ValueError, impl, np.ones(1), None, **kwargs)
            # centers and widths must have the same shape (i.e., ndim)
            self.assertRaises(ValueError, impl, np.ones(1), np.ones((2, 1)), **kwargs)
            # centers and widths must not be empty
            self.assertRaises(ValueError, impl, np.array(1), np.array(1), **kwargs)

    def test_grad_fn_is_not_none(self) -> None:
        """
        Sanity check that the grad_fn is not None for the FuzzySet implementations.

        Returns:
            None
        """
        for impl in all_subclasses(FuzzySet):
            if inspect.isabstract(impl):  # skip abstract classes (e.g., FuzzySet)
                continue
            fuzzy_set = impl(
                centers=np.array([0.0, 0.5, 1.0]),
                widths=np.array([0.5, 0.75, 1.0]),
                device=AVAILABLE_DEVICE,
            )
            # check that the grad_fn is not None
            membership = fuzzy_set(
                torch.tensor([[0.0], [0.25], [0.75]], device=AVAILABLE_DEVICE)
            )
            self.assertIsNotNone(membership.degrees.grad_fn)

            # check that the grad_fn is not None if we made a FuzzySet from .stack()
            fuzzy_set = impl.stack([fuzzy_set, fuzzy_set])
            membership = fuzzy_set(
                torch.tensor([[0.0], [0.25], [0.75]], device=AVAILABLE_DEVICE)
            )
            self.assertIsNotNone(membership.degrees.grad_fn)

    def test_move_to_device(self) -> None:
        """
        Test that the FuzzySet can be moved to a device.

        Returns:
            None
        """
        for impl in all_subclasses(FuzzySet):
            if inspect.isabstract(impl):  # skip abstract classes (e.g., FuzzySet)
                continue
            fuzzy_set = impl(
                centers=np.array([0.0, 0.5, 1.0]),
                widths=np.array([0.5, 0.75, 1.0]),
                device=AVAILABLE_DEVICE,
            )
            fuzzy_set.to(device=torch.device("cpu"))
            self.assertEqual(torch.device("cpu").type, fuzzy_set.device.type)
            # check parameters are on the same device
            self.assertEqual(
                torch.device("cpu").type, fuzzy_set.get_centers().device.type
            )
            self.assertEqual(
                torch.device("cpu").type, fuzzy_set.get_widths().device.type
            )
            fuzzy_set.to(device=AVAILABLE_DEVICE)
            self.assertEqual(AVAILABLE_DEVICE.type, fuzzy_set.device.type)
            # check parameters are on the same device
            self.assertEqual(AVAILABLE_DEVICE.type, fuzzy_set.get_centers().device.type)
            self.assertEqual(AVAILABLE_DEVICE.type, fuzzy_set.get_widths().device.type)

    def test_plot(self) -> None:
        """
        Test that the FuzzySet implementations can be plotted without errors.

        Returns:
            None
        """
        for impl in all_subclasses(FuzzySet):
            if inspect.isabstract(impl):  # skip abstract classes (e.g., FuzzySet)
                continue
            fuzzy_set = impl(
                centers=np.array([[0.0, 0.5, 1.0], [2.0, 2.5, 3.0]]),
                widths=np.array([[0.5, 0.75, 1.0], [1.25, 1.5, 1.75]]),
                device=AVAILABLE_DEVICE,
            )
            _, _ = fuzzy_set.plot(
                output_dir=Path(__file__).parent / "plots" / impl.__name__
            )
            # check the directory exists
            self.assertTrue((Path(__file__).parent / "plots" / impl.__name__).is_dir())
            # check a plot exists for each variable dimension
            for i in range(fuzzy_set.get_centers().shape[0]):
                self.assertTrue(
                    (
                        Path(__file__).parent / "plots" / impl.__name__ / f"mu_{i}.png"
                    ).is_file()
                )

        # delete the plots directory
        shutil.rmtree(Path(__file__).parent / "plots")
