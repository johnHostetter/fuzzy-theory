"""
Test that various continuous fuzzy set implementations are working as intended, such as the
Gaussian fuzzy set (i.e., membership function), and the Triangular fuzzy set (i.e., membership
function).
"""

import inspect
import os
import unittest
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
import torch

from fuzzy.sets import Membership
from fuzzy.sets.abstract import (
    DynamicParameterList,
    FuzzySet,
    FuzzySetInitMethod,
    FuzzySetShape,
)
from fuzzy.sets.impl.basic import NoOp
from fuzzy.sets.impl.gauss_variants.cmf import Gaussian

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
            FuzzySet.create(
                shape=FuzzySetShape(n_variables=4, n_terms=2),
                device=torch.device("cpu"),
                method=FuzzySetInitMethod.LINEAR,
            )

    def test_save_and_load(self) -> None:
        """
        Test that saving and loading a FuzzySet works as intended.

        Returns:
            None
        """
        for subclass in FuzzySet.__subclasses__():
            if inspect.isabstract(subclass) or subclass == NoOp:
                continue
            membership_func = subclass.create(
                shape=FuzzySetShape(n_variables=4, n_terms=4),
                device=AVAILABLE_DEVICE,
                method=FuzzySetInitMethod.LINEAR,
            )
            state_dict: MutableMapping = membership_func.state_dict()

            # test that the path must be valid
            with self.assertRaises(ValueError):
                membership_func.save(Path(""))
            with self.assertRaises(ValueError):
                membership_func.save(Path("test"))
            with self.assertRaises(ValueError):
                # this file extension is not supported; see error message to
                # learn why
                membership_func.save(Path("test.pth"))

            # test that saving the state dict works
            saved_state_dict: MutableMapping[str, Any] = membership_func.save(
                Path("membership_func.pt")
            )

            # check that the saved state dict is the same as the original state
            # dict
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

    def test_save_and_load_of_no_op(self) -> None:
        """
        Test that saving and loading a NoOp (i.e., no operation) works as intended.

        Returns:
            None
        """
        no_op = NoOp(n_elements=10, membership=0.75, device=AVAILABLE_DEVICE)
        membership = no_op(
            torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=AVAILABLE_DEVICE)
        )
        self.save_and_load_of_no_op_helper(membership, no_op)
        # test saving and loading of the NoOp
        tmp_file = Path("tmp.pt")
        no_op.save(tmp_file)
        loaded_no_op = NoOp.load(tmp_file, device=AVAILABLE_DEVICE)
        # delete the temporary file
        tmp_file.unlink()

        self.save_and_load_of_no_op_helper(membership, loaded_no_op)

    def save_and_load_of_no_op_helper(
        self, membership: Membership, no_op: "FuzzySet"
    ) -> None:
        """
        A simple helper method to check the behavior of no operation (i.e., do nothing)
        membership function works as intended both before and after saving/loading.

        Args:
            membership: The membership degrees.
            no_op: The no operation.

        Returns:
            None
        """
        self.assertEqual(no_op.get_centers().size()[0], 10)
        self.assertEqual(membership.degrees.size()[1], 4)
        self.assertNotEqual(no_op.get_centers().size()[0], membership.degrees.size()[1])
        self.assertAlmostEqual(no_op.membership, membership.degrees.mean().item())

    def test_hash_eq_contract(self) -> None:
        """
        Regression test: FuzzySet.__hash__ used to hash the centers/widths *tensors*
        themselves, which torch.Tensor hashes by identity (id()) rather than by value. Since
        __eq__ compares by value (torch.equal), two separately constructed but value-equal
        fuzzy sets were '==' yet had different hashes - a violation of Python's hash contract
        (equal objects must report equal hashes) that breaks their use as dict keys or set
        members.

        Returns:
            None
        """
        for subclass in FuzzySet.__subclasses__():
            if inspect.isabstract(subclass) or subclass == NoOp:
                continue
            first = subclass.create(
                shape=FuzzySetShape(n_variables=2, n_terms=3),
                device=AVAILABLE_DEVICE,
                method=FuzzySetInitMethod.LINEAR,
            )
            second = subclass.create(
                shape=FuzzySetShape(n_variables=2, n_terms=3),
                device=AVAILABLE_DEVICE,
                method=FuzzySetInitMethod.LINEAR,
            )
            # two separately constructed fuzzy sets, built the same way, have identical
            # parameter values but are distinct objects (and therefore distinct underlying
            # tensors) - this is exactly the case that must not violate the hash contract
            self.assertEqual(
                first,
                second,
                f"{subclass.__name__} instances with identical parameters should be equal",
            )
            self.assertEqual(
                hash(first),
                hash(second),
                f"{subclass.__name__}.__hash__ violates the hash contract: "
                f"equal instances must have equal hashes",
            )
            # a fuzzy set must also consistently hash the same as itself across repeated calls
            self.assertEqual(hash(first), hash(first))

    def test_dynamic_parameter_list_to_dtype_only(self) -> None:
        """
        Regression test: DynamicParameterList.to() assumed the first positional argument was
        always a device (self._device = args[0] if args else self._device). torch.nn.Module.to()
        also accepts a dtype-only call (e.g. .to(torch.float64)), which used to corrupt
        _device with a dtype object - later breaking add_parameter() and the empty-list branch
        of the 'tensor' property, both of which pass _device to torch.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0]), widths=np.array([1.0, 1.0]), device=AVAILABLE_DEVICE
        )
        # resolve the expected device the same way .to() would (e.g. an unindexed "cuda"
        # resolves to a concrete "cuda:0" once actually applied to a tensor), so the
        # comparison below is correct on both CPU-only and CUDA machines
        resolved_device = torch.empty(0, device=AVAILABLE_DEVICE).device

        gaussian_mf.to(torch.float64)

        self.assertEqual(gaussian_mf._centers._device, resolved_device)
        self.assertEqual(gaussian_mf._centers._dtype, torch.float64)
        self.assertEqual(gaussian_mf.get_centers().dtype, torch.float64)

        # adding a parameter afterward must use the up-to-date dtype, not a stale one
        gaussian_mf._centers.add_parameter(np.array([[2.0, 3.0]]))
        self.assertEqual(gaussian_mf._centers.params[-1].dtype, torch.float64)

        # an empty DynamicParameterList must not raise when .to() is given a dtype only
        empty = DynamicParameterList(device=AVAILABLE_DEVICE, dtype=torch.float32)
        empty.to(torch.float64)
        self.assertEqual(empty._dtype, torch.float64)
        self.assertEqual(empty.tensor.dtype, torch.float64)
