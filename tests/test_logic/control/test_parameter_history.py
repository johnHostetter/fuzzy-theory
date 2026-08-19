"""
Test the fuzzy.logic.control.parameter_history module.
"""

import unittest

import torch

from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import ZeroOrder
from fuzzy.logic.control.parameter_history import (
    ParameterSnapshot,
    capture_parameter_snapshot,
)
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import Product

from .demo_flcs import AVAILABLE_DEVICE
from .demo_flcs import build_mamdani_flc as _build_mamdani_flc
from .demo_flcs import build_tsk_flc as _build_tsk_flc
from .demo_flcs import toy_tsk


def _build_zero_order_flc() -> FLC:
    """
    A defuzzification method _flc_named_parameters has no method-specific tracking
    for, to exercise its generic fallback (input_granulation-only) branch.

    Returns:
        A small toy ZeroOrder FuzzyLogicController.
    """
    antecedents, _, rules = toy_tsk(t_norm=Product, device=AVAILABLE_DEVICE)
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
        rules=rules,
    )
    return FLC(source=knowledge_base, inference=ZeroOrder, device=AVAILABLE_DEVICE)


class TestCaptureParameterSnapshot(unittest.TestCase):
    """
    Test capture_parameter_snapshot.
    """

    def test_tsk_snapshot_has_expected_parameter_names_and_shapes(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_tsk_flc()
        snapshot = capture_parameter_snapshot(flc, step=0)
        self.assertIsInstance(snapshot, ParameterSnapshot)
        self.assertEqual(snapshot.step, 0)
        self.assertEqual(
            set(snapshot.parameters),
            {
                "input_granulation.centers",
                "input_granulation.widths",
                "defuzzification.consequences",
            },
        )
        self.assertEqual(
            tuple(snapshot.parameters["input_granulation.centers"].shape),
            (flc.shape.n_inputs, flc.input_granulation.centers.shape[1]),
        )
        self.assertEqual(
            tuple(snapshot.parameters["defuzzification.consequences"].shape),
            (flc.shape.n_outputs, flc.shape.n_rules, flc.shape.n_inputs + 1),
        )

    def test_mamdani_snapshot_has_expected_parameter_names_and_shapes(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_mamdani_flc()
        snapshot = capture_parameter_snapshot(flc, step=3)
        self.assertEqual(
            set(snapshot.parameters),
            {
                "input_granulation.centers",
                "input_granulation.widths",
                "defuzzification.centers",
                "defuzzification.widths",
            },
        )
        self.assertEqual(snapshot.step, 3)

    def test_zero_order_snapshot_omits_defuzzification_parameters(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_zero_order_flc()
        snapshot = capture_parameter_snapshot(flc, step=0)
        self.assertEqual(
            set(snapshot.parameters),
            {"input_granulation.centers", "input_granulation.widths"},
        )

    def test_snapshot_is_a_detached_clone_not_a_live_view(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_tsk_flc()
        snapshot = capture_parameter_snapshot(flc, step=0)
        before = snapshot.parameters["input_granulation.centers"].clone()

        with torch.no_grad():
            flc.input_granulation.centers.add_(100.0)

        self.assertTrue(
            torch.equal(snapshot.parameters["input_granulation.centers"], before)
        )
        self.assertFalse(
            torch.equal(
                snapshot.parameters["input_granulation.centers"],
                flc.input_granulation.centers.cpu(),
            )
        )

    def test_snapshot_tensors_do_not_require_grad(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_tsk_flc()
        snapshot = capture_parameter_snapshot(flc, step=0)
        for tensor in snapshot.parameters.values():
            self.assertFalse(tensor.requires_grad)


if __name__ == "__main__":
    unittest.main()
