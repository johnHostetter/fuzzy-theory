"""
Test the fuzzy.logic.control.trace module.
"""

import unittest

import torch

from fuzzy.logic.control.trace import FLCTrace, StageTrace, capture_flc_trace
from fuzzy.logic.rule_helpers import tensor_to_rules
from fuzzy.sets.membership import Membership

from .demo_flcs import AVAILABLE_DEVICE
from .demo_flcs import build_mamdani_flc as _build_mamdani_flc
from .demo_flcs import build_tsk_flc as _build_tsk_flc


class TestCaptureFLCTrace(unittest.TestCase):
    """
    Test capture_flc_trace.
    """

    def test_stages_are_in_pipeline_order(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_tsk_flc()
        observations = torch.rand((3, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
        trace = capture_flc_trace(flc, observations)
        self.assertIsInstance(trace, FLCTrace)
        self.assertEqual(
            [stage.name for stage in trace.stages],
            ["input_granulation", "engine", "defuzzification"],
        )
        for stage in trace.stages:
            self.assertIsInstance(stage, StageTrace)
        self.assertEqual(trace.stages[0].module_type, "FuzzySetGroup")
        self.assertEqual(trace.stages[1].module_type, "Product")
        self.assertEqual(trace.stages[2].module_type, "TSK")
        self.assertIsInstance(trace.stages[0].output, Membership)
        self.assertIsInstance(trace.stages[1].output, Membership)

    def test_output_shape_matches_flc_shape(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_tsk_flc()
        observations = torch.rand((4, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
        trace = capture_flc_trace(flc, observations)
        self.assertEqual(tuple(trace.output.shape), (4, flc.shape.n_outputs))
        self.assertTrue(torch.equal(trace.observations, observations))
        self.assertEqual(trace.shape, flc.shape)

    def test_rule_tensor_and_rule_strings_match_tensor_to_rules(self) -> None:
        """
        Returns:
            None
        """
        flc = _build_tsk_flc()
        observations = torch.rand((2, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
        trace = capture_flc_trace(flc, observations)
        # rule_tensor's variable/rule dims must match the FLC's static shape; its
        # term dim must match the granulation layer's own observed term count for
        # this forward pass (NOT flc.shape.n_input_terms, which only reflects
        # rule-referenced terms and can be smaller than the granulation layer's
        # actual term count)
        self.assertEqual(trace.rule_tensor.shape[0], flc.shape.n_inputs)
        self.assertEqual(
            trace.rule_tensor.shape[1], trace.stages[0].output.mask.shape[1]
        )
        self.assertEqual(trace.rule_tensor.shape[2], flc.shape.n_rules)
        self.assertEqual(trace.rule_strings, tensor_to_rules(trace.rule_tensor))
        self.assertEqual(len(trace.rule_strings), flc.shape.n_rules)

    def test_works_for_mamdani_flc_too(self) -> None:
        """
        Detail panels are meant to be defuzzification-implementation-agnostic -
        confirm capture_flc_trace works the same way for Mamdani as for TSK.

        Returns:
            None
        """
        flc = _build_mamdani_flc()
        observations = torch.rand((2, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
        trace = capture_flc_trace(flc, observations)
        self.assertEqual(trace.stages[2].module_type, "Mamdani")
        self.assertEqual(tuple(trace.output.shape), (2, flc.shape.n_outputs))
        self.assertEqual(len(trace.rule_strings), flc.shape.n_rules)

    def test_rejects_flc_missing_expected_stages(self) -> None:
        """
        Returns:
            None
        """
        not_an_flc = torch.nn.Sequential()
        with self.assertRaises(ValueError):
            capture_flc_trace(not_an_flc, torch.rand((1, 1), device=AVAILABLE_DEVICE))


if __name__ == "__main__":
    unittest.main()
