"""
Test the Defuzzification hierarchy (ZeroOrder, NormalizedZeroOrder, TSK, Mamdani)
directly - construction, save/load round trips, and forward-pass correctness -
independent of a full FuzzyLogicController.
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from fuzzy.logic.control.configurations.data import Shape
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import (
    TSK,
    Defuzzification,
    Mamdani,
    NormalizedZeroOrder,
    ZeroOrder,
)
from fuzzy.sets.group import FuzzySetGroup
from fuzzy.sets.impl import Gaussian
from fuzzy.sets.membership import Membership
from tests import AVAILABLE_DEVICE

from .common import build_mamdani_knowledge_base


class TestZeroOrder(unittest.TestCase):
    """
    Test ZeroOrder's save()/load() round trip, construction from a FuzzySetGroup
    source, and Defuzzification.load()'s UnicodeDecodeError fallback (used when
    reading an older, differently-encoded saved file).
    """

    def setUp(self) -> None:
        self.shape = Shape(
            n_inputs=2, n_input_terms=3, n_rules=4, n_outputs=2, n_output_terms=0
        )
        self.zero_order = ZeroOrder(
            shape=self.shape,
            source=None,
            device=AVAILABLE_DEVICE,
            rule_base=None,
        )

    def test_construction_from_fuzzy_set_group_source(self) -> None:
        """
        Coverage/regression test: ZeroOrder's source=FuzzySetGroup branch (used when
        consequences are tied to actual fuzzy sets, e.g. an output granulation
        layer, rather than raw numbers) was never exercised - every FLC-level
        construction path only ever passes source=None (see
        configurations/abstract.py's FuzzySystem.defuzzification(), which only
        forwards the output granulation layer for Mamdani, not ZeroOrder/TSK).

        Returns:
            None
        """
        shape = Shape(
            n_inputs=2, n_input_terms=3, n_rules=1, n_outputs=3, n_output_terms=0
        )
        source = FuzzySetGroup(
            modules_list=[
                Gaussian(
                    centers=np.array([0.1, 0.2, 0.3]),
                    widths=np.array([0.1, 0.1, 0.1]),
                    device=AVAILABLE_DEVICE,
                )
            ]
        )
        zero_order = ZeroOrder(
            shape=shape, source=source, device=AVAILABLE_DEVICE, rule_base=None
        )
        self.assertTrue(torch.equal(zero_order.consequences, source.centers))

    def test_save_and_load_round_trip(self) -> None:
        """
        Coverage/regression test: ZeroOrder.load() (which used to be its own,
        broken override of the base Defuzzification.load() - see git history) had
        no test coverage at all.

        Returns:
            None
        """
        path = Path("test_zero_order_save_and_load.pt")
        self.zero_order.save(path)
        try:
            loaded = ZeroOrder.load(path, device=AVAILABLE_DEVICE)
            self.assertTrue(
                torch.allclose(self.zero_order.consequences, loaded.consequences)
            )
            self.assertEqual(loaded.shape, self.shape)
        finally:
            path.unlink(missing_ok=True)

    def test_load_falls_back_to_latin1_on_unicode_decode_error(self) -> None:
        """
        Coverage/regression test: Defuzzification.load() must retry with
        encoding="latin1" when torch.load() raises UnicodeDecodeError (the
        documented symptom of reading an older-format saved file), rather than
        propagating the error.

        Returns:
            None
        """
        path = Path("test_defuzzification_unicode_fallback.pt")
        # save it the normal way, then verify load() still succeeds even when the
        # first torch.load() attempt is forced to raise UnicodeDecodeError
        state_dict = self.zero_order.state_dict()
        state_dict["class_name"] = "ZeroOrder"
        state_dict["shape"] = tuple(self.shape)
        state_dict["source"] = (
            self.zero_order.consequences.detach()  # pylint: disable=not-callable
            .cpu()
            .numpy()
        )
        torch.save(state_dict, path)
        try:
            real_torch_load = torch.load
            call_count = 0

            def flaky_load(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise UnicodeDecodeError(
                        "utf-8", b"\xde", 0, 1, "invalid continuation byte"
                    )
                return real_torch_load(*args, **kwargs)

            with mock.patch("torch.load", side_effect=flaky_load):
                loaded = Defuzzification.load(path, device=AVAILABLE_DEVICE)
            self.assertEqual(call_count, 2)
            self.assertTrue(
                torch.allclose(self.zero_order.consequences, loaded.consequences)
            )
        finally:
            path.unlink(missing_ok=True)


class TestNormalizedZeroOrder(unittest.TestCase):
    """
    Test NormalizedZeroOrder's forward pass - not exercised anywhere previously
    (no FLC test builds one).
    """

    def test_forward_matches_manual_normalization(self) -> None:
        """
        Returns:
            None
        """
        shape = Shape(
            n_inputs=2, n_input_terms=3, n_rules=3, n_outputs=2, n_output_terms=0
        )
        source = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        normalized = NormalizedZeroOrder(
            shape=shape, source=source, device=AVAILABLE_DEVICE, rule_base=None
        )
        degrees = torch.tensor(
            [[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]], device=AVAILABLE_DEVICE
        )
        rule_activations = Membership(
            degrees=degrees, mask=torch.ones(3, device=AVAILABLE_DEVICE)
        )

        output = normalized(rule_activations)

        numerator = degrees.unsqueeze(dim=-1) * normalized.consequences
        expected = numerator.sum(dim=1) / (degrees.sum(dim=1, keepdim=True) + 1e-32)
        self.assertTrue(torch.allclose(output, expected))


class TestTSK(unittest.TestCase):
    """
    Test TSK's construction-from-explicit-source path, its consequences property,
    save()/load() round trip, and .to() device move - none of which are exercised
    by the FLC-level TSK tests (those only ever build TSK with source=None).
    """

    def setUp(self) -> None:
        self.shape = Shape(
            n_inputs=2, n_input_terms=3, n_rules=3, n_outputs=2, n_output_terms=0
        )
        # (n_outputs, n_rules, n_inputs + 1): column 0 is bias, the rest are weights
        self.source = (
            np.random.default_rng(0)
            .random((self.shape.n_outputs, self.shape.n_rules, self.shape.n_inputs + 1))
            .astype(np.float32)
        )
        self.tsk = TSK(
            shape=self.shape,
            source=self.source,
            device=AVAILABLE_DEVICE,
            rule_base=None,
        )

    def test_consequences_property_reconstructs_source(self) -> None:
        """
        Returns:
            None
        """
        self.assertTrue(
            torch.allclose(
                self.tsk.consequences,
                torch.as_tensor(self.source, device=AVAILABLE_DEVICE),
            )
        )

    def test_save_and_load_round_trip(self) -> None:
        """
        Returns:
            None
        """
        path = Path("test_tsk_save_and_load.pt")
        self.tsk.save(path)
        try:
            loaded = TSK.load(path, device=AVAILABLE_DEVICE)
            self.assertTrue(torch.allclose(self.tsk.consequences, loaded.consequences))
        finally:
            path.unlink(missing_ok=True)

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires a second device (CUDA) to move to"
    )
    def test_to_moves_weights_and_bias(self) -> None:
        """
        Regression guard: TSK.to() must move both self.weights and self.bias -
        easy to miss since neither is registered the same way a plain nn.Parameter
        assigned directly in __init__ normally would be picked up automatically by
        the default nn.Module.to() for every attribute (it is, but the point is
        this override exists and must not silently no-op).

        Returns:
            None
        """
        tsk_cpu = TSK(
            shape=self.shape,
            source=self.source,
            device=torch.device("cpu"),
            rule_base=None,
        )
        self.assertEqual("cpu", tsk_cpu.weights.device.type)
        self.assertEqual("cpu", tsk_cpu.bias.device.type)

        tsk_cpu.to(torch.device("cuda"))

        self.assertEqual("cuda", tsk_cpu.weights.device.type)
        self.assertEqual("cuda", tsk_cpu.bias.device.type)


class _FakeConsequenceMask:  # pylint: disable=too-few-public-methods
    """
    A minimal stand-in for RuleBase.consequences, exposing only the get_mask()
    method Mamdani.__init__ actually reads - avoids building a full KnowledgeBase
    just to get a rule/term link mask with a specific, hand-chosen shape.
    """

    def __init__(self, mask: torch.Tensor):
        self._mask = mask

    def get_mask(self) -> torch.Tensor:
        """
        Returns:
            The (vars, terms, rules) link mask this instance was built with.
        """
        return self._mask


class _FakeRuleBase:  # pylint: disable=too-few-public-methods
    """
    A minimal stand-in for RuleBase, exposing only the .consequences attribute
    Mamdani.__init__ actually reads.
    """

    def __init__(self, mask: torch.Tensor):
        self.consequences = _FakeConsequenceMask(mask)


class TestMamdani(unittest.TestCase):
    """
    Test Mamdani's documented not-yet-implemented save() guard, and its forward()
    "height method" center-of-gravity approximation.
    """

    def test_forward_matches_height_method_formula(self) -> None:
        """
        Golden-value/drift-detection test, and regression guard for a real bug:
        Mamdani.forward() used to compute each rule's own
        (link * center * width) / (link * width) ratio before multiplying by
        firing strength and summing (unnormalized) over rules. Since output_links
        is one-hot for the standard one-term-per-rule case, width canceled out of
        that ratio identically every time - consequent widths could never receive
        a gradient, and the final sum over rules was never normalized by the total
        weight either (so the output scaled with the number of active rules
        instead of being a valid weighted average).

        Fixed to the standard "height method" approximation of center-of-gravity
        defuzzification: each rule's clipped-consequent area is approximated as
        proportional to its width, and the output is a single, properly
        normalized, firing-strength-and-width-weighted average of centers -
        computed here independently (not by calling any of Mamdani's own code)
        for two rules mapping into the same single output variable via two
        different terms.

        Returns:
            None
        """
        shape = Shape(
            n_inputs=1, n_input_terms=2, n_rules=2, n_outputs=1, n_output_terms=2
        )
        consequent = Gaussian(
            centers=np.array([-2.0, 4.0]),
            widths=np.array([1.0, 3.0]),
            device=AVAILABLE_DEVICE,
        )
        # (vars=1, terms=2, rules=2): rule 0 selects term 0, rule 1 selects term 1
        mask = torch.tensor([[[1, 0], [0, 1]]], device=AVAILABLE_DEVICE)
        mamdani = Mamdani(
            shape=shape,
            source=FuzzySetGroup(modules_list=[consequent]),
            device=AVAILABLE_DEVICE,
            rule_base=_FakeRuleBase(mask),
        )

        degrees = torch.tensor([[0.6, 0.9]], device=AVAILABLE_DEVICE)
        rule_activations = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )
        result = mamdani(rule_activations)

        firing_strengths, centers, widths = [0.6, 0.9], [-2.0, 4.0], [1.0, 3.0]
        numerator = sum(f * w * c for f, w, c in zip(firing_strengths, widths, centers))
        denominator = sum(f * w for f, w in zip(firing_strengths, widths))
        expected = torch.tensor([[numerator / denominator]], device=AVAILABLE_DEVICE)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_widths_receive_nonzero_gradient(self) -> None:
        """
        Regression guard for the same bug test_forward_matches_height_method_formula
        documents: consequent widths are real, trainable Parameters
        (requires_grad=True) - confirms backward() actually reaches them with a
        non-zero, NaN-free gradient through a full FuzzyLogicController, not just
        the isolated Mamdani.forward() call above.

        Returns:
            None
        """
        knowledge_base, _ = build_mamdani_knowledge_base(AVAILABLE_DEVICE)
        flc = FLC(source=knowledge_base, inference=Mamdani, device=AVAILABLE_DEVICE)
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1]], device=AVAILABLE_DEVICE
        )

        predicted_y = flc(input_data)
        predicted_y.sum().backward()

        widths = flc.defuzzification.consequences.widths
        self.assertIsNotNone(widths.grad)
        self.assertFalse(bool(widths.grad.isnan().any()))
        self.assertFalse(bool((widths.grad == 0).all()))

    def test_save_raises_not_implemented(self) -> None:
        """
        Coverage/regression test: Mamdani.save() must raise NotImplementedError with
        an informative message, rather than e.g. silently doing nothing or crashing
        with an unrelated error, since Mamdani defuzzification does not yet support
        saving.

        Returns:
            None
        """
        knowledge_base, _ = build_mamdani_knowledge_base(AVAILABLE_DEVICE)
        flc = FLC(source=knowledge_base, inference=Mamdani, device=AVAILABLE_DEVICE)

        with self.assertRaises(NotImplementedError):
            flc.defuzzification.save(Path("should_not_be_created"))
        self.assertFalse(Path("should_not_be_created").exists())


if __name__ == "__main__":
    unittest.main()
