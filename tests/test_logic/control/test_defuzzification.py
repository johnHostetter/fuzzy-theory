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
            degrees=degrees,
            mask=torch.ones(3, device=AVAILABLE_DEVICE),
            formula="test",
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


class TestTSKRandomInitFanInScaling(unittest.TestCase):
    """
    Regression coverage for TSK's source=None consequent initialization: plain
    torch.randn (no fan-in scaling) gave every weight unit variance regardless of
    n_inputs, so for a large n_inputs (e.g. a CNN feature vector) the resulting
    output logits scaled as sqrt(n_inputs) - confirmed directly to prevent even
    memorizing 64 training examples across 150 epochs, independent of learning
    rate. Only the WEIGHT portion should be scaled - the bias (index 0 along the
    consequences' last dim) has no fan-in dependency and must stay unscaled,
    matching the historical torch.zero-initialized bias this replaced.
    """

    def test_weight_std_scales_down_with_n_inputs(self) -> None:
        torch.manual_seed(0)
        small_shape = Shape(
            n_inputs=4, n_input_terms=3, n_rules=8, n_outputs=2, n_output_terms=0
        )
        small_tsk = TSK(
            shape=small_shape, source=None, device=AVAILABLE_DEVICE, rule_base=None
        )

        torch.manual_seed(0)
        large_shape = Shape(
            n_inputs=512, n_input_terms=3, n_rules=8, n_outputs=2, n_output_terms=0
        )
        large_tsk = TSK(
            shape=large_shape, source=None, device=AVAILABLE_DEVICE, rule_base=None
        )

        # same seed, same torch.randn call shape-for-shape up to n_inputs - the
        # LARGER n_inputs' weight std should be smaller by very close to the
        # expected sqrt(4/512) ratio, not identical (unscaled) or arbitrary.
        expected_ratio = (4 / 512) ** 0.5
        actual_ratio = large_tsk.weights.std().item() / small_tsk.weights.std().item()
        self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.05)

    def test_weight_std_is_close_to_one_over_sqrt_n_inputs(self) -> None:
        n_inputs = 512
        shape = Shape(
            n_inputs=n_inputs,
            n_input_terms=3,
            n_rules=16,
            n_outputs=4,
            n_output_terms=0,
        )
        tsk = TSK(shape=shape, source=None, device=AVAILABLE_DEVICE, rule_base=None)
        expected_std = 1.0 / (n_inputs**0.5)
        self.assertAlmostEqual(tsk.weights.std().item(), expected_std, delta=0.02)

    def test_bias_is_not_rescaled_by_n_inputs(self) -> None:
        """The bias (unlike the weights) has no fan-in dependency - it must keep
        its original unit-ish variance from torch.randn regardless of n_inputs,
        not be caught by the same scaling applied to the weight portion."""
        shape = Shape(
            n_inputs=512, n_input_terms=3, n_rules=16, n_outputs=4, n_output_terms=0
        )
        tsk = TSK(shape=shape, source=None, device=AVAILABLE_DEVICE, rule_base=None)
        # plain torch.randn has std ~1.0 - well outside the ~0.044 the weights
        # themselves land at (1/sqrt(512)) if the bias were wrongly scaled too.
        self.assertGreater(tsk.bias.std().item(), 0.5)

    def test_source_provided_is_never_rescaled(self) -> None:
        """The fan-in scaling only applies to the source=None random-init path -
        an explicitly provided source (e.g. a previously-trained or caller-
        constructed consequence tensor) must be used exactly as given."""
        shape = Shape(
            n_inputs=512, n_input_terms=3, n_rules=3, n_outputs=2, n_output_terms=0
        )
        source = np.ones(
            (shape.n_outputs, shape.n_rules, shape.n_inputs + 1), dtype=np.float32
        )
        tsk = TSK(shape=shape, source=source, device=AVAILABLE_DEVICE, rule_base=None)
        self.assertTrue(torch.allclose(tsk.weights, torch.ones_like(tsk.weights)))


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
            degrees=degrees,
            mask=torch.ones(2, 2, device=AVAILABLE_DEVICE),
            formula="test",
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


class _ThirdPartyTSKStyle(Defuzzification):
    """
    Stands in for a TSK-style Defuzzification subclass defined outside
    fuzzy-theory entirely (e.g. a downstream project's own CP-decomposed TSK
    variant) - forward() requires observations (no default), matching TSK's
    own contract, without fuzzy-theory needing to import or know about this
    class at all.
    """

    def forward(
        self, rule_activations: Membership, observations: torch.Tensor
    ) -> torch.Tensor:
        return observations.sum(dim=-1, keepdim=True) * rule_activations.degrees.sum(
            dim=-1, keepdim=True
        )

    def save(self, path: Path) -> None:
        raise NotImplementedError("Not needed for this test stand-in.")


class _ThirdPartyMamdaniStyle(Defuzzification):
    """
    Stands in for a Mamdani-style (i.e. non-TSK) Defuzzification subclass
    defined outside fuzzy-theory - forward() gives observations a default of
    None (unused, present only for a uniform call signature).
    """

    def forward(
        self, rule_activations: Membership, observations: torch.Tensor = None
    ) -> torch.Tensor:
        return rule_activations.degrees.sum(dim=-1, keepdim=True)

    def save(self, path: Path) -> None:
        raise NotImplementedError("Not needed for this test stand-in.")


class _MissingObservationsParam(Defuzzification):
    """
    Violates the Defuzzification contract: forward() doesn't declare an
    "observations" parameter at all, so the FLC has no way to determine which
    call convention to use.
    """

    def forward(  # pylint: disable=arguments-differ
        self, rule_activations: Membership
    ) -> torch.Tensor:
        return rule_activations.degrees

    def save(self, path: Path) -> None:
        raise NotImplementedError("Not needed for this test stand-in.")


class TestFLCDefuzzifyDispatch(unittest.TestCase):
    """
    Regression tests for FuzzyLogicController's TSK-vs-Mamdani dispatch: it
    must be decided by inspecting the concrete defuzzification class's own
    forward() signature (whether "observations" is required or defaulted),
    not by an isinstance() check against a fixed, closed list of known
    subclasses - the latter would require fuzzy-theory to import every
    downstream subclass that ever wants TSK-style dispatch, creating a
    backwards dependency (this was a real bug: controller.py used to import
    a PySoft-only class for exactly this purpose).
    """

    def test_dispatches_a_third_party_tsk_style_subclass_correctly(self) -> None:
        """
        A subclass fuzzy-theory has never heard of, whose forward() requires
        observations, must still be routed through _defuzzify_tsk (so
        observations actually get passed through) purely because of its
        signature.
        """
        knowledge_base, _ = build_mamdani_knowledge_base(AVAILABLE_DEVICE)
        flc = FLC(
            source=knowledge_base,
            inference=_ThirdPartyTSKStyle,
            device=AVAILABLE_DEVICE,
        )
        self.assertEqual(flc._defuzzify, flc._defuzzify_tsk)

        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1]], device=AVAILABLE_DEVICE
        )
        output = flc(input_data)  # must not raise (observations reach forward())
        self.assertEqual(output.shape[0], input_data.shape[0])

    def test_dispatches_a_third_party_mamdani_style_subclass_correctly(self) -> None:
        """
        A subclass whose forward() gives observations a default must be
        routed through _defuzzify_standard.
        """
        knowledge_base, _ = build_mamdani_knowledge_base(AVAILABLE_DEVICE)
        flc = FLC(
            source=knowledge_base,
            inference=_ThirdPartyMamdaniStyle,
            device=AVAILABLE_DEVICE,
        )
        self.assertEqual(flc._defuzzify, flc._defuzzify_standard)

        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1]], device=AVAILABLE_DEVICE
        )
        output = flc(input_data)
        self.assertEqual(output.shape[0], input_data.shape[0])

    def test_raises_a_clear_error_when_forward_has_no_observations_parameter(
        self,
    ) -> None:
        """
        A Defuzzification subclass that omits "observations" from forward()
        entirely violates the contract the FLC's dispatch relies on - this
        must fail loudly and informatively at FLC construction time, not with
        a bare KeyError or a silent misdispatch.
        """
        knowledge_base, _ = build_mamdani_knowledge_base(AVAILABLE_DEVICE)
        with self.assertRaisesRegex(TypeError, "observations"):
            FLC(
                source=knowledge_base,
                inference=_MissingObservationsParam,
                device=AVAILABLE_DEVICE,
            )


if __name__ == "__main__":
    unittest.main()
