"""
Test the fuzzy.logic.control.cuda_graph module.
"""

import math
import unittest
from unittest import mock

import torch

from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.cuda_graph import (
    GraphedTrainingStep,
    force_cuda_graph_safe_branches,
)
from fuzzy.logic.control.defuzzification import TSK
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.relations.triton_kernels import TRITON_AVAILABLE
from fuzzy.sets.abstract import FuzzySet
from tests import AVAILABLE_DEVICE

from .demo_flcs import build_tsk_flc, get_premises

SKIP_REASON = "requires a CUDA device with triton installed"
CUDA_GRAPH_AVAILABLE = TRITON_AVAILABLE and AVAILABLE_DEVICE.type == "cuda"


def _build_partial_coverage_flc() -> FLC:
    """
    A toy TSK FLC where no single rule references every input variable - unlike
    build_tsk_flc's rules, which all use every variable. Gives engine._all_active ==
    False, exercising _forced_gather_product_forward's general-path fallback (the
    branch it is only safe to take because torch.prod's own backward, not this
    module's code, has the real unconditional sync - see
    force_cuda_graph_safe_branches's docstring).

    Returns:
        A small toy TSK FuzzyLogicController with partial variable coverage.
    """
    premises = get_premises(AVAILABLE_DEVICE)
    rules = [
        Rule(
            premise=Product((0, 0), device=AVAILABLE_DEVICE),
            consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
        ),
        Rule(
            premise=Product((1, 0), device=AVAILABLE_DEVICE),
            consequence=NAryRelation((0, 1), device=AVAILABLE_DEVICE),
        ),
    ]
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=premises, targets=[]),
        rules=rules,
    )
    return FLC(source=knowledge_base, inference=TSK, device=AVAILABLE_DEVICE)


class TestGraphedTrainingStepValidation(unittest.TestCase):
    """
    Test GraphedTrainingStep's constructor argument validation - these checks run
    before anything CUDA-specific, so they are exercised on whatever device is
    available in this test environment.
    """

    def test_rejects_a_non_capturable_optimizer(self) -> None:
        """
        Returns:
            None
        """
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        with self.assertRaises(ValueError):
            GraphedTrainingStep(
                model,
                optimizer,
                torch.nn.MSELoss(),
                (4, 2),
                (4, 2),
                torch.device("cpu"),
            )

    def test_rejects_a_non_cuda_device(self) -> None:
        """
        Returns:
            None
        """
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, capturable=True)
        with self.assertRaises(ValueError):
            GraphedTrainingStep(
                model,
                optimizer,
                torch.nn.MSELoss(),
                (4, 2),
                (4, 2),
                torch.device("cpu"),
            )


class TestForceCudaGraphSafeBranchesWithoutTriton(unittest.TestCase):
    """
    Test force_cuda_graph_safe_branches's Triton requirement - exercised via mocking
    TRITON_AVAILABLE rather than requiring an environment without Triton installed.
    """

    def test_raises_when_triton_is_unavailable(self) -> None:
        """
        Returns:
            None
        """
        flc = build_tsk_flc()
        with mock.patch("fuzzy.logic.control.cuda_graph.TRITON_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                with force_cuda_graph_safe_branches(flc):
                    pass  # pragma: no cover - must never be reached


@unittest.skipUnless(CUDA_GRAPH_AVAILABLE, SKIP_REASON)
class TestForcedGatherProductForward(unittest.TestCase):
    """
    Test that _forced_gather_product_forward (installed by
    force_cuda_graph_safe_branches) reproduces the same values as the real
    Product.forward for NaN-free data - it must be numerically transparent, only
    skipping the sync check.
    """

    def test_matches_real_forward_for_nan_free_data(self) -> None:
        """
        Returns:
            None
        """
        flc = build_tsk_flc()
        observations = torch.rand(8, flc.shape.n_inputs, device=AVAILABLE_DEVICE)

        with torch.no_grad():
            granulated = flc.input_granulation(observations)
            real = flc.engine(granulated)

            with force_cuda_graph_safe_branches(flc):
                forced = flc.engine(granulated)

        self.assertTrue(torch.allclose(forced.degrees, real.degrees))

    def test_matches_real_forward_on_the_general_path_fallback(self) -> None:
        """
        Partial variable coverage (engine._all_active == False) must fall through to
        the general apply_mask()+.prod() path, not the fused Triton kernel (which only
        covers the every-variable-structurally-active case) - and still match the
        real, unpatched forward()'s values.

        Returns:
            None
        """
        flc = _build_partial_coverage_flc()
        self.assertFalse(flc.engine._all_active)  # pylint: disable=protected-access
        observations = torch.rand(8, flc.shape.n_inputs, device=AVAILABLE_DEVICE)

        with torch.no_grad():
            granulated = flc.input_granulation(observations)
            real = flc.engine(granulated)

            with force_cuda_graph_safe_branches(flc):
                forced = flc.engine(granulated)

        self.assertTrue(torch.allclose(forced.degrees, real.degrees))


@unittest.skipUnless(CUDA_GRAPH_AVAILABLE, SKIP_REASON)
class TestForceCudaGraphSafeBranches(unittest.TestCase):
    """
    Test force_cuda_graph_safe_branches's patch/restore behavior.
    """

    def test_sets_eval_mode_and_restores_training_mode_after(self) -> None:
        """
        Returns:
            None
        """
        flc = build_tsk_flc()
        flc.train()
        with force_cuda_graph_safe_branches(flc):
            self.assertFalse(flc.training)
        self.assertTrue(flc.training)

    def test_does_not_restore_training_mode_if_it_started_in_eval(self) -> None:
        """
        Returns:
            None
        """
        flc = build_tsk_flc()
        flc.eval()
        with force_cuda_graph_safe_branches(flc):
            self.assertFalse(flc.training)
        self.assertFalse(flc.training)

    def test_restores_nan_safe_thresholds_after(self) -> None:
        """
        Returns:
            None
        """
        flc = build_tsk_flc()
        fuzzy_sets = [m for m in flc.modules() if isinstance(m, FuzzySet)]
        originals = [
            m._nan_safe_sync_threshold_numel  # pylint: disable=protected-access
            for m in fuzzy_sets
        ]
        with force_cuda_graph_safe_branches(flc):
            for m in fuzzy_sets:
                self.assertEqual(
                    m._nan_safe_sync_threshold_numel,
                    2**62,  # pylint: disable=protected-access
                )
        for m, original in zip(fuzzy_sets, originals):
            self.assertEqual(
                m._nan_safe_sync_threshold_numel,
                original,  # pylint: disable=protected-access
            )

    def test_restores_product_forward_after(self) -> None:
        """
        Returns:
            None
        """
        original_forward = Product.forward
        flc = build_tsk_flc()
        with force_cuda_graph_safe_branches(flc):
            self.assertIsNot(Product.forward, original_forward)
        self.assertIs(Product.forward, original_forward)

    def test_restores_state_even_if_the_body_raises(self) -> None:
        """
        Returns:
            None
        """
        original_forward = Product.forward
        flc = build_tsk_flc()
        flc.train()
        with self.assertRaises(ZeroDivisionError):
            with force_cuda_graph_safe_branches(flc):
                raise ZeroDivisionError("simulated failure during capture")
        self.assertIs(Product.forward, original_forward)
        self.assertTrue(flc.training)


@unittest.skipUnless(CUDA_GRAPH_AVAILABLE, SKIP_REASON)
class TestGraphedTrainingStep(unittest.TestCase):
    """
    Test GraphedTrainingStep end-to-end: capture, replay, and cleanup.
    """

    def _build_graphed(self) -> GraphedTrainingStep:
        flc = build_tsk_flc()
        optimizer = torch.optim.Adam(flc.parameters(), lr=3e-2, capturable=True)
        with force_cuda_graph_safe_branches(flc):
            return GraphedTrainingStep(
                flc,
                optimizer,
                torch.nn.MSELoss(),
                static_input_shape=(8, flc.shape.n_inputs),
                static_target_shape=(8, flc.shape.n_outputs),
                device=AVAILABLE_DEVICE,
                n_warmup=3,
            )

    def test_replay_produces_finite_decreasing_loss(self) -> None:
        """
        Returns:
            None
        """
        graphed = self._build_graphed()
        try:
            losses = []
            for _ in range(50):
                x = torch.rand(8, 2, device=AVAILABLE_DEVICE)
                y = torch.rand(8, 1, device=AVAILABLE_DEVICE)
                losses.append(graphed.step(x, y, sync=True))
            self.assertTrue(all(not math.isnan(loss) for loss in losses))
            self.assertLess(losses[-1], losses[0])
        finally:
            graphed.close()

    def test_step_without_sync_returns_a_live_tensor(self) -> None:
        """
        Returns:
            None
        """
        graphed = self._build_graphed()
        try:
            x = torch.rand(8, 2, device=AVAILABLE_DEVICE)
            y = torch.rand(8, 1, device=AVAILABLE_DEVICE)
            loss = graphed.step(x, y, sync=False)
            self.assertIsInstance(loss, torch.Tensor)
        finally:
            graphed.close()

    def test_works_with_a_plain_non_flc_model(self) -> None:
        """
        GraphedTrainingStep is model-agnostic - only force_cuda_graph_safe_branches is
        FLC-specific.

        Returns:
            None
        """
        model = torch.nn.Linear(4, 2).to(AVAILABLE_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, capturable=True)
        graphed = GraphedTrainingStep(
            model,
            optimizer,
            torch.nn.MSELoss(),
            static_input_shape=(8, 4),
            static_target_shape=(8, 2),
            device=AVAILABLE_DEVICE,
            n_warmup=3,
        )
        try:
            x = torch.rand(8, 4, device=AVAILABLE_DEVICE)
            y = torch.rand(8, 2, device=AVAILABLE_DEVICE)
            loss = graphed.step(x, y, sync=True)
            self.assertFalse(math.isnan(loss))
        finally:
            graphed.close()

    def test_context_manager_closes_on_exit(self) -> None:
        """
        Returns:
            None
        """
        model = torch.nn.Linear(4, 2).to(AVAILABLE_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, capturable=True)
        with GraphedTrainingStep(
            model,
            optimizer,
            torch.nn.MSELoss(),
            static_input_shape=(8, 4),
            static_target_shape=(8, 2),
            device=AVAILABLE_DEVICE,
            n_warmup=3,
        ) as graphed:
            x = torch.rand(8, 4, device=AVAILABLE_DEVICE)
            y = torch.rand(8, 2, device=AVAILABLE_DEVICE)
            graphed.step(x, y)
        self.assertFalse(hasattr(graphed, "graph"))


if __name__ == "__main__":
    unittest.main()
