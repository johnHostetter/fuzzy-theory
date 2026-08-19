"""
Smoke tests for fuzzy.dashboard.flc_viewer.

No Selenium/dash[testing] - matches the precedent set by activation_viewer.py (no
tests at all, since Dash/matplotlib rendering glue isn't where the bugs that matter
live). Instead: the pure data-prep helpers (_cytoscape_elements,
_render_stage_detail) are unit-tested directly, and one smoke test confirms
build_flc_dashboard assembles a working dash.Dash app.
"""

import unittest
from typing import Tuple

import dash
import numpy as np
import torch

from fuzzy.dashboard.flc_viewer import (
    _cytoscape_elements,
    _default_bin_width,
    _defuzzification_detail,
    _firing_strength_histogram_figure,
    _input_granulation_curve_figure,
    _input_granulation_variable_plots,
    _mamdani_rule_consequence_figure,
    _membership_histogram_figure,
    _render_stage_detail,
    _rule_dropdown_options,
    _tsk_rule_weights_figure,
    _variable_dropdown_options,
    _variable_histogram_figure,
    build_flc_dashboard,
    generate_flc_report,
)
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import TSK, Mamdani, ZeroOrder
from fuzzy.logic.control.trace import FLCTrace, capture_flc_trace
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.sets.impl import Gaussian

from ..test_logic.control.demo_flcs import AVAILABLE_DEVICE, toy_mamdani, toy_tsk
from .common import collect_component_ids as _collect_component_ids


def _toy_flc_and_observations() -> Tuple[FLC, torch.Tensor]:
    """
    Returns:
        A small toy TSK FuzzyLogicController and a batch of observations for it.
    """
    antecedents, _, rules = toy_tsk(t_norm=Product, device=AVAILABLE_DEVICE)
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
        rules=rules,
    )
    flc = FLC(source=knowledge_base, inference=TSK, device=AVAILABLE_DEVICE)
    observations = torch.rand((3, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
    return flc, observations


def _toy_trace() -> FLCTrace:
    """
    Returns:
        A captured trace of a small toy TSK FuzzyLogicController's forward pass.
    """
    flc, observations = _toy_flc_and_observations()
    return capture_flc_trace(flc, observations)


def _toy_mamdani_trace() -> FLCTrace:
    """
    Returns:
        A captured trace of a small toy Mamdani FuzzyLogicController's forward pass.
    """
    antecedents, consequents, rules = toy_mamdani(
        t_norm=Product, device=AVAILABLE_DEVICE
    )
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=antecedents, targets=consequents
        ),
        rules=rules,
    )
    flc = FLC(source=knowledge_base, inference=Mamdani, device=AVAILABLE_DEVICE)
    observations = torch.rand((3, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
    return capture_flc_trace(flc, observations)


def _toy_zero_order_trace() -> FLCTrace:
    """
    A defuzzification method _defuzzification_detail has no method-specific
    rendering for, to exercise its generic fallback branch.

    Returns:
        A captured trace of a small toy ZeroOrder FuzzyLogicController's forward
        pass.
    """
    antecedents, _, rules = toy_tsk(t_norm=Product, device=AVAILABLE_DEVICE)
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
        rules=rules,
    )
    flc = FLC(source=knowledge_base, inference=ZeroOrder, device=AVAILABLE_DEVICE)
    observations = torch.rand((3, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
    return capture_flc_trace(flc, observations)


def _toy_mamdani_trace_with_unlinked_output_variable() -> FLCTrace:
    """
    A Mamdani FLC with 2 output variables where rule 0's consequence only links
    output variable 0, leaving output variable 1 with no linked term at all - the
    edge case _mamdani_rule_consequence_figure's "skip this output variable
    entirely" branch handles (unreachable via demo_flcs.toy_mamdani, where every
    rule links every output variable).

    Returns:
        A captured trace of that FLC's forward pass.
    """
    antecedent = Gaussian(
        centers=np.array([0.0, 2.0]),
        widths=np.array([1.0, 1.0]),
        device=AVAILABLE_DEVICE,
    )
    consequent_0 = Gaussian(
        centers=np.array([0.5, -0.5]),
        widths=np.array([0.3, 0.3]),
        device=AVAILABLE_DEVICE,
    )
    consequent_1 = Gaussian(
        centers=np.array([1.0, -1.0]),
        widths=np.array([0.3, 0.3]),
        device=AVAILABLE_DEVICE,
    )
    rules = [
        Rule(
            premise=Product((0, 0), device=AVAILABLE_DEVICE),
            consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
        ),
        Rule(
            premise=Product((0, 1), device=AVAILABLE_DEVICE),
            consequence=NAryRelation((0, 1), (1, 0), device=AVAILABLE_DEVICE),
        ),
    ]
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=[antecedent], targets=[consequent_0, consequent_1]
        ),
        rules=rules,
    )
    flc = FLC(source=knowledge_base, inference=Mamdani, device=AVAILABLE_DEVICE)
    observations = torch.rand((3, flc.shape.n_inputs), device=AVAILABLE_DEVICE)
    return capture_flc_trace(flc, observations)


class TestCytoscapeElements(unittest.TestCase):
    """
    Test _cytoscape_elements.
    """

    def test_five_nodes_and_four_edges(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        elements = _cytoscape_elements(trace)
        nodes = [e for e in elements if "source" not in e["data"]]
        edges = [e for e in elements if "source" in e["data"]]
        self.assertEqual(len(nodes), 5)
        self.assertEqual(len(edges), 4)
        node_ids = {node["data"]["id"] for node in nodes}
        self.assertEqual(
            node_ids,
            {
                "observations",
                "input_granulation",
                "engine",
                "defuzzification",
                "output",
            },
        )

    def test_edges_chain_the_pipeline_in_order(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        elements = _cytoscape_elements(trace)
        edges = [e["data"] for e in elements if "source" in e["data"]]
        self.assertEqual(
            [(e["source"], e["target"]) for e in edges],
            [
                ("observations", "input_granulation"),
                ("input_granulation", "engine"),
                ("engine", "defuzzification"),
                ("defuzzification", "output"),
            ],
        )


class TestRenderStageDetail(unittest.TestCase):
    """
    Test _render_stage_detail.
    """

    def test_renders_every_known_node(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        for node_id in (
            "observations",
            "input_granulation",
            "engine",
            "defuzzification",
            "output",
        ):
            with self.subTest(node_id=node_id):
                detail = _render_stage_detail(trace, node_id)
                self.assertIsNotNone(detail)

    def test_rejects_unknown_node_id(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        with self.assertRaises(ValueError):
            _render_stage_detail(trace, "not-a-real-node")


class TestBuildFlcDashboard(unittest.TestCase):
    """
    Test build_flc_dashboard.
    """

    def test_returns_dash_app_with_expected_component_ids(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        app = build_flc_dashboard(trace)
        self.assertIsInstance(app, dash.Dash)

        component_ids = _collect_component_ids(app.layout)
        self.assertIn("flc-graph", component_ids)
        self.assertIn("detail-panel", component_ids)

    def test_no_component_id_is_shared_across_panels(self) -> None:
        """
        Regression guard: every per-panel dropdown/bin-width/histogram id used to be
        shared across all four batched-per-item panels (observations/output/
        input_granulation/engine), e.g. "variable-dropdown" existed simultaneously
        in every panel's rendered tree, disambiguated only by a stored context key.
        That broke with a "nonexistent object" error the moment you interacted with
        a control while viewing a panel that didn't render every id the shared
        callback's Output list referenced (e.g. engine's dropdown, since
        input_granulation's curve-figure output wasn't part of engine's rendered
        tree) - Dash's client-side callback dispatch requires every Output a
        callback declares to currently exist in the DOM whenever its Input fires.
        Each panel now gets its own dedicated ids instead; this directly asserts
        that root property: no id appears in more than one panel's component tree.

        Returns:
            None
        """
        trace = _toy_trace()
        panel_ids = {
            node_id: _collect_component_ids(_render_stage_detail(trace, node_id))
            for node_id in (
                "observations",
                "input_granulation",
                "engine",
                "defuzzification",
                "output",
            )
        }
        panels = list(panel_ids.items())
        for i, (name_a, ids_a) in enumerate(panels):
            for name_b, ids_b in panels[i + 1 :]:
                with self.subTest(panels=(name_a, name_b)):
                    self.assertFalse(
                        ids_a & ids_b,
                        f"{name_a} and {name_b} share ids: {ids_a & ids_b}",
                    )

    def test_every_callbacks_inputs_and_outputs_share_one_panel(self) -> None:
        """
        Follow-on check to test_no_component_id_is_shared_across_panels: given ids
        aren't shared across panels, every registered callback's Input/Output ids
        must all belong to the SAME single panel's rendered component tree (except
        the graph-click callback, which legitimately spans the whole page) - i.e.
        no callback was accidentally wired to ids drawn from more than one panel.

        Returns:
            None
        """
        trace = _toy_trace()
        panel_ids = {
            node_id: _collect_component_ids(_render_stage_detail(trace, node_id))
            for node_id in (
                "observations",
                "input_granulation",
                "engine",
                "defuzzification",
                "output",
            )
        }

        app = build_flc_dashboard(trace)
        for key, callback in app.callback_map.items():
            outputs = callback["output"]
            if not isinstance(outputs, list):
                outputs = [outputs]
            io_ids = {output.component_id for output in outputs}
            io_ids |= {inp["id"] for inp in callback.get("inputs", [])}
            if "flc-graph" in io_ids or "detail-panel" in io_ids:
                continue  # the graph-click callback spans the whole page by design
            with self.subTest(callback=key):
                self.assertTrue(
                    any(io_ids <= ids for ids in panel_ids.values()),
                    f"callback {key} mixes ids from more than one panel: {io_ids}",
                )


class TestVariableDropdownOptions(unittest.TestCase):
    """
    Test _variable_dropdown_options.
    """

    def test_one_option_per_variable(self) -> None:
        """
        Returns:
            None
        """
        options = _variable_dropdown_options(3)
        self.assertEqual(
            options,
            [
                {"label": "variable 0", "value": 0},
                {"label": "variable 1", "value": 1},
                {"label": "variable 2", "value": 2},
            ],
        )

    def test_scales_to_many_variables(self) -> None:
        """
        Returns:
            None
        """
        options = _variable_dropdown_options(1500)
        self.assertEqual(len(options), 1500)
        self.assertEqual(options[-1], {"label": "variable 1499", "value": 1499})


class TestVariableHistogramFigure(unittest.TestCase):
    """
    Test _variable_histogram_figure.
    """

    def test_histogram_uses_the_selected_column_only(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.tensor([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        fig = _variable_histogram_figure(tensor, 0, "variable")
        (histogram,) = fig.data
        self.assertEqual(sorted(histogram.x), [0.0, 1.0, 2.0])

    def test_title_reflects_variable_index_and_batch_size(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.rand(7, 2)
        fig = _variable_histogram_figure(tensor, 1, "output")
        self.assertIn("output 1", fig.layout.title.text)
        self.assertIn("n=7", fig.layout.title.text)

    def test_custom_bin_width_is_used(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.linspace(0, 10, 100).reshape(-1, 1)
        fig = _variable_histogram_figure(tensor, 0, "variable", bin_width=2.0)
        self.assertEqual(fig.data[0].xbins.size, 2.0)

    def test_invalid_bin_width_falls_back_to_default(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.linspace(0, 10, 100).reshape(-1, 1)
        for invalid in (None, 0, -5):
            with self.subTest(bin_width=invalid):
                fig = _variable_histogram_figure(
                    tensor, 0, "variable", bin_width=invalid
                )
                self.assertGreater(fig.data[0].xbins.size, 0)


class TestDefaultBinWidth(unittest.TestCase):
    """
    Test _default_bin_width.
    """

    def test_roughly_forty_bins_across_the_range(self) -> None:
        """
        Returns:
            None
        """
        values = torch.linspace(0, 4, 100).numpy()
        self.assertAlmostEqual(_default_bin_width(values), 0.1)

    def test_constant_values_do_not_divide_by_zero(self) -> None:
        """
        Returns:
            None
        """
        values = torch.full((10,), 3.0).numpy()
        self.assertEqual(_default_bin_width(values), 1.0)


class TestRuleDropdownOptions(unittest.TestCase):
    """
    Test _rule_dropdown_options.
    """

    def test_labels_include_the_rule_string(self) -> None:
        """
        Returns:
            None
        """
        options = _rule_dropdown_options(
            ["var_0 IS term_0 THEN ___", "var_0 IS term_1 THEN ___"]
        )
        self.assertEqual(
            options,
            [
                {"label": "rule 0: var_0 IS term_0 THEN ___", "value": 0},
                {"label": "rule 1: var_0 IS term_1 THEN ___", "value": 1},
            ],
        )


class TestInputGranulationVariablePlots(unittest.TestCase):
    """
    Test _input_granulation_variable_plots.
    """

    def test_one_variable_plot_per_input_variable(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        variable_plots = _input_granulation_variable_plots(trace)
        self.assertEqual(len(variable_plots), trace.shape.n_inputs)
        self.assertEqual([vp.variable_idx for vp in variable_plots], [0, 1])


class TestInputGranulationCurveFigure(unittest.TestCase):
    """
    Test _input_granulation_curve_figure.
    """

    def test_one_line_and_one_scatter_trace_per_term(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        variable_plots = _input_granulation_variable_plots(trace)
        degrees = trace.stages[0].output.degrees.cpu().detach().numpy()
        observations = trace.observations.cpu().detach().numpy()

        variable_plot = variable_plots[0]
        fig = _input_granulation_curve_figure(variable_plot, degrees, observations, 0)
        n_terms = len(variable_plot.terms)
        self.assertEqual(len(fig.data), 2 * n_terms)
        modes = sorted(trace_.mode for trace_ in fig.data)
        self.assertEqual(modes, sorted(["lines"] * n_terms + ["markers"] * n_terms))


class TestMembershipHistogramFigure(unittest.TestCase):
    """
    Test _membership_histogram_figure.
    """

    def test_flattens_across_batch_and_real_terms_only(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        variable_plots = _input_granulation_variable_plots(trace)
        degrees = trace.stages[0].output.degrees.cpu().detach().numpy()

        variable_plot = variable_plots[0]
        fig = _membership_histogram_figure(variable_plot, degrees, 0)
        batch_size = degrees.shape[0]
        expected_n = batch_size * len(variable_plot.terms)
        self.assertIn(f"n={expected_n}", fig.layout.title.text)


class TestFiringStrengthHistogramFigure(unittest.TestCase):
    """
    Test _firing_strength_histogram_figure.
    """

    def test_uses_only_the_selected_rules_firing_strengths(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        degrees = trace.stages[1].output.degrees.cpu().detach().numpy()
        fig = _firing_strength_histogram_figure(trace, 0)
        self.assertEqual(sorted(fig.data[0].x), sorted(degrees[:, 0].tolist()))
        self.assertIn("rule 0", fig.layout.title.text)

    def test_custom_bin_width_is_used(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        fig = _firing_strength_histogram_figure(trace, 0, bin_width=0.02)
        self.assertEqual(fig.data[0].xbins.size, 0.02)


class TestTskRuleWeightsFigure(unittest.TestCase):
    """
    Test _tsk_rule_weights_figure.
    """

    def test_bars_are_bias_then_one_weight_per_input(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        module = trace.stages[2].module
        self.assertIsInstance(module, TSK)
        fig = _tsk_rule_weights_figure(module, 0)
        n_inputs = trace.shape.n_inputs
        self.assertEqual(
            list(fig.data[0].x), ["bias"] + [f"x{i}" for i in range(n_inputs)]
        )

    def test_values_match_the_selected_rules_consequences(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        module = trace.stages[2].module
        fig = _tsk_rule_weights_figure(module, 1)
        expected = module.consequences.detach().cpu().numpy()[0, 1]
        self.assertEqual(list(fig.data[0].y), list(expected))

    def test_title_reflects_rule_index(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        module = trace.stages[2].module
        fig = _tsk_rule_weights_figure(module, 2)
        self.assertIn("rule 2", fig.layout.title.text)


class TestMamdaniRuleConsequenceFigure(unittest.TestCase):
    """
    Test _mamdani_rule_consequence_figure.
    """

    def test_only_the_rules_linked_term_is_plotted(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_mamdani_trace()
        module = trace.stages[2].module
        self.assertIsInstance(module, Mamdani)
        for rule_idx in range(trace.shape.n_rules):
            with self.subTest(rule_idx=rule_idx):
                fig = _mamdani_rule_consequence_figure(module, rule_idx)
                linked = module.output_links[rule_idx].cpu().numpy()
                expected_n_curves = int((linked.sum(axis=1) > 0).sum())
                self.assertEqual(len(fig.data), expected_n_curves)

    def test_title_reflects_rule_index(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_mamdani_trace()
        module = trace.stages[2].module
        fig = _mamdani_rule_consequence_figure(module, 0)
        self.assertIn("rule 0", fig.layout.title.text)

    def test_output_variables_with_no_linked_term_are_skipped(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_mamdani_trace_with_unlinked_output_variable()
        module = trace.stages[2].module
        # rule 0 only links output variable 0 - exactly one curve, not two
        fig = _mamdani_rule_consequence_figure(module, 0)
        self.assertEqual(len(fig.data), 1)
        self.assertTrue(fig.data[0].name.startswith("out_var 0"))


class TestDefuzzificationDetail(unittest.TestCase):
    """
    Test _defuzzification_detail.
    """

    def test_tsk_gets_a_rule_dropdown_and_figure(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_trace()
        detail = _defuzzification_detail(trace)
        component_ids = _collect_component_ids(detail)
        self.assertIn("defuzz-dropdown", component_ids)
        self.assertIn("defuzz-figure", component_ids)

    def test_mamdani_gets_a_rule_dropdown_and_figure(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_mamdani_trace()
        detail = _defuzzification_detail(trace)
        component_ids = _collect_component_ids(detail)
        self.assertIn("defuzz-dropdown", component_ids)
        self.assertIn("defuzz-figure", component_ids)

    def test_unrecognized_defuzzification_type_falls_back_to_generic_view(self) -> None:
        """
        Returns:
            None
        """
        trace = _toy_zero_order_trace()
        self.assertNotIsInstance(trace.stages[2].module, (TSK, Mamdani))
        detail = _defuzzification_detail(trace)
        component_ids = _collect_component_ids(detail)
        self.assertNotIn("defuzz-dropdown", component_ids)
        self.assertNotIn("defuzz-figure", component_ids)


class TestGenerateFlcReport(unittest.TestCase):
    """
    Test generate_flc_report.
    """

    def test_captures_and_builds_in_one_call(self) -> None:
        """
        Returns:
            None
        """
        flc, observations = _toy_flc_and_observations()
        app = generate_flc_report(flc, observations)
        self.assertIsInstance(app, dash.Dash)


if __name__ == "__main__":
    unittest.main()
