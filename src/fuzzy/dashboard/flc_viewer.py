"""
Interactive Dash + dash-cytoscape visualization of a single captured
FuzzyLogicController (FLC) forward pass (see fuzzy.logic.control.trace).

Requires the optional "dashboard" extra: pip install fuzzy-theory[dashboard].

Usage:
    from fuzzy.dashboard.flc_viewer import generate_flc_report

    app = generate_flc_report(flc, observations)
    app.run(debug=True)
"""

from typing import Any, Dict, List, Union

import dash
import dash_cytoscape as cyto
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Input, Output, dash_table, dcc, html

from fuzzy.logic.control.controller import FuzzyLogicController
from fuzzy.logic.control.defuzzification import TSK, Defuzzification, Mamdani
from fuzzy.logic.control.trace import FLCTrace, capture_flc_trace
from fuzzy.logic.rule_helpers import rule_uniqueness_metrics
from fuzzy.sets.visualization import FuzzySetPlot, VariablePlot

_STAGE_LABELS: Dict[str, str] = {
    "observations": "Observations",
    "input_granulation": "Input Granulation",
    "engine": "Engine",
    "defuzzification": "Defuzzification",
    "output": "Output",
}
_NODE_ORDER: List[str] = list(_STAGE_LABELS)


def _densify(tensor: torch.Tensor) -> torch.Tensor:
    """Sparse tensors (see FuzzySet.enable_sparse) need densifying before use with
    numpy/Plotly - mirrors the same guard in activation_viewer.fuzzy_set_curves_img.
    """
    return tensor.to_dense() if tensor.is_sparse else tensor


def _stats_table(tensor: torch.Tensor) -> dash_table.DataTable:
    """A small shape/min/max/mean table - the raw-value analog of
    activation_viewer.layer_summary, reimplemented locally since that helper
    returns pre-formatted strings shaped for the static-HTML report, not the raw
    values a DataTable wants."""
    tensor = _densify(tensor)
    stats = {
        "shape": str(list(tensor.shape)),
        "min": f"{tensor.min().item():.4f}",
        "max": f"{tensor.max().item():.4f}",
        "mean": f"{tensor.float().mean().item():.4f}",
    }
    return dash_table.DataTable(
        data=[{"stat": key, "value": value} for key, value in stats.items()],
        columns=[{"name": "stat", "id": "stat"}, {"name": "value", "id": "value"}],
        style_cell={"textAlign": "left"},
    )


def _variable_dropdown_options(n_variables: int) -> List[Dict[str, Any]]:
    """
    Dropdown options for selecting one of a tensor's n_variables columns.

    Plain integer values, no special virtualization - dcc.Dropdown's own
    search/typeahead already makes a 1000+-option list usable without needing to
    scroll through it, which is what lets the histogram below scale to that many
    variables (rendering all of them at once would not).
    """
    return [{"label": f"variable {i}", "value": i} for i in range(n_variables)]


def _rule_dropdown_options(rule_strings: List[str]) -> List[Dict[str, Any]]:
    """
    Dropdown options for selecting one of n_rules rules, labeled with the rule's
    own antecedent string (not just its index) so a 1000+-rule dropdown stays
    searchable by content via dcc.Dropdown's own typeahead.
    """
    return [
        {"label": f"rule {i}: {rule_string}", "value": i}
        for i, rule_string in enumerate(rule_strings)
    ]


def _default_bin_width(values: np.ndarray) -> float:
    """A starting bin width that reproduces roughly 40 bins across the data's range,
    rounded to a few significant figures so it reads reasonably in a textbox."""
    data_range = float(values.max() - values.min())
    if data_range <= 0:
        return 1.0
    return float(f"{data_range / 40:.4g}")


def _histogram_figure(
    values: np.ndarray, title: str, xaxis_title: str, bin_width: Union[float, None]
) -> go.Figure:
    """Shared histogram-figure builder: every histogram in this module (raw values,
    membership degrees, firing strengths) is one column/selection of values, a
    title, and a bin width - just the x-axis's meaning differs."""
    if not bin_width or bin_width <= 0:
        bin_width = _default_bin_width(values)
    fig = go.Figure(
        data=[go.Histogram(x=values, xbins={"size": bin_width}, marker_color="#6c63ff")]
    )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="count",
        height=320,
        margin={"t": 40, "b": 40},
    )
    return fig


def _variable_histogram_figure(
    tensor: torch.Tensor,
    variable_idx: int,
    label: str,
    bin_width: Union[float, None] = None,
) -> go.Figure:
    """A histogram of one variable/column's values across the batch dimension."""
    values = _densify(tensor)[:, variable_idx].cpu().detach().numpy()
    return _histogram_figure(
        values,
        title=f"{label} {variable_idx} — distribution across batch (n={len(values)})",
        xaxis_title="value",
        bin_width=bin_width,
    )


def _selector_control_row(
    dropdown_id: str,
    dropdown_options: List[Dict[str, Any]],
    bin_width_id: str,
    initial_bin_width: float,
) -> html.Div:
    """
    A dropdown + bin-width textbox row.

    Each panel that uses this gets its own dedicated pair of ids, never shared
    across panels: Dash's client-side callback dispatch requires every Output a
    callback declares to currently exist in the DOM whenever its Input fires, so a
    shared id whose paired Output only exists in *some* of the panels reusing that
    Input breaks (a "nonexistent object" error) the moment you interact with it
    from a panel where that Output is absent - even if the callback would have
    returned dash.no_update for it. Giving every panel its own ids instead means
    each callback's Input and Output are always rendered together (see
    build_flc_dashboard).
    """
    return html.Div(
        [
            dcc.Dropdown(
                id=dropdown_id,
                options=dropdown_options,
                value=0,
                clearable=False,
                style={"width": "100%", "maxWidth": "500px"},
            ),
            html.Label("bin width:", style={"marginLeft": "16px"}),
            dcc.Input(
                id=bin_width_id,
                type="number",
                value=initial_bin_width,
                min=0,
                step="any",
                style={"width": "100px", "marginLeft": "8px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginTop": "12px"},
    )


def _batched_tensor_detail(
    tensor: torch.Tensor, id_prefix: str, label: str
) -> html.Div:
    """
    Overall stats table plus a per-variable histogram, matching
    activation_viewer.py's convention of showing both together - the histogram is
    dropdown-selected (see _variable_dropdown_options) rather than one-per-variable,
    so this scales to 1000+ variables. Bin width is user-editable (see
    _default_bin_width for the starting value).

    Args:
        tensor: The (batch, n_variables) tensor to summarize (observations/output).
        id_prefix: A short, panel-unique prefix (e.g. "obs"/"out") for this panel's
            dropdown/bin-width/histogram component ids - see _selector_control_row's
            docstring for why these must not be shared across panels.
        label: Human-readable name for the histogram title, e.g. "variable"/"output".
    """
    n_variables = tensor.shape[1]
    initial_values = _densify(tensor)[:, 0].cpu().detach().numpy()
    return html.Div(
        [
            _stats_table(tensor),
            _selector_control_row(
                f"{id_prefix}-dropdown",
                _variable_dropdown_options(n_variables),
                f"{id_prefix}-bin-width",
                _default_bin_width(initial_values),
            ),
            dcc.Graph(
                id=f"{id_prefix}-histogram",
                figure=_variable_histogram_figure(tensor, 0, label),
            ),
        ]
    )


def _cytoscape_elements(trace: FLCTrace) -> List[dict]:
    """Build the fixed 5-node pipeline graph's node/edge elements."""
    labels = {
        "observations": f"{_STAGE_LABELS['observations']}\n{tuple(trace.observations.shape)}",
        "input_granulation": (
            f"{_STAGE_LABELS['input_granulation']}\n{trace.stages[0].module_type}"
        ),
        "engine": (
            f"{_STAGE_LABELS['engine']}\n"
            f"{trace.stages[1].module_type} ({trace.shape.n_rules} rules)"
        ),
        "defuzzification": (
            f"{_STAGE_LABELS['defuzzification']}\n{trace.stages[2].module_type}"
        ),
        "output": f"{_STAGE_LABELS['output']}\n{tuple(trace.output.shape)}",
    }
    nodes = [
        {
            "data": {"id": node_id, "label": labels[node_id]},
            "position": {"x": 160 * i, "y": 0},
        }
        for i, node_id in enumerate(_NODE_ORDER)
    ]
    edges = [
        {"data": {"source": src, "target": dst}}
        for src, dst in zip(_NODE_ORDER[:-1], _NODE_ORDER[1:])
    ]
    return nodes + edges


def _input_granulation_variable_plots(trace: FLCTrace) -> List[VariablePlot]:
    """
    Build input_granulation's per-variable formula-curve data once (see
    FuzzySetPlot.build()), so both the curve figure and the membership histogram
    can index into the same list by variable_idx.

    Assumes input_granulation's FuzzySetGroup wraps a single FuzzySet covering all
    variables (true for every FLC this library's own demos/tests build) - a group
    combining multiple FuzzySet types per variable would need each constituent's
    locally-indexed variable_idx mapped back to the group's global variable index,
    which this does not attempt.
    """
    module = trace.stages[0].module  # FuzzySetGroup
    variable_plots = []
    for fuzzy_set in module.modules_list:
        variable_plots.extend(FuzzySetPlot(fuzzy_set).build())
    return variable_plots


def _input_granulation_curve_figure(
    variable_plot: VariablePlot,
    degrees: np.ndarray,
    observations: np.ndarray,
    variable_idx: int,
) -> go.Figure:
    """The selected variable's formula curves (one line per term, via
    FuzzySetPlot.build() - the same data prep activation_viewer.fuzzy_set_curves_img
    uses for its matplotlib rendering, here natively in Plotly instead), overlaid
    with a scatter of where the observations landed on them."""
    fig = go.Figure()
    for term in variable_plot.terms:
        fig.add_trace(
            go.Scatter(
                x=variable_plot.x_values,
                y=term.y_values,
                mode="lines",
                name=term.label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=observations[:, variable_idx],
                y=degrees[:, variable_idx, term.term_idx],
                mode="markers",
                name=f"{term.label} obs",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"variable {variable_idx} — formula curves",
        xaxis_title="value",
        yaxis_title="membership degree",
        height=320,
        margin={"t": 40, "b": 40},
    )
    return fig


def _membership_histogram_figure(
    variable_plot: VariablePlot,
    degrees: np.ndarray,
    variable_idx: int,
    bin_width: Union[float, None] = None,
) -> go.Figure:
    """A histogram of the selected variable's membership degrees, across the batch
    and all of its real (mask == 1) terms."""
    term_indices = [term.term_idx for term in variable_plot.terms]
    values = degrees[:, variable_idx, term_indices].flatten()
    return _histogram_figure(
        values,
        title=f"variable {variable_idx} — membership degree distribution (n={len(values)})",
        xaxis_title="membership degree",
        bin_width=bin_width,
    )


def _input_granulation_detail(trace: FLCTrace) -> html.Div:
    """
    The input_granulation analog of _batched_tensor_detail: a dropdown-selected
    variable's formula curves plus a histogram of its membership degrees across the
    batch, needing FuzzySetPlot's formula-curve data (see
    _input_granulation_variable_plots) rather than just the raw tensor.
    """
    variable_plots = _input_granulation_variable_plots(trace)
    degrees = _densify(trace.stages[0].output.degrees).cpu().detach().numpy()
    observations = trace.observations.cpu().detach().numpy()

    initial_plot = variable_plots[0]
    initial_term_indices = [term.term_idx for term in initial_plot.terms]
    initial_values = degrees[:, 0, initial_term_indices].flatten()

    return html.Div(
        [
            _selector_control_row(
                "ig-dropdown",
                _variable_dropdown_options(len(variable_plots)),
                "ig-bin-width",
                _default_bin_width(initial_values),
            ),
            dcc.Graph(
                id="ig-curve-figure",
                figure=_input_granulation_curve_figure(
                    initial_plot, degrees, observations, 0
                ),
            ),
            dcc.Graph(
                id="ig-histogram",
                figure=_membership_histogram_figure(initial_plot, degrees, 0),
            ),
        ]
    )


def _firing_strength_histogram_figure(
    trace: FLCTrace, rule_idx: int, bin_width: Union[float, None] = None
) -> go.Figure:
    """A histogram of one rule's firing strength across the batch."""
    values = (
        _densify(trace.stages[1].output.degrees).cpu().detach().numpy()[:, rule_idx]
    )
    return _histogram_figure(
        values,
        title=f"rule {rule_idx} — firing strength across batch (n={len(values)})",
        xaxis_title="firing strength",
        bin_width=bin_width,
    )


def _engine_detail(trace: FLCTrace) -> html.Div:
    """A rule table (antecedent string + first sample's firing strength) plus a
    one-line rule-diversity summary from rule_uniqueness_metrics, and a
    dropdown-selected rule's firing-strength histogram across the whole batch."""
    degrees = _densify(trace.stages[1].output.degrees).cpu().detach().numpy()
    rows = [
        {"rule": rule_string, "firing strength (sample 0)": f"{strength:.4f}"}
        for rule_string, strength in zip(trace.rule_strings, degrees[0])
    ]
    metrics = rule_uniqueness_metrics(trace.rule_tensor)
    summary = html.P(
        f"{trace.shape.n_rules} rules — effective_rules={metrics.effective_rules:.2f}, "
        f"base_diversity={metrics.base_diversity:.2f}"
    )
    table = dash_table.DataTable(
        data=rows,
        columns=[
            {"name": "rule", "id": "rule"},
            {"name": "firing strength (sample 0)", "id": "firing strength (sample 0)"},
        ],
        style_cell={"textAlign": "left"},
        # scroll (not paginate) through the table - with 512+ rules, pagination
        # would just hide most rules behind page-number clicks; virtualization
        # keeps this fast by only rendering the rows currently in view
        page_action="none",
        virtualization=True,
        style_table={"height": "40vh", "overflowY": "auto"},
    )
    return html.Div(
        [
            summary,
            table,
            _selector_control_row(
                "engine-dropdown",
                _rule_dropdown_options(trace.rule_strings),
                "engine-bin-width",
                _default_bin_width(degrees[:, 0]),
            ),
            dcc.Graph(
                id="engine-histogram",
                figure=_firing_strength_histogram_figure(trace, 0),
            ),
        ]
    )


def _tsk_rule_weights_figure(module: TSK, rule_idx: int) -> go.Figure:
    """
    A bar chart of the selected rule's local linear consequence: its bias and
    per-input-variable weight, one grouped set of bars per output variable (see
    TSK.consequences, shape (n_outputs, n_rules, n_inputs + 1) - index 0 of the
    last dim is the bias, the rest are weights).
    """
    consequences = module.consequences.detach().cpu().numpy()
    n_outputs, _, n_inputs_plus_bias = consequences.shape
    x_labels = ["bias"] + [f"x{i}" for i in range(n_inputs_plus_bias - 1)]

    fig = go.Figure()
    for output_idx in range(n_outputs):
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=consequences[output_idx, rule_idx],
                name=f"output {output_idx}",
            )
        )
    fig.update_layout(
        title=f"rule {rule_idx} — TSK linear consequence (y = bias + weights · x)",
        xaxis_title="coefficient",
        yaxis_title="value",
        height=320,
        margin={"t": 40, "b": 40},
        barmode="group",
    )
    return fig


def _mamdani_rule_consequence_figure(module: Mamdani, rule_idx: int) -> go.Figure:
    """
    The formula curve(s) of the output term(s) the selected rule links to (via
    Mamdani.output_links, shape (n_rules, n_out_vars, n_out_terms)), reusing the
    same FuzzySetPlot-based rendering as _input_granulation_curve_figure - Mamdani's
    consequences are a FuzzySetGroup too, just over the output space instead of the
    input space.
    """
    linked = module.output_links[rule_idx].cpu().numpy()  # (n_out_vars, n_out_terms)
    variable_plots = []
    for fuzzy_set in module.consequences.modules_list:
        variable_plots.extend(FuzzySetPlot(fuzzy_set).build())

    fig = go.Figure()
    for out_var_idx in range(linked.shape[0]):
        linked_term_indices = set(linked[out_var_idx].nonzero()[0].tolist())
        if not linked_term_indices:
            continue
        for term in variable_plots[out_var_idx].terms:
            if term.term_idx not in linked_term_indices:
                continue
            fig.add_trace(
                go.Scatter(
                    x=variable_plots[out_var_idx].x_values,
                    y=term.y_values,
                    mode="lines",
                    name=f"out_var {out_var_idx}: {term.label}",
                )
            )
    fig.update_layout(
        title=f"rule {rule_idx} — Mamdani consequent term(s)",
        xaxis_title="value",
        yaxis_title="membership degree",
        height=320,
        margin={"t": 40, "b": 40},
    )
    return fig


def _defuzzification_rule_figure(module: Defuzzification, rule_idx: int) -> go.Figure:
    """Dispatch to the right rule-consequence figure for the defuzzification
    method actually in use - only called for module types _defuzzification_detail
    already confirmed have one (TSK or Mamdani)."""
    if isinstance(module, TSK):
        return _tsk_rule_weights_figure(module, rule_idx)
    return _mamdani_rule_consequence_figure(module, rule_idx)


def _defuzzification_detail(trace: FLCTrace) -> html.Div:
    """
    The defuzzification stage's own contribution to the final output, on top of the
    generic module-type + stats view: for TSK, a dropdown-selected rule's local
    linear consequence (weights + bias); for Mamdani, that rule's linked consequent
    term's formula curve. Any other Defuzzification subclass falls back to the
    generic view alone, since there is nothing method-specific this module knows
    how to render for it yet.
    """
    module = trace.stages[2].module
    generic = html.Div(
        [html.P(trace.stages[2].module_type), _stats_table(trace.output)]
    )
    if not isinstance(module, (TSK, Mamdani)):
        return generic

    return html.Div(
        [
            generic,
            dcc.Dropdown(
                id="defuzz-dropdown",
                options=_rule_dropdown_options(trace.rule_strings),
                value=0,
                clearable=False,
                style={"width": "100%", "maxWidth": "500px", "marginTop": "12px"},
            ),
            dcc.Graph(
                id="defuzz-figure",
                figure=_defuzzification_rule_figure(module, 0),
            ),
        ]
    )


def _render_stage_detail(trace: FLCTrace, node_id: str) -> html.Div:
    """
    Build the detail panel content for the given clicked graph node.

    Args:
        trace: The captured FLC forward pass to render detail from.
        node_id: One of the fixed pipeline node ids (see _NODE_ORDER).

    Returns:
        The detail panel's component tree.
    """
    if node_id not in _STAGE_LABELS:
        raise ValueError(f"Unknown node id: {node_id!r}")

    header = html.H4(_STAGE_LABELS[node_id])
    if node_id == "observations":
        return html.Div(
            [header, _batched_tensor_detail(trace.observations, "obs", "variable")]
        )
    if node_id == "output":
        return html.Div([header, _batched_tensor_detail(trace.output, "out", "output")])
    if node_id == "input_granulation":
        return html.Div([header, _input_granulation_detail(trace)])
    if node_id == "engine":
        return html.Div([header, _engine_detail(trace)])
    # node_id == "defuzzification"
    return html.Div([header, _defuzzification_detail(trace)])


def build_flc_dashboard(trace: FLCTrace) -> dash.Dash:
    """
    Assemble (but do not run) a Dash app visualizing one captured FLC forward pass.

    Args:
        trace: The captured FLC forward pass to visualize.

    Returns:
        The assembled Dash app - call app.run(...) to serve it.
    """
    # each panel's dropdown/bin-width/histogram ids only exist in the DOM while
    # that panel is the one shown in detail-panel (see _render_stage_detail); only
    # "observations" is present in the initial layout Dash validates callbacks
    # against at startup, so the rest need callback-exception suppression - the
    # standard Dash pattern for dynamically-swapped component trees
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            html.H2("Fuzzy Logic Controller — Forward Pass Viewer"),
            cyto.Cytoscape(
                id="flc-graph",
                layout={"name": "preset", "fit": True, "padding": 30},
                elements=_cytoscape_elements(trace),
                style={"width": "100%", "height": "220px"},
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "content": "data(label)",
                            "text-wrap": "wrap",
                            "text-max-width": "120px",
                            "font-size": "10px",
                            "text-valign": "center",
                            "background-color": "#6c63ff",
                        },
                    },
                    {
                        "selector": "edge",
                        "style": {
                            "target-arrow-shape": "triangle",
                            "curve-style": "bezier",
                        },
                    },
                ],
            ),
            html.Div(
                id="detail-panel",
                children=_render_stage_detail(trace, "observations"),
                style={"marginTop": "24px"},
            ),
        ],
        style={"padding": "16px 24px"},
    )

    @app.callback(Output("detail-panel", "children"), Input("flc-graph", "tapNodeData"))
    def _on_node_click(node_data: Any) -> html.Div:
        node_id = node_data["id"] if node_data else "observations"
        return _render_stage_detail(trace, node_id)

    @app.callback(
        Output("obs-histogram", "figure"),
        Input("obs-dropdown", "value"),
        Input("obs-bin-width", "value"),
    )
    def _on_observations_selection_change(
        index: int, bin_width: Union[float, None]
    ) -> go.Figure:
        return _variable_histogram_figure(
            trace.observations, index, "variable", bin_width
        )

    @app.callback(
        Output("out-histogram", "figure"),
        Input("out-dropdown", "value"),
        Input("out-bin-width", "value"),
    )
    def _on_output_selection_change(
        index: int, bin_width: Union[float, None]
    ) -> go.Figure:
        return _variable_histogram_figure(trace.output, index, "output", bin_width)

    @app.callback(
        Output("ig-histogram", "figure"),
        Output("ig-curve-figure", "figure"),
        Input("ig-dropdown", "value"),
        Input("ig-bin-width", "value"),
    )
    def _on_input_granulation_selection_change(
        index: int, bin_width: Union[float, None]
    ):
        variable_plots = _input_granulation_variable_plots(trace)
        degrees = _densify(trace.stages[0].output.degrees).cpu().detach().numpy()
        observations = trace.observations.cpu().detach().numpy()
        variable_plot = variable_plots[index]
        return (
            _membership_histogram_figure(variable_plot, degrees, index, bin_width),
            _input_granulation_curve_figure(
                variable_plot, degrees, observations, index
            ),
        )

    @app.callback(
        Output("engine-histogram", "figure"),
        Input("engine-dropdown", "value"),
        Input("engine-bin-width", "value"),
    )
    def _on_engine_selection_change(
        index: int, bin_width: Union[float, None]
    ) -> go.Figure:
        return _firing_strength_histogram_figure(trace, index, bin_width)

    # only ever rendered (and therefore only ever fires) when defuzzification is
    # TSK or Mamdani - see _defuzzification_detail
    @app.callback(Output("defuzz-figure", "figure"), Input("defuzz-dropdown", "value"))
    def _on_defuzzification_selection_change(index: int) -> go.Figure:
        return _defuzzification_rule_figure(trace.stages[2].module, index)

    return app


def generate_flc_report(
    flc: FuzzyLogicController, observations: torch.Tensor
) -> dash.Dash:
    """
    Capture one FLC forward pass and build a Dash app to visualize it.

    Args:
        flc: The FuzzyLogicController to run and visualize.
        observations: The observations to run through the FLC.

    Returns:
        The assembled Dash app - call app.run(...) to serve it.
    """
    trace = capture_flc_trace(flc, observations)
    return build_flc_dashboard(trace)
