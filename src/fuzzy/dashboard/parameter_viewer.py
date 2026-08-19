"""
Interactive Dash visualization of how a FuzzyLogicController (FLC)'s learnable
parameters changed over a captured training-time history (see
fuzzy.logic.control.parameter_history).

Requires the optional "dashboard" extra: pip install fuzzy-theory[dashboard].

Usage:
    from fuzzy.dashboard.parameter_viewer import build_parameter_dashboard

    app = build_parameter_dashboard(history)  # history: List[ParameterSnapshot]
    app.run(debug=True)
"""

from typing import Any, Dict, List, Tuple

import dash
import plotly.graph_objects as go
import torch
from dash import Input, Output, dcc, html

from fuzzy.logic.control.parameter_history import ParameterSnapshot

# friendly (row_label, col_label) per known parameter name - see
# fuzzy.logic.control.parameter_history._flc_named_parameters for where these
# names come from. Falls back to generic "row"/"col" for anything unrecognized.
_AXIS_LABELS: Dict[str, Tuple[str, str]] = {
    "input_granulation.centers": ("variable", "term"),
    "input_granulation.widths": ("variable", "term"),
    "defuzzification.consequences": ("(output, rule)", "coefficient"),
    "defuzzification.centers": ("output variable", "output term"),
    "defuzzification.widths": ("output variable", "output term"),
}


def _as_indexable_2d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Collapse a parameter tensor down to 2D for row/col dropdown selection: all but
    the last dimension are flattened together into "rows" - e.g. TSK's 3D
    (n_outputs, n_rules, n_inputs + 1) consequences becomes
    (n_outputs * n_rules, n_inputs + 1), the same shape of selection as a plain 2D
    tensor like input_granulation's (n_inputs, n_input_terms) centers.
    """
    if tensor.dim() <= 1:
        return tensor.reshape(1, -1)
    if tensor.dim() == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def _parameter_dropdown_options(
    history: List[ParameterSnapshot],
) -> List[Dict[str, Any]]:
    """
    Dropdown options for selecting which tracked parameter to inspect - derived
    from the first snapshot, since which named parameters exist is fixed by the
    FLC's defuzzification type (chosen once at construction, see
    _flc_named_parameters), even though a given parameter's shape may grow over
    training (see _row_col_dropdown_options).
    """
    return [{"label": name, "value": name} for name in history[0].parameters]


def _row_col_dropdown_options(
    history: List[ParameterSnapshot], parameter_name: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Row/col dropdown options sized to the max shape seen for this parameter across
    the whole history, not just one snapshot - rule/term counts can grow during
    training, and every index that was ever valid should stay selectable.
    """
    max_rows, max_cols = 0, 0
    for snapshot in history:
        tensor = snapshot.parameters.get(parameter_name)
        if tensor is None:
            continue
        indexable = _as_indexable_2d(tensor)
        max_rows = max(max_rows, indexable.shape[0])
        max_cols = max(max_cols, indexable.shape[1])

    row_label, col_label = _AXIS_LABELS.get(parameter_name, ("row", "col"))
    row_options = [{"label": f"{row_label} {i}", "value": i} for i in range(max_rows)]
    col_options = [{"label": f"{col_label} {i}", "value": i} for i in range(max_cols)]
    return row_options, col_options


def _column_series(
    history: List[ParameterSnapshot], parameter_name: str, row_idx: int, col_idx: int
) -> Tuple[List[int], List[float]]:
    """
    One (row, col) element's value at each training step it was valid for.

    Snapshots where this parameter is absent, or this row/col index is out of
    bounds for that snapshot's shape (e.g. before a rule was added mid-training),
    are skipped rather than erroring - a training history is not guaranteed to
    have a uniform parameter shape throughout.
    """
    steps: List[int] = []
    values: List[float] = []
    for snapshot in history:
        tensor = snapshot.parameters.get(parameter_name)
        if tensor is None:
            continue
        indexable = _as_indexable_2d(tensor)
        if row_idx >= indexable.shape[0] or col_idx >= indexable.shape[1]:
            continue
        steps.append(snapshot.step)
        values.append(indexable[row_idx, col_idx].item())
    return steps, values


def _parameter_line_figure(
    history: List[ParameterSnapshot],
    parameter_name: str,
    row_idx: int,
    col_indices: List[int],
) -> go.Figure:
    """
    A line chart of one row's selected columns across training steps, one line per
    column, overlaid on the same plot - e.g. a variable's different terms, or a
    rule's different coefficients, so it's easy to see how they relate to each
    other (and confirm they aren't all just tracking the same value) rather than
    inspecting them one at a time.
    """
    row_label, col_label = _AXIS_LABELS.get(parameter_name, ("row", "col"))
    fig = go.Figure()
    for col_idx in col_indices:
        steps, values = _column_series(history, parameter_name, row_idx, col_idx)
        fig.add_trace(
            go.Scatter(
                x=steps, y=values, mode="lines+markers", name=f"{col_label} {col_idx}"
            )
        )
    fig.update_layout(
        title=f"{parameter_name} [{row_label} {row_idx}]",
        xaxis_title="step",
        yaxis_title="value",
        height=400,
        margin={"t": 40, "b": 40},
    )
    return fig


def build_parameter_dashboard(history: List[ParameterSnapshot]) -> dash.Dash:
    """
    Assemble (but do not run) a Dash app visualizing a training-time parameter
    history.

    Args:
        history: The captured snapshots (see
            fuzzy.logic.control.parameter_history.capture_parameter_snapshot), in
            the order they were recorded.

    Returns:
        The assembled Dash app - call app.run(...) to serve it.
    """
    if not history:
        raise ValueError("history must contain at least one ParameterSnapshot.")

    initial_parameter = next(iter(history[0].parameters))
    initial_row_options, initial_col_options = _row_col_dropdown_options(
        history, initial_parameter
    )
    initial_col_values = [option["value"] for option in initial_col_options]

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("FLC Training Parameter Viewer"),
            html.Div(
                [
                    dcc.Dropdown(
                        id="param-dropdown",
                        options=_parameter_dropdown_options(history),
                        value=initial_parameter,
                        clearable=False,
                        style={"width": "320px"},
                    ),
                    dcc.Dropdown(
                        id="row-dropdown",
                        options=initial_row_options,
                        value=0,
                        clearable=False,
                        style={"width": "160px", "marginLeft": "12px"},
                    ),
                    dcc.Dropdown(
                        id="col-dropdown",
                        options=initial_col_options,
                        value=initial_col_values,
                        multi=True,
                        placeholder="compare columns…",
                        style={"width": "320px", "marginLeft": "12px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginTop": "12px",
                },
            ),
            dcc.Graph(
                id="parameter-line-chart",
                figure=_parameter_line_figure(
                    history, initial_parameter, 0, initial_col_values
                ),
            ),
        ],
        style={"padding": "16px 24px"},
    )

    @app.callback(
        Output("row-dropdown", "options"),
        Output("row-dropdown", "value"),
        Output("col-dropdown", "options"),
        Output("col-dropdown", "value"),
        Input("param-dropdown", "value"),
    )
    def _on_parameter_change(parameter_name: str):
        row_options, col_options = _row_col_dropdown_options(history, parameter_name)
        # default to comparing every column for the (also reset-to-0) row - that's
        # the "how do these relate to each other" view this feature is for
        col_values = [option["value"] for option in col_options]
        return row_options, 0, col_options, col_values

    @app.callback(
        Output("parameter-line-chart", "figure"),
        Input("param-dropdown", "value"),
        Input("row-dropdown", "value"),
        Input("col-dropdown", "value"),
    )
    def _on_selection_change(
        parameter_name: str, row_idx: int, col_indices: List[int]
    ) -> go.Figure:
        return _parameter_line_figure(
            history, parameter_name, row_idx, col_indices or []
        )

    return app
