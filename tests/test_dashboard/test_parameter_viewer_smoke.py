"""
Smoke tests for fuzzy.dashboard.parameter_viewer.

No Selenium/dash[testing] - matches the precedent set by flc_viewer.py/
activation_viewer.py: the pure data-prep helpers are unit-tested directly, and one
smoke test confirms build_parameter_dashboard assembles a working dash.Dash app.
"""

import unittest

import dash
import torch

from fuzzy.dashboard.parameter_viewer import (
    _as_indexable_2d,
    _parameter_dropdown_options,
    _parameter_line_figure,
    _row_col_dropdown_options,
    build_parameter_dashboard,
)
from fuzzy.logic.control.parameter_history import ParameterSnapshot

from .common import collect_component_ids


def _uniform_history() -> list:
    """
    Returns:
        A 3-step history where every snapshot has the same shapes throughout.
    """
    return [
        ParameterSnapshot(
            step=step,
            parameters={
                "input_granulation.centers": torch.tensor(
                    [[1.0 + step, 2.0 + step], [3.0 + step, 4.0 + step]]
                ),
                "defuzzification.consequences": torch.arange(
                    6, dtype=torch.float32
                ).reshape(1, 2, 3)
                + step,
            },
        )
        for step in range(3)
    ]


def _growing_history() -> list:
    """
    Returns:
        A 3-step history simulating mid-training rule growth: step 0 has 1 row,
        steps 1-2 have 2 rows.
    """
    return [
        ParameterSnapshot(
            step=0, parameters={"input_granulation.centers": torch.tensor([[1.0, 2.0]])}
        ),
        ParameterSnapshot(
            step=1,
            parameters={
                "input_granulation.centers": torch.tensor([[1.1, 2.1], [3.0, 4.0]])
            },
        ),
        ParameterSnapshot(
            step=2,
            parameters={
                "input_granulation.centers": torch.tensor([[1.2, 2.2], [3.1, 4.1]])
            },
        ),
    ]


class TestAsIndexable2D(unittest.TestCase):
    """
    Test _as_indexable_2d.
    """

    def test_2d_tensor_is_unchanged(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.rand(3, 4)
        self.assertEqual(tuple(_as_indexable_2d(tensor).shape), (3, 4))

    def test_1d_tensor_becomes_one_row(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.rand(5)
        self.assertEqual(tuple(_as_indexable_2d(tensor).shape), (1, 5))

    def test_3d_tensor_flattens_all_but_last_dim_into_rows(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.rand(2, 3, 4)
        self.assertEqual(tuple(_as_indexable_2d(tensor).shape), (6, 4))


class TestParameterDropdownOptions(unittest.TestCase):
    """
    Test _parameter_dropdown_options.
    """

    def test_one_option_per_tracked_parameter(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        options = _parameter_dropdown_options(history)
        self.assertEqual(
            {opt["value"] for opt in options},
            {"input_granulation.centers", "defuzzification.consequences"},
        )


class TestRowColDropdownOptions(unittest.TestCase):
    """
    Test _row_col_dropdown_options.
    """

    def test_sizes_options_to_the_max_shape_across_the_whole_history(self) -> None:
        """
        Regression guard: dropdown sizing must reflect the max shape seen ACROSS
        the whole history, not just the first snapshot, since rule/term counts can
        grow mid-training.

        Returns:
            None
        """
        history = _growing_history()
        row_options, col_options = _row_col_dropdown_options(
            history, "input_granulation.centers"
        )
        # max across history, not step 0's 1
        self.assertEqual(len(row_options), 2)
        self.assertEqual(len(col_options), 2)

    def test_skips_snapshots_missing_the_parameter(self) -> None:
        """
        Returns:
            None
        """
        history = [
            ParameterSnapshot(
                step=0, parameters={"input_granulation.centers": torch.rand(3, 2)}
            ),
            ParameterSnapshot(step=1, parameters={}),
        ]
        row_options, col_options = _row_col_dropdown_options(
            history, "input_granulation.centers"
        )
        self.assertEqual(len(row_options), 3)
        self.assertEqual(len(col_options), 2)

    def test_uses_friendly_labels_for_known_parameters(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        row_options, _ = _row_col_dropdown_options(history, "input_granulation.centers")
        self.assertEqual(row_options[0]["label"], "variable 0")

    def test_falls_back_to_generic_labels_for_unknown_parameters(self) -> None:
        """
        Returns:
            None
        """
        history = [
            ParameterSnapshot(step=0, parameters={"custom.thing": torch.rand(2, 2)})
        ]
        row_options, _ = _row_col_dropdown_options(history, "custom.thing")
        self.assertEqual(row_options[0]["label"], "row 0")


class TestParameterLineFigure(unittest.TestCase):
    """
    Test _parameter_line_figure.
    """

    def test_values_match_the_selected_row_and_col_at_each_step(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        fig = _parameter_line_figure(history, "input_granulation.centers", 1, [0])
        self.assertEqual(list(fig.data[0].x), [0, 1, 2])
        self.assertEqual(list(fig.data[0].y), [3.0, 4.0, 5.0])

    def test_skips_snapshots_missing_the_parameter(self) -> None:
        """
        Returns:
            None
        """
        history = [
            ParameterSnapshot(
                step=0, parameters={"input_granulation.centers": torch.rand(1, 1)}
            ),
            ParameterSnapshot(step=1, parameters={}),
        ]
        fig = _parameter_line_figure(history, "input_granulation.centers", 0, [0])
        self.assertEqual(list(fig.data[0].x), [0])

    def test_skips_snapshots_where_the_index_is_out_of_bounds(self) -> None:
        """
        Regression guard: a row/col only valid starting from a later snapshot (mid-
        training growth) must not be plotted for the earlier snapshots that don't
        have it, and must not raise an IndexError.

        Returns:
            None
        """
        history = _growing_history()
        fig = _parameter_line_figure(history, "input_granulation.centers", 1, [0])
        # not step 0, which lacks row 1
        self.assertEqual(list(fig.data[0].x), [1, 2])

    def test_empty_result_for_a_parameter_present_in_no_snapshot(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        fig = _parameter_line_figure(history, "not.a.real.parameter", 0, [0])
        self.assertEqual(list(fig.data[0].x), [])

    def test_one_trace_per_selected_column_for_comparison(self) -> None:
        """
        Multiple columns (e.g. a variable's different terms) selected together must
        each become their own trace on the same figure, so they can be compared
        directly rather than one at a time.

        Returns:
            None
        """
        history = _uniform_history()
        fig = _parameter_line_figure(history, "input_granulation.centers", 0, [0, 1])
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(list(fig.data[0].y), [1.0, 2.0, 3.0])
        self.assertEqual(list(fig.data[1].y), [2.0, 3.0, 4.0])

    def test_trace_names_identify_which_column_each_line_is(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        fig = _parameter_line_figure(history, "input_granulation.centers", 0, [0, 1])
        self.assertEqual([trace.name for trace in fig.data], ["term 0", "term 1"])

    def test_no_columns_selected_yields_an_empty_figure(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        fig = _parameter_line_figure(history, "input_granulation.centers", 0, [])
        self.assertEqual(len(fig.data), 0)

    def test_title_no_longer_names_a_single_column(self) -> None:
        """
        Returns:
            None
        """
        history = _uniform_history()
        fig = _parameter_line_figure(history, "input_granulation.centers", 1, [0, 1])
        self.assertEqual(
            fig.layout.title.text, "input_granulation.centers [variable 1]"
        )


class TestBuildParameterDashboard(unittest.TestCase):
    """
    Test build_parameter_dashboard.
    """

    def test_returns_dash_app_with_expected_component_ids(self) -> None:
        """
        Returns:
            None
        """
        app = build_parameter_dashboard(_uniform_history())
        self.assertIsInstance(app, dash.Dash)

        component_ids = collect_component_ids(app.layout)
        self.assertEqual(
            component_ids,
            {"param-dropdown", "row-dropdown", "col-dropdown", "parameter-line-chart"},
        )

    def test_rejects_empty_history(self) -> None:
        """
        Returns:
            None
        """
        with self.assertRaises(ValueError):
            build_parameter_dashboard([])


if __name__ == "__main__":
    unittest.main()
