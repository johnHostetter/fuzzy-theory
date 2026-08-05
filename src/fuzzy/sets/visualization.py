"""
Plotting support for FuzzySet, kept separate from fuzzy.sets.abstract so that FuzzySet
itself (the model) does not need matplotlib/scienceplots as a dependency.

FuzzySetPlot is a Model/ViewModel/View arrangement collapsed into a single class,
since the ViewModel (data preparation) and View (matplotlib rendering) halves each
have too few public responsibilities on their own to justify being separate classes:

    - Model: a FuzzySet instance (see fuzzy.sets.abstract) - untouched by this module.
    - ViewModel: FuzzySetPlot.build() turns a FuzzySet's parameters and membership
      curves into plain data (VariablePlot/TermCurve).
    - View: FuzzySetPlot.render()/.render_formula() take that data (or the fuzzy set
      class itself, for the formula) and render/save it via matplotlib.

FuzzySet.plot() is a thin convenience wrapper around FuzzySetPlot, kept for backward
compatibility with existing callers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# import scienceplots is used via plt.style.context(["science", "no-latex",
# "high-contrast"])
import scienceplots  # noqa # pylint: disable=unused-import
import sympy
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    # avoids a real circular import (fuzzy.sets.abstract imports this module to
    # implement FuzzySet.plot()) - only needed for the type hint below, never at
    # runtime
    from fuzzy.sets.abstract import FuzzySet


@dataclass
class TermCurve:
    """
    Plotting data for a single fuzzy term's membership curve within one variable.
    """

    term_idx: int
    y_values: np.ndarray
    label: str
    selected: bool


@dataclass
class VariablePlot:
    """
    Plotting data for one variable's full set of term curves.
    """

    variable_idx: int
    x_values: np.ndarray
    terms: List[TermCurve] = field(default_factory=list)
    legend_ncol: int = 0


class FuzzySetPlot:
    """
    Prepares plotting data for a FuzzySet's membership functions and renders it via
    matplotlib, saving the result to disk.
    """

    def __init__(self, fuzzy_set: "FuzzySet"):
        self.fuzzy_set = fuzzy_set

    @staticmethod
    def render_formula(
        fuzzy_set_cls: Type["FuzzySet"], latex: bool
    ) -> Union[str, sympy.Expr]:
        """
        Render a fuzzy set class' membership function formula.

        Args:
            fuzzy_set_cls: The FuzzySet subclass whose formula should be rendered.
            latex: Use False for Python Console or Jupyter Notebook. Otherwise, True
                will return a LaTeX representation of the fuzzy set's membership
                function.

        Returns:
            Render of the fuzzy set's membership function.
        """
        if latex:
            return sympy.latex(fuzzy_set_cls.sympy_formula())

        sympy.init_printing(use_unicode=True)
        return fuzzy_set_cls.sympy_formula()

    def build(
        self, selected_terms: Optional[List[Tuple[int, int]]] = None
    ) -> List[VariablePlot]:
        """
        Build the per-variable plotting data for this fuzzy set: which terms are
        real (per the mask), the x-range to sample each variable over, and the
        resulting membership curves.

        Args:
            selected_terms: The (variable_idx, term_idx) terms to highlight.

        Returns:
            One VariablePlot per input variable.
        """
        if selected_terms is None:
            selected_terms = []

        fuzzy_set = self.fuzzy_set
        centers = fuzzy_set.get_centers()
        widths = fuzzy_set.get_widths()
        mask = fuzzy_set.get_mask()

        variable_plots: List[VariablePlot] = []
        for variable_idx in range(centers.shape[0]):
            real_centers: List[float] = [
                centers[variable_idx, term_idx].item()
                for term_idx, mask_value in enumerate(mask[variable_idx])
                if mask_value == 1
            ]
            real_widths: List[float] = [
                widths[variable_idx, term_idx].item()
                for term_idx, mask_value in enumerate(mask[variable_idx])
                if mask_value == 1
            ]
            x_values = torch.linspace(
                min(real_centers) - 2 * max(real_widths),
                max(real_centers) + 2 * max(real_widths),
                steps=1000,
                device=fuzzy_set.device,
            )

            if centers.ndim == 1 or centers.shape[0] == 1:
                x_values = x_values[:, None]
            elif centers.ndim == 2 or centers.shape[0] > 1:
                x_values = x_values[:, None, None]

            memberships: torch.Tensor = fuzzy_set.calculate_membership(
                x_values)
            if memberships.ndim == 2:
                memberships = memberships.unsqueeze(
                    dim=1
                )  # add a temporary dimension for the variable

            memberships = memberships.cpu().detach().numpy()
            x_values_np = x_values.squeeze().cpu().detach().numpy()

            terms: List[TermCurve] = []
            for term_idx in range(memberships.shape[-1]):
                if mask[variable_idx, term_idx] == 0:
                    continue  # not a real fuzzy set
                terms.append(
                    TermCurve(
                        term_idx=term_idx,
                        y_values=memberships[:, variable_idx, term_idx],
                        label=r"$\mu_{"
                        + str(variable_idx + 1)
                        + ","
                        + str(term_idx + 1)
                        + "}$",
                        selected=(variable_idx, term_idx) in selected_terms,
                    )
                )

            variable_plots.append(
                VariablePlot(
                    variable_idx=variable_idx,
                    x_values=x_values_np,
                    terms=terms,
                    legend_ncol=len(real_centers),
                )
            )

        return variable_plots

    def render(
        self, variable_plots: List[VariablePlot], output_dir: Path
    ) -> Tuple[List[Any], Union[Axes, np.ndarray]]:
        """
        Render the given per-variable plotting data to a combined figure (saved as
        "mu.png") and, per variable, individually cropped figures.

        Args:
            variable_plots: The data to plot, one entry per input variable.
            output_dir: The directory to save the plot(s) to.

        Returns:
            A 2-tuple containing the figures and axes of the plot for each variable.
        """
        figures: List[Any] = []
        mpl.rcParams["figure.figsize"] = (6, 4)
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 100
        mpl.rcParams["font.size"] = 24
        mpl.rcParams["legend.fontsize"] = "medium"
        mpl.rcParams["figure.titlesize"] = "medium"
        mpl.rcParams["lines.linewidth"] = 2
        with plt.style.context(["science", "no-latex", "high-contrast"]):
            fig, axes = plt.subplots(1, 4, figsize=(28, 4), dpi=100)
            for variable_plot in variable_plots:
                self._render_variable(axes, variable_plot)
                output_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_dir / "mu.png")

        self._save_individual_plots(axes, fig, output_dir, len(variable_plots))

        return figures, axes

    @staticmethod
    def _render_variable(
        axes: Union[Axes, np.ndarray], variable_plot: VariablePlot
    ) -> None:
        """
        Draw one variable's term curves onto its axis.

        Args:
            axes: The axes shared across all variables being plotted.
            variable_plot: The data to plot for this variable.

        Returns:
            None
        """
        variable_idx = variable_plot.variable_idx
        axes[variable_idx].tick_params(width=2, length=6)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        for term in variable_plot.terms:
            if term.selected:
                axes[variable_idx].fill_between(
                    variable_plot.x_values,
                    term.y_values,
                    alpha=0.5,
                    hatch="///",
                    label=term.label,
                )
            else:
                axes[variable_idx].plot(
                    variable_plot.x_values,
                    term.y_values,
                    alpha=0.5,
                    label=term.label)
        axes[variable_idx].legend(
            bbox_to_anchor=(0.5, -0.2),
            loc="upper center",
            ncol=variable_plot.legend_ncol,
            handletextpad=0.1,
            # reduce spacing b/w legend markers & label (default=0.8)
            columnspacing=0.5,  # reduce spacing b/w legend entries
            borderaxespad=-0.5,  # reduce the spacing b/w the legend and the plot
        )
        plt.subplots_adjust(bottom=0.3, wspace=0.33)

    @staticmethod
    def _save_individual_plots(
        axes: Union[Axes, np.ndarray], fig: Figure, output_dir: Path, n_variables: int
    ) -> None:
        """
        Save just the portion _inside_ each variable's axis boundaries. Why do it
        this way? Because the axis is not always the same size if each plot is
        different. So, the area inside the axis's boundaries is saved, and then it
        can be padded to make it look nice in papers.

        Args:
            axes: The axes to use for the plots.
            fig: The figure to continue referencing when plotting.
            output_dir: The directory to save the figure(s).
            n_variables: The number of variables that were plotted.

        Returns:
            None
        """
        for variable_idx in range(n_variables):
            extent = (
                axes[variable_idx]
                .get_window_extent()
                .transformed(fig.dpi_scale_trans.inverted())
            )
            fig.savefig(
                output_dir / f"mu_{variable_idx}.png",
                bbox_inches=extent,
            )

            # Pad the saved area by 20% in the x-direction and 10% in the
            # y-direction
            fig.savefig(
                output_dir / "ax2_figure_expanded.png",
                bbox_inches=extent.expanded(1.2, 1.2),
            )
            expanded_bbox = mpl.transforms.Bbox(
                [
                    (extent.x0 - 0.15 * extent.width, extent.y0 - 0.35 * extent.height),
                    (extent.x1 + 0.15 * extent.width, extent.y1 + 0.05 * extent.height),
                ]
            )
            fig.savefig(
                output_dir / f"mu_{variable_idx}_expanded.png",
                bbox_inches=expanded_bbox,
            )
