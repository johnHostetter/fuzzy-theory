"""
Captures a sequence of snapshots of a FuzzyLogicController (FLC)'s learnable
parameters over the course of training, for later inspection (e.g. by
fuzzy.dashboard.parameter_viewer). Pure torch/dataclasses - no visualization
dependency - so this stays importable from a training loop without installing the
optional "dashboard" extra.

This library has no Trainer/fit() of its own (see
tests/test_logic/control/demo_flcs.py's train_model for the established
hand-rolled-loop idiom), so there is nowhere to attach an automatic snapshotting
hook. capture_parameter_snapshot is instead a plain function meant to be called
periodically from your own training loop:

    history = []
    for epoch in range(n_epochs):
        ...  # your usual forward/backward/optimizer.step()
        if epoch % 10 == 0:
            history.append(capture_parameter_snapshot(flc, step=epoch))

    from fuzzy.dashboard.parameter_viewer import build_parameter_dashboard
    build_parameter_dashboard(history).run(debug=True)
"""

from dataclasses import dataclass
from typing import Dict

import torch

from fuzzy.logic.control.controller import FuzzyLogicController
from fuzzy.logic.control.defuzzification import TSK, Mamdani


@dataclass
class ParameterSnapshot:
    """One point-in-time snapshot of an FLC's learnable parameters."""

    # a caller-chosen label (e.g. epoch) - only used as the x-axis later
    step: int
    # stable name -> detached, cloned tensor
    parameters: Dict[str, torch.Tensor]


def _flc_named_parameters(flc: FuzzyLogicController) -> Dict[str, torch.Tensor]:
    """
    The learnable parameters this library's own training/visualization tooling
    tracks for an FLC, under clean, stable names - not FuzzyLogicController.
    named_parameters()'s dotted paths, which leak internal wrapper structure (e.g.
    "input_granulation.modules_list.0._params.centers.params.0") and put Mamdani's
    consequences under ".source" rather than ".consequences".

    Returns:
        A mapping from a stable parameter name to its current tensor value.
    """
    parameters = {
        "input_granulation.centers": flc.input_granulation.centers,
        "input_granulation.widths": flc.input_granulation.widths,
    }

    defuzzification = flc.defuzzification
    if isinstance(defuzzification, TSK):
        # the (n_outputs, n_rules, n_inputs + 1) property, not the raw .weights/
        # .bias nn.Parameters - same values, but .consequences is the properly
        # laid-out, human-interpretable view (see _tsk_rule_weights_figure in
        # fuzzy.dashboard.flc_viewer, which reads from the same property)
        parameters["defuzzification.consequences"] = defuzzification.consequences
    elif isinstance(defuzzification, Mamdani):
        # defuzzification.consequences is itself a FuzzySetGroup over the output
        # space - same .centers/.widths properties as input_granulation
        parameters["defuzzification.centers"] = defuzzification.consequences.centers
        parameters["defuzzification.widths"] = defuzzification.consequences.widths

    return parameters


def capture_parameter_snapshot(
    flc: FuzzyLogicController, step: int
) -> ParameterSnapshot:
    """
    Capture the FLC's current learnable parameter values.

    Args:
        flc: The FuzzyLogicController to snapshot.
        step: A caller-chosen step label (e.g. epoch number) for this snapshot -
            used only for the x-axis when plotting the resulting history.

    Returns:
        The captured snapshot. Each tensor is detached and cloned, so later
        optimizer steps cannot mutate an already-captured snapshot in place.
    """
    parameters = {
        name: tensor.detach().clone().cpu()
        for name, tensor in _flc_named_parameters(flc).items()
    }
    return ParameterSnapshot(step=step, parameters=parameters)
