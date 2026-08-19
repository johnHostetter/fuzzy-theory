"""
Captures a structured, stage-by-stage trace of a single FuzzyLogicController (FLC)
forward pass, for later inspection (e.g. by fuzzy.dashboard.flc_viewer). Pure
torch/dataclasses - no visualization dependency - so this stays importable from a
training loop without installing the optional "dashboard" extra.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch

from fuzzy.logic.control.configurations.data import Shape
from fuzzy.logic.control.controller import FuzzyLogicController
from fuzzy.logic.rule_helpers import tensor_to_rules
from fuzzy.sets.membership import Membership
from fuzzy.utils.functions import capture

# the exact named children FuzzyLogicController.disable_parameters_and_build()
# always registers, in pipeline order - see controller.py's
# __init__/_forward_impl
_STAGE_NAMES: Tuple[str, str, str] = ("input_granulation", "engine", "defuzzification")


@dataclass
class StageTrace:
    """
    One pipeline stage's captured output from a single FuzzyLogicController forward
    pass.

    Keeps a reference to the live producing module (not just its type name) - unlike
    Membership.formula elsewhere in this library, which deliberately stores only a
    class-name str to stay memory-efficient on the hot training-time forward path.
    A trace is built once, on request, purely for inspection (mirroring how
    fuzzy.dashboard.activation_viewer's capture() already keeps live module
    references around for its own detail rendering) - so the module itself (e.g. to
    re-derive a FuzzySet's formula curve) is worth keeping.
    """

    name: str  # one of _STAGE_NAMES
    module: torch.nn.Module
    # type(module).__name__, e.g. "FuzzySetGroup", "Product", "TSK"
    module_type: str
    output: Union[torch.Tensor, Membership]


@dataclass
class FLCTrace:
    """
    One FuzzyLogicController forward pass, captured stage-by-stage.

    Deliberately shaped as "one complete frame" (every stage captured together, from
    one forward pass) rather than coupled to how it will be displayed, so a future
    live/streaming mode could emit the same StageTrace shape incrementally (one per
    completed stage) without redesigning this type - see
    fuzzy.dashboard.flc_viewer for the (replay-only, for now) consumer.
    """

    # (batch, n_inputs) - the forward pass's argument
    observations: torch.Tensor
    stages: List[StageTrace]  # length 3, in pipeline order (see _STAGE_NAMES)
    # (batch, n_outputs) - the forward pass's return value
    output: torch.Tensor
    rule_tensor: torch.Tensor  # (n_vars, n_terms, n_rules) binary, dense
    rule_strings: List[str]  # tensor_to_rules(rule_tensor)
    shape: Shape


def capture_flc_trace(
    flc: FuzzyLogicController, observations: torch.Tensor
) -> FLCTrace:
    """
    Run one forward pass through an FLC and capture each pipeline stage's output.

    Captures outputs only (not inputs): FuzzyLogicController._defuzzify_tsk calls
    self.defuzzification(...) with keyword arguments while _defuzzify_standard calls
    it positionally, so a forward hook's captured *positional* input args would be
    inconsistent (empty for TSK) across defuzzification methods. Each stage's
    logical input is instead reconstructed by the caller from the previous stage's
    output (or observations, for the first stage), which is unambiguous regardless
    of how a given Defuzzification subclass's forward() is invoked.

    Args:
        flc: The FuzzyLogicController to run and capture a trace of.
        observations: The observations to run through the FLC (its forward()
            argument).

    Returns:
        The captured trace.
    """
    named_children = dict(flc.named_children())
    missing_stages = [name for name in _STAGE_NAMES if name not in named_children]
    if missing_stages:
        raise ValueError(
            f"The given FLC is missing expected stage(s) {missing_stages}; a "
            f"FuzzyLogicController must have {_STAGE_NAMES} as named children."
        )

    with capture(flc, layers=list(_STAGE_NAMES), include_inputs=False) as activations:
        output = flc(observations)

    stages = [
        StageTrace(
            name=name,
            module=named_children[name],
            module_type=type(named_children[name]).__name__,
            output=activations[name],
        )
        for name in _STAGE_NAMES
    ]

    engine_mask = stages[1].output.mask
    rule_tensor = engine_mask.to_dense() if engine_mask.is_sparse else engine_mask
    rule_strings = tensor_to_rules(rule_tensor)

    return FLCTrace(
        observations=observations,
        stages=stages,
        output=output,
        rule_tensor=rule_tensor,
        rule_strings=rule_strings,
        shape=flc.shape,
    )
