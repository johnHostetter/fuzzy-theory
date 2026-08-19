"""
Trains a small built-in demo FuzzyLogicController (FLC) for a few epochs,
periodically capturing a fuzzy.logic.control.parameter_history.ParameterSnapshot,
and serves the resulting training history through fuzzy.dashboard.parameter_viewer.

Usage:
    python examples/fuzzy/dashboard/parameter_viewer_demo.py

Then open http://127.0.0.1:8050, pick a tracked parameter, and a row/col within
it, to see how that value moved over training.

To use with your own FLC and training loop, see
fuzzy.logic.control.parameter_history.capture_parameter_snapshot and
fuzzy.dashboard.parameter_viewer.build_parameter_dashboard directly instead of
this script - this library has no Trainer/fit() of its own, so
capture_parameter_snapshot is meant to be called periodically from a plain,
hand-rolled training loop like the one below.
"""

from typing import List

import numpy as np
import torch

from fuzzy.dashboard.parameter_viewer import build_parameter_dashboard
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import TSK
from fuzzy.logic.control.parameter_history import (
    ParameterSnapshot,
    capture_parameter_snapshot,
)
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.sets.impl import Gaussian

_DEVICE = torch.device("cpu")
_N_EPOCHS = 100
_SNAPSHOT_EVERY = 5


def _build_tsk_flc() -> FLC:
    """
    Build a small toy TSK FLC (2 input variables, 2 terms each, 4 rules).

    Returns:
        The built FuzzyLogicController.
    """
    antecedents = [
        Gaussian(
            centers=np.array([1.0, 5.0]), widths=np.array([1.0, 1.0]), device=_DEVICE
        ),
        Gaussian(
            centers=np.array([0.2, 0.9]), widths=np.array([0.3, 0.3]), device=_DEVICE
        ),
    ]
    rules = [
        Rule(
            premise=Product((0, 0), (1, 0), device=_DEVICE),
            consequence=NAryRelation((0, 0), device=_DEVICE),
        ),
        Rule(
            premise=Product((0, 0), (1, 1), device=_DEVICE),
            consequence=NAryRelation((0, 1), device=_DEVICE),
        ),
        Rule(
            premise=Product((0, 1), (1, 0), device=_DEVICE),
            consequence=NAryRelation((0, 2), device=_DEVICE),
        ),
        Rule(
            premise=Product((0, 1), (1, 1), device=_DEVICE),
            consequence=NAryRelation((0, 3), device=_DEVICE),
        ),
    ]
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
        rules=rules,
    )
    return FLC(source=knowledge_base, inference=TSK, device=_DEVICE)


def train_and_capture_history() -> List[ParameterSnapshot]:
    """
    Train the demo FLC for a few epochs against a fixed random target, capturing a
    ParameterSnapshot every _SNAPSHOT_EVERY epochs - a plain hand-rolled training
    loop, since this library has no Trainer/fit() of its own.

    Returns:
        The captured training history, in epoch order.
    """
    torch.manual_seed(42)
    flc = _build_tsk_flc()

    observations = torch.rand((32, flc.shape.n_inputs), device=_DEVICE) * torch.tensor(
        [6.0, 1.0]
    )
    target = torch.rand((32, flc.shape.n_outputs), device=_DEVICE)

    optimizer = torch.optim.Adam(flc.parameters(), lr=0.05)
    history: List[ParameterSnapshot] = [capture_parameter_snapshot(flc, step=0)]
    for epoch in range(1, _N_EPOCHS + 1):
        optimizer.zero_grad()
        prediction = flc(observations)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

        if epoch % _SNAPSHOT_EVERY == 0:
            history.append(capture_parameter_snapshot(flc, step=epoch))

    return history


def main() -> None:
    """Train the demo FLC, capture its parameter history, and serve the viewer."""
    history: List[ParameterSnapshot] = train_and_capture_history()
    print(f"captured {len(history)} snapshots over {_N_EPOCHS} epochs")
    app = build_parameter_dashboard(history)
    app.run(debug=True)


if __name__ == "__main__":
    main()
