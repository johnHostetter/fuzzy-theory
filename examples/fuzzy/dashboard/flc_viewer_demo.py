"""
Runs a small built-in demo FuzzyLogicController (FLC) through
fuzzy.dashboard.flc_viewer and serves the interactive report.

Usage:
    python examples/fuzzy/dashboard/flc_viewer_demo.py [--flc {tsk,mamdani}]

Then open http://127.0.0.1:8050 and click a pipeline node to inspect that stage.

To use with your own FLC, see fuzzy.dashboard.flc_viewer.generate_flc_report and
fuzzy.logic.control.trace.capture_flc_trace directly instead of this script.
"""

import argparse

import dash
import numpy as np
import torch

from fuzzy.dashboard.flc_viewer import generate_flc_report
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import TSK, Mamdani
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.sets.impl import Gaussian

_DEVICE = torch.device("cpu")


def demo_tsk_flc() -> dash.Dash:
    """
    Build a small toy TSK FLC (2 input variables, 2 terms each, 4 rules) and
    return a Dash app visualizing one forward pass through it.

    Returns:
        The assembled Dash app - call app.run(...) to serve it.
    """
    torch.manual_seed(42)

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
    flc = FLC(source=knowledge_base, inference=TSK, device=_DEVICE)

    observations = torch.rand((32, flc.shape.n_inputs), device=_DEVICE) * torch.tensor(
        [6.0, 1.0]
    )
    return generate_flc_report(flc, observations)


def demo_mamdani_flc() -> dash.Dash:
    """
    Build a small toy Mamdani FLC (2 input variables, 2 terms each, 4 rules, 1
    output variable) and return a Dash app visualizing one forward pass through it.

    Returns:
        The assembled Dash app - call app.run(...) to serve it.
    """
    torch.manual_seed(42)

    antecedents = [
        Gaussian(
            centers=np.array([1.0, 5.0]), widths=np.array([1.0, 1.0]), device=_DEVICE
        ),
        Gaussian(
            centers=np.array([0.2, 0.9]), widths=np.array([0.3, 0.3]), device=_DEVICE
        ),
    ]
    consequent = Gaussian(
        centers=np.array([-1.0, 1.0]), widths=np.array([0.5, 0.5]), device=_DEVICE
    )
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
            consequence=NAryRelation((0, 1), device=_DEVICE),
        ),
        Rule(
            premise=Product((0, 1), (1, 1), device=_DEVICE),
            consequence=NAryRelation((0, 0), device=_DEVICE),
        ),
    ]
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=antecedents, targets=[consequent]
        ),
        rules=rules,
    )
    flc = FLC(source=knowledge_base, inference=Mamdani, device=_DEVICE)

    observations = torch.rand((32, flc.shape.n_inputs), device=_DEVICE) * torch.tensor(
        [6.0, 1.0]
    )
    return generate_flc_report(flc, observations)


def main() -> None:
    """Parse --flc and run the corresponding demo's Dash app."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flc",
        choices=["tsk", "mamdani"],
        default="tsk",
        help="Which toy FLC's defuzzification method to demo (default: tsk).",
    )
    args = parser.parse_args()

    app = demo_tsk_flc() if args.flc == "tsk" else demo_mamdani_flc()
    app.run(debug=True)


if __name__ == "__main__":
    main()
