"""
Runs small built-in demo models through fuzzy.dashboard.activation_viewer and
writes out their activation reports.

Usage:
    python examples/fuzzy/dashboard/activation_viewer_demo.py

To use with your own model, see fuzzy.dashboard.activation_viewer.generate_report
and fuzzy.utils.functions.capture directly instead of this script.
"""

import numpy as np
import torch
from torch import nn

from fuzzy.dashboard.activation_viewer import generate_report
from fuzzy.sets import Gaussian
from fuzzy.utils.functions import capture


def demo_mlp():
    """Run a small built-in MLP and write out its activation report."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 10),
    )
    model.eval()

    x = torch.randn(16, 784)  # 16 sample "images"

    with capture(model) as acts:
        model(x)

    generate_report(model, acts, "activations_mlp.html")


def demo_fuzzy_set():
    """
    Run a small Gaussian FuzzySet and write out its activation report.

    Demonstrates activation_viewer's FuzzySet-aware rendering: each variable gets
    its own subplot of term curves, overlaid with a scatter of where these
    observations landed on them - which requires include_inputs=True, since the
    scatter needs the raw observation values, not just the resulting degrees.
    """
    torch.manual_seed(42)

    # 2 variables, 3 terms each
    fuzzy_set = Gaussian(
        centers=np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
        widths=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        device=torch.device("cpu"),
    )
    model = nn.Sequential(fuzzy_set)
    model.eval()

    x = torch.randn(64, 2)

    with capture(model, include_inputs=True) as acts:
        model(x)

    generate_report(model, acts, "activations_fuzzy_set.html")


def main():
    """Run both demos and write out their activation reports."""
    demo_mlp()
    demo_fuzzy_set()


if __name__ == "__main__":
    main()
