"""
activation_viewer.py

Captures intermediate activations from a PyTorch Sequential model
and generates a self-contained HTML page that displays them as images.

Usage: import and call generate_report() with a model and captured activations
(see fuzzy.utils.functions.capture) - or run examples/fuzzy/dashboard for a
working end-to-end demo.
"""

import base64
import html as html_mod
import io
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
from torch import nn

from fuzzy.sets import FuzzySet, Membership

# headless backend for a report generator that only ever saves figures to
# base64 PNGs - must be selected before pyplot is imported anywhere, so
# pyplot is imported lazily inside the two functions that need it (below)
# rather than at module level here, where a write-time autopep8 pass would
# otherwise hoist it above this call and silently break the backend switch
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# 1. Rendering helpers
# ---------------------------------------------------------------------------


def _unpack_activation(entry):
    """
    Split a captured activation into (inputs, output).

    capture(model, include_inputs=True) stores a plain 2-tuple (inputs, output) per
    layer instead of just the output; a Membership is itself a 3-element tuple, so it
    is distinguished from that (inputs, output) pairing by type, not just length.

    Returns:
        A 2-tuple: the layer's captured input args (or None, if not captured/available),
        and the captured output.
    """
    if (
        isinstance(entry, tuple)
        and not isinstance(entry, Membership)
        and len(entry) == 2
    ):
        inputs, output = entry
        return inputs, output
    return None, entry


def tensor_to_base64_img(
    tensor: torch.Tensor, title: str = "", cmap: str = "viridis"
) -> str:
    """Render a 1-D or 2-D tensor as a base64-encoded PNG heatmap."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    if isinstance(tensor, Membership):
        tensor = tensor.degrees

    arr = tensor.cpu().numpy()

    # If it's a single sample with one dim, reshape to a row
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # For batched data, show the first 32 samples max
    if arr.ndim == 2:
        arr = arr[:32]

    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim == 3:
        shape = arr.shape
        # Find the axis with the largest dimension — keep it
        keep = np.argmax(shape)
        # Move it to the last axis, flatten the other two
        arr = np.moveaxis(arr, keep, -1)  # (small1, small2, big)
        a, b, c = arr.shape
        arr = arr.reshape(a * b, c)
        title = f"{title}  [{shape} → {arr.shape}, axis {keep} kept]"

    fig, ax = plt.subplots(
        figsize=(max(4, arr.shape[1] / 16), max(1.2, arr.shape[0] / 6))
    )
    im = ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("neuron")
    ax.set_ylabel("sample")
    ax.set_title(title, fontsize=10, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def layer_summary(name: str, module: nn.Module, tensor: torch.Tensor) -> dict:
    """Build a summary dict for one layer."""

    if isinstance(tensor, Membership):
        tensor = tensor.degrees

    return {
        "name": name,
        "type": type(module).__name__,
        "shape": list(tensor.shape),
        "min": f"{tensor.min().item():.4f}",
        "max": f"{tensor.max().item():.4f}",
        "mean": f"{tensor.float().mean().item():.4f}",
        "std": f"{tensor.float().std().item():.4f}",
        "zeros_pct": f"{(tensor == 0).float().mean().item() * 100:.1f}%",
    }


def distribution_img(tensor: torch.Tensor) -> str:
    """Render a histogram of values as a base64 PNG."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    if isinstance(tensor, Membership):
        tensor = tensor.degrees

    arr = tensor.cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    ax.hist(arr, bins=60, color="#6c63ff", edgecolor="none", alpha=0.85)
    ax.set_ylabel("count", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_variable_curves(ax, variable_plot, degrees, obs, colors) -> None:
    """Draw one variable's term curves, plus observation scatter if available."""
    for color_idx, term in enumerate(variable_plot.terms):
        color = colors[color_idx % len(colors)]
        ax.plot(
            variable_plot.x_values,
            term.y_values,
            color=color,
            alpha=0.85,
            label=term.label,
        )
        if obs is not None:
            ax.scatter(
                obs[:, variable_plot.variable_idx],
                degrees[:, variable_plot.variable_idx, term.term_idx],
                color=color,
                s=14,
                alpha=0.7,
                edgecolors="none",
            )
    ax.set_title(f"variable {variable_plot.variable_idx}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best")


def fuzzy_set_curves_img(
    mod: FuzzySet, membership: Membership, observations: Optional[torch.Tensor] = None
) -> str:
    """
    Render each of a FuzzySet's variables as its own subplot: one line per term,
    tracing the term's formula curve, overlaid with a scatter of where the actual
    captured observations landed on it (if observations were captured - see
    _unpack_activation) in the matching color.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # pylint: disable-next=import-outside-toplevel
    from fuzzy.sets.visualization import FuzzySetPlot

    variable_plots = FuzzySetPlot(mod).build()

    degrees = (
        membership.degrees.to_dense()
        if membership.degrees.is_sparse
        else membership.degrees
    )
    degrees = degrees.cpu().detach().numpy()  # (batch, variable, term)
    obs = observations.cpu().detach().numpy() if observations is not None else None

    fig, axes = plt.subplots(
        1, len(variable_plots), figsize=(4 * len(variable_plots), 3.2), squeeze=False
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for variable_plot in variable_plots:
        _render_variable_curves(
            axes[0][variable_plot.variable_idx], variable_plot, degrees, obs, colors
        )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# 2. HTML generation
# ---------------------------------------------------------------------------


def _build_layer_tree(names):
    """Build a nested dict from dot-separated layer names.

    Leaf nodes store the full layer name under the key `__leaf__`.
    """
    tree = {}
    for name in names:
        parts = name.split(".")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node["__leaf__"] = name  # mark as a concrete layer
    return tree


def _render_nav(tree, modules, depth=0):
    """Recursively render sidebar nav with collapsible groups."""
    html = ""
    for key, subtree in tree.items():
        if key == "__leaf__":
            continue

        is_leaf = "__leaf__" in subtree
        children = {k: v for k, v in subtree.items() if k != "__leaf__"}
        full_name = subtree.get("__leaf__", "")

        if is_leaf and not children:
            # Pure leaf — just a link
            mod_type = type(modules.get(full_name)).__name__
            html += (
                f'<a href="#layer-{html_mod.escape(full_name)}" '
                f'style="padding-left: {20 + depth * 12}px">'
                f'<span class="nav-idx">{html_mod.escape(key)}</span>'
                f'<span class="nav-type">{mod_type}</span></a>\n'
            )
        else:
            # Group node (may also be a leaf itself)
            html += f'<details class="nav-group" {"open" if depth < 2 else ""}>\n'
            if is_leaf:
                mod_type = type(modules.get(full_name)).__name__
                html += (
                    f'<summary style="padding-left: {20 + depth * 12}px">'
                    f'<a href="#layer-{html_mod.escape(full_name)}" '
                    f'class="nav-group-link">'
                    f'<span class="nav-idx">{html_mod.escape(key)}</span>'
                    f'<span class="nav-type">{mod_type}</span></a></summary>\n'
                )
            else:
                html += (
                    f'<summary style="padding-left: {20 + depth * 12}px">'
                    f'<span class="nav-idx">{html_mod.escape(key)}</span></summary>\n'
                )
            html += _render_nav(children, modules, depth + 1)
            html += "</details>\n"

    return html


def _render_leaf_card(full_name, modules, activations) -> str:
    """Render a single leaf's card, or "" if it has no captured activation."""
    if full_name not in activations:
        return ""

    inputs, tensor = _unpack_activation(activations[full_name])
    mod = modules.get(full_name)
    info = layer_summary(full_name, mod, tensor)
    heatmap_b64 = tensor_to_base64_img(
        tensor,
        title=f"{info['type']}  [{', '.join(str(s) for s in info['shape'])}]",
    )
    hist_b64 = distribution_img(tensor)

    curves_b64 = None
    if (
        isinstance(mod, FuzzySet)
        and isinstance(tensor, Membership)
        and tensor.degrees.dim() == 3
    ):
        # observations, if captured (see capture(..., include_inputs=True)), are the
        # FuzzySet's sole forward() argument
        observations = inputs[0] if inputs else None
        curves_b64 = fuzzy_set_curves_img(mod, tensor, observations)

    return _card_html(full_name, info, heatmap_b64, hist_b64, curves_b64)


def _render_cards(tree, modules, activations, depth=0):
    """Recursively render cards, wrapping groups in collapsible sections."""
    html = ""
    for key, subtree in tree.items():
        if key == "__leaf__":
            continue

        is_leaf = "__leaf__" in subtree
        children = {k: v for k, v in subtree.items() if k != "__leaf__"}
        full_name = subtree.get("__leaf__", "")

        # Render this node's own card if it's a leaf
        card = _render_leaf_card(full_name, modules, activations) if is_leaf else ""

        if not children:
            # Pure leaf
            html += card
        else:
            # Group wrapper
            html += (
                f'<details class="group" {"open" if depth < 2 else ""} '
                f'style="--depth: {depth}">\n'
                f'<summary class="group-header">'
                f'<span class="group-name">{html_mod.escape(key)}</span>'
                f'<span class="group-count">'
                f"{_count_leaves(subtree)} layers</span></summary>\n"
                f'<div class="group-body">\n'
            )
            html += card  # own card first if it's also a leaf
            html += _render_cards(children, modules, activations, depth + 1)
            html += "</div>\n</details>\n"

    return html


def _count_leaves(tree):
    """Count leaf nodes in the tree."""
    count = 0
    for key, val in tree.items():
        if key == "__leaf__":
            count += 1
        elif isinstance(val, dict):
            count += _count_leaves(val)
    return count


def _card_html(name, info, heatmap_b64, hist_b64, curves_b64=None):
    curves_section = ""
    if curves_b64:
        curves_section = f"""
        <div class="curves-wrap">
            <img src="data:image/png;base64,{curves_b64}" alt="fuzzy set formula curves" />
        </div>
        """
    return f"""
    <div class="card" id="layer-{html_mod.escape(name)}">
        <div class="card-header">
            <span class="layer-name">{html_mod.escape(name)}</span>
            <span class="layer-type">{html_mod.escape(info['type'])}</span>
        </div>
        <div class="card-body">
            <div class="heatmap-wrap">
                <img src="data:image/png;base64,{heatmap_b64}" alt="activation heatmap" />
            </div>
            <div class="sidebar">
                <div class="stat-grid">
                    <div class="stat"><span class="stat-label">shape</span><span class="stat-value">{info['shape']}</span></div>
                    <div class="stat"><span class="stat-label">min</span><span class="stat-value">{info['min']}</span></div>
                    <div class="stat"><span class="stat-label">max</span><span class="stat-value">{info['max']}</span></div>
                    <div class="stat"><span class="stat-label">mean</span><span class="stat-value">{info['mean']}</span></div>
                    <div class="stat"><span class="stat-label">std</span><span class="stat-value">{info['std']}</span></div>
                    <div class="stat"><span class="stat-label">zeros</span><span class="stat-value">{info['zeros_pct']}</span></div>
                </div>
                <img class="hist" src="data:image/png;base64,{hist_b64}" alt="value distribution" />
            </div>
        </div>
        {curves_section}
    </div>
    """


def _build_html(model, activations: OrderedDict) -> str:
    """Return a complete HTML string for the activation report."""
    modules = dict(model.named_modules())
    tree = _build_layer_tree(activations.keys())

    nav_html = _render_nav(tree, modules)
    cards_html = _render_cards(tree, modules, activations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Activation Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: #f4f4f8;
    color: #22222e;
    line-height: 1.5;
  }}

  .page {{
    display: flex;
    min-height: 100vh;
  }}

  /* --- sidebar nav --- */
  .nav {{
    position: sticky;
    top: 0;
    height: 100vh;
    width: 220px;
    flex-shrink: 0;
    background: #eceef5;
    border-right: 1px solid #d8d8e3;
    padding: 24px 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0;
  }}
  .nav-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6c63ff;
    padding: 0 20px 16px;
  }}
  .nav a {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 20px;
    text-decoration: none;
    color: #5a5a6e;
    font-size: 13px;
    transition: background 0.15s, color 0.15s;
  }}
  .nav a:hover {{
    background: #dfe0ec;
    color: #1a1a2e;
  }}
  .nav-idx {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 12px;
    color: #6c63ff;
    min-width: 20px;
  }}
  .nav-type {{
    font-size: 11px;
    color: #8a8a9a;
  }}

  /* nav groups */
  .nav-group {{
    display: flex;
    flex-direction: column;
  }}
  .nav-group > summary {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 20px;
    font-size: 13px;
    color: #5a5a6e;
    cursor: pointer;
    list-style: none;
    user-select: none;
    transition: background 0.15s;
  }}
  .nav-group > summary::-webkit-details-marker {{ display: none; }}
  .nav-group > summary::before {{
    content: '▸';
    font-size: 10px;
    color: #9a9aae;
    transition: transform 0.15s;
  }}
  .nav-group[open] > summary::before {{
    transform: rotate(90deg);
  }}
  .nav-group > summary:hover {{
    background: #dfe0ec;
    color: #1a1a2e;
  }}
  .nav-group-link {{
    display: contents;
  }}

  /* --- main content --- */
  .main {{
    flex: 1;
    padding: 40px 48px;
    max-width: 960px;
  }}
  .main h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 6px;
  }}
  .main .subtitle {{
    font-size: 14px;
    color: #5a5a70;
    margin-bottom: 36px;
  }}

  /* --- collapsible groups in main --- */
  .group {{
    margin-bottom: 16px;
    border: 1px solid #d8d8e3;
    border-radius: 10px;
    background: rgba(230, 230, 240, calc(0.4 + var(--depth, 0) * 0.15));
    overflow: hidden;
  }}
  .group-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 20px;
    cursor: pointer;
    list-style: none;
    user-select: none;
    border-bottom: 1px solid transparent;
    transition: background 0.15s;
  }}
  .group-header::-webkit-details-marker {{ display: none; }}
  .group-header::before {{
    content: '▸';
    font-size: 11px;
    color: #6c63ff;
    transition: transform 0.15s;
  }}
  .group[open] > .group-header {{
    border-bottom-color: #d8d8e3;
  }}
  .group[open] > .group-header::before {{
    transform: rotate(90deg);
  }}
  .group-header:hover {{
    background: #e4e4f0;
  }}
  .group-name {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 14px;
    color: #4c46b3;
  }}
  .group-count {{
    font-size: 11px;
    color: #6a6a80;
    margin-left: auto;
  }}
  .group-body {{
    padding: 12px;
  }}

  /* --- cards --- */
  .card {{
    background: #ffffff;
    border: 1px solid #e0e0ea;
    border-radius: 10px;
    margin-bottom: 16px;
    overflow: hidden;
    scroll-margin-top: 24px;
  }}
  .card:last-child {{
    margin-bottom: 0;
  }}
  .card-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    border-bottom: 1px solid #e0e0ea;
  }}
  .layer-name {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 15px;
    color: #4c46b3;
  }}
  .layer-type {{
    font-size: 12px;
    background: #eceef5;
    color: #5a5a6e;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }}

  .card-body {{
    display: flex;
    gap: 20px;
    padding: 20px;
    flex-wrap: wrap;
  }}
  .heatmap-wrap {{
    flex: 1;
    min-width: 280px;
  }}
  .heatmap-wrap img {{
    width: 100%;
    border-radius: 6px;
  }}

  .sidebar {{
    width: 220px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .stat-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  .stat {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .stat-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8a8a9a;
    font-family: 'IBM Plex Mono', monospace;
  }}
  .stat-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #2a2a3a;
  }}

  .hist {{
    width: 100%;
    border-radius: 6px;
  }}

  .curves-wrap {{
    padding: 0 20px 20px;
  }}
  .curves-wrap img {{
    width: 100%;
    border-radius: 6px;
  }}

  @media (max-width: 800px) {{
    .nav {{ display: none; }}
    .main {{ padding: 20px; }}
    .sidebar {{ width: 100%; }}
  }}
</style>
</head>
<body>
<div class="page">
  <nav class="nav">
    <div class="nav-title">Layers</div>
    {nav_html}
  </nav>
  <div class="main">
    <h1>Activation Viewer</h1>
    <p class="subtitle">Intermediate outputs captured from forward pass &mdash; {len(activations)} layers</p>
    {cards_html}
  </div>
</div>
</body>
</html>"""


def generate_report(model, activations: OrderedDict, path: str = "activations.html"):
    """Write the activation report to an HTML file."""
    html = _build_html(model, activations)
    Path(path).write_text(html, encoding="utf-8")
    print(f"wrote {path}  ({Path(path).stat().st_size / 1024:.0f} KB)")
