"""
Benchmark CUDA graph capture-and-replay (torch.cuda.graph) against plain eager
execution and torch.compile(fullgraph=True), under a synthetic "grows early, stable
later" FLC training schedule - to measure whether CUDA graphs are worth adopting when
torch.compile is not an option (e.g. because of custom, third-party TNorm/FuzzySet
subclasses that break fullgraph=True tracing).

Unlike torch.compile, CUDA graph capture does not need to trace or understand Python
*semantics* at all - it only records the literal sequence of CUDA kernel launches
actually executed once, then replays them verbatim. That may survive third-party
overrides that break Dynamo tracing. But it comes with its own hard constraint: the
FLC may grow new fuzzy-set/rule parameters mid-training (see
fuzzy.logic.control.controller and tests/test_logic/control/demo_flcs.py's train_model
for the established growth idiom - always a brand new FuzzyLogicController object, not
an in-place mutation), and a captured graph cannot survive that; it must be discarded
and recaptured from scratch. This benchmark measures whether that recapture cost pays
for itself against a schedule where growth is frequent early and rare-to-never later -
exactly the pattern reported by this library's users.

Requires Triton (see fuzzy.logic.control.cuda_graph.force_cuda_graph_safe_branches): backing
out of the Triton fused rule-combination kernel throws the computation onto
torch.prod's built-in CUDA backward kernel, which was confirmed (via
torch.cuda.set_sync_debug_mode) to sync unconditionally regardless of any patch this
codebase could make - a hard PyTorch-level blocker on capturing an FLC's full
forward+backward step without Triton, not a tuning knob. The cuda_graph condition is
skipped, with a warning, if Triton or CUDA is unavailable.

Each new growth segment calls build_flc(config, device) fresh (random re-init), not a
weight-preserving morph from the previous architecture - this benchmark measures
wall-clock timing behavior only, not training convergence/accuracy under real
network-morphism growth. The printed sanity_loss values confirm each segment is
actually training, not stuck; they are not a real learning curve.

Usage:
    .venv/bin/python -m benchmarks.bench_cuda_graphs
    .venv/bin/python -m benchmarks.bench_cuda_graphs --quick
    .venv/bin/python -m benchmarks.bench_cuda_graphs --conditions eager cuda_graph
    .venv/bin/python -m benchmarks.bench_cuda_graphs --total-steps 5000 --checkpoint-every 5
    .venv/bin/python -m benchmarks.bench_cuda_graphs --csv-path out.csv

What this measures:

  For each condition (eager / torch_compile / cuda_graph), the FLC is rebuilt fresh at
  each growth-schedule boundary (a fresh module object naturally triggers a fresh
  Dynamo trace for torch_compile, and forces a fresh GraphedTrainingStep capture for
  cuda_graph - "recapture/recompile on growth" falls out for free from growth already
  meaning "a new model object" in this codebase). Wall-clock time is measured
  cumulatively across the *entire* schedule, including every recapture/recompile/
  construction cost at every segment boundary - that cost paying for itself is exactly
  the question being answered, so it is never excluded or warmed away. A
  torch.cuda.synchronize() is taken once per checkpoint, uniformly across all three
  conditions, before recording the cumulative timestamp - a small constant cost applied
  equally everywhere (see run_condition's docstring for why this is a deliberate,
  documented limitation rather than an oversight).

  The crossover table (compute_crossover_table) reports, per growth boundary and
  per non-eager condition, how many steps of the following stable stretch it takes for
  that condition's absolute cumulative time to drop back under eager's - or "never"
  if the stretch is too short relative to the recapture/recompile cost to pay it back
  within this run, an expected and reportable outcome, not a bug.

A known, documented confound (see force_cuda_graph_safe_branches): the cuda_graph
condition runs with cache_membership=False and three data-dependent sync branches
forced onto their slower alternative for the ENTIRE run (not just capture) - eager, by
contrast, gets its real fast paths (its own cache, and the Triton kernel's real NaN
check). Any place cuda_graph looks worse than expected should be checked against this
confound before concluding it is pure replay overhead.
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch

from benchmarks.common import FLCShapeConfig, build_flc, make_batch
from fuzzy.logic.control.cuda_graph import (
    GraphedTrainingStep,
    force_cuda_graph_safe_branches,
)
from fuzzy.relations.triton_kernels import TRITON_AVAILABLE

GrowthSchedule = List[Tuple[int, FLCShapeConfig]]

DEFAULT_SCHEDULE: GrowthSchedule = [
    (0, FLCShapeConfig(batch_size=256, n_inputs=8, n_terms=3, n_rules=8, n_outputs=4)),
    (20, FLCShapeConfig(batch_size=256, n_inputs=8, n_terms=3, n_rules=12, n_outputs=4)),
    (40, FLCShapeConfig(batch_size=256, n_inputs=8, n_terms=4, n_rules=16, n_outputs=4)),
    (60, FLCShapeConfig(batch_size=256, n_inputs=8, n_terms=4, n_rules=24, n_outputs=4)),
    (80, FLCShapeConfig(batch_size=256, n_inputs=8, n_terms=5, n_rules=32, n_outputs=4)),
]
DEFAULT_TOTAL_STEPS = 2000

QUICK_SCHEDULE: GrowthSchedule = [
    (0, FLCShapeConfig(batch_size=64, n_inputs=4, n_terms=2, n_rules=4, n_outputs=2)),
    (2, FLCShapeConfig(batch_size=64, n_inputs=4, n_terms=2, n_rules=6, n_outputs=2)),
    (4, FLCShapeConfig(batch_size=64, n_inputs=4, n_terms=3, n_rules=8, n_outputs=2)),
]
QUICK_TOTAL_STEPS = 40

CONDITIONS = ("eager", "torch_compile", "cuda_graph")


def _segment_bounds(schedule: GrowthSchedule, total_steps: int) -> List[Tuple[int, int, FLCShapeConfig]]:
    """
    Returns:
        (segment_start_step, segment_end_step, config) triples, one per schedule
        entry, where segment_end_step is the next entry's start (or total_steps for
        the last segment).
    """
    bounds: List[Tuple[int, int, FLCShapeConfig]] = []
    for i, (start_step, config) in enumerate(schedule):
        end_step = schedule[i + 1][0] if i + 1 < len(schedule) else total_steps
        bounds.append((start_step, end_step, config))
    return bounds


def _build_eager_step(config: FLCShapeConfig, device: torch.device):
    model = build_flc(config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    criterion = torch.nn.MSELoss()

    def _step(x: torch.Tensor, y: torch.Tensor, sync: bool) -> float:
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item() if sync else None

    return _step, lambda: None


def _build_compile_step(config: FLCShapeConfig, device: torch.device):
    model = torch.compile(build_flc(config, device=device), fullgraph=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    criterion = torch.nn.MSELoss()

    def _step(x: torch.Tensor, y: torch.Tensor, sync: bool) -> float:
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item() if sync else None

    return _step, lambda: None


def _build_cuda_graph_step(config: FLCShapeConfig, device: torch.device):
    graph_config = FLCShapeConfig(
        batch_size=config.batch_size,
        n_inputs=config.n_inputs,
        n_terms=config.n_terms,
        n_rules=config.n_rules,
        n_outputs=config.n_outputs,
        cache_membership=False,  # see force_cuda_graph_safe_branches's docstring
        seed=config.seed,
    )
    model = build_flc(graph_config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2, capturable=True)
    criterion = torch.nn.MSELoss()
    with force_cuda_graph_safe_branches(model):
        graphed = GraphedTrainingStep(
            model,
            optimizer,
            criterion,
            (config.batch_size, config.n_inputs),
            (config.batch_size, config.n_outputs),
            device,
        )

    def _step(x: torch.Tensor, y: torch.Tensor, sync: bool) -> float:
        return graphed.step(x, y, sync=sync)

    return _step, graphed.close


def run_condition(
    condition: str,
    schedule: GrowthSchedule,
    total_steps: int,
    device: torch.device,
    checkpoint_every: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Run one condition across the whole growth schedule, recording cumulative
    wall-clock time at every checkpoint_every-th step.

    A torch.cuda.synchronize() is taken once per checkpoint, uniformly across all
    three conditions, immediately before recording the cumulative timestamp. This is
    a deliberate, documented choice, not an oversight: it adds a small, constant
    per-checkpoint cost to every condition equally, so it does not bias the relative
    comparison (the crossover point) even though it makes each condition's absolute
    per-step number slightly less flattering than a sync-free measurement would show -
    most notably for cuda_graph, whose true per-replay cost may be small enough for
    this to dominate. A documented follow-up, if this turns out to matter, would be
    checkpointing every K>1 steps and interpolating rather than checkpointing every
    step.

    Returns:
        A DataFrame with columns (step, segment_start_step, wall_clock_cumulative_s,
        sanity_loss), or None if this condition was skipped (cuda_graph on a non-CUDA
        device, or without Triton).
    """
    if condition == "cuda_graph" and (device.type != "cuda" or not TRITON_AVAILABLE):
        warnings.warn(
            "Skipping cuda_graph: requires a CUDA device and Triton (see "
            "force_cuda_graph_safe_branches's docstring for why Triton specifically "
            "is required, not just CUDA)."
        )
        return None

    builders = {
        "eager": _build_eager_step,
        "torch_compile": _build_compile_step,
        "cuda_graph": _build_cuda_graph_step,
    }
    build_step = builders[condition]

    rows: List[Dict[str, object]] = []
    close_fn: Callable[[], None] = lambda: None
    cumulative_before_segment = 0.0
    for segment_start, segment_end, config in _segment_bounds(schedule, total_steps):
        close_fn()

        # construction/recapture/recompile cost is charged to the segment's own
        # elapsed time below (never excluded or warmed away) by starting the clock
        # before build_step, not after - that cost paying for itself is exactly the
        # question this benchmark answers.
        segment_wall_start = time.perf_counter()
        step_fn, close_fn = build_step(config, device)

        for step in range(segment_start, segment_end):
            x = make_batch(config.batch_size, config.n_inputs, device=device)
            y = torch.rand(config.batch_size, config.n_outputs, device=device)
            do_checkpoint = (step - segment_start) % checkpoint_every == 0
            loss = step_fn(x, y, do_checkpoint)
            if do_checkpoint:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                cumulative_s = cumulative_before_segment + (
                    time.perf_counter() - segment_wall_start
                )
                rows.append(
                    {
                        "step": step,
                        "segment_start_step": segment_start,
                        "wall_clock_cumulative_s": cumulative_s,
                        "sanity_loss": loss,
                    }
                )
        if rows:
            cumulative_before_segment = rows[-1]["wall_clock_cumulative_s"]
    close_fn()
    return pd.DataFrame(rows)


def compute_crossover_table(
    results: Dict[str, Optional[pd.DataFrame]], schedule: GrowthSchedule
) -> pd.DataFrame:
    """
    For each growth boundary (after the first, which every condition pays "cold") and
    each non-eager condition, find the first step at or after that boundary where the
    condition's absolute cumulative time drops back under eager's absolute cumulative
    time - i.e. the recapture/recompile cost from this boundary (and any earlier debt)
    has been fully paid back. None if it never does within the run - an expected,
    reportable outcome when growth is too frequent relative to the stable stretch that
    follows it.

    Returns:
        A DataFrame with columns (segment_start_step, condition, recovers_by_step).
    """
    eager_df = results.get("eager")
    if eager_df is None:
        raise ValueError("compute_crossover_table requires the eager condition's results.")

    rows: List[Dict[str, object]] = []
    boundaries = [start for start, _ in schedule][1:]  # skip the first, all pay cold
    for condition in ("torch_compile", "cuda_graph"):
        df = results.get(condition)
        if df is None:
            continue
        merged = pd.merge(df, eager_df, on="step", suffixes=("_cond", "_eager"))
        for boundary in boundaries:
            after = merged[merged["step"] >= boundary]
            recovered = after[
                after["wall_clock_cumulative_s_cond"]
                <= after["wall_clock_cumulative_s_eager"]
            ]
            recovers_by_step = (
                int(recovered.iloc[0]["step"]) if not recovered.empty else None
            )
            rows.append(
                {
                    "segment_start_step": boundary,
                    "condition": condition,
                    "recovers_by_step": recovers_by_step,
                }
            )
    return pd.DataFrame(rows)


def _print_and_maybe_save(name: str, df: pd.DataFrame, csv_path: "Path | None") -> None:
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))
    if csv_path is not None:
        out_path = csv_path.with_name(f"{csv_path.stem}_{name}{csv_path.suffix}")
        df.to_csv(out_path, index=False)
        print(f"(saved to {out_path})")


def main() -> None:
    """
    CLI entry point; see this module's docstring for usage examples.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a much shorter schedule, for a fast smoke test.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override the schedule's total step count.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Record a cumulative-time checkpoint every this many steps.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="Which conditions to run (default: all three).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="If given, save the raw series and crossover table as CSVs next to this path.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    schedule = QUICK_SCHEDULE if args.quick else DEFAULT_SCHEDULE
    total_steps = args.total_steps or (QUICK_TOTAL_STEPS if args.quick else DEFAULT_TOTAL_STEPS)

    print(f"Device: {device}")
    print(f"Schedule: {schedule}")
    print(f"Total steps: {total_steps}")

    results: Dict[str, Optional[pd.DataFrame]] = {}
    for condition in args.conditions:
        print(f"\nRunning condition={condition}...", flush=True)
        df = run_condition(
            condition,
            schedule,
            total_steps,
            device=device,
            checkpoint_every=args.checkpoint_every,
        )
        results[condition] = df
        if df is not None:
            _print_and_maybe_save(condition, df, args.csv_path)

    if "eager" in results and results["eager"] is not None:
        crossover_df = compute_crossover_table(results, schedule)
        _print_and_maybe_save("crossover", crossover_df, args.csv_path)


if __name__ == "__main__":
    main()
