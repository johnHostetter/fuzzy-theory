"""
Benchmark FuzzyLogicController (FLC) eager execution against
torch.compile(flc, fullgraph=True), to measure how much torch.compile speeds up the
FLC's forward-only ("eval") and forward+backward ("train") steps.

fullgraph=True is used deliberately rather than the default (graph-break-tolerant)
mode: it is the stricter guarantee that Dynamo traced the *entire* forward pass into
a single graph with no fallback to eager PyTorch anywhere inside it (see
sets/abstract.py, sets/group.py, relations/t_norm.py, and relations/n_ary.py for the
graph breaks that had to be fixed to make this possible).

Usage:
    .venv/bin/python -m benchmarks.bench_torch_compile
    .venv/bin/python -m benchmarks.bench_torch_compile --quick
    .venv/bin/python -m benchmarks.bench_torch_compile --axes batch_size n_rules
    .venv/bin/python -m benchmarks.bench_torch_compile --device cpu --csv-path out.csv

What this measures:

  1. Axis sweeps (--axes): for each of the five FLCShapeConfig knobs, hold the other
     four at the baseline and vary that one, timing eager vs. compiled eval and train
     steps. torch.compile's win is expected to be largest where the FLC is dominated
     by Python-level dispatch overhead rather than raw FLOPs (small batch_size/n_rules)
     - see benchmarks/common.py's H1 - since compilation fuses that overhead away.
  2. gradient_checkpointing ablation (always run, at the baseline config): compares
     compiled vs. eager train-step timing with gradient_checkpointing enabled, since
     that combination only recently became compatible with fullgraph=True (the applied
     mask used to be stashed on a mutated module attribute inside the checkpoint's
     traced subgraph, which Dynamo forbids - see relations/n_ary.py's
     _apply_mask_with_mask).

The first call to the compiled model in each timed function pays Dynamo's tracing/
compilation cost; n_warmup absorbs this before any timed repetition runs.

Results are printed as tables and, if --csv-path is given, written to CSV files.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch

from benchmarks.common import (
    FLCShapeConfig,
    benchmark_forward_backward,
    benchmark_forward_only,
    build_flc,
    make_batch,
)
from fuzzy.logic.control.configurations.data import ExecutionOptions
from fuzzy.logic.control.controller import FuzzyLogicController as FLC

BASELINE = FLCShapeConfig(
    batch_size=256, n_inputs=8, n_terms=5, n_rules=32, n_outputs=4
)

DEFAULT_SWEEP_VALUES: Dict[str, List[int]] = {
    "batch_size": [8, 32, 128, 512, 2048],
    "n_inputs": [4, 16, 64, 256],
    "n_terms": [2, 5, 12],
    "n_rules": [4, 32, 256, 2048],
    "n_outputs": [1, 4, 16],
}

QUICK_SWEEP_VALUES: Dict[str, List[int]] = {
    "batch_size": [8, 64],
    "n_inputs": [2, 4],
    "n_terms": [2, 3],
    "n_rules": [4, 16],
    "n_outputs": [1, 2],
}


def _with_override(config: FLCShapeConfig, axis: str, value: int) -> FLCShapeConfig:
    kwargs = dict(
        batch_size=config.batch_size,
        n_inputs=config.n_inputs,
        n_terms=config.n_terms,
        n_rules=config.n_rules,
        n_outputs=config.n_outputs,
        cache_membership=config.cache_membership,
        seed=config.seed,
    )
    kwargs[axis] = value
    return FLCShapeConfig(**kwargs)


def _run_one_config(
    config: FLCShapeConfig,
    device: torch.device,
    n_repeats: int,
    n_warmup: int,
) -> Dict[str, float]:
    eager_flc = build_flc(config, device=device)
    compiled_flc = torch.compile(build_flc(config, device=device), fullgraph=True)

    def make_input() -> torch.Tensor:
        return make_batch(config.batch_size, config.n_inputs, device=device)

    eager_eval = benchmark_forward_only(
        eager_flc, make_input, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    compiled_eval = benchmark_forward_only(
        compiled_flc, make_input, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    eager_train = benchmark_forward_backward(
        eager_flc,
        make_input,
        n_outputs=config.n_outputs,
        device=device,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
    )
    compiled_train = benchmark_forward_backward(
        compiled_flc,
        make_input,
        n_outputs=config.n_outputs,
        device=device,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
    )

    return {
        "eager_eval_mean_s": eager_eval.mean,
        "eager_eval_std_s": eager_eval.std,
        "compiled_eval_mean_s": compiled_eval.mean,
        "compiled_eval_std_s": compiled_eval.std,
        "eval_speedup_x": eager_eval.mean / compiled_eval.mean,
        "eager_train_mean_s": eager_train.mean,
        "eager_train_std_s": eager_train.std,
        "compiled_train_mean_s": compiled_train.mean,
        "compiled_train_std_s": compiled_train.std,
        "train_speedup_x": eager_train.mean / compiled_train.mean,
    }


def run_axis_sweep(
    axis: str,
    values: Sequence[int],
    baseline: FLCShapeConfig,
    device: torch.device,
    n_repeats: int,
    n_warmup: int,
) -> pd.DataFrame:
    """
    Sweep a single shape axis, holding the rest of the baseline config fixed, timing
    eager vs. compiled eval and train steps at each value.
    """
    rows: List[Dict[str, float]] = []
    for value in values:
        config = _with_override(baseline, axis, value)
        print(f"  [{axis}={value}] building and timing...", flush=True)
        result = _run_one_config(
            config, device=device, n_repeats=n_repeats, n_warmup=n_warmup
        )
        rows.append({axis: value, **result})
    return pd.DataFrame(rows)


def run_gradient_checkpointing_ablation(
    baseline: FLCShapeConfig, device: torch.device, n_repeats: int, n_warmup: int
) -> pd.DataFrame:
    """
    Compare eager vs. compiled(fullgraph=True) train-step timing with
    gradient_checkpointing enabled, at the baseline config.
    """

    def make_input() -> torch.Tensor:
        return make_batch(baseline.batch_size, baseline.n_inputs, device=device)

    def build(compiled: bool) -> FLC:
        flc = build_flc(baseline, device=device)
        # rebuild with gradient_checkpointing on, reusing the same source
        flc = FLC(
            source=flc.source,
            inference=type(flc.defuzzification),
            device=device,
            execution=ExecutionOptions(gradient_checkpointing=True),
        )
        return torch.compile(flc, fullgraph=True) if compiled else flc

    eager_flc = build(compiled=False)
    compiled_flc = build(compiled=True)

    eager_train = benchmark_forward_backward(
        eager_flc,
        make_input,
        n_outputs=baseline.n_outputs,
        device=device,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
    )
    compiled_train = benchmark_forward_backward(
        compiled_flc,
        make_input,
        n_outputs=baseline.n_outputs,
        device=device,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
    )
    return pd.DataFrame(
        [
            {
                "eager_train_mean_s": eager_train.mean,
                "eager_train_std_s": eager_train.std,
                "compiled_train_mean_s": compiled_train.mean,
                "compiled_train_std_s": compiled_train.std,
                "train_speedup_x": eager_train.mean / compiled_train.mean,
            }
        ]
    )


def _print_and_maybe_save(
    name: str, df: pd.DataFrame, csv_path: "Path | None"
) -> None:
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
        help="Device to run the benchmark on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=list(DEFAULT_SWEEP_VALUES.keys()),
        choices=list(DEFAULT_SWEEP_VALUES.keys()),
        help="Which shape axes to sweep (default: all five).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=50,
        help="Timed repetitions per configuration.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Untimed warmup calls beforehand (also absorbs the compiled model's "
        "first-call Dynamo tracing/compilation cost).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a much smaller sweep and fewer repeats, for a fast smoke test.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="If given, save each results table as a CSV next to this path.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    sweep_values = QUICK_SWEEP_VALUES if args.quick else DEFAULT_SWEEP_VALUES
    n_repeats = 3 if args.quick else args.n_repeats
    n_warmup = 2 if args.quick else args.n_warmup

    print(f"Device: {device}")
    print(f"Baseline config: {BASELINE}")

    for axis in args.axes:
        print(f"\nSweeping {axis} over {sweep_values[axis]}...")
        df = run_axis_sweep(
            axis,
            sweep_values[axis],
            BASELINE,
            device=device,
            n_repeats=n_repeats,
            n_warmup=n_warmup,
        )
        _print_and_maybe_save(axis, df, args.csv_path)

    print("\nRunning gradient_checkpointing ablation at the baseline config...")
    gc_df = run_gradient_checkpointing_ablation(
        BASELINE, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    _print_and_maybe_save("gradient_checkpointing", gc_df, args.csv_path)


if __name__ == "__main__":
    main()
