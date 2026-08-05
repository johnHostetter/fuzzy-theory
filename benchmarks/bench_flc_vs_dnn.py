"""
Benchmark FuzzyLogicController (FLC) against a comparably-shaped plain deep neural
network (DNN), sweeping batch size, input dimensionality, number of fuzzy sets (terms)
per input dimension, number of rules, and output size - to locate where the FLC's
forward/backward cost comes from relative to a DNN of the same batch size, input size,
and output size (3 hidden layers of 512 neurons, by default).

Usage:
    .venv/bin/python -m benchmarks.bench_flc_vs_dnn
    .venv/bin/python -m benchmarks.bench_flc_vs_dnn --quick          # fast smoke test
    .venv/bin/python -m benchmarks.bench_flc_vs_dnn --axes batch_size n_rules
    .venv/bin/python -m benchmarks.bench_flc_vs_dnn --device cpu --csv-path out.csv

What this measures, and why (see benchmarks/common.py's module docstring for the full
hypothesis writeup this is meant to test):

  1. Axis sweeps (--axes): for each of the five knobs, hold the other four at a fixed
     baseline and vary that one knob, timing FLC vs. DNN forward-only ("eval") and
     forward+backward ("train") steps. This is the headline "how does the slowdown
     scale" data.
  2. Stage breakdown (always run, at the baseline config): splits one FLC forward pass
     into fuzzification / rule engine / defuzzification, to localize where time goes
     within the FLC itself.
  3. Cache ablation (always run, at the baseline config): re-times eval-mode forward
     passes with the FLC's membership cache enabled vs. disabled, to check whether the
     cache (which is keyed on tensor identity, and therefore always misses when every
     call gets a fresh mini-batch) is pure overhead in a typical training loop.

Results are printed as tables and, if --csv-path is given, written to CSV files
(one per section, suffixed by section name) for later analysis/plotting.
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
    build_dnn,
    build_flc,
    make_batch,
    profile_flc_stages,
)

BASELINE = FLCShapeConfig(
    batch_size=256, n_inputs=8, n_terms=5, n_rules=32, n_outputs=4
)

DEFAULT_SWEEP_VALUES: Dict[str, List[int]] = {
    "batch_size": [32, 128, 512, 2048],
    "n_inputs": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "n_terms": [2, 3, 5, 8, 12, 24],
    "n_rules": [4, 16, 64, 256, 1024, 2048],
    "n_outputs": [1, 2, 4, 8, 16],
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
    flc = build_flc(config, device=device)
    dnn = build_dnn(n_inputs=config.n_inputs, n_outputs=config.n_outputs, device=device)

    def make_input() -> torch.Tensor:
        return make_batch(config.batch_size, config.n_inputs, device=device)

    flc_eval = benchmark_forward_only(
        flc, make_input, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    dnn_eval = benchmark_forward_only(
        dnn, make_input, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    flc_train = benchmark_forward_backward(
        flc,
        make_input,
        n_outputs=config.n_outputs,
        device=device,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
    )
    dnn_train = benchmark_forward_backward(
        dnn,
        make_input,
        n_outputs=config.n_outputs,
        device=device,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
    )

    return {
        "flc_eval_mean_s": flc_eval.mean,
        "flc_eval_std_s": flc_eval.std,
        "dnn_eval_mean_s": dnn_eval.mean,
        "dnn_eval_std_s": dnn_eval.std,
        "eval_slowdown_x": flc_eval.mean / dnn_eval.mean,
        "flc_train_mean_s": flc_train.mean,
        "flc_train_std_s": flc_train.std,
        "dnn_train_mean_s": dnn_train.mean,
        "dnn_train_std_s": dnn_train.std,
        "train_slowdown_x": flc_train.mean / dnn_train.mean,
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
    FLC vs. DNN eval and train steps at each value.
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


def run_stage_profile(
    baseline: FLCShapeConfig, device: torch.device, n_repeats: int, n_warmup: int
) -> pd.DataFrame:
    """
    Break the FLC's forward pass into fuzzification/engine/defuzzification stages at
    the baseline config, to localize where time is spent within the FLC itself.
    """
    flc = build_flc(baseline, device=device)

    def make_input() -> torch.Tensor:
        return make_batch(baseline.batch_size, baseline.n_inputs, device=device)

    stages = profile_flc_stages(
        flc, make_input, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    return pd.DataFrame(
        [
            {"stage": "fuzzification", "seconds": stages.fuzzification},
            {"stage": "rule_engine", "seconds": stages.engine},
            {"stage": "defuzzification", "seconds": stages.defuzzification},
            {"stage": "end_to_end (measured directly)", "seconds": stages.end_to_end},
        ]
    )


def run_cache_ablation(
    baseline: FLCShapeConfig, device: torch.device, n_repeats: int, n_warmup: int
) -> pd.DataFrame:
    """
    Compare eval-mode forward timings with the FLC's membership cache enabled vs.
    disabled, at the baseline config, to check whether the cache pays for itself when
    every call gets a fresh mini-batch (see H3 in benchmarks/common.py).
    """
    rows: List[Dict[str, float]] = []
    for cache_membership in (True, False):
        config = _with_override(baseline, "cache_membership", cache_membership)
        flc = build_flc(config, device=device)

        def make_input(config=config) -> torch.Tensor:
            return make_batch(config.batch_size, config.n_inputs, device=device)

        timing = benchmark_forward_only(
            flc, make_input, device=device, n_repeats=n_repeats, n_warmup=n_warmup
        )
        rows.append(
            {
                "cache_membership": cache_membership,
                "eval_mean_s": timing.mean,
                "eval_std_s": timing.std,
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
        default=100,
        help="Timed repetitions per configuration. This environment's GPU timings are "
        "noisy enough (std comparable to the mean) that 20 repeats produces "
        "differences that look real but vanish at higher repeat counts - see the "
        "cache_membership ablation in the findings writeup. Prefer erring high.",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=10, help="Untimed warmup calls beforehand."
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
        help="If given, save each results table as a CSV next to this path "
        "(e.g. results.csv -> results_batch_size.csv, results_stage_profile.csv, ...).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    sweep_values = QUICK_SWEEP_VALUES if args.quick else DEFAULT_SWEEP_VALUES
    n_repeats = 3 if args.quick else args.n_repeats
    n_warmup = 1 if args.quick else args.n_warmup

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

    print("\nProfiling FLC pipeline stages at the baseline config...")
    stage_df = run_stage_profile(
        BASELINE, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    _print_and_maybe_save("stage_profile", stage_df, args.csv_path)

    print("\nRunning membership-cache ablation at the baseline config...")
    cache_df = run_cache_ablation(
        BASELINE, device=device, n_repeats=n_repeats, n_warmup=n_warmup
    )
    _print_and_maybe_save("cache_ablation", cache_df, args.csv_path)


if __name__ == "__main__":
    main()
