"""
Shared builders and timing utilities for the FuzzyLogicController (FLC) vs. plain
deep neural network (DNN) benchmarks.

Design notes / hypotheses this benchmark suite exists to test (see bench_flc_vs_dnn.py's
module docstring for the full writeup):

  H1. Python-level dispatch overhead: the FLC forward pass is composed of many small
      Python-level method calls and object indirections (FuzzySetGroup -> FuzzySet ->
      cache lookup -> parameter_signature -> get_centers/get_widths/get_mask, engine ->
      NAryRelation -> GroupedLinks -> BinaryLinks, ...), versus a DNN's handful of large,
      fused matmul kernel launches. This overhead is roughly constant per call, not
      per-FLOP, so it should show up most clearly at small batch sizes.
  H2. Redundant parameter_signature() computation: computed once in the cache lookup and
      again in the cache store on every cache miss, both at the per-fuzzy-set and the
      FuzzySetGroup level.
  H3. Membership caching overhead without benefit: the cache is keyed on tensor identity,
      so a fresh mini-batch tensor every training step always misses, paying bookkeeping
      cost for zero hit rate. Compare cache_membership=True vs False to isolate this.
  H4. GPU synchronization points: `bool(tensor.any())` calls in the NaN-safety path
      (FuzzySet._calculate_membership_nan_safe) and the engine's gather-based mask
      application force the GPU pipeline to drain on every forward call (since the cache
      almost always misses during training), preventing the asynchronous kernel queuing
      that keeps a DNN's forward pass cheap on GPU.
  H5. Per-module Python loop in FuzzySetGroup.forward() when modules_list has more than
      one module.
  H6. The engine's mask application falls back to materializing a full
      (batch, vars, terms, rules) tensor when the optimized gather path is not
      applicable, scaling with vars * terms * rules.

Run `python -m benchmarks.bench_flc_vs_dnn --help` for the CLI that exercises these.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import TSK
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.sets.impl import Gaussian
from fuzzy.sets.membership import Membership


@dataclass
class FLCShapeConfig:
    """
    The five knobs this benchmark suite sweeps over, plus everything else needed to
    reproducibly build an FLC and a comparable DNN of that shape.
    """

    batch_size: int = 256
    n_inputs: int = 8
    n_terms: int = 5  # number of fuzzy sets (linguistic terms) per input dimension
    n_rules: int = 32
    n_outputs: int = 4
    cache_membership: bool = True
    seed: int = 0


def build_flc(config: FLCShapeConfig, device: torch.device) -> FLC:
    """
    Build a TSK FuzzyLogicController of the given shape.

    Every rule's premise uses all n_inputs variables (one randomly chosen term per
    variable), matching the library's dominant grid-partition TSK use case and giving
    exact control over n_rules independent of the combinatorial n_terms ** n_inputs
    grid size. Rules are assigned to output variables round-robin so every output is
    driven by at least one rule when n_rules >= n_outputs.

    Args:
        config: The shape to build.
        device: The device to build the FLC on.

    Returns:
        A TSK FuzzyLogicController of the requested shape.
    """
    rng = random.Random(config.seed)

    antecedents = [
        Gaussian(
            centers=np.linspace(0.0, 1.0, config.n_terms, dtype=np.float32),
            widths=np.full(config.n_terms, 1.0 / config.n_terms, dtype=np.float32),
            device=device,
        )
        for _ in range(config.n_inputs)
    ]

    rules: List[Rule] = []
    for rule_idx in range(config.n_rules):
        premise_indices: List[Tuple[int, int]] = [
            (var_idx, rng.randrange(config.n_terms))
            for var_idx in range(config.n_inputs)
        ]
        output_idx = rule_idx % config.n_outputs
        rules.append(
            Rule(
                premise=Product(*premise_indices, device=device),
                consequence=NAryRelation((output_idx, 0), device=device),
            )
        )

    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
        rules=rules,
    )

    flc = FLC(source=knowledge_base, inference=TSK, device=device)
    flc.input_granulation.cache_membership = config.cache_membership
    flc.input_granulation._membership_cache.enabled = (  # pylint: disable=protected-access
        config.cache_membership
    )
    for module in flc.input_granulation.modules_list:
        cache = getattr(module, "_membership_cache", None)
        if cache is not None:
            cache.enabled = config.cache_membership
    return flc


def build_dnn(
    n_inputs: int,
    n_outputs: int,
    device: torch.device,
    hidden_size: int = 512,
    n_hidden_layers: int = 3,
) -> torch.nn.Module:
    """
    Build a plain feedforward DNN with the given number of hidden layers, used as the
    "how fast could this be" baseline for the FLC benchmark.

    Args:
        n_inputs: Number of input features.
        n_outputs: Number of output features.
        device: The device to build the DNN on.
        hidden_size: The width of each hidden layer.
        n_hidden_layers: The number of hidden layers.

    Returns:
        A torch.nn.Sequential DNN on the given device.
    """
    layers: List[torch.nn.Module] = []
    in_features = n_inputs
    for _ in range(n_hidden_layers):
        layers.append(torch.nn.Linear(in_features, hidden_size))
        layers.append(torch.nn.ReLU())
        in_features = hidden_size
    layers.append(torch.nn.Linear(in_features, n_outputs))
    return torch.nn.Sequential(*layers).to(device)


def make_batch(
    batch_size: int, n_inputs: int, device: torch.device, seed: Optional[int] = None
) -> torch.Tensor:
    """
    Returns:
        A fresh random observation batch of shape (batch_size, n_inputs), in [0, 1) to
        match the antecedents built by build_flc.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.rand(batch_size, n_inputs, generator=generator)
    return data.to(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class TimingResult:
    """
    Wall-clock timings (in seconds) for repeated calls to some function, after warmup.
    """

    samples: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Returns: the arithmetic mean of the samples, in seconds."""
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        """Returns: the sample standard deviation of the samples, in seconds."""
        return float(np.std(self.samples))

    @property
    def min(self) -> float:
        """Returns: the minimum (least noisy) sample, in seconds."""
        return float(np.min(self.samples))


def time_calls(
    fn: Callable[[], None],
    n_repeats: int,
    n_warmup: int,
    device: torch.device,
) -> TimingResult:
    """
    Time repeated calls to fn(), each call individually bracketed by device
    synchronization so GPU work is actually measured (not just Python-side dispatch).

    Args:
        fn: A zero-argument callable to time; should perform one full unit of work
            (e.g., one forward pass, or one forward+backward step) per call, including
            generating any fresh random input it needs internally.
        n_repeats: Number of timed repetitions.
        n_warmup: Number of untimed warmup calls beforehand (CUDA kernel
            compilation/caching, allocator warmup, etc.).
        device: The device the work runs on.

    Returns:
        A TimingResult with one sample per repetition.
    """
    for _ in range(n_warmup):
        fn()
    _synchronize(device)

    samples: List[float] = []
    for _ in range(n_repeats):
        _synchronize(device)
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    return TimingResult(samples=samples)


def time_calls_with_setup(
    setup_fn: Callable[[], object],
    timed_fn: Callable[[object], None],
    n_repeats: int,
    n_warmup: int,
    device: torch.device,
) -> TimingResult:
    """
    Like time_calls, but each repetition first calls setup_fn() (untimed) to produce an
    argument, then times only timed_fn(argument).

    This isolates a single pipeline stage's own cost precisely: naively timing
    cumulative prefixes of a pipeline (e.g. "fuzzify", "fuzzify+engine",
    "fuzzify+engine+defuzzify") and subtracting consecutive means to back out each
    stage's share amplifies noise (variances add on subtraction) and can go negative
    for cheap stages - timing each stage's own call directly avoids that entirely,
    while setup_fn still runs fresh (untimed) upstream work each repetition so a
    downstream stage is never accidentally measured against a stale, cached upstream
    result.

    Args:
        setup_fn: Zero-argument callable producing the input the timed stage needs;
            not included in the timing.
        timed_fn: One-argument callable performing the work to be timed.
        n_repeats: Number of timed repetitions.
        n_warmup: Number of untimed warmup repetitions beforehand.
        device: The device the work runs on.

    Returns:
        A TimingResult with one sample per repetition.
    """
    for _ in range(n_warmup):
        timed_fn(setup_fn())
    _synchronize(device)

    samples: List[float] = []
    for _ in range(n_repeats):
        argument = setup_fn()
        _synchronize(device)
        start = time.perf_counter()
        timed_fn(argument)
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    return TimingResult(samples=samples)


def benchmark_forward_only(
    model: torch.nn.Module,
    make_input: Callable[[], torch.Tensor],
    device: torch.device,
    n_repeats: int = 30,
    n_warmup: int = 5,
) -> TimingResult:
    """
    Time inference-only forward passes (model.eval(), torch.no_grad()), each on a
    freshly generated input (as a real inference-serving workload would see).
    """
    model.eval()

    def _step() -> None:
        with torch.no_grad():
            model(make_input())

    return time_calls(_step, n_repeats=n_repeats, n_warmup=n_warmup, device=device)


def benchmark_forward_backward(
    model: torch.nn.Module,
    make_input: Callable[[], torch.Tensor],
    n_outputs: int,
    device: torch.device,
    n_repeats: int = 30,
    n_warmup: int = 5,
) -> TimingResult:
    """
    Time forward+backward training steps (model.train()), each on a freshly generated
    input/target pair. No optimizer step is taken, to isolate compute cost from weight
    update cost (comparable between models either way).
    """
    model.train()

    def _step() -> None:
        for param in model.parameters():
            param.grad = None
        observations = make_input()
        target = torch.rand(
            observations.shape[0], n_outputs, device=device
        )
        output = model(observations)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

    return time_calls(_step, n_repeats=n_repeats, n_warmup=n_warmup, device=device)


@dataclass
class StageTimings:
    """
    Per-stage breakdown of one FLC forward pass, in seconds (mean over n_repeats).
    """

    fuzzification: float
    engine: float
    defuzzification: float
    end_to_end: float


def profile_flc_stages(
    flc: FLC,
    make_input: Callable[[], torch.Tensor],
    device: torch.device,
    n_repeats: int = 30,
    n_warmup: int = 5,
) -> StageTimings:
    """
    Break an FLC's forward pass down into its three pipeline stages
    (fuzzification -> rule engine -> defuzzification), timing each stage's own cost
    directly (via time_calls_with_setup) rather than by subtracting cumulative
    prefixes - see that function's docstring for why. Each repetition still uses a
    fresh observations tensor, so no stage benefits from a stale cached upstream
    result.
    """
    flc.eval()

    def _make_observations() -> torch.Tensor:
        return make_input()

    def _make_granulated() -> Membership:
        with torch.no_grad():
            return flc.input_granulation(make_input())

    def _make_observations_and_rule_strengths() -> Tuple[torch.Tensor, Membership]:
        with torch.no_grad():
            observations = make_input()
            granulated = flc.input_granulation(observations)
            rule_strengths = flc.engine(granulated)
            return observations, rule_strengths

    def _fuzzify(observations: torch.Tensor) -> None:
        with torch.no_grad():
            flc.input_granulation(observations)

    def _engine(granulated: Membership) -> None:
        with torch.no_grad():
            flc.engine(granulated)

    def _defuzzify(observations_and_rule_strengths: Tuple[torch.Tensor, Membership]) -> None:
        observations, rule_strengths = observations_and_rule_strengths
        with torch.no_grad():
            flc._defuzzify(  # pylint: disable=protected-access
                observations, rule_strengths
            )

    def _end_to_end(observations: torch.Tensor) -> None:
        with torch.no_grad():
            flc(observations)

    fuzzification = time_calls_with_setup(
        _make_observations, _fuzzify, n_repeats=n_repeats, n_warmup=n_warmup, device=device
    ).mean
    engine_only = time_calls_with_setup(
        _make_granulated, _engine, n_repeats=n_repeats, n_warmup=n_warmup, device=device
    ).mean
    defuzzification_only = time_calls_with_setup(
        _make_observations_and_rule_strengths,
        _defuzzify,
        n_repeats=n_repeats,
        n_warmup=n_warmup,
        device=device,
    ).mean
    end_to_end = time_calls_with_setup(
        _make_observations, _end_to_end, n_repeats=n_repeats, n_warmup=n_warmup, device=device
    ).mean

    return StageTimings(
        fuzzification=fuzzification,
        engine=engine_only,
        defuzzification=defuzzification_only,
        end_to_end=end_to_end,
    )
