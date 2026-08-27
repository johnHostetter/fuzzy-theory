"""
CUDA graph capture-and-replay for training a FuzzyLogicController (FLC), as an
alternative to torch.compile(fullgraph=True) for eliminating the per-step Python/
kernel-launch dispatch overhead a training loop otherwise pays on every call. Unlike
torch.compile, CUDA graph capture does not need to trace or understand Python
*semantics* - it only records the literal sequence of CUDA kernel launches actually
executed once, then replays them verbatim - which can survive custom, third-party
TNorm/FuzzySet subclasses that break Dynamo's fullgraph tracing.

This trades away flexibility that torch.compile keeps: a captured graph requires the
exact same op sequence and tensor shapes on every replay, and an FLC can grow new
fuzzy-set/rule/variable parameters mid-training (see
tests/test_logic/control/demo_flcs.py's train_model for the established growth idiom).
There is no in-place "grow the captured graph" operation - growth always means
discarding the stale GraphedTrainingStep and building a fresh one.

GraphedTrainingStep captures whatever a caller-supplied, zero-argument step_fn does -
not just a fixed model(x)/criterion(pred, y) shape - so it accommodates an arbitrary
training step: multiple forward passes (e.g. a target network), gradient clipping
between backward() and optimizer.step(), custom loss shapes, and so on. step_fn reads
its input from "static" buffers the caller allocates once and owns; the caller copies
fresh data into them before every replay() call:

    from fuzzy.logic.control.cuda_graph import (
        GraphedTrainingStep,
        force_cuda_graph_safe_branches,
    )

    static_obs = torch.zeros(batch_size, flc.shape.n_inputs, device=device)
    static_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
    static_targets = torch.zeros(batch_size, device=device)

    # step_fn closes over `flc` and `optimizer` by reference, read fresh from this
    # mutable holder on every call - so rebinding holder["optimizer"] on growth
    # (below) is enough to make the NEXT captured graph use the new one, without
    # having to redefine step_fn itself.
    holder = {"flc": flc, "optimizer": optimizer}

    def step_fn():
        # zero_grad(set_to_none=False) is required - set_to_none=True would
        # allocate a fresh grad tensor on the next backward(), an address a
        # replayed graph cannot see.
        holder["optimizer"].zero_grad(set_to_none=False)
        action_values = (
            holder["flc"](static_obs)
            .gather(1, static_actions.unsqueeze(1))
            .squeeze(1)
        )
        td_error = criterion(static_targets, action_values)
        td_error.backward()
        torch.nn.utils.clip_grad_norm_(holder["flc"].parameters(), max_norm=1.0)
        holder["optimizer"].step()
        return td_error

    def build_graphed():
        with force_cuda_graph_safe_branches(holder["flc"]):
            return GraphedTrainingStep(step_fn, holder["optimizer"], device=device)

    def flc_signature(flc):
        # changes on ANY growth: a new input variable (changes n_inputs), a new
        # term/rule (changes a parameter's shape/count), or a wholesale new FLC
        # object - all three change this.
        return flc.shape.n_inputs, flc.shape.n_outputs, sum(
            p.numel() for p in flc.parameters()
        )

    graphed = build_graphed()
    last_signature = flc_signature(holder["flc"])

    for obs, actions, targets in training_data:
        current_signature = flc_signature(holder["flc"])
        if current_signature != last_signature:
            graphed.close()
            # growth means new parameters, so the optimizer must be rebuilt too -
            # step_fn will pick up the new one via `holder` on its next call. If
            # growth replaced `flc` with an entirely new object (rather than
            # mutating the existing one in place - see this module's top-level
            # docstring section on the two ways growth can happen in this
            # codebase), also set holder["flc"] = new_flc here before rebuilding.
            holder["optimizer"] = torch.optim.Adam(
                holder["flc"].parameters(), lr=3e-2, capturable=True
            )
            graphed = build_graphed()
            last_signature = current_signature
        static_obs.copy_(obs, non_blocking=True)
        static_actions.copy_(actions, non_blocking=True)
        static_targets.copy_(targets, non_blocking=True)
        loss = graphed.replay()

Requires Triton (see force_cuda_graph_safe_branches's docstring for why: PyTorch's
own built-in torch.prod CUDA backward kernel - the fallback used when Triton is
unavailable - was confirmed, via torch.cuda.set_sync_debug_mode, to sync
unconditionally regardless of any patch this codebase could make, which torch.cuda.
graph capture cannot tolerate).
"""

import contextlib
from typing import Callable, Iterator, List

import torch

import fuzzy.relations.n_ary as n_ary_module
import fuzzy.relations.t_norm as t_norm_module
from fuzzy.logic.control.controller import FuzzyLogicController
from fuzzy.relations.t_norm import Product
from fuzzy.relations.triton_kernels import TRITON_AVAILABLE, gather_prod
from fuzzy.sets.abstract import FuzzySet
from fuzzy.sets.membership import Membership


# duplicate-code: this deliberately mirrors Product.forward's own structure (see
# t_norm.py) - the whole point is to replay the same operations minus one check, so
# some near-identical lines are expected and not a sign of accidental copy-paste.
# pylint: disable-next=duplicate-code
def _forced_gather_product_forward(self: Product, membership: Membership) -> Membership:
    """
    Replacement for Product.forward used only while force_cuda_graph_safe_branches is
    active (see its docstring for the full story). Takes the fused Triton
    gather-then-product kernel path whenever the structural preconditions hold,
    WITHOUT the real forward()'s "not bool(membership.degrees.isnan().any())" check -
    that check is a hard CUDA-graph-capture blocker (any bool()/.item() on a CUDA
    tensor forces a host sync, which torch.cuda.graph capture cannot tolerate), and is
    safe to skip only because callers of this replacement guarantee NaN-free synthetic
    or otherwise-validated data. Falls back to the ordinary general path
    (_apply_mask_with_mask + .prod()) when the structural preconditions do not hold -
    that fallback's backward has its own, unrelated sync (torch.prod's built-in CUDA
    backward kernel syncs unconditionally, confirmed empirically; not something this
    module can work around), which is why this replacement is only installed when
    TRITON_AVAILABLE is True.

    Args:
        self: The Product instance forward() was called on.
        membership: The memberships to apply the algebraic product relation to.

    Returns:
        The algebraic product membership value, identical to Product.forward's own
        result for NaN-free input.
    """
    if (
        self._use_gather  # pylint: disable=protected-access
        and self._all_active  # pylint: disable=protected-access
        and membership.degrees.is_cuda
    ):
        return Membership(
            degrees=gather_prod(
                membership.degrees,
                self._gather_indices,  # pylint: disable=protected-access
            ),
            mask=self._cached_mask,  # pylint: disable=protected-access
            formula=type(self).__name__,
        )
    after_mask, applied_mask = (
        self._apply_mask_with_mask(  # pylint: disable=protected-access
            membership=membership
        )
    )
    return Membership(
        degrees=after_mask.prod(dim=-2, keepdim=False),
        mask=applied_mask,
        formula=type(self).__name__,
    )


@contextlib.contextmanager
def force_cuda_graph_safe_branches(flc: FuzzyLogicController) -> Iterator[None]:
    """
    Temporarily force every data-dependent CPU-sync branch this FLC's forward pass can
    take onto its sync-free alternative, for the duration of CUDA graph warmup+capture
    only (see GraphedTrainingStep). torch.cuda.graph capture raises RuntimeError on any
    CPU-GPU sync issued while a stream is capturing, and this codebase has three:

      1. FuzzySet._calculate_membership_nan_safe's bool(nan_mask.any()), gated by
         _nan_safe_sync_threshold_numel (per-instance) - forced off by setting the
         threshold effectively infinite, so the unconditional (sync-free) branch is
         always taken.
      2. NAryRelation._gather_apply_mask's bool(any_nan_per_variable.any()), gated by
         the module-level GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL - forced off the same
         way.
      3. Product.forward's Triton fast-path check, "not bool(degrees.isnan().any())" -
         unlike the other two, this one is NOT size-gated, and disabling it (rather
         than forcing it) would be actively counterproductive: doing so falls through
         to the general apply_mask()+.prod(dim=-2) path, and torch.prod's built-in CUDA
         backward kernel was confirmed (empirically, via torch.cuda.set_sync_debug_mode)
         to sync unconditionally, regardless of this module's own code - a hard
         PyTorch-level blocker, not a fuzzy-theory design choice, and not fixable by
         patching anything in this codebase. The fused Triton kernel's custom
         autograd.Function backward (_GatherProdReduce, in triton_kernels.py) is pure
         Triton kernel launches with no host sync, so this context manager instead
         *forces* the Triton branch to fire unconditionally (via
         _forced_gather_product_forward, monkeypatched onto the Product class), skipping
         only the isnan() check - safe here because callers only ever feed a captured
         graph guaranteed-NaN-free data. Requires TRITON_AVAILABLE; raises RuntimeError
         immediately (before attempting capture, which would otherwise fail confusingly
         deep inside a captured backward()) if Triton is not installed, since there is
         no other sync-free way to backward through a Product engine's rule combination
         on CUDA.

    Also, temporarily sets self.training = False on the FLC (via flc.eval()) for the
    same reason: Defuzzification.forward's "if self.training: assert not
    defuzzification.isnan().any()" is unconditional (not gated by any threshold or
    is_compiling() check) whenever self.training is True. eval() only affects this one
    assert in this codebase (no dropout/batchnorm-style train/eval-dependent behavior
    elsewhere in the forward path), so gradients are computed identically either way.

    All patches are restored in a finally block, including on exception, so a failed
    capture attempt never leaves global module state mutated for the rest of the
    process.

    Args:
        flc: The FLC about to be warmed up and captured.

    Yields:
        None.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "force_cuda_graph_safe_branches requires Triton: without it, Product's "
            "rule-combination backward pass (torch.prod's built-in CUDA kernel) syncs "
            "unconditionally, which torch.cuda.graph capture cannot tolerate, and "
            "there is no sync-free fallback available in this codebase."
        )

    was_training = flc.training
    flc.eval()

    fuzzy_sets: List[FuzzySet] = [m for m in flc.modules() if isinstance(m, FuzzySet)]
    original_nan_safe_thresholds = [
        module._nan_safe_sync_threshold_numel  # pylint: disable=protected-access
        for module in fuzzy_sets
    ]
    for module in fuzzy_sets:
        module._nan_safe_sync_threshold_numel = (  # pylint: disable=protected-access
            2**62
        )

    original_gather_mask_threshold = n_ary_module.GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL
    n_ary_module.GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL = 2**62

    original_product_forward = t_norm_module.Product.forward
    t_norm_module.Product.forward = _forced_gather_product_forward

    try:
        yield
    finally:
        t_norm_module.Product.forward = original_product_forward
        n_ary_module.GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL = (
            original_gather_mask_threshold
        )
        for module, original_threshold in zip(fuzzy_sets, original_nan_safe_thresholds):
            module._nan_safe_sync_threshold_numel = (  # pylint: disable=protected-access
                original_threshold
            )
        if was_training:
            flc.train()


class GraphedTrainingStep:
    """
    Captures one training step, as performed by a caller-supplied step_fn, as a CUDA
    graph, then replays it on every replay() call - eliminating the per-call Python/
    kernel-launch dispatch overhead a plain eager training step pays every time, at
    the cost of requiring the exact same op sequence and tensor shapes on every
    replay. See force_cuda_graph_safe_branches for why capturing an FLC's full train
    step also requires Triton and forcing a couple of data-dependent branches closed.

    step_fn is a zero-argument callable performing one complete training step -
    zero_grad, any forward pass(es), loss computation, backward(), any gradient
    transformation (e.g. torch.nn.utils.clip_grad_norm_), and optimizer.step() -
    returning the loss tensor. It is responsible for its own zero_grad(set_to_none=
    False) call (set_to_none=True would allocate a fresh grad tensor on the next
    backward(), an address a replayed graph cannot see). Unlike a fixed model(x)/
    criterion(pred, y) shape, step_fn can do anything a real training step needs:
    multiple forward passes (e.g. a target network), gradient clipping between
    backward() and optimizer.step(), or a loss computed from more than one model's
    output. It must read its input from "static" buffers the caller allocates once,
    outside step_fn, and owns - the caller copies fresh data into those buffers
    before every replay() call (see this module's docstring for a full example).

    Not reusable across a model architecture change (e.g. FLC rule/term/variable
    growth mid-training): construct a fresh GraphedTrainingStep (with a fresh
    optimizer, since growth means new parameters, and typically a fresh step_fn
    closing over that new optimizer) instead - there is no in-place "grow the
    captured graph" operation. See this module's docstring for a growth-detection
    pattern.

    Model-agnostic - nothing about the capture/replay mechanics below is FLC-specific
    (only force_cuda_graph_safe_branches is) - so this also works for a plain
    torch.nn.Module with no fuzzy-theory-specific sync branches to worry about.
    """

    def __init__(
        self,
        step_fn: Callable[[], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        n_warmup: int = 11,
    ) -> None:
        """
        Args:
            step_fn: A zero-argument callable performing one complete training step
                and returning the loss tensor - see the class docstring for the full
                contract.
            optimizer: The optimizer step_fn calls .step() on (passed separately only
                so this constructor can validate it up front, before paying for
                warmup). Either a capturable=True Adam-family optimizer (e.g.
                torch.optim.Adam(params, lr=..., capturable=True) - a non-capturable
                Adam/AdamW keeps its step counter as a plain Python int with float
                bias-correction math, both illegal to record into a CUDA graph) OR a
                plain torch.optim.SGD, which has no such step-counter/bias-correction
                state at all and was confirmed (empirically, capturing and replaying
                directly against torch.cuda.graph) to capture and replay cleanly with
                no capturable-style flag needed. SGD's lr, unlike Adam's, cannot be a
                mutable Tensor under capture either (confirmed empirically: its
                _foreach_add_(..., alpha=-lr) call forces a host-side scalar
                conversion, raising "operation not permitted when stream is
                capturing") - a caller changing SGD's learning rate (e.g. a per-epoch
                schedule) must construct a fresh GraphedTrainingStep to pick up the
                new value, exactly like any other structural change (see this
                class's own docstring on growth).
            device: Must be a CUDA device.
            n_warmup: Untimed step_fn() calls run on a side stream before capture,
                letting cuDNN/cuBLAS algorithm selection and the CUDA caching
                allocator's memory pool both reach steady state - per torch.cuda.
                graph's documented capture protocol - before the actual capture
                records the final, stable kernel sequence. Also, incidentally,
                exceeds this library's membership cache's default maxsize (2) several
                times over, so any cache staleness from warmup has long since resolved
                through real optimizer steps before the captured call.
        """
        is_sgd = isinstance(optimizer, torch.optim.SGD)
        if not is_sgd and not optimizer.param_groups[0].get("capturable", False):
            raise ValueError(
                "GraphedTrainingStep requires either a plain torch.optim.SGD or an "
                "optimizer constructed with capturable=True (e.g. torch.optim.Adam"
                "(..., capturable=True)) - a non-capturable Adam/AdamW's step "
                "counter is a plain Python int with float bias-correction math, "
                "both illegal to record into a CUDA graph."
            )
        if device.type != "cuda":
            raise ValueError("GraphedTrainingStep requires a CUDA device.")

        self.device = device

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(n_warmup):
                step_fn()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_loss = step_fn()

    def replay(self, sync: bool = False) -> torch.Tensor:
        """
        Replay the captured step against whatever data is currently in step_fn's
        static buffers - the caller must copy fresh data into them before calling
        this.

        Args:
            sync: If True, return loss.item() (forces a device sync - use only for
                occasional sanity-check logging, not on the timed hot path). If False
                (default), return the live static_loss tensor with no forced sync.

        Returns:
            The loss (a live tensor referencing the graph's fixed output buffer, or a
            plain float if sync=True).
        """
        self.graph.replay()
        return self.static_loss.item() if sync else self.static_loss

    def close(self) -> None:
        """
        Explicitly drop the captured graph and its output buffer, so the graph's
        private memory pool is freed deterministically rather than waiting on GC.
        The caller's own static input buffers (allocated outside step_fn) are not
        this object's to manage, and are left untouched.

        Returns:
            None
        """
        self.graph.reset()
        del self.graph
        del self.static_loss

    def __enter__(self) -> "GraphedTrainingStep":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
