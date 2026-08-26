"""
A fused, Triton-accelerated implementation of `entmax.entmax_bisect` (Peters et al.,
2019; see https://arxiv.org/pdf/1905.05702), specialized for the way this codebase
actually calls it: reduction along the last dimension, with a single scalar alpha
shared across the whole tensor (see BoundAlphaEntmax in
fuzzy/utils/options/impl/impl_options.py).

entmax_bisect's forward pass is a fixed-iteration-count bisection (binary) search: each
of its n_iter iterations narrows a per-row interval [tau_lo, tau_hi] by evaluating a
sum-reduction over the row and comparing it to zero. The eager implementation runs
this as n_iter sequential Python-level iterations, each launching several small CUDA
kernels (a subtract, a power, a sum-reduction, a comparison, a `where`) that
materialize a fresh (batch, ..., D) intermediate every iteration. Profiling this
codebase's CO-FIS training step (see git history / PR description) found this
accounted for roughly 86% of GPU compute time: not because any single iteration is
expensive, but because 30 iterations x ~6 tiny kernel launches x 12 calls per training
step adds up to thousands of kernel launches, each dominated by fixed per-kernel
overhead rather than real work on a (batch<=64, D<=256)-sized tensor.

This module fuses the entire bisection loop - for one row - into a single Triton
kernel: the row is loaded into registers once and reused across every iteration,
eliminating both the repeated global-memory round-trips and the many small kernel
launches. It covers only the forward pass; entmax_bisect's own backward
(EntmaxBisectFunction.backward in the entmax package) is already a single, small,
loop-free set of reductions - the part worth fusing is exclusively the n_iter loop in
forward, so the backward here is a direct, unmodified reimplementation of that same
formula (see _EntmaxBisectTritonFunction.backward), not a second Triton kernel.

Falls back to the reference `entmax.entmax_bisect` whenever a precondition doesn't
hold - non-CUDA tensors, TRITON_AVAILABLE is False, `dim` is not the last dimension,
`alpha` is not a scalar (this codebase's BoundAlphaEntmax always uses a single learned
scalar, but entmax_bisect's own public API allows a per-row alpha tensor), or D
(the reduction dimension's size) exceeds MAX_BLOCK_D - so this module never changes
behavior for a caller outside its documented fast-path preconditions.
"""

from typing import Optional, Tuple, Union

import torch

from entmax import entmax_bisect as _reference_entmax_bisect

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on installs without triton
    TRITON_AVAILABLE = False


# Above this, a single row's D elements no longer comfortably fit in one Triton
# program's registers/shared memory as one block - fall back to the reference
# implementation rather than risk a slow or failing large-BLOCK_D kernel. Every
# n_rules/n_terms value this codebase's hyperparameter search can produce (see
# RuleStructureConfig.n_rules, PremiseStructureConfig - up to a few hundred) is far
# below this.
MAX_BLOCK_D = 16384


if TRITON_AVAILABLE:

    # pragma: no cover justification (applies to the @triton.jit kernel below): Triton
    # traces this into GPU code at compile time, never executing as CPython bytecode,
    # so coverage.py cannot observe it regardless of how thoroughly
    # test_triton_entmax.py exercises it (forward value parity across many shapes,
    # dtypes, alphas, and edge cases) - it just cannot move this function's line
    # coverage number. Mirrors the same, already-established justification in
    # triton_kernels.py for _gather_prod_fwd_kernel.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
    @triton.jit
    def _entmax_bisect_fwd_kernel(
        x_ptr,
        out_ptr,
        alpha_ptr,
        d,
        stride_row,
        stride_col,
        out_stride_row,
        out_stride_col,
        n_iter: tl.constexpr,
        ensure_sum_one: tl.constexpr,
        block_d: tl.constexpr,
    ):  # pragma: no cover
        row = tl.program_id(0)
        offs = tl.arange(0, block_d)
        mask = offs < d

        x_ptrs = x_ptr + row * stride_row + offs * stride_col
        # padding lanes (offs >= d) are filled with -inf: harmless below, since
        # clamp(-inf - tau, min=0) == 0 and 0 ** positive == 0, so they never
        # contribute to any sum, and -inf never wins the initial max reduction
        # (assuming real inputs are finite, exactly as the reference implementation
        # itself assumes).
        x = tl.load(x_ptrs, mask=mask, other=float("-inf")).to(tl.float32)
        alpha = tl.load(alpha_ptr).to(tl.float32)

        alpha_m1 = alpha - 1.0
        inv_alpha_m1 = 1.0 / alpha_m1

        x = x * alpha_m1
        max_val = tl.max(x, axis=0)

        # _gp(1, alpha) == 1 ** (alpha - 1) == 1.0 unconditionally (any power of 1 is
        # 1), so tau_lo simplifies to max_val - 1.0 without needing a pow at all.
        tau_lo = max_val - 1.0
        inv_d = 1.0 / d
        # _gp(1 / d, alpha) == (1 / d) ** (alpha - 1)
        tau_hi = max_val - tl.exp(alpha_m1 * tl.log(inv_d))
        dm = tau_hi - tau_lo

        p_m = tl.zeros([block_d], dtype=tl.float32)
        for _ in range(n_iter):
            dm = dm / 2.0
            tau_m = tau_lo + dm
            clamped = tl.maximum(x - tau_m, 0.0)
            # _gp_inv(y, alpha) == y ** (1 / (alpha - 1)); guard the clamped-to-zero
            # case explicitly (clamped>0 branch is exact; the other branch matches
            # what clamped ** inv_alpha_m1 would give at clamped==0 for the
            # inv_alpha_m1>0 range this codebase's bounding strategies produce, but is
            # written explicitly rather than relying on log(0)==-inf/exp(-inf)==0
            # IEEE-754 semantics holding under Triton's compiler).
            p_m = tl.where(
                clamped > 0, tl.exp(tl.log(clamped) * inv_alpha_m1), 0.0
            )
            p_m = tl.where(mask, p_m, 0.0)
            f_m = tl.sum(p_m, axis=0) - 1.0
            take = f_m >= 0
            tau_lo = tl.where(take, tau_m, tau_lo)

        if ensure_sum_one:
            denom = tl.sum(p_m, axis=0)
            p_m = p_m / denom

        out_ptrs = out_ptr + row * out_stride_row + offs * out_stride_col
        tl.store(out_ptrs, p_m, mask=mask)

    def _entmax_bisect_forward_triton(
        x2d: torch.Tensor,
        alpha_scalar: torch.Tensor,
        n_iter: int,
        ensure_sum_one: bool,
        block_d: int,
    ) -> torch.Tensor:
        rows, d = x2d.shape
        out2d = torch.empty(rows, d, device=x2d.device, dtype=torch.float32)
        _entmax_bisect_fwd_kernel[(rows,)](
            x2d,
            out2d,
            alpha_scalar,
            d,
            x2d.stride(0),
            x2d.stride(1),
            out2d.stride(0),
            out2d.stride(1),
            n_iter=n_iter,
            ensure_sum_one=ensure_sum_one,
            block_d=block_d,
        )
        return out2d.to(x2d.dtype)

    # pylint: disable-next=abstract-method,arguments-differ
    class _EntmaxBisectTritonFunction(torch.autograd.Function):
        """
        Forward: the fused Triton bisection kernel above (see module docstring for why
        only forward is fused). Backward: an unmodified reimplementation of
        entmax.root_finding.EntmaxBisectFunction.backward - the same formula, applied
        to this Function's own saved output, so gradients w.r.t. both X and alpha are
        bit-for-bit derivations of the same published closed form the reference uses,
        not an approximation of it.
        """

        @staticmethod
        def forward(  # pylint: disable=arguments-differ
            ctx,
            x: torch.Tensor,
            alpha: torch.Tensor,
            n_iter: int,
            ensure_sum_one: bool,
            block_d: int,
        ) -> torch.Tensor:
            orig_shape = x.shape
            d = orig_shape[-1]
            x2d = x.reshape(-1, d)
            alpha_scalar = alpha.reshape(1).to(device=x.device, dtype=torch.float32)

            out2d = _entmax_bisect_forward_triton(
                x2d, alpha_scalar, n_iter, ensure_sum_one, block_d
            )
            out = out2d.reshape(orig_shape)

            ctx.save_for_backward(out, alpha)
            ctx.alpha_needs_grad = alpha.requires_grad
            return out

        @staticmethod
        def backward(  # pylint: disable=arguments-differ
            ctx, d_y: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None, None, None]:
            y, alpha = ctx.saved_tensors
            dim = -1

            # identical formula to EntmaxBisectFunction.backward (root_finding.py):
            # gppr = Y ** (2 - alpha) where Y > 0, else 0
            gppr = torch.where(
                y > 0, y ** (2 - alpha), y.new_zeros(())
            )
            d_x = d_y * gppr
            q = d_x.sum(dim) / gppr.sum(dim)
            q = q.unsqueeze(dim)
            d_x = d_x - q * gppr

            d_alpha = None
            if ctx.alpha_needs_grad:
                shannon = torch.where(
                    y > 0, y * torch.log(y), y.new_zeros(())
                )
                entropy = shannon.sum(dim).unsqueeze(dim)
                y_skewed = gppr / gppr.sum(dim).unsqueeze(dim)

                d_alpha = d_y * (y - y_skewed) / ((alpha - 1) ** 2)
                d_alpha = d_alpha - d_y * (shannon - y_skewed * entropy) / (alpha - 1)
                d_alpha = d_alpha.sum(dim).unsqueeze(dim)

            return d_x, d_alpha, None, None, None


def _is_fast_path_eligible(
    x: torch.Tensor, alpha: torch.Tensor, dim: int
) -> bool:
    if not TRITON_AVAILABLE or not x.is_cuda:
        return False
    if dim not in (-1, x.ndim - 1):
        return False
    if alpha.numel() != 1:
        return False
    if x.shape[-1] > MAX_BLOCK_D:
        return False
    return True


def entmax_bisect_triton(
    x: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 1.5,
    dim: int = -1,
    n_iter: int = 50,
    ensure_sum_one: bool = True,
) -> torch.Tensor:
    """
    Fused-Triton-kernel equivalent of `entmax.entmax_bisect`, matching its forward
    value and both its X- and alpha- gradients exactly (see module docstring), for the
    common case in this codebase: a CUDA tensor, reduction along the last dimension,
    and a single scalar alpha shared across the whole tensor. Falls back to the
    unmodified reference `entmax.entmax_bisect` for everything else (CPU tensors, a
    non-scalar per-row alpha, a `dim` other than the last one, or an unusually large
    reduction dimension) - see _is_fast_path_eligible.

    Args:
        x: The input tensor.
        alpha: Scalar Tsallis alpha parameter (> 1); may be a Python float or a 0-
            or 1-element torch.Tensor (matching entmax_bisect's own accepted forms
            for this codebase's scalar-alpha usage).
        dim: The dimension to reduce along.
        n_iter: Number of bisection iterations.
        ensure_sum_one: Whether to renormalize the result so it sums to exactly 1
            along `dim` (see entmax_bisect's own docstring for why this can matter).

    Returns:
        The same tensor entmax.entmax_bisect(x, alpha, dim, n_iter, ensure_sum_one)
        would return.
    """
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)

    if not _is_fast_path_eligible(x, alpha, dim):
        return _reference_entmax_bisect(
            x, alpha=alpha, dim=dim, n_iter=n_iter, ensure_sum_one=ensure_sum_one
        )

    block_d = triton.next_power_of_2(x.shape[-1])
    return _EntmaxBisectTritonFunction.apply(x, alpha, n_iter, ensure_sum_one, block_d)
