"""
A fused, Triton-accelerated implementation of "gather one term per variable per rule,
then take the product across variables" - the core computation of a gather-eligible
Product t-norm (see NAryRelation._gather_apply_mask and t_norm.Product.forward).

The eager implementation materializes an intermediate (batch, vars, rules) tensor via
torch.gather, then reduces it with torch.prod - two separate kernels, with the
intermediate written and then immediately re-read. This module fuses both into a
single kernel that never materializes that intermediate, and pairs it with a
numerically-safe (division-free) backward that handles exact-zero factors correctly -
verified to match torch.prod's own backward, including at exact zeros, to floating
point precision.

This module only covers the common case: no NaN present in `degrees`, and every
variable structurally active for every rule (relation._all_active). Callers are
responsible for checking those preconditions (see t_norm.Product.forward) and falling
back to the general-purpose eager path otherwise - this module does not attempt to
replicate the NaN-poisoning-via-any-unselected-term or partial-variable-coverage
semantics of the general path.

Requires Triton and a CUDA tensor; TRITON_AVAILABLE is False (and the public function
raises) when triton cannot be imported, so callers must guard on it before use.
"""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on installs without triton
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _mul_combine(a, b):
        return a * b

    @triton.jit
    def _add_combine(a, b):
        return a + b

    @triton.jit
    def _gather_prod_fwd_kernel(
        degrees_ptr,
        idx_ptr,
        out_ptr,
        V,
        stride_db,
        stride_dv,
        stride_dt,
        stride_iv,
        stride_ir,
        stride_ob,
        stride_or,
        BLOCK_V: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)

        acc = 1.0
        for v_start in range(0, V, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offsets < V

            idx_ptrs = idx_ptr + v_offsets * stride_iv + pid_r * stride_ir
            term_idx = tl.load(idx_ptrs, mask=v_mask, other=0)

            d_ptrs = (
                degrees_ptr
                + pid_b * stride_db
                + v_offsets * stride_dv
                + term_idx * stride_dt
            )
            vals = tl.load(d_ptrs, mask=v_mask, other=1.0)
            vals = tl.where(v_mask, vals, 1.0)
            block_prod = tl.reduce(vals, axis=0, combine_fn=_mul_combine)
            acc = acc * block_prod

        out_ptrs = out_ptr + pid_b * stride_ob + pid_r * stride_or
        tl.store(out_ptrs, acc)

    @triton.jit
    def _gather_prod_bwd_kernel(
        degrees_ptr,
        idx_ptr,
        grad_out_ptr,
        grad_degrees_ptr,
        V,
        stride_db,
        stride_dv,
        stride_dt,
        stride_iv,
        stride_ir,
        stride_gob,
        stride_gor,
        BLOCK_V: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)

        # pass 1: nz_count (how many of the V selected values are exactly zero) and
        # prod_nonzero (product of only the nonzero selected values) - together these
        # give a closed-form, division-free leave-one-out product below, avoiding a
        # sequential prefix/suffix scan
        nz_count = 0
        prod_nonzero = 1.0
        for v_start in range(0, V, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offsets < V
            idx_ptrs = idx_ptr + v_offsets * stride_iv + pid_r * stride_ir
            term_idx = tl.load(idx_ptrs, mask=v_mask, other=0)
            d_ptrs = (
                degrees_ptr
                + pid_b * stride_db
                + v_offsets * stride_dv
                + term_idx * stride_dt
            )
            vals = tl.load(d_ptrs, mask=v_mask, other=1.0)
            is_zero = (vals == 0.0) & v_mask
            nz_count += tl.reduce(is_zero.to(tl.int32),
                                  axis=0, combine_fn=_add_combine)
            safe_vals = tl.where(is_zero | (~v_mask), 1.0, vals)
            block_prod = tl.reduce(safe_vals, axis=0, combine_fn=_mul_combine)
            prod_nonzero = prod_nonzero * block_prod

        grad_y_ptr = grad_out_ptr + pid_b * stride_gob + pid_r * stride_gor
        grad_y = tl.load(grad_y_ptr)

        # pass 2: leave-one-out product per position, scatter-added into
        # grad_degrees (accumulation is required: multiple rules may select the
        # same (variable, term) pair)
        for v_start in range(0, V, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offsets < V
            idx_ptrs = idx_ptr + v_offsets * stride_iv + pid_r * stride_ir
            term_idx = tl.load(idx_ptrs, mask=v_mask, other=0)
            d_ptrs = (
                degrees_ptr
                + pid_b * stride_db
                + v_offsets * stride_dv
                + term_idx * stride_dt
            )
            vals = tl.load(d_ptrs, mask=v_mask, other=1.0)
            is_zero = vals == 0.0

            # x != 0: leave-one-out is prod_nonzero/x only if there is no OTHER
            # zero in the group (else some other zero kills it); x == 0: leave-one-out
            # is prod_nonzero only if x is the group's *only* zero (else another
            # zero kills it)
            loo_nonzero_case = tl.where(
                nz_count == 0, prod_nonzero / tl.where(is_zero, 1.0, vals), 0.0
            )
            loo_zero_case = tl.where(nz_count == 1, prod_nonzero, 0.0)
            leave_one_out = tl.where(is_zero, loo_zero_case, loo_nonzero_case)

            grad_contribution = grad_y * leave_one_out
            out_ptrs = (
                grad_degrees_ptr
                + pid_b * stride_db
                + v_offsets * stride_dv
                + term_idx * stride_dt
            )
            tl.atomic_add(out_ptrs, grad_contribution, mask=v_mask)

    def _gather_prod_forward(
            degrees: torch.Tensor,
            idx: torch.Tensor) -> torch.Tensor:
        batch_size, n_vars, _ = degrees.shape
        n_rules = idx.shape[1]
        out = torch.empty(
            (batch_size, n_rules), device=degrees.device, dtype=degrees.dtype
        )
        _gather_prod_fwd_kernel[(batch_size, n_rules)](
            degrees,
            idx,
            out,
            n_vars,
            degrees.stride(0),
            degrees.stride(1),
            degrees.stride(2),
            idx.stride(0),
            idx.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_V=128,
        )
        return out

    def _gather_prod_backward(
        degrees: torch.Tensor, idx: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_vars, _ = degrees.shape
        n_rules = idx.shape[1]
        grad_degrees = torch.zeros_like(degrees)
        _gather_prod_bwd_kernel[(batch_size, n_rules)](
            degrees,
            idx,
            grad_output,
            grad_degrees,
            n_vars,
            degrees.stride(0),
            degrees.stride(1),
            degrees.stride(2),
            idx.stride(0),
            idx.stride(1),
            grad_output.stride(0),
            grad_output.stride(1),
            BLOCK_V=128,
        )
        return grad_degrees

    class _GatherProdReduce(torch.autograd.Function):
        """See module docstring; forward/backward for a gather-then-product reduction."""

        @staticmethod
        def forward(
                ctx,
                degrees: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
            out = _gather_prod_forward(degrees, idx)
            ctx.save_for_backward(degrees, idx)
            return out

        @staticmethod
        def backward(
                ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
            degrees, idx = ctx.saved_tensors
            grad_degrees = _gather_prod_backward(
                degrees, idx, grad_output.contiguous())
            return grad_degrees, None


def gather_prod(degrees: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Fused, differentiable equivalent of:

        selected = torch.gather(degrees, dim=2, index=idx.unsqueeze(0).expand(B, -1, -1))
        return selected.prod(dim=1)

    without materializing `selected`. Only valid when TRITON_AVAILABLE is True and
    degrees is a CUDA tensor - callers must check both themselves.

    Args:
        degrees: (batch, vars, terms) membership degrees.
        idx: (vars, rules) term index each variable contributes to each rule, e.g.
            NAryRelation._gather_indices.

    Returns:
        (batch, rules) product, one value per (observation, rule).
    """
    return _GatherProdReduce.apply(degrees, idx)  # pylint: disable=no-member
