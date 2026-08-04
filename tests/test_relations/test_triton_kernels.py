"""
Test the fused Triton gather+product-reduction kernel (relations/triton_kernels.py).

These tests only run meaningfully on CUDA with triton installed - the kernel is a
CUDA-only optimization for the common case of Product.forward() (see t_norm.py), with
a full, unmodified fallback for everything else, so there is nothing device-specific
to verify on CPU.
"""

import unittest

import torch

from fuzzy.relations.triton_kernels import TRITON_AVAILABLE, gather_prod

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SKIP_REASON = "requires a CUDA device with triton installed"


def _eager_gather_prod(degrees: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size = degrees.shape[0]
    selected = torch.gather(
        degrees, dim=2, index=idx.unsqueeze(0).expand(batch_size, -1, -1)
    )
    return selected.prod(dim=1)


@unittest.skipUnless(TRITON_AVAILABLE and AVAILABLE_DEVICE.type == "cuda", SKIP_REASON)
class TestGatherProd(unittest.TestCase):
    """
    Test gather_prod against the eager torch.gather(...).prod(...) it replaces.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch_size, self.n_vars, self.n_terms, self.n_rules = 16, 24, 5, 12
        self.degrees = (
            torch.rand(
                self.batch_size, self.n_vars, self.n_terms, device=AVAILABLE_DEVICE
            )
            * 0.9
            + 0.05
        ).detach()
        self.idx = torch.randint(
            0, self.n_terms, (self.n_vars, self.n_rules), device=AVAILABLE_DEVICE
        )

    def test_forward_matches_eager(self) -> None:
        """
        Returns:
            None
        """
        triton_out = gather_prod(self.degrees, self.idx)
        eager_out = _eager_gather_prod(self.degrees, self.idx)
        self.assertTrue(torch.allclose(triton_out, eager_out, atol=1e-5))

    def test_backward_matches_eager_without_zeros(self) -> None:
        """
        Returns:
            None
        """
        degrees_a = self.degrees.clone().requires_grad_(True)
        degrees_b = self.degrees.clone().requires_grad_(True)
        gather_prod(degrees_a, self.idx).sum().backward()
        _eager_gather_prod(degrees_b, self.idx).sum().backward()
        self.assertTrue(torch.allclose(degrees_a.grad, degrees_b.grad, atol=1e-4))

    def test_backward_matches_eager_with_zeros(self) -> None:
        """
        Regression-guard: the closed-form leave-one-out gradient must match
        torch.prod's own backward exactly at exact-zero factors, not just away from
        them - this is the case a naive division-based gradient (grad/x) gets wrong
        (produces NaN/Inf at x == 0).

        Returns:
            None
        """
        degrees_with_zeros = self.degrees.clone()
        # an entire variable's terms are zero for one batch element
        degrees_with_zeros[0, 3, :] = 0.0
        # exactly the selected term for one (variable, rule) pair is zero
        degrees_with_zeros[1, 7, self.idx[7, 2].item()] = 0.0

        degrees_a = degrees_with_zeros.clone().requires_grad_(True)
        degrees_b = degrees_with_zeros.clone().requires_grad_(True)
        out_a = gather_prod(degrees_a, self.idx)
        out_b = _eager_gather_prod(degrees_b, self.idx)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5))

        out_a.sum().backward()
        out_b.sum().backward()
        self.assertFalse(bool(degrees_a.grad.isnan().any()))
        self.assertFalse(bool(degrees_b.grad.isnan().any()))
        self.assertTrue(torch.allclose(degrees_a.grad, degrees_b.grad, atol=1e-4))

    def test_forward_matches_eager_all_zero_group(self) -> None:
        """
        Every selected value in a (batch, rule) group is zero - the product (and its
        gradient) must still be well-defined (zero product, well-defined gradient),
        not NaN.

        Returns:
            None
        """
        degrees_all_zero = torch.zeros_like(self.degrees).requires_grad_(True)
        out = gather_prod(degrees_all_zero, self.idx)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        out.sum().backward()
        self.assertFalse(bool(degrees_all_zero.grad.isnan().any()))

    def test_single_variable_and_rule(self) -> None:
        """
        Degenerate shape (one variable, one rule) - the loop bounds inside the kernel
        must not assume more than one iteration.

        Returns:
            None
        """
        degrees = torch.rand(4, 1, 3, device=AVAILABLE_DEVICE)
        idx = torch.zeros(1, 1, dtype=torch.long, device=AVAILABLE_DEVICE)
        triton_out = gather_prod(degrees, idx)
        eager_out = _eager_gather_prod(degrees, idx)
        self.assertTrue(torch.allclose(triton_out, eager_out, atol=1e-5))

    def test_n_vars_not_a_multiple_of_block_size(self) -> None:
        """
        The kernel loops over variables in blocks of 128 - a variable count that
        doesn't divide evenly must still be handled correctly by the mask on the
        final, partial block.

        Returns:
            None
        """
        n_vars = 130  # 128 + a partial second block of 2
        degrees = (
            torch.rand(8, n_vars, 5, device=AVAILABLE_DEVICE) * 0.9 + 0.05
        ).detach()
        idx = torch.randint(0, 5, (n_vars, 6), device=AVAILABLE_DEVICE)
        triton_out = gather_prod(degrees, idx)
        eager_out = _eager_gather_prod(degrees, idx)
        self.assertTrue(torch.allclose(triton_out, eager_out, atol=1e-4))
