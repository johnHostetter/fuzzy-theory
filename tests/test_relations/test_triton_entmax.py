"""
Test the fused Triton entmax_bisect kernel (relations/triton_entmax.py) against the
reference `entmax.entmax_bisect` (Peters et al., 2019) it accelerates.

These tests only run meaningfully on CUDA with triton installed - the kernel is a
CUDA-only optimization for the common case of a scalar alpha reduced along the last
dimension (see BoundAlphaEntmax in impl_options.py), with a full, unmodified fallback
to the reference implementation for everything else, so there is nothing
device-specific to verify on CPU.
"""

import unittest

import torch
from entmax import entmax_bisect as reference_entmax_bisect

from fuzzy.relations.triton_entmax import (
    MAX_BLOCK_D,
    TRITON_AVAILABLE,
    _is_fast_path_eligible,
    entmax_bisect_triton,
)
from tests import AVAILABLE_DEVICE

SKIP_REASON = "requires a CUDA device with triton installed"


@unittest.skipUnless(TRITON_AVAILABLE and AVAILABLE_DEVICE.type == "cuda", SKIP_REASON)
class TestEntmaxBisectTritonForward(unittest.TestCase):
    """
    Forward-value parity between entmax_bisect_triton and entmax.entmax_bisect.
    """

    def _compare(
        self,
        shape,
        alpha,
        n_iter=50,
        ensure_sum_one=True,
        dtype=torch.float32,
        scale=1.0,
        atol=1e-4,
        rtol=1e-4,
        seed=0,
    ):
        torch.manual_seed(seed)
        x = (torch.randn(*shape, device=AVAILABLE_DEVICE, dtype=dtype) * scale)
        alpha_t = torch.tensor(alpha, dtype=dtype, device=AVAILABLE_DEVICE)

        triton_out = entmax_bisect_triton(
            x, alpha=alpha_t, dim=-1, n_iter=n_iter, ensure_sum_one=ensure_sum_one
        )
        reference_out = reference_entmax_bisect(
            x, alpha=alpha_t, dim=-1, n_iter=n_iter, ensure_sum_one=ensure_sum_one
        )
        self.assertEqual(triton_out.shape, reference_out.shape)
        self.assertEqual(triton_out.dtype, x.dtype)
        torch.testing.assert_close(
            triton_out, reference_out, atol=atol, rtol=rtol
        )
        # entmax_bisect's defining constraint: every row sums to (very nearly) 1
        if ensure_sum_one:
            row_sums = triton_out.sum(dim=-1)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
            )

    def test_typical_rules_and_batch(self) -> None:
        """
        The shape this codebase actually exercises: (n_batch, n_rules).

        Returns:
            None
        """
        self._compare(shape=(40, 128), alpha=1.5)

    def test_non_power_of_two_d(self) -> None:
        """
        D need not be a power of two - block_d padding (masked lanes) must not leak
        into the sum or the max.

        Returns:
            None
        """
        for d in (1, 2, 3, 5, 7, 33, 65, 100, 129, 255):
            with self.subTest(d=d):
                self._compare(shape=(8, d), alpha=1.5, seed=d)

    def test_batch_size_one(self) -> None:
        """
        Returns:
            None
        """
        self._compare(shape=(1, 64), alpha=1.5)

    def test_extra_leading_dimensions(self) -> None:
        """
        Anything with >= 2 dims should work - the wrapper flattens all leading dims
        before dispatching to the kernel.

        Returns:
            None
        """
        self._compare(shape=(4, 6, 32), alpha=1.5)
        self._compare(shape=(2, 3, 4, 16), alpha=1.5)

    def test_alpha_near_one(self) -> None:
        """
        entmax_bisect approaches softmax as alpha -> 1 (but is documented as not
        supporting alpha == 1 exactly); this exercises the near-degenerate,
        numerically sensitive end of the supported range.

        Returns:
            None
        """
        self._compare(shape=(16, 64), alpha=1.001, atol=1e-3, rtol=1e-3)

    def test_alpha_near_two(self) -> None:
        """
        entmax_bisect approaches sparsemax as alpha -> 2, which produces exactly-zero
        entries - the other numerically sensitive end of the supported range.

        Returns:
            None
        """
        self._compare(shape=(16, 64), alpha=1.999)

    def test_sparsemax_alpha_two_produces_exact_zeros(self) -> None:
        """
        At alpha == 2 (sparsemax), entmax_bisect is expected to zero out some entries
        exactly - confirms the fused kernel reproduces that sparsity, not just
        matching values everywhere they happen to already be nonzero.

        Returns:
            None
        """
        torch.manual_seed(0)
        x = torch.randn(32, 64, device=AVAILABLE_DEVICE)
        alpha_t = torch.tensor(2.0, device=AVAILABLE_DEVICE)
        triton_out = entmax_bisect_triton(x, alpha=alpha_t, dim=-1, n_iter=50)
        reference_out = reference_entmax_bisect(x, alpha=alpha_t, dim=-1, n_iter=50)
        torch.testing.assert_close(triton_out, reference_out, atol=1e-4, rtol=1e-4)
        self.assertGreater((reference_out == 0).sum().item(), 0)
        self.assertTrue(
            torch.equal(triton_out == 0, reference_out == 0),
            "the fused kernel's exact-zero pattern must match the reference's",
        )

    def test_extreme_input_scales(self) -> None:
        """
        The bisection interval is derived from max(X), so very large or very small
        inputs stress tau_lo/tau_hi and the pow/log/exp reformulation differently than
        a unit-scale input would.

        Returns:
            None
        """
        for scale in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            with self.subTest(scale=scale):
                self._compare(shape=(8, 64), alpha=1.5, scale=scale, seed=int(scale))

    def test_all_equal_inputs_tie(self) -> None:
        """
        A row of identical values is a degenerate case (every element ties for the
        max); the result should be the uniform distribution.

        Returns:
            None
        """
        x = torch.full((4, 32), 3.0, device=AVAILABLE_DEVICE)
        alpha_t = torch.tensor(1.5, device=AVAILABLE_DEVICE)
        triton_out = entmax_bisect_triton(x, alpha=alpha_t, dim=-1, n_iter=50)
        reference_out = reference_entmax_bisect(x, alpha=alpha_t, dim=-1, n_iter=50)
        torch.testing.assert_close(triton_out, reference_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            triton_out, torch.full_like(triton_out, 1.0 / 32), atol=1e-4, rtol=1e-4
        )

    def test_reduced_n_iter_matches_this_codebases_choice(self) -> None:
        """
        This codebase calls entmax_bisect with n_iter=30 (see BoundAlphaEntmax.forward
        in impl_options.py) rather than the library default of 50; the fused kernel
        must match the reference at that same, reduced iteration count too.

        Returns:
            None
        """
        self._compare(shape=(40, 128), alpha=1.5, n_iter=30)

    def test_ensure_sum_one_false(self) -> None:
        """
        Returns:
            None
        """
        self._compare(shape=(16, 64), alpha=1.5, ensure_sum_one=False)

    def test_float16_input(self) -> None:
        """
        The no_grad, autocast(float16) forward calls in this codebase's training step
        (computing Double-DQN targets) reach entmax_bisect in float16; the fused
        kernel accumulates internally in float32 regardless of input dtype (see
        module docstring), so this is a looser-tolerance comparison than the float32
        cases, not a bit-exact one, and only forward is checked (no backward is ever
        needed in that no_grad call site).

        Returns:
            None
        """
        self._compare(
            shape=(40, 128),
            alpha=1.5,
            dtype=torch.float16,
            atol=2e-2,
            rtol=2e-2,
        )

    def test_deterministic_across_repeated_calls(self) -> None:
        """
        Returns:
            None
        """
        torch.manual_seed(0)
        x = torch.randn(20, 96, device=AVAILABLE_DEVICE)
        alpha_t = torch.tensor(1.5, device=AVAILABLE_DEVICE)
        first = entmax_bisect_triton(x, alpha=alpha_t, dim=-1, n_iter=30)
        second = entmax_bisect_triton(x, alpha=alpha_t, dim=-1, n_iter=30)
        self.assertTrue(torch.equal(first, second))


@unittest.skipUnless(TRITON_AVAILABLE and AVAILABLE_DEVICE.type == "cuda", SKIP_REASON)
class TestEntmaxBisectTritonBackward(unittest.TestCase):
    """
    Backward (dX and d_alpha) parity between entmax_bisect_triton and
    entmax.entmax_bisect.
    """

    def _compare_grads(
        self,
        shape,
        alpha_value,
        alpha_requires_grad,
        n_iter=50,
        atol=1e-4,
        rtol=1e-4,
        seed=0,
        non_leaf_alpha_transform=False,
    ):
        torch.manual_seed(seed)
        x_data = torch.randn(*shape, device=AVAILABLE_DEVICE)
        grad_output = torch.randn(*shape, device=AVAILABLE_DEVICE)

        def build(entmax_fn):
            x = x_data.clone().requires_grad_(True)
            leaf_alpha = torch.tensor(
                alpha_value, device=AVAILABLE_DEVICE, requires_grad=alpha_requires_grad
            )
            if non_leaf_alpha_transform:
                # exercises the actual usage pattern in this codebase: alpha passed
                # into entmax_bisect is itself the output of a differentiable
                # reparameterization (BoundAlphaEntmax.bound_alpha), not the raw leaf
                # Parameter - the returned gradient must still reduce correctly back
                # through that transform to the leaf.
                alpha = 1.0 + torch.sigmoid(leaf_alpha)
            else:
                alpha = leaf_alpha
            out = entmax_fn(x, alpha=alpha, dim=-1, n_iter=n_iter)
            out.backward(grad_output)
            return x.grad, leaf_alpha.grad

        triton_dx, triton_dalpha = build(entmax_bisect_triton)
        reference_dx, reference_dalpha = build(reference_entmax_bisect)

        torch.testing.assert_close(triton_dx, reference_dx, atol=atol, rtol=rtol)
        if alpha_requires_grad:
            self.assertIsNotNone(triton_dalpha)
            self.assertIsNotNone(reference_dalpha)
            torch.testing.assert_close(
                triton_dalpha, reference_dalpha, atol=atol, rtol=rtol
            )
        else:
            self.assertIsNone(triton_dalpha)
            self.assertIsNone(reference_dalpha)

    def test_dx_matches_without_alpha_grad(self) -> None:
        """
        Returns:
            None
        """
        self._compare_grads(shape=(40, 128), alpha_value=1.5, alpha_requires_grad=False)

    def test_dx_and_dalpha_match_leaf_alpha(self) -> None:
        """
        Matches this codebase's actual usage: alpha is a torch.nn.Parameter that
        requires grad (see BoundAlphaEntmax.__init__).

        Returns:
            None
        """
        self._compare_grads(shape=(40, 128), alpha_value=1.5, alpha_requires_grad=True)

    def test_dx_and_dalpha_match_through_bound_alpha_reparameterization(self) -> None:
        """
        Returns:
            None
        """
        self._compare_grads(
            shape=(24, 64),
            alpha_value=0.0,  # pre-sigmoid; bound_alpha maps this to alpha == 1.5
            alpha_requires_grad=True,
            non_leaf_alpha_transform=True,
        )

    def test_grads_match_at_reduced_n_iter(self) -> None:
        """
        Returns:
            None
        """
        self._compare_grads(
            shape=(40, 128), alpha_value=1.5, alpha_requires_grad=True, n_iter=30
        )

    def test_grads_match_with_extra_leading_dimensions(self) -> None:
        """
        Returns:
            None
        """
        self._compare_grads(
            shape=(4, 6, 32), alpha_value=1.5, alpha_requires_grad=True
        )

    def test_grads_match_at_sparsemax_alpha(self) -> None:
        """
        At alpha == 2, gppr's `Y > 0` mask is doing real work (some entries are
        exactly zero) - this is the branch most likely to diverge if the fused
        kernel's forward didn't reproduce the reference's exact-zero pattern.

        Returns:
            None
        """
        self._compare_grads(shape=(16, 64), alpha_value=1.999, alpha_requires_grad=True)

    def test_end_to_end_gradient_step_matches(self) -> None:
        """
        A fuller integration check: build a tiny linear layer feeding into
        entmax_bisect, run one backward pass, and confirm every learnable tensor's
        gradient (the linear layer's weight and bias, plus alpha) matches between the
        fused and reference paths - not just entmax's own direct inputs.

        Returns:
            None
        """
        torch.manual_seed(0)
        x_data = torch.randn(20, 10, device=AVAILABLE_DEVICE)
        grad_output = torch.randn(20, 96, device=AVAILABLE_DEVICE)
        weight_data = torch.randn(96, 10, device=AVAILABLE_DEVICE) * 0.1
        bias_data = torch.randn(96, device=AVAILABLE_DEVICE) * 0.1

        def build(entmax_fn):
            weight = weight_data.clone().requires_grad_(True)
            bias = bias_data.clone().requires_grad_(True)
            leaf_alpha = torch.zeros(1, device=AVAILABLE_DEVICE, requires_grad=True)
            alpha = 1.0 + torch.sigmoid(leaf_alpha)
            logits = x_data @ weight.T + bias
            out = entmax_fn(logits, alpha=alpha, dim=-1, n_iter=30)
            out.backward(grad_output)
            return weight.grad, bias.grad, leaf_alpha.grad

        triton_grads = build(entmax_bisect_triton)
        reference_grads = build(reference_entmax_bisect)
        for triton_grad, reference_grad, name in zip(
            triton_grads, reference_grads, ("weight", "bias", "alpha")
        ):
            with self.subTest(param=name):
                torch.testing.assert_close(
                    triton_grad, reference_grad, atol=1e-4, rtol=1e-4
                )


@unittest.skipUnless(TRITON_AVAILABLE and AVAILABLE_DEVICE.type == "cuda", SKIP_REASON)
class TestEntmaxBisectTritonFallback(unittest.TestCase):
    """
    Confirms the wrapper falls back to the unmodified reference implementation
    outside its documented fast-path preconditions, and that the fallback result is
    therefore correct by construction (it does not run the Triton kernel at all).
    """

    def test_non_last_dim_falls_back(self) -> None:
        """
        Returns:
            None
        """
        torch.manual_seed(0)
        x = torch.randn(8, 16, 32, device=AVAILABLE_DEVICE)
        alpha_t = torch.tensor(1.5, device=AVAILABLE_DEVICE)
        self.assertFalse(_is_fast_path_eligible(x, alpha_t, dim=1))
        triton_out = entmax_bisect_triton(x, alpha=alpha_t, dim=1, n_iter=30)
        reference_out = reference_entmax_bisect(x, alpha=alpha_t, dim=1, n_iter=30)
        torch.testing.assert_close(triton_out, reference_out)

    def test_non_scalar_alpha_falls_back(self) -> None:
        """
        entmax_bisect's public API allows a per-row alpha tensor; this codebase never
        constructs one (BoundAlphaEntmax always uses a single scalar), but the
        wrapper must still behave correctly - by falling back - if ever given one.

        Returns:
            None
        """
        torch.manual_seed(0)
        x = torch.randn(8, 32, device=AVAILABLE_DEVICE)
        alpha_t = torch.full((8, 1), 1.5, device=AVAILABLE_DEVICE)
        self.assertFalse(_is_fast_path_eligible(x, alpha_t, dim=-1))
        triton_out = entmax_bisect_triton(x, alpha=alpha_t, dim=-1, n_iter=30)
        reference_out = reference_entmax_bisect(x, alpha=alpha_t, dim=-1, n_iter=30)
        torch.testing.assert_close(triton_out, reference_out)

    def test_oversized_last_dim_falls_back(self) -> None:
        """
        Returns:
            None
        """
        x = torch.randn(2, MAX_BLOCK_D + 1, device=AVAILABLE_DEVICE)
        alpha_t = torch.tensor(1.5, device=AVAILABLE_DEVICE)
        self.assertFalse(_is_fast_path_eligible(x, alpha_t, dim=-1))

    def test_cpu_tensor_falls_back(self) -> None:
        """
        Returns:
            None
        """
        x = torch.randn(8, 32)
        alpha_t = torch.tensor(1.5)
        self.assertFalse(_is_fast_path_eligible(x, alpha_t, dim=-1))
        triton_out = entmax_bisect_triton(x, alpha=alpha_t, dim=-1, n_iter=30)
        reference_out = reference_entmax_bisect(x, alpha=alpha_t, dim=-1, n_iter=30)
        torch.testing.assert_close(triton_out, reference_out)


@unittest.skipUnless(TRITON_AVAILABLE and AVAILABLE_DEVICE.type == "cuda", SKIP_REASON)
class TestEntmaxBisectTritonCudaGraph(unittest.TestCase):
    """
    The whole point of fusing this into one kernel launch is to make it cheap to
    include inside a torch.cuda.graph-captured training step (see
    fuzzy/logic/control/cuda_graph.py) - confirm it is actually capturable and that a
    replay reproduces the eager result.
    """

    def test_capture_and_replay(self) -> None:
        """
        Returns:
            None
        """
        torch.manual_seed(0)
        static_x = torch.randn(40, 128, device=AVAILABLE_DEVICE)
        alpha_t = torch.tensor(1.5, device=AVAILABLE_DEVICE)

        def step():
            return entmax_bisect_triton(static_x, alpha=alpha_t, dim=-1, n_iter=30)

        eager_out = step()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = step()
        graph.replay()

        torch.testing.assert_close(static_out, eager_out, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
