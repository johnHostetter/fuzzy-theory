"""
Test the function-selection behavior behind premise aggregation/activation:
PremiseAggregation's own construction, PremiseActivation.func()'s ENTMAX_BISECT
branch, and the BoundAlphaEntmax module that branch builds - split out of
test_options.py (which had grown past pylint's too-many-lines limit) since this
class is self-contained.
"""

import unittest

import torch

from fuzzy.utils.options.impl.impl_enums import (
    BoundAlphaEntmaxEnum,
    PremiseActivationEnum,
    PremiseAggregationEnum,
)
from fuzzy.utils.options.impl.impl_options import (
    BoundAlphaEntmax,
    PremiseActivation,
    PremiseAggregation,
)


class TestPremiseActivationFunctions(unittest.TestCase):
    """
    Test the function-selection behavior behind premise aggregation/activation:
    PremiseAggregation's own construction, PremiseActivation.func()'s ENTMAX_BISECT
    branch, and the BoundAlphaEntmax module that branch builds.
    """

    def test_premise_aggregation_construction(self) -> None:
        """
        Coverage/regression test: PremiseAggregation's own __init__ (as opposed to
        its .func() classmethod, already exercised elsewhere) had no test coverage.

        Returns:
            None
        """
        premise_aggregation = PremiseAggregation()
        self.assertIsNotNone(premise_aggregation)
        sum_fn = PremiseAggregation.func(PremiseAggregationEnum.SUM)
        mean_fn = PremiseAggregation.func(PremiseAggregationEnum.MEAN)
        degrees = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(sum_fn(degrees), -1 * degrees.sum(dim=1)))
        self.assertTrue(torch.equal(
            mean_fn(degrees), -1 * degrees.mean(dim=1)))

    def test_bound_alpha_entmax(self) -> None:
        """
        Coverage/regression test: BoundAlphaEntmax (construction, every bound_alpha()
        strategy, its ValueError for an unrecognized strategy, and forward()) had no
        test coverage at all.

        Returns:
            None
        """
        alpha = torch.zeros(1)
        for strategy in (
            BoundAlphaEntmaxEnum.SIGMOID,
            BoundAlphaEntmaxEnum.TANH,
            BoundAlphaEntmaxEnum.HARD_TANH,
            BoundAlphaEntmaxEnum.SOFTPLUS,
        ):
            module = BoundAlphaEntmax(
                bounding_strategy=strategy,
                alpha=alpha.clone())
            bounded = module.bound_alpha()
            # every strategy must keep alpha strictly within (1, 2), per the
            # class' documented contract
            self.assertTrue(bool((bounded > 1.0).all()))
            self.assertTrue(bool((bounded < 2.0).all()))

            tensor = torch.rand(3, 4)
            output = module(tensor)
            self.assertEqual(output.shape, tensor.shape)

        # an unrecognized bounding strategy must raise, not silently return
        # None
        module = BoundAlphaEntmax(
            bounding_strategy=BoundAlphaEntmaxEnum.SIGMOID, alpha=alpha.clone()
        )
        module.bounding_strategy = "not_a_real_strategy"
        with self.assertRaises(ValueError):
            module.bound_alpha()

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires a second device (CUDA) to move to"
    )
    def test_bound_alpha_entmax_forward_moves_alpha_to_input_device(
            self) -> None:
        """
        Coverage/regression test: forward() must move its bounded alpha onto the
        input tensor's device when they differ, rather than letting entmax_bisect
        fail on a device mismatch.

        Returns:
            None
        """
        module = BoundAlphaEntmax(
            bounding_strategy=BoundAlphaEntmaxEnum.SIGMOID,
            device=torch.device("cpu"),
        )
        tensor = torch.rand(3, 4, device=torch.device("cuda"))
        output = module(tensor)
        self.assertEqual("cuda", output.device.type)

    def test_premise_activation_func_entmax_bisect(self) -> None:
        """
        Coverage/regression test: PremiseActivation.func()'s ENTMAX_BISECT branch
        (which builds and returns a BoundAlphaEntmax instance, rather than looking up
        a plain function like the other transforms) had no test coverage, nor did its
        assertion that a bounding strategy must be given for it.

        Returns:
            None
        """
        result = PremiseActivation.func(
            transform=PremiseActivationEnum.ENTMAX_BISECT,
            bound=BoundAlphaEntmaxEnum.SIGMOID,
        )
        self.assertIsInstance(result, BoundAlphaEntmax)

        with self.assertRaises(AssertionError):
            PremiseActivation.func(
                transform=PremiseActivationEnum.ENTMAX_BISECT, bound=None
            )

    def test_bound_alpha_entmax_gradient_flows_to_alpha_and_input(self) -> None:
        """
        Golden-value/drift-detection test: no gradient test existed for
        BoundAlphaEntmax at all. Its output is an entmax (sparse-softmax-family)
        activation, whose row sums are always exactly 1 regardless of the input
        (the same normalization property softmax has), so a bare .sum() loss
        would have zero gradient even for a correct implementation - a dot
        product against a non-uniform weight vector is used instead. Confirms
        both alpha and the input tensor receive a real, non-zero, NaN-free
        gradient for every bounding strategy.

        Returns:
            None
        """
        for strategy in (
            BoundAlphaEntmaxEnum.SIGMOID,
            BoundAlphaEntmaxEnum.TANH,
            BoundAlphaEntmaxEnum.HARD_TANH,
            BoundAlphaEntmaxEnum.SOFTPLUS,
        ):
            module = BoundAlphaEntmax(
                bounding_strategy=strategy, alpha=torch.zeros(1)
            )
            x = torch.rand(3, 4, requires_grad=True)
            output = module(x)
            loss = (output * torch.linspace(0.5, 2.0, 4)).sum()
            loss.backward()

            self.assertFalse(bool(x.grad.isnan().any()), strategy)
            self.assertFalse(bool((x.grad == 0).all()), strategy)
            self.assertFalse(bool(module.alpha.grad.isnan().any()), strategy)
            self.assertFalse(bool((module.alpha.grad == 0).all()), strategy)

    def test_entmax15_gradient_flows_to_input(self) -> None:
        """
        Golden-value/drift-detection test: PremiseActivation.func()'s ENTMAX15
        branch (looking up the plain entmax15 function, as opposed to the
        ENTMAX_BISECT branch's BoundAlphaEntmax module) had no gradient test -
        same non-uniform-weight-loss rationale as
        test_bound_alpha_entmax_gradient_flows_to_alpha_and_input.

        Returns:
            None
        """
        entmax15_fn = PremiseActivation.func(
            transform=PremiseActivationEnum.ENTMAX15, bound=None
        )
        x = torch.rand(3, 4, requires_grad=True)
        output = entmax15_fn(x, dim=-1)
        loss = (output * torch.linspace(0.5, 2.0, 4)).sum()
        loss.backward()
        self.assertFalse(bool(x.grad.isnan().any()))
        self.assertFalse(bool((x.grad == 0).all()))
