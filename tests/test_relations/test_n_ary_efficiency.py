"""
Test the performance-oriented optimizations of the n-ary relations: raw scale, the
links cache, the gather-based fast path for apply_mask(), and the alternative (claimed
more efficient) computation strategies EXP_SUM_LOG and LINEAR_SUM. Core NAryRelation
behavior (construction, save/load, graph representation, basic mask application) lives
in test_n_ary.py instead.
"""

# white-box tests deliberately reach into private/internal attributes to verify
# implementation details
# pylint: disable=protected-access

import unittest
from typing import Tuple
from unittest import mock

import numpy as np
import torch

from fuzzy.relations.linkage import BinaryLinks, GroupedLinks
from fuzzy.relations.n_ary import (
    GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL,
    NAryMaskMethods,
    NAryRelation,
)
from fuzzy.sets import FuzzySet, Gaussian, Membership
from fuzzy.sets.shape import FuzzySetInitMethod, FuzzySetShape
from tests.test_relations.test_n_ary import (
    AVAILABLE_DEVICE,
    N_OBSERVATIONS,
    TestNAryRelation,
)


class TestComputationalAbilities(unittest.TestCase):
    """
    This class tests the computational abilities of the n-ary relation, particularly when dealing
    with very large relations. It pushes the limits of the n-ary relation to see if it can handle
    extremely large fuzzy inference systems.

    Failing this test does not necessarily mean that the n-ary relation is not working as expected,
    but it may indicate that the n-ary relation is not optimized for very large fuzzy inference
    systems (e.g., those with thousands of features, such as in computer vision).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_terms: int = 16
        self.n_variables: int = 24000
        self.n_relations: int = 256

    def test_very_large_n_ary_relation(self) -> None:
        """
        Test the n-ary relation can handle very large relations involving thousands of features.

        Essentially, this is to check that memory management is working as expected; particularly
        for CUDA devices.

        Returns:
            None
        """
        # random indices
        indices: np.ndarray = np.random.choice(
            [0, 1], size=(self.n_variables * self.n_terms * self.n_relations)
        ).reshape(self.n_variables, self.n_terms, self.n_relations)
        n_ary = NAryRelation(
            grouped_links=GroupedLinks(
                modules_list=[
                    BinaryLinks(
                        indices,
                        device=AVAILABLE_DEVICE,
                    )
                ]
            ),
            device=AVAILABLE_DEVICE,
        )
        # example membership
        membership_function: FuzzySet = Gaussian.create(
            FuzzySetShape(
                n_variables=self.n_variables,
                n_terms=self.n_terms,
            ),
            device=AVAILABLE_DEVICE,
            method=FuzzySetInitMethod.RANDOM,
        )
        # max terms used in the above N-ary relation
        membership: Membership = membership_function(
            torch.randn(
                N_OBSERVATIONS, self.n_variables, self.n_terms, device=AVAILABLE_DEVICE
            )
        )
        # check that the apply_mask works
        n_ary.apply_mask(membership)


class _StochasticLinks(torch.nn.Module):
    """
    A minimal stand-in for a hypothetical non-deterministic links module (e.g. a future
    Gumbel-Softmax-resampled logits module - see GroupedLinks' own docstring). Its forward()
    returns a DIFFERENT value on every call, unlike BinaryLinks, which always returns the
    same fixed tensor regardless of its argument. Used only to prove that NAryRelation's
    links cache refuses to memoize anything that isn't provably a BinaryLinks.
    """

    def __init__(self, shape: torch.Size, device: torch.device):
        super().__init__()
        self._shape = shape
        self._device = device
        self.call_count = 0

    @property
    def shape(self) -> torch.Size:
        """
        Returns:
            The shape of the (fake) links tensor this module would produce.
        """
        return self._shape

    def forward(self, *_) -> torch.Tensor:
        """
        Returns:
            A tensor filled with the current call count, so each call is trivially
            distinguishable from the last.
        """
        self.call_count += 1
        return torch.full(self._shape, float(self.call_count), device=self._device)


class TestNAryRelationEfficiency(TestNAryRelation):
    """
    Test the performance-oriented mechanisms shared by all n-ary relations: the links
    cache (avoids recomputing BinaryLinks-derived links every call) and the gather-based
    fast path for apply_mask() (avoids materializing the full
    (batch, vars, terms, rules) intermediate tensor the general PROD path builds).
    """

    def test_links_are_cacheable_false_when_grouped_links_none(self) -> None:
        """
        _links_are_cacheable() must report False (never crash) when grouped_links has not
        been set up yet.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), device=AVAILABLE_DEVICE)
        n_ary.grouped_links = None
        self.assertFalse(
            n_ary._links_are_cacheable()  # pylint: disable=protected-access
        )

    def test_precompute_gather_indices_no_grouped_links(self) -> None:
        """
        _precompute_gather_indices() must report the relation as not gather-eligible
        (never crash) when grouped_links has not been set up yet.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), device=AVAILABLE_DEVICE)
        n_ary.grouped_links = None
        n_ary._precompute_gather_indices()  # pylint: disable=protected-access
        self.assertFalse(n_ary._use_gather)  # pylint: disable=protected-access

    def test_links_cache_reused_on_second_call(self) -> None:
        """
        _ensure_links_cache() must not recompute the links on a second call when nothing
        has changed (the common, BinaryLinks-only case): the exact same tensor object
        should be reused.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()
        # pylint: disable=protected-access
        n_ary._ensure_links_cache(membership)
        first = n_ary._cached_links
        n_ary._ensure_links_cache(membership)
        second = n_ary._cached_links
        # pylint: enable=protected-access
        self.assertIs(first, second)

    def test_links_not_cached_for_non_binary_links(self) -> None:
        """
        Regression/defensive test: a links module that is not BinaryLinks (e.g. a future
        stochastic, Gumbel-Softmax-resampled module) must never be treated as cacheable,
        since it could legitimately produce a different result on every call and this
        cache has no way to detect that on its own.

        Returns:
            None
        """
        stochastic = _StochasticLinks(
            shape=torch.Size([2, 2, 1]), device=AVAILABLE_DEVICE
        )
        grouped_links = GroupedLinks(modules_list=[stochastic])
        n_ary = NAryRelation(grouped_links=grouped_links, device=AVAILABLE_DEVICE)
        self.assertFalse(
            n_ary._links_are_cacheable()  # pylint: disable=protected-access
        )

        membership = Membership(
            degrees=torch.rand(3, 2, 2, device=AVAILABLE_DEVICE),
            mask=torch.ones(2, 2, device=AVAILABLE_DEVICE),
        )
        # pylint: disable=protected-access
        n_ary._ensure_links_cache(membership)
        first = n_ary._cached_links
        n_ary._ensure_links_cache(membership)
        second = n_ary._cached_links
        # pylint: enable=protected-access
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(stochastic.call_count, 2)

    def test_gather_and_prod_paths_agree_on_nan_observations(self) -> None:
        """
        Regression test: _gather_apply_mask (the optimized path, chosen automatically
        whenever a relation's links are all BinaryLinks with at most one active term per
        variable/rule) used to disagree with _prod_apply_mask (the fallback) whenever an
        observation was NaN on a term that is not the one a given rule selects, but
        belongs to a variable another rule DOES select via a different term. Both must
        give identical results, since which one runs is an invisible implementation
        detail the caller has no control over.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            [(0, 2)],
            device=AVAILABLE_DEVICE,
            method=NAryMaskMethods.PROD,
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access

        degrees = torch.rand(4, 2, 3, device=AVAILABLE_DEVICE)
        # var1-term2: not selected by any rule here
        degrees[2, 1, 2] = float("nan")
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 3, device=AVAILABLE_DEVICE)
        )

        # pylint: disable=protected-access
        gather_result, _ = relation._gather_apply_mask(membership)
        prod_result, _ = relation._prod_apply_mask(membership)
        # pylint: enable=protected-access
        self.assertTrue(torch.allclose(gather_result, prod_result, equal_nan=True))

    def test_gather_apply_mask_matches_manual_gather_when_no_nan(self) -> None:
        """
        Characterization test locking down _gather_apply_mask's exact output (not just
        agreement with _prod_apply_mask) for the common, NaN-free case, via an
        independent, manually-indexed computation that does not share any code path
        with the implementation under test.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
            device=AVAILABLE_DEVICE,
            method=NAryMaskMethods.PROD,
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access
        self.assertTrue(relation._all_active)  # pylint: disable=protected-access

        degrees = torch.rand(5, 2, 2, device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )

        # pylint: disable=protected-access
        result, _ = relation._gather_apply_mask(membership)
        # pylint: enable=protected-access

        # manually reproduce "for each rule, for each variable, pick the one term that
        # rule's premise selects for that variable" without using
        # gather/masking at all
        expected = torch.empty(5, 2, 2, device=AVAILABLE_DEVICE)
        for rule_idx, relation_indices in enumerate(relation.indices):
            for var_idx, term_idx in relation_indices:
                expected[:, var_idx, rule_idx] = degrees[:, var_idx, term_idx]
        self.assertTrue(torch.equal(result, expected))

    def test_gather_apply_mask_handles_partial_variable_coverage(self) -> None:
        """
        When a variable is not used by every rule's premise (a realistic, sparse rule
        set - not every rule needs to reference every variable), _all_active is False
        and the inactive (variable, rule) pairs must resolve to 1.0 (the product
        identity), while active pairs still resolve to the selected degree.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0)],  # rule 0 only references variable 0
            [(1, 0)],  # rule 1 only references variable 1
            device=AVAILABLE_DEVICE,
            method=NAryMaskMethods.PROD,
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access
        self.assertFalse(relation._all_active)  # pylint: disable=protected-access

        degrees = torch.tensor(
            [[[0.2], [0.9]], [[0.4], [0.1]]], device=AVAILABLE_DEVICE
        )  # (batch=2, vars=2, terms=1)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE)
        )

        # pylint: disable=protected-access
        result, _ = relation._gather_apply_mask(membership)
        # pylint: enable=protected-access

        expected = torch.tensor(
            [[[0.2, 1.0], [1.0, 0.9]], [[0.4, 1.0], [1.0, 0.1]]],
            device=AVAILABLE_DEVICE,
        )  # (batch, vars, rules): var0 only active for rule0, var1 only for rule1
        self.assertTrue(torch.equal(result, expected))

    @staticmethod
    def _large_gather_relation_and_degrees(
        with_nan: bool,
    ) -> Tuple[NAryRelation, torch.Tensor, Membership]:
        """
        Build a gather-eligible relation (2 variables, 1 rule spanning both - a single
        list of 2 tuples passed to NAryRelation is one rule, not two) with a large
        enough batch size that selected.numel() exceeds
        GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL, to exercise _gather_apply_mask's
        sync-and-maybe-skip branch (as opposed to its always-unconditional small-tensor
        branch).

        Returns:
            The relation, the degrees tensor, and the Membership wrapping it.
        """
        relation = NAryRelation(
            [(0, 0), (1, 0)], device=AVAILABLE_DEVICE, method=NAryMaskMethods.PROD
        )
        # selected.numel() == batch_size * n_vars(2) * n_rules(1)
        batch_size = GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL // 2 + 10
        degrees = torch.rand(batch_size, 2, 1, device=AVAILABLE_DEVICE)
        if with_nan:
            degrees[0, 1, 0] = float("nan")
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE)
        )
        return relation, degrees, membership

    def test_gather_apply_mask_large_tensor_skips_nan_to_num_when_no_nan_present(
        self,
    ) -> None:
        """
        Regression/performance test: _gather_apply_mask used to call
        selected.nan_to_num(...) unconditionally on every call, even though it already
        computes any_nan_per_variable and therefore knows in advance whether there is
        anything for nan_to_num to do. Profiling a FuzzyLogicController with many input
        variables showed this unconditional nan_to_num call over the full
        (batch, vars, rules) tensor was the single largest cost in the rule engine
        (~33% of the engine's GPU time) - pure waste on the common, NaN-free path,
        since nan_to_num is a no-op there. Above GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL
        elements, it must be skipped when there is no NaN to replace.

        Returns:
            None
        """
        relation, _, membership = self._large_gather_relation_and_degrees(
            with_nan=False
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access

        with mock.patch.object(
            torch.Tensor, "nan_to_num", autospec=True
        ) as mocked_nan_to_num:
            relation._gather_apply_mask(membership)  # pylint: disable=protected-access
        mocked_nan_to_num.assert_not_called()

    def test_gather_apply_mask_small_tensor_always_calls_nan_to_num(self) -> None:
        """
        Below GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL, calibration showed a CUDA sync to
        decide whether nan_to_num is needed (~400-750us, dominated by the reduction
        kernel and blocking scalar readback, not the sync primitive itself - see
        GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL's definition) costs far more than just
        always doing the cheap NaN-handling unconditionally. So below the threshold,
        _gather_apply_mask must skip the sync and always call nan_to_num, even when
        there is no NaN present (a harmless no-op at this size).

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0), (1, 0)], device=AVAILABLE_DEVICE, method=NAryMaskMethods.PROD
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access
        degrees = torch.rand(
            4, 2, 1, device=AVAILABLE_DEVICE
        )  # no NaN anywhere; tiny tensor
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE)
        )

        with mock.patch.object(
            torch.Tensor, "nan_to_num", autospec=True
        ) as mocked_nan_to_num:
            relation._gather_apply_mask(membership)  # pylint: disable=protected-access
        mocked_nan_to_num.assert_called_once()

    def test_gather_apply_mask_calls_nan_to_num_when_nan_present(self) -> None:
        """
        Regardless of tensor size, nan_replacement must still be honored whenever
        there is something to actually replace.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0), (1, 0)], device=AVAILABLE_DEVICE, method=NAryMaskMethods.PROD
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access
        degrees = torch.rand(4, 2, 1, device=AVAILABLE_DEVICE)
        degrees[0, 1, 0] = float("nan")
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE)
        )

        with mock.patch.object(
            torch.Tensor, "nan_to_num", autospec=True
        ) as mocked_nan_to_num:
            relation._gather_apply_mask(membership)  # pylint: disable=protected-access
        mocked_nan_to_num.assert_called_once()

        # and, unmocked, nan_replacement is actually honored (correctness, not just
        # that nan_to_num was invoked)
        result, _ = relation._gather_apply_mask(  # pylint: disable=protected-access
            membership
        )
        self.assertFalse(bool(result.isnan().any()))

    def test_gather_apply_mask_large_tensor_calls_nan_to_num_when_nan_present(
        self,
    ) -> None:
        """
        The large-tensor sync-and-maybe-skip branch must still honor nan_replacement
        whenever NaN is actually present, not just in the small-tensor branch.

        Returns:
            None
        """
        relation, _, membership = self._large_gather_relation_and_degrees(with_nan=True)
        result, _ = relation._gather_apply_mask(  # pylint: disable=protected-access
            membership
        )
        self.assertFalse(bool(result.isnan().any()))

    def test_gather_apply_mask_large_tensor_matches_manual_gather(self) -> None:
        """
        Correctness of the large-tensor branch, independent of the small-tensor
        branch's own characterization test above - both branches must agree with an
        independently-written manual computation, not just with each other.

        Returns:
            None
        """
        relation, degrees, membership = self._large_gather_relation_and_degrees(
            with_nan=False
        )
        result, _ = relation._gather_apply_mask(  # pylint: disable=protected-access
            membership
        )
        expected = torch.empty_like(result)
        for rule_idx, relation_indices in enumerate(relation.indices):
            for var_idx, term_idx in relation_indices:
                expected[:, var_idx, rule_idx] = degrees[:, var_idx, term_idx]
        self.assertTrue(torch.equal(result, expected))

    def test_gather_apply_mask_no_nan_skip_preserves_gradient(self) -> None:
        """
        The small-tensor branch's unconditional NaN-handling must not change the
        gradient computed with respect to the degrees that fed into it, relative to
        what nan_to_num's own (identity, away from NaN/inf) gradient would give.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0), (1, 0)], device=AVAILABLE_DEVICE, method=NAryMaskMethods.PROD
        )
        self.assertTrue(relation._use_gather)  # pylint: disable=protected-access

        degrees = torch.rand(4, 2, 1, device=AVAILABLE_DEVICE, requires_grad=True)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 1, device=AVAILABLE_DEVICE)
        )
        result, _ = relation._gather_apply_mask(  # pylint: disable=protected-access
            membership
        )
        result.sum().backward()
        self.assertIsNotNone(degrees.grad)
        self.assertTrue(torch.equal(degrees.grad, torch.ones_like(degrees)))

    def test_exp_sum_log_matches_prod(self) -> None:
        """
        NAryMaskMethods.EXP_SUM_LOG is documented as mathematically equivalent to PROD,
        but was never exercised by any existing test.

        Returns:
            None
        """
        prod_relation = NAryRelation(
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            device=AVAILABLE_DEVICE,
            method=NAryMaskMethods.PROD,
        )
        exp_sum_log_relation = NAryRelation(
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            device=AVAILABLE_DEVICE,
            method=NAryMaskMethods.EXP_SUM_LOG,
        )
        degrees = torch.rand(4, 2, 2, device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )
        # call the method bodies directly to guarantee coverage regardless of whether
        # this particular link structure happens to be gather-eligible
        # pylint: disable=protected-access
        prod_result, _ = prod_relation._prod_apply_mask(membership)
        exp_sum_log_result, _ = exp_sum_log_relation._exp_sum_log_apply_mask(membership)
        # pylint: enable=protected-access
        self.assertTrue(torch.allclose(prod_result, exp_sum_log_result, atol=1e-5))

    def test_linear_sum_method(self) -> None:
        """
        NAryMaskMethods.LINEAR_SUM was not exercised by any existing test. Verify it
        computes the documented linear combination of degrees and links directly.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            device=AVAILABLE_DEVICE,
            method=NAryMaskMethods.LINEAR_SUM,
        )
        degrees = torch.rand(4, 2, 2, device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=degrees, mask=torch.ones(2, 2, device=AVAILABLE_DEVICE)
        )
        result, mask_component = (
            relation._linear_sum_apply_mask(  # pylint: disable=protected-access
                membership
            )
        )
        self.assertIsNone(mask_component)
        mask = relation.grouped_links(membership=membership)
        expected = (degrees.unsqueeze(-1) * mask).sum(dim=(1, 2))
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))
