"""
Test the fuzzy.logic.rule_helpers module.
"""

# white-box test of _effective_rule_count's single-rule edge case deliberately
# reaches into a private function
# pylint: disable=protected-access

import unittest

import numpy as np
import torch

from fuzzy.logic.rule_helpers import (
    RuleUniquenessMetrics,
    _effective_rule_count,
    jaccard,
    rule_to_clause_set,
    rule_uniqueness_metrics,
    tensor_to_rules,
)
from tests import AVAILABLE_DEVICE


def _rule_tensor(antecedents) -> torch.Tensor:
    """
    Build a (num_vars, num_terms, num_rules) binary rule tensor from a list of rules,
    where each rule is a list of (var_idx, term_idx) pairs.

    Returns:
        The binary rule tensor.
    """
    num_vars = max((v for rule in antecedents for v, _ in rule), default=-1) + 1
    num_terms = max((t for rule in antecedents for _, t in rule), default=-1) + 1
    num_rules = len(antecedents)
    tensor = torch.zeros((num_vars, num_terms, num_rules), device=AVAILABLE_DEVICE)
    for rule_idx, rule in enumerate(antecedents):
        for var_idx, term_idx in rule:
            tensor[var_idx, term_idx, rule_idx] = 1
    return tensor


class TestTensorToRules(unittest.TestCase):
    """
    Test tensor_to_rules.
    """

    def test_default_names(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0), (1, 1)], [(0, 1)]])
        rules = tensor_to_rules(tensor)
        self.assertEqual(
            rules,
            [
                "var_0 IS term_0 AND var_1 IS term_1 THEN ___",
                "var_0 IS term_1 THEN ___",
            ],
        )

    def test_accepts_numpy_rule_tensor(self) -> None:
        """
        numpy.ndarray.nonzero() returns a tuple (unlike torch.Tensor.nonzero()) -
        exercise that branch directly.

        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0), (1, 1)], [(0, 1)]]).cpu().numpy()
        rules = tensor_to_rules(tensor)
        self.assertEqual(
            rules,
            [
                "var_0 IS term_0 AND var_1 IS term_1 THEN ___",
                "var_0 IS term_1 THEN ___",
            ],
        )

    def test_custom_names(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)]])
        rules = tensor_to_rules(
            tensor, var_names=["temperature"], term_names=["low", "high"]
        )
        self.assertEqual(rules, ["temperature IS low THEN ___"])

    def test_empty_antecedent(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.zeros((2, 2, 1), device=AVAILABLE_DEVICE)
        rules = tensor_to_rules(tensor)
        self.assertEqual(rules, ["(empty antecedent) THEN ___"])

    def test_with_consequences(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)], [(1, 1)]])
        consequences = torch.tensor([[1.0], [2.0]], device=AVAILABLE_DEVICE)
        rules = tensor_to_rules(tensor, consequences=consequences)
        self.assertEqual(rules[0], "var_0 IS term_0 THEN [1.0]")
        self.assertEqual(rules[1], "var_1 IS term_1 THEN [2.0]")

    def test_rejects_wrong_rule_tensor_dimensionality(self) -> None:
        """
        Returns:
            None
        """
        with self.assertRaises(ValueError):
            tensor_to_rules(torch.zeros((2, 2), device=AVAILABLE_DEVICE))

    def test_rejects_wrong_consequences_dimensionality(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)]])
        with self.assertRaises(ValueError):
            tensor_to_rules(
                tensor, consequences=torch.zeros(1, device=AVAILABLE_DEVICE)
            )

    def test_rejects_mismatched_consequences_rule_count(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)]])
        with self.assertRaises(ValueError):
            tensor_to_rules(
                tensor, consequences=torch.zeros((2, 1), device=AVAILABLE_DEVICE)
            )


class TestRuleToClauseSet(unittest.TestCase):
    """
    Test rule_to_clause_set.
    """

    def test_extracts_clauses(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0), (1, 1)]])
        self.assertEqual(rule_to_clause_set(tensor, 0), frozenset({(0, 0), (1, 1)}))

    def test_empty_rule_has_no_clauses(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.zeros((2, 2, 1), device=AVAILABLE_DEVICE)
        self.assertEqual(rule_to_clause_set(tensor, 0), frozenset())


class TestJaccard(unittest.TestCase):
    """
    Test jaccard.
    """

    def test_both_empty(self) -> None:
        """
        Returns:
            None
        """
        self.assertEqual(jaccard(frozenset(), frozenset()), 1.0)

    def test_identical_sets(self) -> None:
        """
        Returns:
            None
        """
        clause_set = frozenset({(0, 0), (1, 1)})
        self.assertEqual(jaccard(clause_set, clause_set), 1.0)

    def test_disjoint_sets(self) -> None:
        """
        Returns:
            None
        """
        self.assertEqual(jaccard(frozenset({(0, 0)}), frozenset({(1, 1)})), 0.0)

    def test_partial_overlap(self) -> None:
        """
        Returns:
            None
        """
        set_a = frozenset({(0, 0), (1, 1)})
        set_b = frozenset({(0, 0), (2, 2)})
        # intersection = {(0, 0)}, union = {(0,0), (1,1), (2,2)}
        self.assertAlmostEqual(jaccard(set_a, set_b), 1.0 / 3.0)


class TestRuleUniquenessMetrics(unittest.TestCase):
    """
    Test rule_uniqueness_metrics.
    """

    def test_identical_rules_have_zero_diversity(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)], [(0, 0)], [(0, 0)]])
        metrics = rule_uniqueness_metrics(tensor)
        self.assertIsInstance(metrics, RuleUniquenessMetrics)
        self.assertAlmostEqual(metrics.base_diversity, 0.0)
        np.testing.assert_allclose(metrics.nn_distance, np.zeros(3))
        np.testing.assert_allclose(metrics.mean_distance, np.zeros(3))
        self.assertAlmostEqual(metrics.effective_rules, 1.0)

    def test_disjoint_rules_have_maximal_diversity(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)], [(1, 1)], [(2, 2)]])
        metrics = rule_uniqueness_metrics(tensor)
        self.assertAlmostEqual(metrics.base_diversity, 1.0)
        np.testing.assert_allclose(metrics.nn_distance, np.ones(3))
        self.assertAlmostEqual(metrics.effective_rules, 3.0)

    def test_partially_redundant_rules_fall_between(self) -> None:
        """
        2 of 3 rules are identical, 1 is disjoint from both - effective_rules should
        land strictly between the all-identical (1.0) and all-disjoint (3.0) cases.

        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)], [(0, 0)], [(1, 1)]])
        metrics = rule_uniqueness_metrics(tensor)
        self.assertGreater(metrics.effective_rules, 1.0)
        self.assertLess(metrics.effective_rules, 3.0)

    def test_similarity_matrix_shape_and_symmetry(self) -> None:
        """
        Returns:
            None
        """
        tensor = _rule_tensor([[(0, 0)], [(0, 0), (1, 1)], [(1, 1)]])
        metrics = rule_uniqueness_metrics(tensor)
        self.assertEqual(metrics.similarity_matrix.shape, (3, 3))
        np.testing.assert_allclose(
            metrics.similarity_matrix, metrics.similarity_matrix.T
        )

    def test_single_rule_is_trivially_one_effective_rule(self) -> None:
        """
        A single rule has no other rule to compare against - effective_rules is
        trivially 1.0 rather than dividing by zero.

        Returns:
            None
        """
        self.assertEqual(_effective_rule_count(np.array([[1.0]])), 1.0)


if __name__ == "__main__":
    unittest.main()
