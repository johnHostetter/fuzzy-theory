"""
Helpers for converting a fuzzy rule base's raw rule tensor into human-readable rule
strings, and for measuring how distinct the identified rules are from one another.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import numpy as np
import torch


def tensor_to_rules(
    rule_tensor: torch.Tensor,
    consequences: Optional[torch.Tensor] = None,
    var_names: Optional[List[str]] = None,
    term_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Convert a 3D binary rule tensor into human-readable fuzzy rule strings.

    Args:
        rule_tensor: A (num_vars, num_terms, num_rules) binary tensor, where
            rule_tensor[v, t, r] == 1 means rule r's antecedent includes
            "variable v IS term t".
        consequences: An optional (num_rules, ...) tensor of each rule's consequence,
            rendered as-is after "THEN". Left as "___" per rule if not given.
        var_names: The name of each variable, in order. Defaults to "var_0", "var_1", etc.
        term_names: The name of each term, in order. Defaults to "term_0", "term_1", etc.

    Returns:
        One rule antecedent (and consequence, if given) string per rule.
    """
    if rule_tensor.ndim != 3:
        raise ValueError(
            f"Expected rule_tensor to have 3 dimensions, but got {rule_tensor.ndim}."
        )
    num_vars, num_terms, num_rules = rule_tensor.shape
    if consequences is not None:
        if consequences.ndim != 2:
            raise ValueError(
                f"Expected consequences to have 2 dimensions, but got "
                f"{consequences.ndim}."
            )
        if consequences.shape[0] != num_rules:
            raise ValueError(
                f"Expected consequences' first dimension ({consequences.shape[0]}) "
                f"to equal the number of rules identified in rule_tensor "
                f"({num_rules})."
            )
    var_names = var_names or [f"var_{i}" for i in range(num_vars)]
    term_names = term_names or [f"term_{j}" for j in range(num_terms)]

    rules = []
    for rule_idx in range(num_rules):
        antecedents = []
        consequence_str = "___"
        if consequences is not None:
            consequence_str = consequences[rule_idx].cpu().detach().tolist()
        for var_idx in range(num_vars):
            # torch.Tensor.nonzero() returns a (N, 1) tensor of indices directly;
            # numpy.ndarray.nonzero() instead returns a 1-tuple wrapping that
            # array
            active_terms = rule_tensor[var_idx, :, rule_idx].nonzero()
            if isinstance(active_terms, tuple):
                active_terms = active_terms[0]
            for term_idx in active_terms.flatten():
                antecedents.append(f"{var_names[var_idx]} IS {term_names[term_idx]}")

        if antecedents:
            rules.append(" AND ".join(antecedents) + f" THEN {consequence_str}")
        else:
            rules.append(f"(empty antecedent) THEN {consequence_str}")

    return rules


def rule_to_clause_set(tensor: torch.Tensor, rule_idx: int) -> frozenset:
    """
    Extract the set of (variable, term) index pairs making up one rule's antecedent.

    Args:
        tensor: A (num_vars, num_terms, num_rules) binary rule tensor (see
            tensor_to_rules).
        rule_idx: The index of the rule to extract clauses for.

    Returns:
        A frozenset of (variable_idx, term_idx) pairs.
    """
    num_vars, num_terms, _ = tensor.shape
    return frozenset(
        (var_idx, term_idx)
        for var_idx in range(num_vars)
        for term_idx in range(num_terms)
        if tensor[var_idx, term_idx, rule_idx]
    )


def jaccard(set_a: frozenset, set_b: frozenset) -> float:
    """
    Calculate the Jaccard similarity between two sets: the size of their intersection
    over the size of their union.

    Args:
        set_a: The first set.
        set_b: The second set.

    Returns:
        1.0 if both sets are empty, otherwise len(set_a & set_b) / len(set_a | set_b).
    """
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


@dataclass
class RuleUniquenessMetrics:
    """
    Per-rule and whole-rule-base metrics describing how distinct the rules identified
    in a rule tensor are from one another, based on pairwise Jaccard distance between
    their (variable, term) clause sets.
    """

    # per-rule, shape (num_rules,): distance to the nearest other rule - low means
    # nearly redundant, high means truly unique
    nn_distance: np.ndarray
    # per-rule, shape (num_rules,): mean distance to every other rule
    mean_distance: np.ndarray
    # per-rule, shape (num_rules,): a rule's own mean distance vs. its neighbors' -
    # above 1 means this rule is more isolated than its neighbors are
    isolation: np.ndarray
    # scalar: overall pairwise mean distance, 0 (all rules identical) to 1 (all
    # rules disjoint)
    base_diversity: float
    # scalar: the effective number of distinct rules, from 1 (every rule identical)
    # to num_rules (every rule pairwise disjoint) - see _effective_rule_count
    effective_rules: float
    # (num_rules, num_rules): pairwise Jaccard similarity
    similarity_matrix: np.ndarray


def _pairwise_similarity(clause_sets: List[frozenset]) -> np.ndarray:
    """
    Build the (num_rules, num_rules) pairwise Jaccard similarity matrix between each
    pair of rules' clause sets.
    """
    num_rules = len(clause_sets)
    similarity = np.zeros((num_rules, num_rules))
    for i, j in combinations(range(num_rules), 2):
        score = jaccard(clause_sets[i], clause_sets[j])
        similarity[i, j] = similarity[j, i] = score
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _isolation_scores(distance: np.ndarray, mean_distance: np.ndarray) -> np.ndarray:
    """
    Ratio of each rule's own mean distance to its k-nearest-neighbors' mean distance -
    above 1 means this rule is more isolated than its neighbors are.
    """
    num_rules = distance.shape[0]
    k = min(5, num_rules - 1)
    isolation = np.zeros(num_rules)
    for rule_idx in range(num_rules):
        knn_idx = np.argsort(distance[rule_idx])[:k]
        neighbor_mean = np.mean([mean_distance[j] for j in knn_idx])
        isolation[rule_idx] = (
            mean_distance[rule_idx] / neighbor_mean if neighbor_mean > 0 else 1.0
        )
    return isolation


def _effective_rule_count(similarity: np.ndarray) -> float:
    """
    The effective number of distinct rules, given how similar rules are to one
    another on average: 1.0 if every rule is identical, up to num_rules if every
    rule is pairwise disjoint from every other rule.

    Uses the N / (1 + (N - 1) * mean_offdiag_similarity) form from the "effective
    number of alleles"/inverse-Simpson family of diversity indices, applied to the
    average pairwise Jaccard similarity between rules' clause sets. (A prior version
    of this function instead computed the perplexity of each row of the similarity
    matrix treated as a soft cluster assignment, which - despite the name - produced
    the inverse of the intended result: identical rules scored highest, disjoint
    rules scored lowest.)
    """
    num_rules = similarity.shape[0]
    if num_rules <= 1:
        return 1.0
    off_diagonal_sum = similarity.sum() - np.trace(similarity)
    mean_off_diag_similarity = off_diagonal_sum / (num_rules * (num_rules - 1))
    return num_rules / (1.0 + (num_rules - 1) * mean_off_diag_similarity)


def rule_uniqueness_metrics(tensor: torch.Tensor) -> RuleUniquenessMetrics:
    """
    Measure how distinct each rule (and the rule base as a whole) is from the others,
    based on pairwise Jaccard distance between rules' (variable, term) clause sets.

    Args:
        tensor: A (num_vars, num_terms, num_rules) binary rule tensor (see
            tensor_to_rules).

    Returns:
        The computed per-rule and whole-rule-base uniqueness metrics.
    """
    num_rules = tensor.shape[2]
    clause_sets = [
        rule_to_clause_set(tensor, rule_idx) for rule_idx in range(num_rules)
    ]

    similarity = _pairwise_similarity(clause_sets)
    distance = 1.0 - similarity

    # nearest-neighbor distance: how far is the closest other rule?
    np.fill_diagonal(distance, np.inf)
    nn_distance = distance.min(axis=1)
    np.fill_diagonal(distance, 0.0)

    # mean distance: average distinctness from every other rule
    mean_distance = distance.sum(axis=1) / (num_rules - 1)
    isolation = _isolation_scores(distance, mean_distance)

    # base_diversity: overall pairwise mean distance across the whole rule base
    upper_triangle = distance[np.triu_indices(num_rules, k=1)]
    base_diversity = upper_triangle.mean()

    effective_rules = _effective_rule_count(similarity)

    return RuleUniquenessMetrics(
        nn_distance=nn_distance,
        mean_distance=mean_distance,
        isolation=isolation,
        base_diversity=base_diversity,
        effective_rules=effective_rules,
        similarity_matrix=similarity,
    )
