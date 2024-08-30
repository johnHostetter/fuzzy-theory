"""
Test the various mechanisms in which a fuzzy logic rule can be created.
"""

import unittest

import torch

from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.continuous.t_norm import Product

from examples.continuous.demo_flcs import toy_mamdani


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFuzzyLogicRule(unittest.TestCase):
    """
    Test the operations and functions of a fuzzy logic rule.
    """

    def test_add_mamdani_rules_to_knowledge_base(self) -> None:
        """
        Test that adding Mamdani fuzzy logic rules to a KnowledgeBase object does not break things.

        Returns:
            None
        """
        antecedents, consequents, rules = toy_mamdani(
            t_norm=Product, device=AVAILABLE_DEVICE
        )
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=antecedents, targets=consequents
            ),
            rules=rules,
        )
        self.assertEqual(
            len(knowledge_base.graph.vs.select(tags_eq={"rule"})), len(rules)
        )

        # the recovered rules should be in the same order as the rules
        for expected_rule, actual_rule in zip(rules, knowledge_base.rules):
            self.assertEqual(expected_rule, actual_rule)
