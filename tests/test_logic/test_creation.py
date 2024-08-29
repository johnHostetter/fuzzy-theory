"""
Test the various mechanisms in which a fuzzy logic rule can be created.
"""

import unittest

import torch
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.continuous.t_norm import Product

from examples.supervised.demo_flcs import toy_mamdani


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_rules(test_case: unittest.TestCase, knowledge_base, rules):
    # check that the rules are the same if the id is ignored
    actual_rules = knowledge_base.get_fuzzy_logic_rules()
    for actual_rule, expected_rule in zip(actual_rules, rules):
        test_case.assertEqual(
            actual_rule.premise.indices, expected_rule.premise.indices
        )
        test_case.assertEqual(actual_rule.consequence, expected_rule.consequence)


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

        # check that the rules are the same if the id is ignored
        compare_rules(self, knowledge_base, rules)
