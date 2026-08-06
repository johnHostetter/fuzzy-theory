"""
Test the various mechanisms in which a fuzzy logic rule can be created.
"""

import unittest

from tests import AVAILABLE_DEVICE

from .common import build_mamdani_knowledge_base


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
        knowledge_base, rules = build_mamdani_knowledge_base(AVAILABLE_DEVICE)
        self.assertEqual(
            len(knowledge_base.graph.vs.select(tags_eq={"rule"})), len(rules)
        )

        # the recovered rules should be in the same order as the rules
        for expected_rule, actual_rule in zip(rules, knowledge_base.rules):
            self.assertEqual(expected_rule, actual_rule)
