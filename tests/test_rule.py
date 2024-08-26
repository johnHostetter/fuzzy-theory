"""
Test the Rule class.
"""
import shutil
import unittest
from pathlib import Path
from typing import List, Type

import torch

from fuzzy.logic import Rule
from fuzzy.relations.continuous.n_ary import NAryRelation
from fuzzy.relations.continuous.t_norm import TNorm, Minimum, Product, SoftmaxSum


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRule(unittest.TestCase):
    """
    Test the Rule class.
    """

    def test_invalid_rule_creation(self) -> None:
        """
        Test that a rule cannot be created with a premise and consequence that are not unary.

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            Rule(
                NAryRelation([(0, 1), (1, 0)], [(2, 0)], device=AVAILABLE_DEVICE),
                NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            )
            Rule(
                NAryRelation((0, 1), (1, 0), (2, 0), device=AVAILABLE_DEVICE),
                NAryRelation([(0, 0)], [(1, 0)], device=AVAILABLE_DEVICE),
            )

    def test_create_rule_with_n_ary_relation(self) -> None:
        """
        Test that a rule can be created with a NAryRelation premise and NAryRelation consequence.

        Returns:
            None
        """
        n_rules_created: int = 0
        n_ary_types: List[Type[NAryRelation]] = [
            NAryRelation,
            TNorm,
            Minimum,
            Product,
            SoftmaxSum,
        ]
        # all t-norms should have the same string representation
        expected_str_rules: List[str] = [
            "IF NAryRelation([((0, 1), (1, 0), (2, 0))]) THEN NAryRelation([((0, 0),)])",
        ] + ["IF (0, 1) AND (1, 0) AND (2, 0) THEN (0, 0)"] * len(n_ary_types[1:])
        for n_ary_type, expected_str in zip(n_ary_types, expected_str_rules):
            premise = n_ary_type((0, 1), (1, 0), (2, 0), device=AVAILABLE_DEVICE)
            consequence = n_ary_type((0, 0), device=AVAILABLE_DEVICE)
            rule = Rule(premise, consequence)
            self.assertEqual(rule.premise, premise)
            self.assertEqual(rule.consequence, consequence)
            self.assertEqual(rule.id, n_rules_created)
            self.assertEqual(rule.next_id, n_rules_created + 1)
            self.assertEqual(str(rule), expected_str)
            n_rules_created += 1

    def test_save_and_load(self) -> None:
        """
        Test that we can save and load a Rule object.

        Returns:
            None
        """
        rule = Rule(
            NAryRelation((0, 1), (1, 0), (2, 0), device=AVAILABLE_DEVICE),
            NAryRelation((0, 0), device=AVAILABLE_DEVICE),
        )

        # illegal path name (has a file extension)
        with self.assertRaises(ValueError):
            rule.save(Path("test_rule.txt"))
        # good path (folder) name
        rule.save(Path(__file__).parent / "test_rule")
        # check that the folder exists
        self.assertTrue((Path(__file__).parent / "test_rule").exists())
        # load the file
        loaded_rule = Rule.load(Path(__file__).parent / "test_rule", device=AVAILABLE_DEVICE)
        # check that the loaded rule is the same class as the original rule
        self.assertIsInstance(loaded_rule, Rule)
        # compare devices
        self.assertEqual(rule.premise.device, loaded_rule.premise.device)
        self.assertEqual(rule.consequence.device, loaded_rule.consequence.device)
        # compare the premise indices
        self.assertEqual(rule.premise.indices, loaded_rule.premise.indices)
        # compare the consequence indices
        self.assertEqual(rule.consequence.indices, loaded_rule.consequence.indices)
        # compare the created GroupedLinks objects
        self.assertTrue(
            torch.allclose(
                rule.premise.grouped_links(None),
                loaded_rule.premise.grouped_links(None)
            )
        )
        # compare the ID of the Rule objects
        self.assertEqual(rule.id, loaded_rule.id)
        # delete the folder
        shutil.rmtree(Path(__file__).parent / "test_rule")
