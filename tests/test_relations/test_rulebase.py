"""
Test the RuleBase class.
"""

import shutil
import unittest
from pathlib import Path
from typing import List

import torch

from fuzzy.logic import Rule
from fuzzy.logic.rulebase import RuleBase
from fuzzy.relations.continuous.t_norm import Product
from fuzzy.sets.continuous.membership import Membership
from .test_n_ary import TestNAryRelation

N_RULES: int = 5
N_VARIABLES: int = 4
AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRuleBase(unittest.TestCase):
    """
    Test the RuleBase class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules: List[Rule] = []
        for i in range(N_RULES):
            self.rules.append(
                Rule(
                    premise=Product(
                        *[(j, i) for j in range(N_VARIABLES)], device=AVAILABLE_DEVICE
                    ),
                    consequence=Product((i, 0), device=AVAILABLE_DEVICE),
                )
            )

    def test_create_rule_base(self) -> None:
        """
        Test the creation of a RuleBase object is correct.

        Returns:
            None
        """
        rule_base: RuleBase = RuleBase(self.rules, device=AVAILABLE_DEVICE)
        self.assertEqual(len(self.rules), len(rule_base))
        for expected_rule, actual_rule in zip(self.rules, rule_base.rules):
            self.assertEqual(expected_rule.id, actual_rule.id)
            self.assertEqual(expected_rule.premise.indices, actual_rule.premise.indices)
            self.assertEqual(
                expected_rule.consequence.indices, actual_rule.consequence.indices
            )
            self.assertEqual(expected_rule.premise.device, actual_rule.premise.device)
            self.assertEqual(
                expected_rule.consequence.device, actual_rule.consequence.device
            )
        # the premise_t_norm should be a Product T-Norm (it aggregates the premises)
        self.assertIsInstance(rule_base.premise_t_norm, Product)

    def test_rule_base_output(self) -> None:
        """
        Test the output of the RuleBase object is correct.

        Returns:
            None
        """
        rule_base: RuleBase = RuleBase(self.rules, device=AVAILABLE_DEVICE)
        membership: Membership = TestNAryRelation().test_gaussian_membership()
        rule_base_output: Membership = rule_base(membership)
        expected_output: torch.Tensor = torch.hstack(
            [rule.premise(membership).degrees for rule in self.rules]
        )
        self.assertTrue(torch.allclose(expected_output, rule_base_output.degrees))

    def test_save_rule_base(self) -> None:
        """
        Test the save method of the RuleBase object.

        Returns:
            None
        """
        rule_base: RuleBase = RuleBase(self.rules, device=AVAILABLE_DEVICE)
        rule_base.save(path=Path("test_rule_base"))
        # check if the directory, subdirectories and the files are created
        self.assertTrue(Path("test_rule_base").is_dir())
        for idx, _ in enumerate(self.rules):
            self.assertTrue(Path(f"test_rule_base/rule_{idx}").is_dir())

        loaded_rule_base = RuleBase.load(
            path=Path("test_rule_base"), device=AVAILABLE_DEVICE
        )
        self.assertEqual(len(self.rules), len(loaded_rule_base))
        for expected_rule, actual_rule in zip(rule_base.rules, loaded_rule_base.rules):
            self.assertEqual(expected_rule.id, actual_rule.id)
            self.assertEqual(expected_rule.premise.indices, actual_rule.premise.indices)
            self.assertEqual(
                expected_rule.consequence.indices, actual_rule.consequence.indices
            )
            self.assertEqual(expected_rule.premise.device, actual_rule.premise.device)
            self.assertEqual(
                expected_rule.consequence.device, actual_rule.consequence.device
            )
        # the premise_t_norm should be a Product T-Norm (it aggregates the premises)
        self.assertIsInstance(loaded_rule_base.premise_t_norm, Product)
        # remove the directory
        shutil.rmtree("test_rule_base")
