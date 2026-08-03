"""
Test properties of KnowledgeBase, such as how attributes of observations or granules are stored.
"""

import shutil
import unittest
from pathlib import Path
from typing import List

import igraph
import numpy as np
import torch

from fuzzy.logic.control.configurations.data import GranulationLayers, Shape
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.rulebase import RuleBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import TNorm
from fuzzy.sets.group import FuzzySetGroup
from fuzzy.sets.impl import Lorentzian

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestKnowledgeBase(unittest.TestCase):
    """
    Test the KnowledgeBase class such as checking data attributes are correctly added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linguistic_variables = LinguisticVariables(
            inputs=[
                Lorentzian(
                    centers=np.array([0.0, 0.5, 1.0]),
                    widths=np.array([0.5, 0.75, 1.0]),
                    device=AVAILABLE_DEVICE,
                ),
                Lorentzian(
                    centers=np.array([1.0, 1.5, 2.0, 2.5]),
                    widths=np.array([0.1, 0.15, 0.2, 0.25]),
                    device=AVAILABLE_DEVICE,
                ),
            ],
            targets=[
                Lorentzian(
                    centers=np.array([0.0, 0.5, 1.0, 2.0, 2.5]),
                    widths=np.array([0.5, 0.75, 1.0, 0.2, 0.25]),
                    device=AVAILABLE_DEVICE,
                ),
                Lorentzian(
                    centers=np.array([1.0, 1.5]),
                    widths=np.array([0.1, 0.15]),
                    device=AVAILABLE_DEVICE,
                ),
            ],
        )
        self.rules: List[Rule] = [
            Rule(
                premise=TNorm((0, 0), (1, 0), device=AVAILABLE_DEVICE),
                consequence=TNorm((0, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=TNorm((0, 1), (1, 0), device=AVAILABLE_DEVICE),
                consequence=TNorm((0, 1), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=TNorm((0, 2), (1, 0), device=AVAILABLE_DEVICE),
                consequence=TNorm((0, 2), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=TNorm((1, 0), (1, 2), device=AVAILABLE_DEVICE),
                consequence=TNorm((1, 0), device=AVAILABLE_DEVICE),
            ),
        ]

    def test_attributes(self) -> None:
        """
        Test that attributes, and their corresponding references (i.e., relations in KnowledgeBase),
        are appropriately added to the Knowledgebase graph.

        Returns:
            None
        """
        universe = [f"x{i}" for i in range(1, 11)]
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(universe, tags="element")
        # group up data points that share the same value for the respective
        # attribute
        attribute_groupings = {
            # for example, 'x1', 'x2', 'x10' have the same value for attribute
            # 'a'
            "a": (
                {"x1", "x2", "x10"},
                {"x4", "x6", "x8"},
                {"x3"},
                {"x5", "x7"},
                {"x9"},
            ),
            # also, 'x2', 'x4' have the same value for attribute 'b'
            "b": ({"x1", "x3", "x7"}, {"x2", "x4"}, {"x5", "x6", "x8"}),
            # and so on
            "c": ({"x1", "x5"}, {"x2", "x6"}, {"x3", "x4", "x7", "x8"}),
            "d": ({"x2", "x7", "x8"}, {"x1", "x3", "x4", "x5", "x6"}),
        }
        knowledge_base.add_parent_relation("a", attribute_groupings["a"])
        knowledge_base.add_parent_relation("b", attribute_groupings["b"])
        knowledge_base.add_parent_relation("c", attribute_groupings["c"])
        knowledge_base.add_parent_relation("d", attribute_groupings["d"])

        # pylint: disable=fixme
        # TODO: Fix this; attributes are no longer stored
        # assert knowledge_base.attributes('x1') == {'a': 0, 'b': 0, 'c': 0, 'd': 1}
        # assert knowledge_base.attributes('x2') == {'a': 0, 'b': 1, 'c': 1, 'd': 0}
        # assert knowledge_base.attributes('x3') == {'a': 2, 'b': 0, 'c': 2, 'd': 1}
        # assert knowledge_base.attributes('x4') == {'a': 1, 'b': 1, 'c': 2, 'd': 1}
        # assert knowledge_base.attributes('x5') == {'a': 3, 'b': 2, 'c': 0, 'd': 1}
        # assert knowledge_base.attributes('x6') == {'a': 1, 'b': 2, 'c': 1, 'd': 1}
        # assert knowledge_base.attributes('x7') == {'a': 3, 'b': 0, 'c': 2, 'd': 0}
        # assert knowledge_base.attributes('x8') == {'a': 1, 'b': 2, 'c': 2, 'd': 0}
        # assert knowledge_base.attributes('x9') == {'a': 4}
        # assert knowledge_base.attributes('x10') == {'a': 0}

    def test_empty_knowledge_base(self) -> None:
        """
        Test that if we create a KnowledgeBase with no arguments, that it is indeed empty.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        self.assertEqual(knowledge_base.graph.vcount(), 0)
        self.assertEqual(knowledge_base.graph.ecount(), 0)

    def test_create_knowledge_base(self) -> None:
        """
        Test that if we create a KnowledgeBase with arguments, that it is not empty.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=self.linguistic_variables,
            rules=self.rules,
        )
        knowledge_base.set_granules([1, 2, 3, 4, 5], tags="element")
        self.assertEqual(32, knowledge_base.graph.vcount())
        self.assertEqual(24, knowledge_base.graph.ecount())

        # check the properties of the KnowledgeBase are as we expect
        expected_shape = Shape(
            n_inputs=2,
            n_input_terms=3,  # it only counts the number of input terms involved via rules
            n_rules=4,
            n_outputs=2,
            n_output_terms=3,  # it only counts the number of output terms involved via rules
        )
        self.assertEqual(expected_shape, knowledge_base.shape)

        # check that the rules can easily be retrieved (in the same order as
        # they were added)
        self.assertEqual(self.rules, knowledge_base.rules)

        expected_rule_base = RuleBase(
            rules=self.rules, device=None
        )  # # use the same behavior as KnowledgeBase does when device is None
        self.assertEqual(expected_rule_base, knowledge_base.rule_base)

        # check that the engine can be retrieved (this comes from RuleBase, but
        # it is a shortcut)
        self.assertEqual(
            expected_rule_base.premises,
            knowledge_base.engine,
        )

        # check that the stacked granule representation can easily be retrieved
        expected_granulation_layers: GranulationLayers = GranulationLayers(
            input=FuzzySetGroup(
                modules_list=[
                    Lorentzian.stack(
                        self.linguistic_variables.inputs)], ), output=FuzzySetGroup(
                modules_list=[
                    Lorentzian.stack(
                        self.linguistic_variables.targets)], ), )
        self.assertEqual(
            expected_granulation_layers,
            knowledge_base.granulation_layers,
        )

        # check individual premise granules can be retrieved (from the
        # Knowledgebase.graph)
        actual_granules: igraph.VertexSeq = knowledge_base.get_granules(
            tags="premise")
        self.assertEqual(
            self.linguistic_variables.inputs,
            actual_granules["item"])

        # check individual consequence granules can be retrieved (from the
        # Knowledgebase.graph)
        actual_granules: igraph.VertexSeq = knowledge_base.get_granules(
            tags="consequence"
        )
        self.assertEqual(
            self.linguistic_variables.targets,
            actual_granules["item"])

        # getting granules with nonexistent tag results in no matches
        actual_granules: igraph.VertexSeq = knowledge_base.get_granules(
            tags="nonexistent"
        )
        self.assertEqual(0, len(actual_granules))

        # count of premise terms is correct for each variable
        premise_terms_dim = np.array([3, 4], dtype=np.int32)
        self.assertTrue(
            np.allclose(
                premise_terms_dim,
                knowledge_base.intra_dimensions("premise")))

        # count of consequence terms is correct for each variable
        consequence_terms_dim = np.array([5, 2], dtype=np.int32)
        self.assertTrue(
            np.allclose(
                consequence_terms_dim,
                knowledge_base.intra_dimensions("consequence")))

        # check that we can save and load the KnowledgeBase
        save_dir: Path = Path(__file__).parent / "knowledge_base"
        knowledge_base.save(path=save_dir)

        loaded_knowledge_base = KnowledgeBase.load(
            path=save_dir, device=AVAILABLE_DEVICE
        )

        # remove the directory
        shutil.rmtree(save_dir)
        self.assertFalse(save_dir.exists())

        # check that the loaded KnowledgeBase is the same as the original
        self.assertEqual(
            knowledge_base.attribute_table,
            loaded_knowledge_base.attribute_table)

        for vertex, loaded_vertex in zip(
            knowledge_base.graph.vs, loaded_knowledge_base.graph.vs
        ):
            self.assertEqual(vertex.index, loaded_vertex.index)
            for attribute in vertex.attributes():
                self.assertEqual(vertex[attribute], loaded_vertex[attribute])

        for edge, loaded_edge in zip(
            knowledge_base.graph.es, loaded_knowledge_base.graph.es
        ):
            self.assertEqual(edge.attributes(), loaded_edge.attributes())

    def test_granulation_layers_ambiguous_group_raises(self) -> None:
        """
        granulation_layers must raise a clear ValueError if more than one group vertex
        matches the same layer's tags, rather than silently picking one - this can happen
        if add_hypercube() is (mistakenly) called more than once for the same tag set.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=self.linguistic_variables,
            rules=self.rules,
        )
        # add a second "premise" group vertex directly (bypassing add_hypercube, whose
        # own granule-selection would otherwise pick up the first hypercube itself,
        # since select_by_tags matches by tag subset)
        duplicate_hypercube = FuzzySetGroup(
            modules_list=[Lorentzian.stack(self.linguistic_variables.inputs)],
        )
        knowledge_base.graph.add_vertex(
            item=duplicate_hypercube, tags={"premise", "group"}
        )
        with self.assertRaises(ValueError):
            _ = knowledge_base.granulation_layers

    def test_add_hypercube_with_string_tag(self) -> None:
        """
        add_hypercube() must accept a single string tag (not just a set of tags),
        normalizing it internally.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(
            self.linguistic_variables.inputs,
            tags="premise")
        vertex = knowledge_base.add_hypercube(tags="premise")
        self.assertIsNotNone(vertex)
        self.assertIsInstance(vertex["item"], FuzzySetGroup)

    def test_add_hypercube_with_no_matching_granules(self) -> None:
        """
        add_hypercube() must return None (rather than raising) when no granules match
        the given tags.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        self.assertIsNone(knowledge_base.add_hypercube(tags="nonexistent"))

    def test_create_with_no_rules_warns(self) -> None:
        """
        create() must warn (not raise) when given an empty list of rules.

        Returns:
            None
        """
        with self.assertWarns(UserWarning):
            knowledge_base = KnowledgeBase.create(
                linguistic_variables=self.linguistic_variables, rules=[]
            )
        self.assertEqual(0, len(knowledge_base.rules))

    def test_create_rejects_multi_index_relation(self) -> None:
        """
        create()'s own check for a multi-index premise/consequence is redundant with
        Rule.__init__'s validation for any normally-constructed Rule, but remains
        reachable (and must still raise clearly) if a Rule's relation is mutated to add
        more indices after construction.

        Returns:
            None
        """
        rule = Rule(
            premise=TNorm((0, 0), (1, 0), device=AVAILABLE_DEVICE),
            consequence=TNorm((0, 0), device=AVAILABLE_DEVICE),
        )
        # bypass Rule.__init__'s validation by mutating the premise directly
        rule.premise.indices.append(((0, 1), (1, 1)))
        with self.assertRaises(ValueError):
            KnowledgeBase.create(
                linguistic_variables=self.linguistic_variables, rules=[rule]
            )

    def test_attributes_currently_broken(self) -> None:
        """
        Known, pre-existing limitation (not fixed here - see the conversation that added
        this test): KnowledgeBase.attributes() looks up self.attribute_table, but nothing
        currently populates that table - RoughDecisions.add_parent_relation() (from the
        external rough-theory package) no longer stores into it. The real assertions for
        this in test_attributes() above have been commented out since before this test was
        added, with a 'TODO: Fix this; attributes are no longer stored' note. This test
        documents the current (broken) behavior so a future fix to attribute_table's
        population is visible as an intentional change here, rather than an unnoticed one.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(["x1", "x2"], tags="element")
        knowledge_base.add_parent_relation("a", ({"x1", "x2"},))
        self.assertEqual({}, dict(knowledge_base.attribute_table))
        with self.assertRaises(KeyError):
            knowledge_base.attributes("x1")

        # an element with no equivalence classes at all (no add_parent_relation ever
        # touched it) doesn't hit attribute_table, so it correctly returns an empty dict -
        # confirming the KeyError above is specifically about the unpopulated table, not
        # attributes() being broken in every case
        untouched_knowledge_base = KnowledgeBase()
        untouched_knowledge_base.set_granules(["y1"], tags="element")
        self.assertEqual({}, untouched_knowledge_base.attributes("y1"))
