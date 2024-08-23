"""
Test extensions of the BaseDiscreteFuzzySet, such as AlphaCut or SpecialFuzzySet.
"""

import unittest

from fuzzy.relations.discrete.snorm import StandardUnion
from fuzzy.relations.discrete.tnorm import StandardIntersection
from fuzzy.relations.discrete.extension import AlphaCut, SpecialFuzzySet
from examples.discrete.student import known, learned


class TestAlphaCut(unittest.TestCase):
    """
    Test the alpha cut operation works as intended.
    """

    def test_alpha_cut(self) -> None:
        """
        Test the AlphaCut operation (class) works as intended.

        Returns:
            None
        """
        alpha_cut = AlphaCut(known(), 0.6, "AlphaCut")
        assert alpha_cut.degree(80) == alpha_cut.alpha


class TestSpecialFuzzySet(unittest.TestCase):
    """
    Test that we can create a special fuzzy set.
    """

    def test_special_fuzzy_set(self) -> None:
        """
        Test the SpecialFuzzySet class works as intended.

        Returns:
            None
        """
        special_fuzzy_set = SpecialFuzzySet(known(), 0.5, "Special")
        assert special_fuzzy_set.height() == special_fuzzy_set.alpha
        assert special_fuzzy_set.degree(80) == special_fuzzy_set.alpha


class TestDiscreteFuzzyRelation(unittest.TestCase):
    """
    Test the discrete fuzzy relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terms = [known(), learned()]

    def test_standard_intersection(self) -> None:
        """
        Test the standard intersection of two fuzzy sets.

        Returns:
            None
        """
        standard_intersection = StandardIntersection(fuzzy_sets=self.terms)
        assert standard_intersection.degree(87) == 0.4

    def test_standard_union(self) -> None:
        """
        Test the standard union of two fuzzy sets.

        Returns:
            None
        """
        standard_union = StandardUnion(fuzzy_sets=self.terms)
        assert standard_union.degree(87) == 0.6
