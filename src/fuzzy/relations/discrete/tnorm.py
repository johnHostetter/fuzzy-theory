"""
Implements the t-norm fuzzy relations.
"""

from typing import List

from fuzzy.sets.discrete import DiscreteFuzzySet
from fuzzy.relations.discrete.extension import DiscreteFuzzyRelation


class StandardIntersection(DiscreteFuzzyRelation):
    """
    A standard intersection of one or more ordinary fuzzy sets.
    """

    def __init__(self, fuzzy_sets: List[DiscreteFuzzySet], name=None):
        """
        Parameters
        ----------
        fuzzy_sets : 'list'
            A list of elements each of type OrdinaryDiscreteFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        DiscreteFuzzyRelation.__init__(self, formulas=fuzzy_sets, name=name, mode=min)
        self.fuzzy_sets = fuzzy_sets
