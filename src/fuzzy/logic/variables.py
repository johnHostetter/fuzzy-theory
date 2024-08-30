"""
Implements the LinguisticVariables class to store the input and output fuzzy sets
for fuzzy logic rule(s).
"""

from typing import List
from dataclasses import dataclass

from fuzzy.sets.abstract import FuzzySet


@dataclass
class LinguisticVariables:
    """
    The LinguisticVariables class contains the input and output fuzzy sets for fuzzy logic rule(s).
    """

    inputs: List[FuzzySet]
    targets: List[FuzzySet]

    def __post_init__(self):
        pass  # no post-initialization needed for this dataclass
