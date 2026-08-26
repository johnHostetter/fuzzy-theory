"""
Implements the LinguisticVariables class to store the input and output fuzzy sets
for fuzzy logic rule(s).
"""

from dataclasses import dataclass
from typing import List

from fuzzy.sets.abstract import FuzzySet


@dataclass(eq=False)
class LinguisticVariables:
    """
    The LinguisticVariables class contains the input and output fuzzy sets for fuzzy logic rule(s).

    eq=False keeps this class hashable (identity-based, via the default
    object.__hash__): a plain @dataclass auto-generates __eq__, which disables
    __hash__ unless explicitly restored, and inputs/targets are Lists (of
    FuzzySet objects) anyway - not something a value-based __eq__/__hash__
    could support without also changing them to an immutable type. Confirmed
    nothing in fuzzy-theory, fuzzy-ml, or PySoft currently compares two
    LinguisticVariables instances with ==/!=, so this has no observable
    behavior change.
    """

    inputs: List[FuzzySet]
    targets: List[FuzzySet]

    def __post_init__(self):
        pass  # no post-initialization needed for this dataclass
