"""
This directory contains the implementation of the Rule class, which is used to represent
fuzzy logic rules.
"""

from typing import Union

from fuzzy.relations.continuous.n_ary import NAryRelation, Compound


class Rule:
    """
    A fuzzy logic rule that contains the premise and the consequence. The premise is a n-ary
    fuzzy relation compound, usually involving a t-norm (e.g., minimum, product). The consequence
    is a list of tuples, where the first element is the index of the output variable and the
    second element is the index of the output linguistic term.
    """

    next_id: int = 0

    def __init__(
        self,
        premise: Union[NAryRelation, Compound],
        consequence: NAryRelation,
    ):
        self.premise = premise
        self.consequence = consequence
        self.id = Rule.next_id
        Rule.next_id += 1

    def __str__(self):
        return f"IF {self.premise} THEN {self.consequence}"
