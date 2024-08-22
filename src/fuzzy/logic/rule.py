"""
This directory contains the implementation of the Rule class, which is used to represent
fuzzy logic rules.
"""

from dataclasses import dataclass

from fuzzy.relations.continuous.n_ary import NAryRelation


@dataclass
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
        premise: NAryRelation,
        consequence: NAryRelation,
    ):
        if len(premise.indices) > 1 or len(consequence.indices) > 1:
            raise ValueError("Only unary relations are supported to create a Rule.")
        self.premise = premise
        self.consequence = consequence
        self.id = Rule.next_id
        Rule.next_id += 1

    def __str__(self) -> str:
        return f"IF {self.premise} THEN {self.consequence}"

    def __hash__(self) -> int:
        return hash(self.premise) + hash(self.consequence) + hash(self.id)
