"""
This directory contains the implementation of the Rule class, which is used to represent
fuzzy logic rules.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Union, Type, Any

import torch

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
        premise: Union[NAryRelation, Type[NAryRelation]],
        consequence: Union[NAryRelation, Type[NAryRelation]],
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Rule):
            return False
        return self.premise == other.premise and self.consequence == other.consequence

    def save(self, path: Path) -> None:
        """
        Save the Rule object to a directory.

        Args:
            path: The path (directory) to save the Rule object to.

        Returns:
            None
        """
        if "." in path.name:
            raise ValueError(
                f"The path to save the {self.__class__} must not have a file extension, "
                f"but got {path.name}"
            )

        path.mkdir(parents=True, exist_ok=True)
        self.premise.save(
            path / "premise.pt"
        )  # may result in a .pt file or a directory
        self.consequence.save(
            path / "consequence.pt"
        )  # may result in a .pt file or a directory
        # save the ID of the Rule
        with open(path / "id.txt", "w", encoding="utf-8") as f:
            f.write(str(self.id))

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "Rule":
        """
        Load the Rule object from a directory.

        Args:
            path: The path (directory) to load the Rule object from.
            device: The device to put the Rule object on.

        Returns:
            The Rule object.
        """
        premise_location: str = "premise"
        consequence_location: str = "consequence"
        if (path / "premise.pt").exists():
            premise_location += ".pt"  # add the file extension; it is stored in a file
        if (path / "consequence.pt").exists():
            consequence_location += (
                ".pt"  # add the file extension; it is stored in a file
            )

        premise = NAryRelation.load(path / premise_location, device=device)
        consequence = NAryRelation.load(path / consequence_location, device=device)
        with open(path / "id.txt", "r", encoding="utf-8") as f:
            rule_id = int(f.read())
        obj = cls(premise, consequence)
        obj.id = rule_id  # this may cause a bug if the ID is not unique
        cls.next_id -= 1  # decrement the next ID to counter the constructor's increment
        return obj
