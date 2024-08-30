"""
A fuzzy logic rule base that contains a list of rules. The rule base is a module that can be
used to perform fuzzy logic inference more efficiently than using a list of rules.
"""

from pathlib import Path
from typing import List, Union, Set, Any

import torch
from natsort import natsorted

from fuzzy.logic.rule import Rule
from fuzzy.logic.control.configurations import Shape
from fuzzy.sets.membership import Membership
from fuzzy.relations.t_norm import TNorm


class RuleBase(torch.nn.Module):
    """
    A fuzzy logic rule base that contains a list of rules. The rule base is a module that can be
    used to perform fuzzy logic inference more efficiently than using a list of rules.
    """

    def __init__(
        self, rules: List[Rule], device: Union[None, torch.device], *args, **kwargs
    ):
        """
        Initialize the RuleBase object.

        Args:
            rules: A list of rules.
            device: The device to move the RuleBase object to; if None, the device is not set, and
            is automatically determined by the rules (they should all be on the same device).
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.rules: List[Rule] = rules
        self.device: Union[None, torch.device] = device
        self.premises: TNorm = self._combine_t_norms(attribute="premise")
        self.consequences: TNorm = self._combine_t_norms(attribute="consequence")

    def __len__(self) -> int:
        return len(self.rules)

    def __hash__(self) -> int:
        return hash(tuple(self.rules))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RuleBase) or len(self.rules) != len(other.rules):
            return False
        # order matters here
        return all(
            rule == other_rule for rule, other_rule in zip(self.rules, other.rules)
        )

    def __getitem__(self, idx: int) -> Rule:
        return self.rules[idx]

    @property
    def shape(self) -> Shape:
        """
        Get the shape of the RuleBase object.

        Returns:
            The shape of the RuleBase object.
        """
        return Shape(
            n_inputs=self.premises.shape[0],
            n_input_terms=self.premises.shape[1],
            n_rules=self.premises.shape[2],
            n_outputs=self.consequences.shape[0],
            n_output_terms=self.consequences.shape[1],
        )

    def _combine_t_norms(self, attribute: str) -> TNorm:
        """
        Combine the TNorms of the rules for the given attribute. The TNorms should be of the same
        type. This greatly speeds up the inference process.

        Args:
            attribute: The attribute to combine the TNorms for (e.g., "premise" or "consequence").

        Returns:
            A TNorm object that combines the TNorms of the rules for the given attribute.
        """
        if attribute not in ["premise", "consequence"]:
            raise ValueError(
                f"Attribute {attribute} is not valid. Use 'premise' or 'consequence'."
            )
        t_norm_types = {type(getattr(rule, attribute)) for rule in self.rules}
        if len(t_norm_types) > 1:
            raise NotImplementedError(
                f"The rules have different TNorm types for {attribute}. This is not supported yet."
            )
        t_norm_type: TNorm = t_norm_types.pop()
        # find the device to move the TNorm to
        devices: Set[torch.device] = {
            getattr(rule, attribute).device for rule in self.rules
        }
        device: Union[None, torch.device] = self.device
        if self.device is None:
            # if the device is not set,
            if len(devices) != 1:
                # cannot determine which device to move the TNorm to
                raise ValueError("The rules are on different devices.")
            # move the TNorm to the device of the rules
            device = devices.pop()

        return t_norm_type(
            *[list(getattr(rule, attribute).indices[0]) for rule in self.rules],
            device=device,
        )

    def save(self, path: Path) -> None:
        """
        Save the RuleBase object to a directory.

        Args:
            path: The path (directory) to save the RuleBase object to.

        Returns:
            None
        """
        if "." in path.name:
            raise ValueError("The path should be a directory, not a file.")
        path.mkdir(parents=True, exist_ok=True)
        for idx, rule in enumerate(self.rules):
            rule.save(path / f"rule_{idx}")

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "RuleBase":
        """
        Load a RuleBase object from a directory.

        Args:
            path: The path (directory) to load the RuleBase object from.
            device: The device to move the RuleBase object to.

        Returns:
            A RuleBase object.
        """
        rules = []
        for rule_path in natsorted(path.iterdir()):  # order by rule number
            if rule_path.is_dir():
                rules.append(Rule.load(rule_path, device))
        return cls(rules, device=device)

    def forward(self, membership: Membership) -> Membership:
        """
        Forward pass through the rule base.

        Args:
            membership: The membership values of the input elements.

        Returns:
            The membership values of the rule base given the elements after applying the t-norm(s).
        """
        return self.premises(membership)
