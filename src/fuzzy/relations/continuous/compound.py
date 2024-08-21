"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

from typing import List

import torch

from fuzzy.sets.continuous.membership import Membership
from fuzzy.relations.continuous.n_ary import NAryRelation


class Compound(torch.nn.Module):
    """
    This class represents an n-ary compound relation, where it expects at least 1 or more
    instance of NAryRelation.
    """

    def __init__(self, *relations: NAryRelation, **kwargs):
        """
        Initialize the compound relation with the given n-ary relation(s).

        Args:
            relation: The n-ary compound relation.
        """
        super().__init__(**kwargs)
        # store the relations as a module list (as they are also modules)
        self.relations = torch.nn.ModuleList(relations)

    def forward(self, membership: Membership) -> Membership:
        """
        Apply the compound n-ary relation to the given membership values.

        Args:
            membership: The membership values to apply the compound n-ary relation to.

        Returns:
            The stacked output of the compound n-ary relation; ready for subsequent follow-up.
        """
        # apply the compound n-ary relation to the membership values
        memberships: List[Membership] = [
            relation(membership=membership) for relation in self.relations
        ]
        degrees: torch.Tensor = torch.cat(
            [membership.degrees for membership in memberships], dim=-1
        ).unsqueeze(dim=-1)
        # create a new mask that accounts for the different masks for each relation
        mask = torch.stack([relation.applied_mask for relation in self.relations])
        return Membership(elements=membership.elements, degrees=degrees, mask=mask)
