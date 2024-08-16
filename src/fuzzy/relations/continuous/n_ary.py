"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

from typing import Tuple, List

import torch
import numpy as np
import scipy.sparse as sps

from fuzzy.sets.continuous.membership import Membership


class NAryRelation(torch.nn.Module):
    """
    This class represents an n-ary fuzzy relation. An n-ary fuzzy relation is a relation that takes
    n arguments and returns a (float) value. This class is useful for representing fuzzy relations
    that take multiple arguments, such as a t-norm that takes two or more arguments and returns a
    truth value.
    """

    def __init__(self, *indices: Tuple[int, int], device: torch.device, **kwargs):
        """
        Apply an n-ary relation to the indices (i.e., relation's matrix) on the provided device.

        Args:
            items: The 2-tuple indices to apply the n-ary relation to (e.g., (0, 1), (1, 0)).
            device: The device to use for the relation.
        """
        super().__init__(**kwargs)
        self.indices = indices
        data = np.ones(len(indices))  # a '1' indicates a relation exists
        row, col = zip(*indices)

        self._coo_matrix: sps.sparse._coo.matrix = sps.coo_matrix(
            (data, (row, col)), dtype=np.int8
        )
        self._original_shape: Tuple[int, int] = self._coo_matrix.shape
        # this mask is used to zero out the values that are not part of the relation
        self.mask: torch.Tensor = torch.tensor(
            self._coo_matrix.toarray(), dtype=torch.float32, device=device
        )
        # matrix size can increase (in-place) for more potential rows (vars) and columns (terms)
        # self._coo_matrix.resize(
        #     self._coo_matrix.shape[0] + 1, self._coo_matrix.shape[1] + 1
        # )
        # we can create a graph from the adjacency matrix
        # g = igraph.Graph.Adjacency(self._coo_matrix)

    def resize(self, *shape):
        """
        Resize the matrix in-place to the given shape.

        Args:
            shape: The new shape of the matrix.
        """
        # resize the COO matrix in-place
        self._coo_matrix.resize(*shape)
        # update the mask to reflect the new shape
        self.mask = torch.tensor(
            self._coo_matrix.toarray(), dtype=torch.float32, device=self.mask.device
        )

    def apply_mask(self, membership: Membership) -> torch.Tensor:
        """
        Apply the n-ary relation's mask to the given memberships.

        Args:
            membership: The membership values to apply the minimum n-ary relation to.

        Returns:
            The masked membership values (zero may or may not be a valid degree of truth).
        """
        membership_shape: torch.Size = membership.mask.shape
        if self._coo_matrix.shape != membership_shape:
            if len(membership_shape) > 2:
                # this is for the case where masks have been stacked due to compound relations
                membership_shape = membership_shape[-2:]  # get the last two dimensions
            self.resize(*membership_shape)
        # select the membership values that are not zeroed out (i.e., involved in the relation)
        after_mask = membership.degrees * self.mask
        return after_mask.sum(dim=1, keepdim=True)  # drop the zeroed out values

    def forward(self, membership: Membership) -> torch.Tensor:
        """
        Apply the n-ary relation to the given memberships.

        Args:
            membership: The membership values to apply the minimum n-ary relation to.

        Returns:
            The minimum membership value, according to the n-ary relation (i.e., which truth values
            to actually consider).
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__} has no defined forward function. Please create a class "
            f"and inherit from {self.__class__.__name__}, or use a predefined class."
        )


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
        )
        # create a new mask that accounts for the different masks for each relation
        mask = torch.stack([relation.mask for relation in self.relations])
        return Membership(elements=membership.elements, degrees=degrees, mask=mask)
