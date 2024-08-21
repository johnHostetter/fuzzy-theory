"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

from typing import Union, Tuple, List

import igraph
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

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *indices: Union[Tuple[int, int], List[Tuple[int, int]]],
        device: torch.device,
        nan_replacement: float = 0.0,
        **kwargs,
    ):
        """
        Apply an n-ary relation to the indices (i.e., relation's matrix) on the provided device.

        Args:
            items: The 2-tuple indices to apply the n-ary relation to (e.g., (0, 1), (1, 0)).
            device: The device to use for the relation.
            nan_replacement: The value to use when a value is missing in the relation (i.e., nan);
                this is useful for when input to the relation is not complete. Default is 0.0
                (penalize), a value of 1.0 would ignore missing values (i.e., do not penalize).
        """
        super().__init__(**kwargs)
        self.matrix = None  # this will be created later (via self._rebuild)
        self.graph = None  # this will be created later (via self._rebuild)
        self.device: torch.device = device
        if nan_replacement not in [0.0, 1.0]:
            raise ValueError("The nan_replacement must be either 0.0 or 1.0.")
        self.nan_replacement: float = nan_replacement
        self.indices: List[List[Tuple[int, int]]] = []

        if not isinstance(indices[0], list):
            indices = [indices]

        # this scenario is for when we have multiple compound indices that use the same relation
        # this is useful for computational efficiency (i.e., not having to use a for loop)
        self._coo_matrix: List[sps._coo.coo_matrix] = []
        self._original_shape: List[Tuple[int, int]] = []
        for relation_indices in indices:
            if len(set(relation_indices)) < len(relation_indices):
                raise ValueError(
                    "The indices must be unique for the relation to be well-defined."
                )
            coo_matrix = self.convert_indices_to_matrix(relation_indices)
            self._original_shape.append(coo_matrix.shape)
            self._coo_matrix.append(coo_matrix)
        # now convert to a list of matrices
        max_var = max(t[0] for t in self._original_shape)
        max_term = max(t[1] for t in self._original_shape)
        self.indices.extend(indices)
        self._rebuild(*(max_var, max_term))

    @staticmethod
    def convert_indices_to_matrix(indices) -> sps._coo.coo_matrix:
        """
        Convert the given indices to a COO matrix.

        Args:
            indices: The indices where a '1' will be placed at each index.

        Returns:
            The COO matrix with a '1' at each index.
        """
        data = np.ones(len(indices))  # a '1' indicates a relation exists
        row, col = zip(*indices)
        return sps.coo_matrix((data, (row, col)), dtype=np.int8)

    def create_ndarray(self, max_var: int, max_term: int) -> None:
        """
        Make (or update) the numpy matrix from the COO matrices.

        Args:
            max_var: The maximum number of variables.
            max_term: The maximum number of terms.

        Returns:
            None
        """
        matrices = []
        for coo_matrix in self._coo_matrix:
            # first resize
            coo_matrix.resize(max_var, max_term)
            matrices.append(coo_matrix.toarray())
        # make a new axis and stack long that axis
        self.matrix: np.ndarray = np.stack(matrices).swapaxes(0, 1).swapaxes(1, 2)

    def create_igraph(self) -> None:
        """
        Create the graph representation of the relation(s).

        Returns:
            None
        """
        graphs: List[igraph.Graph] = []
        for relation in self.indices:
            # create a directed (mode="in") star graph with the relation as the center (vertex 0)
            graphs.append(igraph.Graph.Star(n=len(relation) + 1, mode="in", center=0))
            # relation vertices are the first vertices in the graph
            relation_vertex: igraph.Vertex = graphs[-1].vs.find(0)  # located at index 0
            # set item and tags for the relation vertex for easy retrieval; name is for graph union
            (
                relation_vertex["name"],
                relation_vertex["item"],
                relation_vertex["tags"],
            ) = (hash(self) + hash(tuple(relation)), self, {"relation"})
            # anchor vertices are the var-term pairs that are involved in the relation vertex
            anchor_vertices: List[igraph.Vertex] = relation_vertex.predecessors()
            # set anchor vertices' item and tags for easy retrieval; name is for graph union
            for anchor_vertex, index_pair in zip(anchor_vertices, relation):
                anchor_vertex["name"], anchor_vertex["item"], anchor_vertex["tags"] = (
                    index_pair,
                    index_pair,
                    {"anchor"},
                )
        self.graph = igraph.union(graphs, byname=True)

    def _rebuild(self, *shape) -> None:
        """
        Rebuild the relation's matrix and graph.

        Args:
            shape: The new shape of the n-ary fuzzy relation; assuming shape is (max_var, max_term).

        Returns:
            None
        """
        # re-create the self.matrix
        self.create_ndarray(shape[0], shape[1])
        # re-create the self.graph
        self.create_igraph()
        # update the self.mask to reflect the new shape
        # this mask is used to zero out the values that are not part of the relation
        self.mask = torch.tensor(self.matrix, dtype=torch.float32, device=self.device)

    def resize(self, *shape) -> None:
        """
        Resize the matrix in-place to the given shape, and then rebuild the relations' members.

        Args:
            shape: The new shape of the matrix.

        Returns:
            None
        """
        for coo_matrix in self._coo_matrix:
            coo_matrix.resize(*shape)
        self._rebuild(*shape)

    def apply_mask(self, membership: Membership) -> torch.Tensor:
        """
        Apply the n-ary relation's mask to the given memberships.

        Args:
            membership: The membership values to apply the minimum n-ary relation to.

        Returns:
            The masked membership values (zero may or may not be a valid degree of truth).
        """
        membership_shape: torch.Size = membership.degrees.shape
        if self.matrix.shape[:-1] != membership_shape[1:]:
            # if len(membership_shape) > 2:
            # this is for the case where masks have been stacked due to compound relations
            membership_shape = membership_shape[1:]  # get the last two dimensions
            self.resize(*membership_shape)
        # select memberships that are not zeroed out (i.e., involved in the relation)
        after_mask = membership.degrees.unsqueeze(dim=-1) * self.mask.unsqueeze(0)
        # the complement mask adds zeros where the mask is zero, these are not part of the relation
        # nan_to_num is used to replace nan values with the nan_replacement value (often not needed)
        return (
            (after_mask + (1 - self.mask))
            .prod(dim=2, keepdim=False)
            .nan_to_num(self.nan_replacement)
        )

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
        ).unsqueeze(dim=-1)
        # create a new mask that accounts for the different masks for each relation
        mask = torch.stack([relation.mask for relation in self.relations])
        return Membership(elements=membership.elements, degrees=degrees, mask=mask)
