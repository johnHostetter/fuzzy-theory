"""
Classes for representing n-ary fuzzy relations, such as t-norms and t-conorms. These relations
are used to combine multiple membership values into a single value. The n-ary relations (of
differing types) can then be combined into a compound relation.
"""

from pathlib import Path
from typing import Union, Tuple, List, MutableMapping, Any

import igraph
import torch
import numpy as np
import scipy.sparse as sps

from fuzzy.sets.continuous.membership import Membership
from fuzzy.utils import check_path_to_save_torch_module, TorchJitModule
from .linkage import GroupedLinks, BinaryLinks


class NAryRelation(TorchJitModule):
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
        grouped_links: Union[None, GroupedLinks] = None,
        nan_replacement: float = 0.0,
        **kwargs,
    ):
        """
        Apply an n-ary relation to the indices (i.e., relation's matrix) on the provided device.

        Args:
            items: The 2-tuple indices to apply the n-ary relation to (e.g., (0, 1), (1, 0)).
            device: The device to use for the relation.
            grouped_links: The end-user can provide the links to use for the relation; this is
                useful for when the links are already created and the user wants to use them, or
                for a relation that requires more complex setup. Default is None.
            nan_replacement: The value to use when a value is missing in the relation (i.e., nan);
                this is useful for when input to the relation is not complete. Default is 0.0
                (penalize), a value of 1.0 would ignore missing values (i.e., do not penalize).
        """
        super().__init__(**kwargs)
        self.device: torch.device = device
        if nan_replacement not in [0.0, 1.0]:
            raise ValueError("The nan_replacement must be either 0.0 or 1.0.")
        self.nan_replacement: float = nan_replacement

        self.matrix = None  # created later (via self._rebuild)
        self.grouped_links: Union[None, GroupedLinks] = (
            None  # created later (via self._rebuild)
        )
        self.applied_mask: Union[None, torch.Tensor] = (
            None  # created later (at the end of the constructor)
        )
        self.graph = None  # will be created later (via self._rebuild)
        self.indices: List[List[Tuple[int, int]]] = []

        if not indices:  # indices are not given
            if grouped_links is None:
                raise ValueError(
                    "At least one set of indices must be provided, or GroupedLinks must be given."
                )
            # note that many features are not available when using grouped_links
            self.grouped_links = grouped_links
        else:  # indices are given
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

        # test if the relation is well-defined & build it
        # the last index, -1, is the relation index; first 2 are (variable, term) indices
        membership_shape: torch.Size = self.grouped_links.shape[:-1]
        # but we also need to include a dummy batch dimension (32) for the grouped_links
        membership_shape: torch.Size = torch.Size([32] + list(membership_shape))
        self.applied_mask = self.grouped_links(
            Membership(
                elements=torch.empty(membership_shape, device=self.device),
                degrees=torch.empty(membership_shape, device=self.device),
                mask=torch.empty(membership_shape, device=self.device),
            )
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.indices})"

    def __hash__(self) -> int:
        return hash(self.applied_mask) + hash(self.nan_replacement) + hash(self.device)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NAryRelation) or not isinstance(self, type(other)):
            return False
        if self.applied_mask is None:
            return (
                self.indices == other.indices
                and self.nan_replacement == other.nan_replacement
            )
        return (
            self.applied_mask.shape == other.applied_mask.shape
            and torch.allclose(self.applied_mask, other.applied_mask)
            and self.nan_replacement == other.nan_replacement
        )

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

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the n-ary relation to a dictionary given a path.

        Args:
            path: The (requested) path to save the n-ary relation. This may be modified to ensure
            all necessary files are saved (e.g., it may be turned into a directory instead).

        Returns:
            The dictionary representation of the n-ary relation.
        """
        check_path_to_save_torch_module(path)
        state_dict: MutableMapping = self.state_dict()
        state_dict["nan_replacement"] = self.nan_replacement
        state_dict["class_name"] = self.__class__.__name__

        if len(self.indices) == 0:
            dir_path = path.parent / path.name.split(".")[0]
            # we will rebuild from the grouped_links, so we do not need to save the indices
            grouped_links_dir: Path = dir_path / "grouped_links"
            self.grouped_links.save(path=grouped_links_dir)
            state_dict["grouped_links"] = (
                grouped_links_dir  # save the path to the grouped_links
            )
            torch.save(state_dict, dir_path / "state_dict.pt")
        else:
            # we will rebuild from the indices, so we do not need to save the grouped_links
            state_dict["indices"] = (
                self.indices if len(self.indices) > 1 else self.indices[0]
            )
            torch.save(state_dict, path)

        return state_dict

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "NAryRelation":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Returns:
            None
        """
        if path.is_file() and path.suffix == ".pt":
            # load from indices
            state_dict: MutableMapping = torch.load(path, weights_only=False)
        else:
            # load from grouped_links, path is a directory
            state_dict: MutableMapping = torch.load(
                path / "state_dict.pt", weights_only=False
            )
        nan_replacement = state_dict.pop("nan_replacement")
        class_name = state_dict.pop("class_name")

        if "indices" in state_dict:
            indices = state_dict.pop("indices")
            return cls.get_subclass(class_name)(
                *indices,
                device=device,
                nan_replacement=nan_replacement,
            )
        grouped_links: Path = state_dict.pop("grouped_links")
        return cls.get_subclass(class_name)(
            device=device,
            grouped_links=GroupedLinks.load(grouped_links, device=device),
            nan_replacement=nan_replacement,
        )

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
        # update the self.grouped_links to reflect the new shape
        # these links are used to zero out the values that are not part of the relation
        self.grouped_links = GroupedLinks(
            modules_list=[BinaryLinks(links=self.matrix, device=self.device)]
        )

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
        if self.applied_mask.shape[:-1] != membership_shape[1:]:
            # if len(membership_shape) > 2:
            # this is for the case where masks have been stacked due to compound relations
            membership_shape = membership_shape[1:]  # get the last two dimensions
            self.resize(*membership_shape)
        # select memberships that are not zeroed out (i.e., involved in the relation)
        self.applied_mask: torch.Tensor = self.grouped_links(membership=membership)
        after_mask = membership.degrees.unsqueeze(dim=-1) * self.applied_mask.unsqueeze(
            0
        )
        # the complement mask adds zeros where the mask is zero, these are not part of the relation
        # nan_to_num is used to replace nan values with the nan_replacement value (often not needed)
        return (
            (after_mask + (1 - self.applied_mask))
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
