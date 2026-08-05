"""
Test the fuzzy n-ary relations work as expected.
"""

# white-box tests deliberately reach into private/internal attributes to verify
# implementation details
# pylint: disable=protected-access

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
import unittest
from pathlib import Path
from typing import Any, List, MutableMapping, Tuple

import igraph
import numpy as np
import torch

from fuzzy.relations.linkage import BinaryLinks, GroupedLinks
from fuzzy.relations.n_ary import NAryMaskMethods, NAryRelation
from fuzzy.sets.impl import Gaussian
from fuzzy.sets.membership import Membership
from tests import AVAILABLE_DEVICE

N_TERMS: int = 2
N_VARIABLES: int = 4
N_OBSERVATIONS: int = 3
N_COMPOUNDS: int = 5


class _SparseLinks(torch.nn.Module):
    """
    A minimal stand-in for a links module that returns an uncoalesced sparse tensor, used
    only to exercise get_mask()'s defensive coalesce() branch. BinaryLinks always returns a
    dense tensor, so this branch is otherwise unreachable through the public API today.
    """

    def __init__(self, shape: torch.Size, device: torch.device):
        super().__init__()
        self._shape = shape
        self._device = device

    @property
    def shape(self) -> torch.Size:
        """
        Returns:
            The shape of the (fake) sparse links tensor this module would produce.
        """
        return self._shape

    def forward(self, *_) -> torch.Tensor:
        """
        Returns:
            An uncoalesced sparse tensor with a duplicate index, filled with ones.
        """
        # a duplicate index makes the resulting sparse tensor uncoalesced
        indices = torch.zeros(
            (len(self._shape), 2), dtype=torch.long, device=self._device
        )
        values = torch.ones(2, device=self._device)
        return torch.sparse_coo_tensor(
            indices, values, size=self._shape, device=self._device
        )


class TestNAryRelation(unittest.TestCase):
    """
    Test the abstract n-ary relation, including functionality that is common to all n-ary relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_mf = Gaussian(
            centers=np.array([[i, i + 1] for i in range(N_VARIABLES)]),
            widths=np.array([[(i + 1) / 2, (i + 1) / 3] for i in range(N_VARIABLES)]),
            device=AVAILABLE_DEVICE,
        )
        self.data: np.ndarray = np.array(
            [
                [0.0412, 0.4543, 0.1980, 0.3821],
                [0.9327, 0.5900, 0.1569, 0.6902],
                [0.0894, 0.9433, 0.9903, 0.5800],
            ]
        )

    def test_gaussian_membership(self) -> Membership:
        """
        Although this test is not directly related to the NAryRelation class, and is possibly
        redundant due to Gaussian's unit testing, it is used to double-check that the Gaussian
        membership function is working as intended as these unit tests rely on correct values
        from the Gaussian membership function to work.

        Returns:
            The membership values for the Gaussian membership function.
        """
        membership: Membership = self.gaussian_mf(
            torch.tensor(self.data, dtype=torch.float32, device=AVAILABLE_DEVICE)
        )

        self.assertEqual(membership.degrees.shape[0], N_OBSERVATIONS)
        self.assertEqual(membership.degrees.shape[1], N_VARIABLES)
        self.assertEqual(membership.degrees.shape[2], N_TERMS)

        # check that the membership is correct
        expected_membership_degrees: torch.Tensor = torch.tensor(
            [
                [
                    [9.9323326e-01, 2.5514542e-04],
                    [7.4245834e-01, 4.6277978e-03],
                    [2.3617040e-01, 3.8928282e-04],
                    [1.8026091e-01, 6.3449936e-04],
                ],
                [
                    [3.0816132e-02, 9.6005607e-01],
                    [8.4526926e-01, 1.1410457e-02],
                    [2.2095737e-01, 3.0867624e-04],
                    [2.6347569e-01, 2.1079029e-03],
                ],
                [
                    [9.6853620e-01, 5.7408627e-04],
                    [9.9679035e-01, 8.1074789e-02],
                    [6.3564914e-01, 1.7616944e-02],
                    [2.3128603e-01, 1.3889252e-03],
                ],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(
            torch.allclose(membership.degrees.to_dense(), expected_membership_degrees)
        )
        return membership

    def test_invalid_use_of_n_ary_relation(self) -> None:
        """
        Test an Exception is raised when the n-ary relation is used incorrectly.

        Returns:
            None
        """
        # need to provide either indices or grouped_links
        self.assertRaises(ValueError, NAryRelation, device=AVAILABLE_DEVICE)
        # invalid selection for nan_replacement
        self.assertRaises(
            ValueError, NAryRelation, device=AVAILABLE_DEVICE, nan_replacement=3.0
        )

    def test_n_ary_relation(self) -> None:
        """
        Test the abstract n-ary relation.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        # the forward pass should not be implemented
        self.assertRaises(NotImplementedError, n_ary.forward, None)
        # check that the matrix shape is correct
        self.assertEqual(
            n_ary._coo_matrix[0].shape, (2, 2)  # pylint: disable=protected-access
        )
        # check that the original shape is stored
        self.assertEqual(
            n_ary._original_shape[0], (2, 2)  # pylint: disable=protected-access
        )
        # matrix size can increase (in-place) for more potential rows (vars)
        # and columns (terms)
        n_ary._coo_matrix[0].resize(3, 3)  # pylint: disable=protected-access
        self.assertEqual(
            n_ary._coo_matrix[0].shape, (3, 3)  # pylint: disable=protected-access
        )
        # check that the original shape is still kept after resizing
        self.assertEqual(
            n_ary._original_shape[0], (2, 2)  # pylint: disable=protected-access
        )

    def test_duplicates(self) -> None:
        """
        Test that the NAryRelation class throws an error when given duplicate indices. Otherwise, a
        duplicate index will result in a value greater than 1 in the mask, which is not allowed.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            NAryRelation,
            (0, 1),
            (1, 0),
            (1, 0),
            device=AVAILABLE_DEVICE,
        )

    def test_grouped_links(self) -> None:
        """
        Test that the grouped links are created correctly.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()
        # we have not used the relation yet, but it is built from dummy inputs
        self.assertTrue(n_ary.get_mask().to_dense() is not None)
        n_ary.apply_mask(membership=membership)
        self.assertTrue(n_ary.grouped_links is not None)
        self.assertTrue(
            n_ary.get_mask().to_dense() is not None
        )  # we have used the relation

        self.assertTrue(
            torch.allclose(
                n_ary.grouped_links(membership=membership).to_dense(),
                n_ary.get_mask().to_dense(),
            )
        )
        # we can create a new n-ary relation with a GroupedLinks object
        new_n_ary = NAryRelation(
            grouped_links=n_ary.grouped_links, device=AVAILABLE_DEVICE
        )
        # the new n-ary relation should have the same applied mask as the
        # original
        self.assertTrue(
            torch.allclose(
                new_n_ary.grouped_links(membership=membership).to_dense(),
                n_ary.get_mask().to_dense(),
            )
        )

    def test_graph(self) -> None:
        """
        Test that a graph representation of the relation can be created.

        Returns:
            None
        """
        indices: List[Tuple[int, int]] = [(0, 1), (1, 0)]
        single_n_ary = NAryRelation(*indices, device=AVAILABLE_DEVICE)
        single_n_ary_graph: igraph.Graph = single_n_ary.graph
        self.assertTrue(single_n_ary_graph is not None)
        self.assertEqual(
            single_n_ary_graph.vcount(), 3
        )  # 2 index pairs + 1 for relation
        self.assertEqual(single_n_ary_graph.ecount(), 2)  # 2 edges (relations)

        # check vertex attributes are as we expect
        self.assertEqual(single_n_ary_graph.vs[0]["tags"], {"relation"})
        for index in (1, 2):
            self.assertEqual(single_n_ary_graph.vs[index]["tags"], {"anchor"})
            self.assertEqual(single_n_ary_graph.vs[index]["item"], indices[index - 1])

        # check edges are as we expect
        for index in (0, 1):
            self.assertEqual(single_n_ary_graph.es[index].source, index + 1)
            self.assertEqual(single_n_ary_graph.es[index].target, 0)

        indices: List[List[Tuple[int, int]]] = [[(0, 1), (1, 0)], [(1, 1), (2, 1)]]
        multiple_n_ary = NAryRelation(*indices, device=AVAILABLE_DEVICE)
        multiple_n_ary_graph: igraph.Graph = multiple_n_ary.graph
        self.assertTrue(multiple_n_ary_graph is not None)
        self.assertEqual(
            multiple_n_ary_graph.vcount(), 6
        )  # 4 index pairs + 2 for relations
        self.assertEqual(multiple_n_ary_graph.ecount(), 4)  # 4 edges (relations)

        # check vertex attributes are as we expect
        relation_vertices: igraph.VertexSeq = multiple_n_ary_graph.vs.select(
            tags_eq={"relation"}
        )
        self.assertEqual(len(relation_vertices), 2)
        relation_index: int = 0
        for relation_vertex in relation_vertices:
            self.assertEqual(relation_vertex["tags"], {"relation"})
            predecessors: List[igraph.Vertex] = relation_vertex.predecessors()
            for predecessor in predecessors:
                self.assertEqual(predecessor["tags"], {"anchor"})
                # below does not work consistently
                # self.assertEqual(predecessor["item"], indices[relation_index][index])
            relation_index += 1

        # check that relations involving the same index references share the
        # same vertex

        multiple_n_ary = NAryRelation(
            [(0, 1), (1, 0)], [(1, 1), (0, 1)], device=AVAILABLE_DEVICE
        )
        multiple_n_ary_graph_with_uniques: igraph.Graph = multiple_n_ary.graph
        self.assertTrue(multiple_n_ary_graph_with_uniques is not None)
        self.assertEqual(
            multiple_n_ary_graph_with_uniques.vcount(), 5
        )  # 3 unique index pairs + 2 for relations
        self.assertEqual(
            multiple_n_ary_graph_with_uniques.ecount(), 4  # 4 edges (relations)
        )

    def test_save_and_load_from_indices(self) -> None:
        """
        Test that the n-ary relation can be saved and loaded when its source was indices.

        Returns:
            None
        """
        indices: tuple = ((0, 1), (1, 0))
        dir_path: Path = Path("n_ary_relation")
        n_ary = NAryRelation(*indices, device=AVAILABLE_DEVICE)
        state_dict: MutableMapping[str, Any] = n_ary.save(path=dir_path)
        # check that the file was created and exists
        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())
        # check that the state dict contains the necessary keys
        for key in ("indices", "class_name", "nan_replacement"):
            self.assertTrue(
                key in state_dict
            )  # these should appear since they are saved
        # check that the state dict does not contain unnecessary keys
        self.assertTrue(
            "grouped_links" not in state_dict
        )  # not saved; loading from indices here

        # check that the state dict contains the correct values
        self.assertEqual(indices, state_dict["indices"])
        self.assertEqual("NAryRelation", state_dict["class_name"])
        self.assertEqual(n_ary.nan_replacement, state_dict["nan_replacement"])
        loaded_n_ary = NAryRelation.load(path=dir_path, device=AVAILABLE_DEVICE)
        self.assertEqual(n_ary.indices, loaded_n_ary.indices)
        self.assertEqual(n_ary.nan_replacement, loaded_n_ary.nan_replacement)
        # the applied_mask is the resulting output from grouped_links()
        self.assertTrue(
            torch.allclose(
                n_ary.get_mask().to_dense(), loaded_n_ary.get_mask().to_dense()
            )
        )
        # pylint: disable=protected-access
        self.assertTrue(
            np.allclose(
                n_ary._coo_matrix[0].toarray(),
                loaded_n_ary._coo_matrix[0].toarray(),
            )
        )
        self.assertEqual(
            n_ary._coo_matrix[0].shape,
            loaded_n_ary._coo_matrix[0].shape,
        )
        # pylint: enable=protected-access
        # remove the directory
        shutil.rmtree(dir_path)

    def test_save_and_load_from_grouped_links(self) -> None:
        """
        Test the n-ary relation can be saved and loaded when its source was a GroupedLinks object.

        Returns:
            None
        """
        grouped_links: GroupedLinks = GroupedLinks(
            modules_list=[
                BinaryLinks(np.eye(N_TERMS, N_TERMS), device=AVAILABLE_DEVICE),
                BinaryLinks(np.eye(N_TERMS, N_TERMS), device=AVAILABLE_DEVICE),
                BinaryLinks(np.eye(N_TERMS, N_TERMS), device=AVAILABLE_DEVICE),
            ]
        )
        n_ary = NAryRelation(grouped_links=grouped_links, device=AVAILABLE_DEVICE)
        intended_destination: Path = Path(__file__).parent / "n_ary_relation"
        n_ary.save(path=intended_destination)
        # note a .pt file is NOT created, but a directory is created instead
        # (to save the grouped links)
        self.assertTrue(intended_destination.exists())
        self.assertTrue(intended_destination.is_dir())
        loaded_n_ary = NAryRelation.load(intended_destination, device=AVAILABLE_DEVICE)
        self.assertTrue(
            torch.allclose(
                n_ary.get_mask().to_dense(), loaded_n_ary.get_mask().to_dense()
            )
        )  # the applied_mask is the resulting output from grouped_links()
        for actual_module, loaded_module in zip(
            n_ary.grouped_links.modules_list, loaded_n_ary.grouped_links.modules_list
        ):
            # modules are expected to have the shape property
            self.assertEqual(actual_module.shape, loaded_module.shape)
            # the loaded module should be the same as the original per __eq__
            # method
            self.assertEqual(actual_module, loaded_module)
        # remove the directory and its contents
        shutil.rmtree(intended_destination)

    def test_load_from_single_pt_file(self) -> None:
        """
        NAryRelation.load() supports loading from either a directory (the format save()
        produces) or a single .pt file directly; the latter had no test coverage.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        state_dict: MutableMapping[str, Any] = n_ary.state_dict()
        state_dict["nan_replacement"] = n_ary.nan_replacement
        state_dict["class_name"] = "NAryRelation"
        state_dict["indices"] = (
            n_ary.indices if len(n_ary.indices) > 1 else n_ary.indices[0]
        )
        pt_path = Path("n_ary_relation_single_file.pt")
        torch.save(state_dict, pt_path)
        try:
            loaded = NAryRelation.load(pt_path, device=AVAILABLE_DEVICE)
            self.assertEqual(n_ary.indices, loaded.indices)
            self.assertEqual(n_ary.nan_replacement, loaded.nan_replacement)
        finally:
            pt_path.unlink()

    def test_unsupported_method_raises(self) -> None:
        """
        _cache_apply_mask_func() must raise NotImplementedError for any method value it
        does not recognize.

        Returns:
            None
        """
        n_ary = NAryRelation((0, 1), device=AVAILABLE_DEVICE)
        n_ary.method = "not_a_real_method"  # pylint: disable=protected-access
        self.assertRaises(NotImplementedError, n_ary._cache_apply_mask_func)

    def test_get_mask_coalesces_sparse_mask(self) -> None:
        """
        get_mask() must coalesce an uncoalesced sparse mask before returning it.
        BinaryLinks never produces a sparse result, so this is exercised directly with a
        fake links module instead.

        Returns:
            None
        """
        sparse_links = _SparseLinks(
            shape=torch.Size([2, 2, 1]), device=AVAILABLE_DEVICE
        )
        n_ary = NAryRelation(
            grouped_links=GroupedLinks(modules_list=[sparse_links]),
            device=AVAILABLE_DEVICE,
        )
        mask = n_ary.get_mask()
        self.assertTrue(mask.is_sparse)
        self.assertTrue(mask.is_coalesced())

    def test_nan_observation_does_not_corrupt_relation_gradient(self) -> None:
        """
        Regression test: a NaN observation (representing missing data - see
        nan_replacement) used to corrupt the gradient of shared parameters for every
        OTHER, valid observation, via IEEE754's 0 * NaN = NaN - both through
        _prod_apply_mask's internal product reduction and through _gather_apply_mask's
        NaN re-injection. Neither may leak a NaN gradient to a position whose own local
        computation never touched NaN.

        Returns:
            None
        """
        relation = NAryRelation(
            [(0, 0)], device=AVAILABLE_DEVICE, method=NAryMaskMethods.PROD
        )
        degrees = torch.tensor(
            [[[0.3, 0.7]], [[float("nan"), float("nan")]], [[0.9, 0.1]]],
            device=AVAILABLE_DEVICE,
            requires_grad=True,
        )
        membership = Membership(
            degrees=degrees, mask=torch.ones(1, 2, device=AVAILABLE_DEVICE)
        )
        result = relation.apply_mask(membership)
        result.sum().nan_to_num(0.0).backward()
        self.assertFalse(bool(degrees.grad.isnan().any()))
