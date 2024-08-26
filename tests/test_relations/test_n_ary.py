"""
Test the fuzzy n-ary relations work as expected.
"""

import shutil
import unittest
from pathlib import Path
from typing import List, Tuple, MutableMapping, Any

import torch
import igraph
import numpy as np

from fuzzy.relations.continuous.linkage import GroupedLinks, BinaryLinks
from fuzzy.sets.continuous.impl import Gaussian
from fuzzy.sets.continuous.membership import Membership
from fuzzy.sets.continuous.group import GroupedFuzzySets
from fuzzy.sets.continuous.abstract import ContinuousFuzzySet
from fuzzy.relations.continuous.t_norm import Minimum, Product
from fuzzy.relations.continuous.n_ary import NAryRelation
from fuzzy.relations.continuous.compound import Compound

N_TERMS: int = 2
N_VARIABLES: int = 4
N_OBSERVATIONS: int = 3
N_COMPOUNDS: int = 5
AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.assertTrue(torch.allclose(membership.degrees, expected_membership_degrees))
        return membership

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
        self.assertEqual(n_ary._coo_matrix[0].shape, (2, 2))
        # check that the original shape is stored
        self.assertEqual(n_ary._original_shape[0], (2, 2))
        # matrix size can increase (in-place) for more potential rows (vars) and columns (terms)
        n_ary._coo_matrix[0].resize(3, 3)
        self.assertEqual(n_ary._coo_matrix[0].shape, (3, 3))
        # check that the original shape is still kept after resizing
        self.assertEqual(n_ary._original_shape[0], (2, 2))

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
        self.assertTrue(n_ary.applied_mask is not None)
        n_ary.apply_mask(membership=membership)
        self.assertTrue(n_ary.grouped_links is not None)
        self.assertTrue(n_ary.applied_mask is not None)  # we have used the relation
        self.assertTrue(
            torch.allclose(
                n_ary.grouped_links(membership=membership), n_ary.applied_mask
            )
        )
        # we can create a new n-ary relation with a GroupedLinks object
        new_n_ary = NAryRelation(
            grouped_links=n_ary.grouped_links, device=AVAILABLE_DEVICE
        )
        # the new n-ary relation should have the same applied mask as the original
        self.assertTrue(
            torch.allclose(
                new_n_ary.grouped_links(membership=membership), n_ary.applied_mask
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

        # check that relations involving the same index references share the same vertex

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
        file_name: str = "n_ary_relation"
        n_ary = NAryRelation(*indices, device=AVAILABLE_DEVICE)
        self.assertRaises(
            ValueError, n_ary.save, Path(f"{file_name}.txt")
        )  # wrong extension
        self.assertRaises(
            ValueError, n_ary.save, Path(f"{file_name}.pth")
        )  # bad extension
        self.assertRaises(ValueError, n_ary.save, Path(f"{file_name}"))  # no extension
        state_dict: MutableMapping[str, Any] = n_ary.save(Path(f"{file_name}.pt"))
        # check that the file was created and exists

        # check that the state dict contains the necessary keys
        self.assertTrue(Path(f"{file_name}.pt").exists())
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
        loaded_n_ary = NAryRelation.load(
            Path(f"{file_name}.pt"), device=AVAILABLE_DEVICE
        )
        self.assertEqual(n_ary.indices, loaded_n_ary.indices)
        self.assertEqual(n_ary.nan_replacement, loaded_n_ary.nan_replacement)
        # the applied_mask is the resulting output from grouped_links()
        self.assertTrue(torch.allclose(n_ary.applied_mask, loaded_n_ary.applied_mask))
        self.assertTrue(
            np.allclose(
                n_ary._coo_matrix[0].toarray(), loaded_n_ary._coo_matrix[0].toarray()
            )
        )
        self.assertEqual(n_ary._coo_matrix[0].shape, loaded_n_ary._coo_matrix[0].shape)
        # remove the file
        Path(f"{file_name}.pt").unlink()

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
        file_name: str = "n_ary_relation"
        n_ary = NAryRelation(grouped_links=grouped_links, device=AVAILABLE_DEVICE)
        self.assertRaises(ValueError, n_ary.save, Path(f"{file_name}.txt"))
        self.assertRaises(ValueError, n_ary.save, Path(f"{file_name}.pth"))
        intended_destination: Path = Path(__file__).parent / f"{file_name}.pt"
        n_ary.save(path=intended_destination)
        # note a .pt file is NOT created, but a directory is created instead
        # (to save the grouped links)
        actual_destination: Path = Path(__file__).parent / file_name
        self.assertTrue(actual_destination.exists())
        self.assertTrue(actual_destination.is_dir())
        loaded_n_ary = NAryRelation.load(actual_destination, device=AVAILABLE_DEVICE)
        self.assertTrue(
            torch.allclose(n_ary.applied_mask, loaded_n_ary.applied_mask)
        )  # the applied_mask is the resulting output from grouped_links()
        for actual_module, loaded_module in zip(
            n_ary.grouped_links.modules_list, loaded_n_ary.grouped_links.modules_list
        ):
            # modules are expected to have the shape property
            self.assertEqual(actual_module.shape, loaded_module.shape)
            # the loaded module should be the same as the original per __eq__ method
            self.assertEqual(actual_module, loaded_module)
        # remove the directory and its contents
        shutil.rmtree(actual_destination)


class TestProduct(TestNAryRelation):
    """
    Test the Product n-ary relation.
    """

    def test_algebraic_product(self) -> None:
        """
        Test the n-ary product operation given a single relation.

        Returns:

        """
        n_ary = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        # test the mask application
        after_mask = n_ary.apply_mask(membership=membership)
        expected_after_mask = torch.tensor(
            [
                [[2.5514542e-04], [7.4245834e-01], [1.0000000e00], [1.0000000e00]],
                [[9.6005607e-01], [8.4526926e-01], [1.0000000e00], [1.0000000e00]],
                [[5.7408627e-04], [9.9679035e-01], [1.0000000e00], [1.0000000e00]],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(after_mask, expected_after_mask))

        # test the forward pass
        prod_membership: Membership = n_ary.forward(membership)
        expected_prod_values = torch.tensor(
            [
                [7.4245834e-01 * 2.5514542e-04],
                [8.4526926e-01 * 9.6005607e-01],
                [9.9679035e-01 * 5.7408627e-04],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(prod_membership.degrees, expected_prod_values))

        # check that it is torch.jit scriptable (currently not working)
        # n_ary_script = torch.jit.script(n_ary)
        #
        # after_mask_script = n_ary_script.apply_mask(membership=membership)
        # self.assertTrue(torch.allclose(after_mask_script, expected_after_mask))
        #
        # min_values_script = n_ary_script.forward(membership)
        # self.assertTrue(torch.allclose(min_values_script, expected_min_values))

    def test_multiple_indices_passed_as_list(self) -> None:
        """
        Test the Product operation given multiple relations, where some variables are never used
        by those relations. This is a test to ensure that the Product operation can handle
        relations that do not use all variables (i.e., does not wrongly output zeros).

        Returns:
            None
        """
        n_ary = Product(
            [(0, 1), (1, 0)],
            [(1, 1), (2, 1)],
            [(2, 1), (2, 0)],
            [(0, 1), (2, 0)],
            [(1, 1), (0, 1)],
            device=AVAILABLE_DEVICE,
        )
        membership = self.test_gaussian_membership()
        prod_membership: Membership = n_ary(membership)
        expected_prod_values = torch.tensor(
            [
                [
                    membership.degrees[0][0][1].item()
                    * membership.degrees[0][1][0].item(),
                    membership.degrees[0][1][1].item()
                    * membership.degrees[0][2][1].item(),
                    membership.degrees[0][2][1].item()
                    * membership.degrees[0][2][0].item(),
                    membership.degrees[0][0][1].item()
                    * membership.degrees[0][2][0].item(),
                    membership.degrees[0][1][1].item()
                    * membership.degrees[0][0][1].item(),
                ],
                [
                    membership.degrees[1][0][1].item()
                    * membership.degrees[1][1][0].item(),
                    membership.degrees[1][1][1].item()
                    * membership.degrees[1][2][1].item(),
                    membership.degrees[1][2][1].item()
                    * membership.degrees[1][2][0].item(),
                    membership.degrees[1][0][1].item()
                    * membership.degrees[1][2][0].item(),
                    membership.degrees[1][1][1].item()
                    * membership.degrees[1][0][1].item(),
                ],
                [
                    membership.degrees[2][0][1].item()
                    * membership.degrees[2][1][0].item(),
                    membership.degrees[2][1][1].item()
                    * membership.degrees[2][2][1].item(),
                    membership.degrees[2][2][1].item()
                    * membership.degrees[2][2][0].item(),
                    membership.degrees[2][0][1].item()
                    * membership.degrees[2][2][0].item(),
                    membership.degrees[2][1][1].item()
                    * membership.degrees[2][0][1].item(),
                ],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertEqual(prod_membership.degrees.shape[0], N_OBSERVATIONS)
        self.assertEqual(prod_membership.degrees.shape[1], N_COMPOUNDS)
        self.assertEqual(prod_membership.degrees.shape, expected_prod_values.shape)
        self.assertTrue(torch.allclose(prod_membership.degrees, expected_prod_values))


class TestMinimum(TestNAryRelation):
    """
    Test the Minimum n-ary relation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypercube = GroupedFuzzySets(
            modules_list=[
                ContinuousFuzzySet.stack(
                    [
                        Gaussian(
                            centers=np.array([-1, 0.0, 1.0]),
                            widths=np.array([1.0, 1.0, 1.0]),
                            device=AVAILABLE_DEVICE,
                        ),
                        Gaussian(
                            centers=np.array([-1.0, 0.0, 1.0]),
                            widths=np.array([1.0, 1.0, 1.0]),
                            device=AVAILABLE_DEVICE,
                        ),
                    ]
                )
            ]
        )

    def test_minimum(self) -> None:
        """
        Test the n-ary minimum operation given a single relation.

        Returns:
            None
        """
        n_ary = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        # test the mask application
        after_mask = n_ary.apply_mask(membership=membership)
        expected_after_mask = torch.tensor(
            [
                [[2.5514542e-04], [7.4245834e-01], [1.0000000e00], [1.0000000e00]],
                [[9.6005607e-01], [8.4526926e-01], [1.0000000e00], [1.0000000e00]],
                [[5.7408627e-04], [9.9679035e-01], [1.0000000e00], [1.0000000e00]],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(after_mask, expected_after_mask))

        # test the forward pass
        min_membership: Membership = n_ary.forward(membership)
        expected_min_values = torch.tensor(
            [[2.5514542e-04], [8.4526926e-01], [5.7408627e-04]],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(min_membership.degrees, expected_min_values))

        # check that it is torch.jit scriptable (currently not working)
        # n_ary_script = torch.jit.script(n_ary)
        #
        # after_mask_script = n_ary_script.apply_mask(membership=membership)
        # self.assertTrue(torch.allclose(after_mask_script, expected_after_mask))
        #
        # min_values_script = n_ary_script.forward(membership)
        # self.assertTrue(torch.allclose(min_values_script, expected_min_values))

    def test_multiple_indices_passed_as_list(self) -> None:
        """
        Test the Minimum operation given multiple relations, where some variables are never used
        by those relations. This is a test to ensure that the Minimum operation can handle
        relations that do not use all variables (i.e., does not wrongly output zeros).

        Returns:
            None
        """
        data = torch.tensor(
            [
                [1.5409961, -0.2934289],
                [-2.1787894, 0.56843126],
                [-1.0845224, -1.3985955],
                [0.40334684, 0.83802634],
            ],
            device=AVAILABLE_DEVICE,
        )
        minimum = Minimum(
            [(0, 0), (1, 0)],
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
            [(0, 1), (1, 1)],
            [(0, 1), (1, 2)],
            device=AVAILABLE_DEVICE,
        )

        membership: Membership = self.hypercube(data)
        min_membership: Membership = minimum(membership)
        expected_degrees = torch.tensor(
            [
                [0.00157003, 0.00157003, 0.09304529, 0.09304529, 0.09304529],
                [
                    8.5436940e-02,
                    2.4918883e-01,
                    8.6766202e-03,
                    8.6766e-03,
                    8.6766202e-03,
                ],
                [0.8531001, 0.14141318, 0.3084521, 0.14141318, 0.00317242],
                [0.034104, 0.13954304, 0.034104, 0.49545035, 0.8498557],
            ],
            device=AVAILABLE_DEVICE,
        )

        self.assertTrue(torch.allclose(min_membership.degrees, expected_degrees))


class TestCompound(TestNAryRelation):
    """
    Test the Compound n-ary relation, which allows the user to compound/aggregate multiple n-ary
    relations together.
    """

    def test_combination_of_t_norms(self) -> None:
        """
        Test we can create a combination of t-norms to reflect more complex compound propositions.

        Returns:
            None
        """
        n_ary_min = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        n_ary_prod = Product((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        membership = self.test_gaussian_membership()

        t_norm = Compound(n_ary_min, n_ary_prod)
        compound_values = t_norm(membership=membership)
        expected_compound_values = torch.cat(
            [
                n_ary_min(membership=membership).degrees,
                n_ary_prod(membership=membership).degrees,
            ],
            dim=-1,
        ).unsqueeze(dim=-1)
        self.assertTrue(
            torch.allclose(compound_values.degrees, expected_compound_values)
        )

        # we can then follow it up with another t-norm

        n_ary_next_min = Minimum((0, 1), (1, 0), device=AVAILABLE_DEVICE)
        min_membership: Membership = n_ary_next_min(compound_values)
        expected_min_values = torch.tensor(
            [
                [7.4245834e-01 * 2.5514542e-04],
                [8.4526926e-01 * 9.6005607e-01],
                [9.9679035e-01 * 5.7408627e-04],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(min_membership.degrees, expected_min_values))
