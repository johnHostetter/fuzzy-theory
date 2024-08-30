"""
Test the linkage behavior, which helps support the calculations of n-ary fuzzy relations.
"""

import shutil
import unittest
from pathlib import Path

import torch
import numpy as np

from fuzzy.relations.linkage import BinaryLinks, GroupedLinks


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestBinaryLinks(unittest.TestCase):
    """
    Test the BinaryLinks class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.links = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.binary_links = BinaryLinks(self.links, device=AVAILABLE_DEVICE)

    def test_create_linkage(self) -> None:
        """
        Test that a BinaryLinks object is created correctly.

        Returns:
            None
        """
        self.assertIsInstance(self.binary_links, BinaryLinks)
        self.assertIsInstance(self.binary_links.links, torch.Tensor)
        self.assertEqual(AVAILABLE_DEVICE.type, self.binary_links.links.device.type)
        self.assertEqual(torch.int8, self.binary_links.links.dtype)
        self.assertEqual((3, 3), self.binary_links.links.shape)
        self.assertEqual(
            self.links.tolist(), self.binary_links.links.cpu().numpy().tolist()
        )
        self.assertEqual((3, 3), self.binary_links.shape)

        # test that we can move it to a different device
        self.binary_links.to(torch.device("cpu"))
        self.assertEqual(torch.device("cpu"), self.binary_links.device)
        # test that this is reflected in its parameters
        self.assertEqual(torch.device("cpu"), self.binary_links.links.device)

        # test we can move it back
        self.binary_links.to(AVAILABLE_DEVICE)
        self.assertEqual(AVAILABLE_DEVICE.type, self.binary_links.device.type)
        # test that this is reflected in its parameters
        self.assertEqual(AVAILABLE_DEVICE.type, self.binary_links.links.device.type)

        # test we can also create a GroupedLinks object without any modules
        grouped_links = GroupedLinks(modules_list=None)
        # and then later add a module
        grouped_links.modules_list.add_module("0", self.binary_links)
        self.assertEqual(1, len(grouped_links.modules_list))
        self.assertIsInstance(grouped_links.modules_list[0], BinaryLinks)

    def test_save_and_load_linkage(self) -> None:
        """
        Test that we can save and load a BinaryLinks object.

        Returns:
            None
        """
        # illegal path name
        self.assertRaises(ValueError, self.binary_links.save, Path("test.txt"))
        # bad path name
        self.assertRaises(ValueError, self.binary_links.save, Path("test.pth"))
        # good path name
        path = Path("binary_links.pt")
        state_dict = self.binary_links.save(path)
        # check that the file exists
        self.assertTrue(path.exists() and path.is_file())
        self.assertIsInstance(state_dict, dict)
        self.assertTrue("links" in state_dict)
        # load the file
        loaded_binary_links = BinaryLinks.load(path, device=AVAILABLE_DEVICE)
        self.assertIsInstance(loaded_binary_links, BinaryLinks)
        self.assertIsInstance(loaded_binary_links.links, torch.Tensor)
        # test they are on the same device
        self.assertEqual(AVAILABLE_DEVICE.type, loaded_binary_links.links.device.type)
        self.assertEqual(
            self.binary_links.links.device, loaded_binary_links.links.device
        )
        # test the links' dtypes are equal
        self.assertEqual(torch.int8, loaded_binary_links.links.dtype)
        self.assertEqual(self.binary_links.links.dtype, loaded_binary_links.links.dtype)
        # test the links' shapes are equal
        self.assertEqual((3, 3), loaded_binary_links.links.shape)
        self.assertEqual(self.binary_links.shape, loaded_binary_links.shape)
        self.assertEqual(self.binary_links.links.shape, loaded_binary_links.links.shape)
        # test the links' values are equal
        self.assertEqual(
            self.links.tolist(), loaded_binary_links.links.cpu().numpy().tolist()
        )
        self.assertTrue(
            torch.allclose(self.binary_links.links, loaded_binary_links.links)
        )
        # test the __eq__ method
        self.assertEqual(self.binary_links, loaded_binary_links)
        # test the forward pass
        self.assertTrue(torch.allclose(self.binary_links(), loaded_binary_links()))
        # remove the file
        path.unlink()
        self.assertFalse(path.exists())


class TestGroupedLinks(unittest.TestCase):
    """
    Test the GroupedLinks class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.links_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.links_2 = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]])
        self.binary_links_1 = BinaryLinks(self.links_1, device=AVAILABLE_DEVICE)
        self.binary_links_2 = BinaryLinks(self.links_2, device=AVAILABLE_DEVICE)
        self.grouped_links = GroupedLinks(
            modules_list=[self.binary_links_1, self.binary_links_2]
        )

    def test_create_linkage(self) -> None:
        """
        Test that a GroupedLinks object is created correctly.

        Returns:
            None
        """
        self.assertIsInstance(self.grouped_links, GroupedLinks)
        self.assertEqual((3, 6), self.grouped_links.shape)
        expected_grouped_links = torch.cat(
            [self.binary_links_1.links, self.binary_links_2.links], dim=1
        )
        self.assertTrue(
            torch.allclose(expected_grouped_links, self.grouped_links(membership=None))
        )

    def test_save_and_load_linkage(self) -> None:
        """
        Test that we can save and load a GroupedLinks object.

        Returns:
            None
        """
        # illegal path name
        self.assertRaises(ValueError, self.grouped_links.save, Path("test.txt"))
        # bad path name
        self.assertRaises(ValueError, self.grouped_links.save, Path("test.pth"))
        # good path (folder) name
        path = Path("grouped_links")
        self.grouped_links.save(path)
        # check that the folder exists
        self.assertTrue(path.exists() and path.is_dir())
        # load the file
        loaded_grouped_links = GroupedLinks.load(path, device=AVAILABLE_DEVICE)
        self.assertIsInstance(loaded_grouped_links, GroupedLinks)
        self.assertIsInstance(loaded_grouped_links.modules_list, torch.nn.ModuleList)
        self.assertEqual(
            len(self.grouped_links.modules_list), len(loaded_grouped_links.modules_list)
        )
        for idx, module in enumerate(self.grouped_links.modules_list):
            self.assertIsInstance(loaded_grouped_links.modules_list[idx], BinaryLinks)
            self.assertEqual(
                module.links.device, loaded_grouped_links.modules_list[idx].links.device
            )
            self.assertEqual(
                module.links.dtype, loaded_grouped_links.modules_list[idx].links.dtype
            )
            self.assertEqual(
                module.links.shape, loaded_grouped_links.modules_list[idx].links.shape
            )
            self.assertTrue(
                torch.allclose(
                    module.links, loaded_grouped_links.modules_list[idx].links
                )
            )
        # remove the directory
        shutil.rmtree(path)
        self.assertFalse(path.exists())
