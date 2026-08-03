"""
Direct unit tests for the fuzzy.sets.membership module's namedtuples: Membership and
NamedTensor.
"""

import unittest

import torch

from fuzzy.sets.membership import Membership, NamedTensor
from fuzzy.utils.options.impl.impl_enums import DimensionEnum

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMembership(unittest.TestCase):
    """
    Test the Membership namedtuple.
    """

    def test_construction(self) -> None:
        """
        Returns:
            None
        """
        degrees = torch.rand(2, 3, device=AVAILABLE_DEVICE)
        mask = torch.ones(2, 3, device=AVAILABLE_DEVICE)
        membership = Membership(degrees=degrees, mask=mask)
        self.assertIs(membership.degrees, degrees)
        self.assertIs(membership.mask, mask)


class TestNamedTensor(unittest.TestCase):
    """
    Test the NamedTensor namedtuple.
    """

    def test_construction(self) -> None:
        """
        Returns:
            None
        """
        data = torch.rand(2, 3, device=AVAILABLE_DEVICE)
        named_tensor = NamedTensor(data=data, names=["batch", "feature"])
        self.assertIs(named_tensor.data, data)
        self.assertEqual(named_tensor.names, ["batch", "feature"])

    def test_construction_with_dimension_enum_names(self) -> None:
        """
        Returns:
            None
        """
        data = torch.rand(2, device=AVAILABLE_DEVICE)
        named_tensor = NamedTensor(data=data, names=(DimensionEnum.VARIABLE,))
        self.assertEqual(named_tensor.names, (DimensionEnum.VARIABLE,))

    def test_data_must_be_a_tensor(self) -> None:
        """
        Returns:
            None
        """
        with self.assertRaises(AssertionError):
            NamedTensor(data=[1.0, 2.0], names=["batch"])

    def test_shape_must_match_names_length(self) -> None:
        """
        Returns:
            None
        """
        data = torch.rand(2, 3, device=AVAILABLE_DEVICE)
        with self.assertRaises(AssertionError):
            NamedTensor(data=data, names=["only_one_name"])


if __name__ == "__main__":
    unittest.main()
