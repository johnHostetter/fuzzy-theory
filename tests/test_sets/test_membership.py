"""
Direct unit tests for the fuzzy.sets.membership module's namedtuples: Membership and
NamedTensor.
"""

# white-box tests deliberately reach into private/internal attributes to verify
# implementation details
# pylint: disable=protected-access

import unittest

import torch
import torch._dynamo

from fuzzy.sets.membership import Membership, NamedTensor
from fuzzy.utils.options.impl.impl_enums import DimensionEnum
from tests import AVAILABLE_DEVICE


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

    def test_tuple_unpacking(self) -> None:
        """
        Membership must still behave as an ordinary 2-tuple (degrees, mask) - existing
        callers that unpack it positionally must keep working.

        Returns:
            None
        """
        degrees = torch.rand(2, 3, device=AVAILABLE_DEVICE)
        mask = torch.ones(2, 3, device=AVAILABLE_DEVICE)
        membership = Membership(degrees=degrees, mask=mask)
        unpacked_degrees, unpacked_mask = membership
        self.assertIs(unpacked_degrees, degrees)
        self.assertIs(unpacked_mask, mask)

    def test_is_a_plain_namedtuple_not_a_custom_subclass(self) -> None:
        """
        Regression guard: Membership used to be a namedtuple *subclass* with a custom
        (but fully dead - every assert inside it was already commented out) __new__
        override. That subclassing made it opaque to torch.compile's Dynamo tracer,
        which cannot see through a subclassed namedtuple's construction or, separately,
        its field access when an instance crosses a graph boundary as a plain function
        argument - forcing a graph break at both points (see
        test_no_graph_break_on_construction_and_field_access below for the direct
        regression test of that). Guards against silently reintroducing a subclass
        (e.g. to add back validation) without noticing the Dynamo cost.

        Returns:
            None
        """
        # collections.namedtuple("Membership", [...]) directly (no further subclassing)
        # produces a class whose MRO is exactly [Membership, tuple, object] - a
        # hand-written subclass (the old code) would insert an extra level. This is a
        # structural check that doesn't depend on Membership's exact field
        # names.
        self.assertEqual(Membership.__mro__, (Membership, tuple, object))
        # the auto-generated __new__ takes no *args/**kwargs beyond the named fields
        # and has no custom validation logic to bypass
        self.assertEqual(
            Membership.__new__.__code__.co_varnames[
                : Membership.__new__.__code__.co_argcount
            ],
            ("_cls", "degrees", "mask"),
        )

    def test_docstring_preserved(self) -> None:
        """
        Dropping the custom __new__ (and its docstring, which lived on the class body)
        must not lose the human-readable documentation - it is reattached via
        Membership.__doc__ instead.

        Returns:
            None
        """
        self.assertIsNotNone(Membership.__doc__)
        self.assertIn("membership", Membership.__doc__.lower())

    @unittest.skipUnless(
        torch.cuda.is_available(), "graph breaks are a torch.compile/CUDA concern"
    )
    def test_no_graph_break_on_construction_and_field_access(self) -> None:
        """
        Direct regression test for the actual bug: torch.compile's Dynamo tracer must
        not break the graph either when a Membership is constructed, or when one
        constructed in a separate (compiled) call is passed into another function that
        accesses its fields - mirroring the actual pattern in this codebase (a fuzzy
        set constructs a Membership; a t-norm or defuzzification method downstream
        consumes another Membership's .degrees/.mask).

        Returns:
            None
        """

        def make_membership(x: torch.Tensor) -> Membership:
            return Membership(degrees=x * 2, mask=torch.ones_like(x))

        def consume(m: Membership) -> torch.Tensor:
            return m.degrees.sum() + m.mask.sum()

        def full(x: torch.Tensor) -> torch.Tensor:
            return consume(make_membership(x))

        torch._dynamo.reset()
        x = torch.rand(4, device=AVAILABLE_DEVICE)
        explanation = torch._dynamo.explain(full)(x)
        self.assertEqual(explanation.graph_break_count, 0)
        self.assertEqual(explanation.graph_count, 1)


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
