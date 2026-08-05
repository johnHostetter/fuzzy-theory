"""
Direct unit tests for fuzzy.sets.cache: MembershipCache, MembershipCacheEntry, version_of,
and signature_of. See test_membership_cache.py for integration-style tests that exercise the
cache through FuzzySet/FuzzySetGroup instead.
"""

import unittest

import torch

from fuzzy.sets.cache import MembershipCache, MembershipCacheEntry
from fuzzy.sets.membership import Membership
from fuzzy.utils.functions import signature_of, version_of
from tests import AVAILABLE_DEVICE


class TestVersionOf(unittest.TestCase):
    """
    Test the version_of() helper.
    """

    def test_ordinary_tensor(self) -> None:
        """
        Returns:
            None
        """
        tensor = torch.zeros(2, device=AVAILABLE_DEVICE)
        self.assertEqual(version_of(tensor), 0)
        with torch.no_grad():
            tensor.add_(1.0)
        self.assertEqual(version_of(tensor), 1)

    def test_inference_tensor_created_inside_inference_mode(self) -> None:
        """
        A tensor created under torch.inference_mode() does not track a version counter at
        all, so version_of() must report -1 rather than raise, even from inside that
        context.

        Returns:
            None
        """
        with torch.inference_mode():
            tensor = torch.zeros(2, device=AVAILABLE_DEVICE)
            self.assertEqual(version_of(tensor), -1)

    def test_inference_tensor_accessed_outside_inference_mode(self) -> None:
        """
        The "inference tensor" marking sticks to a tensor for its whole lifetime, even
        once the torch.inference_mode() block that created it has been exited. version_of()
        must still report -1 (not raise) for such a tensor when called later, from
        ordinary (non-inference) code - e.g. observations precomputed under
        inference_mode for efficiency, then reused in a context that needs gradients.

        Returns:
            None
        """
        with torch.inference_mode():
            tensor = torch.zeros(2, device=AVAILABLE_DEVICE)
        self.assertFalse(torch.is_inference_mode_enabled())
        self.assertTrue(tensor.is_inference())
        self.assertEqual(version_of(tensor), -1)


class TestSignatureOf(unittest.TestCase):
    """
    Test the signature_of() helper.
    """

    def test_changes_with_identity_version_and_requires_grad(self) -> None:
        """
        Returns:
            None
        """
        first = torch.zeros(2, device=AVAILABLE_DEVICE)
        second = torch.zeros(2, device=AVAILABLE_DEVICE)

        self.assertEqual(signature_of([first]), signature_of([first]))
        self.assertNotEqual(
            signature_of([first]), signature_of([second])
        )  # different identity

        signature_before = signature_of([first])
        with torch.no_grad():
            first.add_(1.0)
        self.assertNotEqual(
            signature_before, signature_of(
                [first]))  # version changed

        signature_before = signature_of([first])
        first.requires_grad_(True)
        self.assertNotEqual(
            signature_before, signature_of([first])
        )  # requires_grad changed


class TestMembershipCacheEntry(unittest.TestCase):
    """
    Test MembershipCacheEntry directly.
    """

    def test_invalidate_accepts_and_ignores_arguments(self) -> None:
        """
        invalidate() must work both as a plain call and as a backward hook (which invokes
        it with the incoming gradient as a positional argument).

        Returns:
            None
        """
        observations = torch.zeros(2, device=AVAILABLE_DEVICE)
        degrees = torch.ones(2, device=AVAILABLE_DEVICE)
        entry = MembershipCacheEntry(
            observations, signature_of(
                []), Membership(
                degrees=degrees, mask=torch.ones(
                    2, device=AVAILABLE_DEVICE)), )
        self.assertTrue(entry.valid)
        # as if it were a hook
        entry.invalidate(torch.zeros(2, device=AVAILABLE_DEVICE))
        self.assertFalse(entry.valid)
        self.assertIsNone(entry.membership)


class TestMembershipCache(unittest.TestCase):
    """
    Test MembershipCache directly.
    """

    def test_maxsize_must_be_at_least_one(self) -> None:
        """
        Returns:
            None
        """
        with self.assertRaises(ValueError):
            MembershipCache(maxsize=0)

    def test_evicts_oldest_entry_when_full(self) -> None:
        """
        Returns:
            None
        """
        cache = MembershipCache(maxsize=1)
        first_observations = torch.zeros(2, device=AVAILABLE_DEVICE)
        second_observations = torch.ones(2, device=AVAILABLE_DEVICE)
        membership = Membership(
            degrees=torch.zeros(2, device=AVAILABLE_DEVICE),
            mask=torch.ones(2, device=AVAILABLE_DEVICE),
        )

        cache.store(first_observations, signature_of([]), membership)
        self.assertEqual(len(cache), 1)
        self.assertIsNotNone(
            cache.lookup(
                first_observations,
                signature_of(
                    [])))

        cache.store(second_observations, signature_of([]), membership)
        self.assertEqual(len(cache), 1)  # still bounded to maxsize
        self.assertIsNone(
            cache.lookup(
                first_observations,
                signature_of(
                    [])))  # evicted
        self.assertIsNotNone(
            cache.lookup(
                second_observations,
                signature_of(
                    [])))


if __name__ == "__main__":
    unittest.main()
