"""
Test functionality relating to FuzzySetGroup.
"""

# white-box tests deliberately reach into private/internal attributes to verify
# implementation details
# pylint: disable=protected-access

import pickle
import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.sets import Membership
from fuzzy.sets.abstract import FuzzySetInitMethod, FuzzySetShape
from fuzzy.sets.group import FuzzySetGroup
from fuzzy.sets.impl import Gaussian
from fuzzy.sets.shape import MembershipConfig
from fuzzy.utils.functions import get_object_attributes

AVAILABLE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class _GenericCentersWidthsMaskModule(torch.nn.Module):
    """
    A plain torch.nn.Module (deliberately NOT a FuzzySet, and without parameter_signature)
    that exposes get_centers/get_widths/get_mask, used to exercise FuzzySetGroup's
    documented support for holding any kind of torch.nn.Module, not just FuzzySet
    instances - see FuzzySetGroup's own docstring.
    """

    def __init__(self, centers: torch.Tensor, widths: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self._centers = centers
        self._widths = widths
        self._mask = mask

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The centers.
        """
        return self._centers

    def get_widths(self) -> torch.Tensor:
        """
        Returns:
            The widths.
        """
        return self._widths

    def get_mask(self) -> torch.Tensor:
        """
        Returns:
            The mask.
        """
        return self._mask

    def forward(self, observations: torch.Tensor) -> Membership:
        """
        Returns:
            A Membership broadcasting the centers across the given observations.
        """
        return Membership(
            degrees=self._centers.expand(observations.shape[0], -1, -1),
            mask=self._mask,
            formula="test",
        )


class _ModuleMissingGetters(torch.nn.Module):
    """
    A plain torch.nn.Module missing get_centers/get_widths/get_mask (and
    parameter_signature) entirely, used to exercise FuzzySetGroup's fallback to "caching
    is not safe" when a module exposes neither.
    """

    def __init__(self, degrees: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self._degrees = degrees
        self._mask = mask

    def forward(self, observations: torch.Tensor) -> Membership:
        """
        Returns:
            A fixed Membership, independent of the given observations.
        """
        assert isinstance(
            observations, torch.Tensor
        ), "The observations should be a torch.Tensor."
        return Membership(degrees=self._degrees, mask=self._mask, formula="test")


class TestFuzzySetGroup(unittest.TestCase):
    """
    Test the FuzzySetGroup class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grouped_fuzzy_sets: FuzzySetGroup = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
            ]
        )

    def test_grad_fn_is_not_none(self):
        """
        Test that the grad_fn attribute is not None.
        """
        # test individual modules
        input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=AVAILABLE_DEVICE)
        for module in self.grouped_fuzzy_sets.modules_list:
            self.assertIsInstance(module, Gaussian)
            output: Membership = module(input_data)
            self.assertIsNotNone(output.degrees.grad_fn)

        # test grouped fuzzy sets
        output: Membership = self.grouped_fuzzy_sets(input_data)
        self.assertIsNotNone(output.degrees.grad_fn)

    def test_save_grouped_fuzzy_sets(self):
        """
        Test saving grouped fuzzy sets.
        """

        # test compatibility with torch.jit.script
        torch.jit.script(self.grouped_fuzzy_sets)

        # test that FuzzySetGroup can be saved and loaded
        self.grouped_fuzzy_sets.save(Path("test_grouped_fuzzy_sets"))
        loaded_grouped_fuzzy_sets: FuzzySetGroup = self.grouped_fuzzy_sets.load(
            Path("test_grouped_fuzzy_sets"), device=AVAILABLE_DEVICE
        )

        for idx, module in enumerate(self.grouped_fuzzy_sets.modules_list):
            assert torch.equal(
                module.get_centers(),
                loaded_grouped_fuzzy_sets.modules_list[idx].get_centers(),
            )
            assert torch.equal(
                module.get_widths(),
                loaded_grouped_fuzzy_sets.modules_list[idx].get_widths(),
            )

        # check the remaining attributes are the same
        for attribute in get_object_attributes(self.grouped_fuzzy_sets):
            value = getattr(self.grouped_fuzzy_sets, attribute)
            if isinstance(value, torch.nn.ModuleList):
                continue  # already checked above
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(loaded_grouped_fuzzy_sets, attribute))
            else:  # for non-tensors
                assert value == getattr(loaded_grouped_fuzzy_sets, attribute)

        # delete the temporary directory using shutil, ignore errors if there are any
        # read-only files

        shutil.rmtree(Path("test_grouped_fuzzy_sets"), ignore_errors=True)

    def test_load_skips_attributes_with_no_setter(self) -> None:
        """
        Coverage/regression test: NestedTorchJitModule.load()'s "setattr() failed
        because it names a read-only property" branch had no test coverage - a
        current save() never actually produces such an attribute (see
        get_object_attributes()'s own _is_read_only_property filter, added
        specifically because centers/widths/mask used to leak through as if they
        were save-able state), so this simulates an older-format save that still has
        one, confirming load() skips it gracefully rather than crashing.

        Returns:
            None
        """
        path = Path("test_group_stale_readonly_attribute")
        self.grouped_fuzzy_sets.save(path)
        try:
            pickle_path = path / f"{FuzzySetGroup.__name__}.pickle"
            with open(pickle_path, "rb") as handle:
                saved_attributes = pickle.load(handle)
            # "centers" is a real read-only property on FuzzySetGroup (no setter) -
            # inject it as if an older save() had included it
            saved_attributes["centers"] = torch.zeros(1)
            with open(pickle_path, "wb") as handle:
                pickle.dump(saved_attributes, handle)

            loaded = FuzzySetGroup.load(path, device=AVAILABLE_DEVICE)
            self.assertIsInstance(loaded, FuzzySetGroup)
            # the injected bogus value must not have overridden the real, computed
            # centers property
            self.assertTrue(
                torch.equal(self.grouped_fuzzy_sets.centers, loaded.centers)
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_hash_eq_contract(self) -> None:
        """
        FuzzySetGroup.__hash__ is identity-based (id(self)) while __eq__ compares member
        FuzzySets by value - the same eq-by-value/hash-by-identity trade-off
        GroupedLinks.__hash__ (linkage.py) and FuzzySet.__hash__ (abstract.py) document.
        Two separately constructed but value-equal groups must still be '==', but are not
        required to hash equal.

        Returns:
            None
        """
        first = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
            ]
        )
        second = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
            ]
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(first))

    def test_to_dtype_only_does_not_corrupt_device(self) -> None:
        """
        Regression test: to() used to assign self.device = device unconditionally, but
        .to() also accepts a dtype-only call (e.g. .to(torch.float64)), in which case
        'device' is actually a dtype - corrupting self.device with it.

        Returns:
            None
        """
        device_before = self.grouped_fuzzy_sets.device
        self.grouped_fuzzy_sets.to(torch.float64)
        self.assertEqual(self.grouped_fuzzy_sets.device, device_before)
        self.assertNotIsInstance(self.grouped_fuzzy_sets.device, torch.dtype)
        # the underlying parameters must still have been converted (via the ordinary
        # recursive nn.Module.to() mechanism, independent of self.device
        # bookkeeping)
        for module in self.grouped_fuzzy_sets.modules_list:
            self.assertEqual(module.get_centers().dtype, torch.float64)

    def test_to_device_only_updates_device(self) -> None:
        """
        A normal device-only .to() call must still update self.device exactly as before.

        Returns:
            None
        """
        self.grouped_fuzzy_sets.to(AVAILABLE_DEVICE)
        self.assertEqual(self.grouped_fuzzy_sets.device, AVAILABLE_DEVICE)

    def test_forward_with_mixed_sparse_and_dense_submodules(self) -> None:
        """
        Regression test: FuzzySetGroup.forward() used to crash when concatenating a group
        that mixes sparse (use_sparse_tensor=True) and dense fuzzy sets, since torch.cat
        cannot mix layouts. This is a realistic configuration - FuzzySetGroup is explicitly
        documented to hold heterogeneous fuzzy sets, and use_sparse_tensor is a legitimate
        per-variable memory choice (e.g. a high-cardinality variable set sparse next to a
        low-cardinality one left dense).

        Returns:
            None
        """
        group = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                    membership_config=MembershipConfig(enable_sparse=True),
                ),
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                    membership_config=MembershipConfig(enable_sparse=False),
                ),
            ]
        )
        input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=AVAILABLE_DEVICE)
        output: Membership = group(input_data)
        self.assertFalse(output.degrees.is_sparse)
        self.assertEqual(output.degrees.shape[-1], 6)

        # an all-sparse group must still produce a sparse result (no unnecessary
        # densification when every submodule agrees)
        all_sparse_group = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                    membership_config=MembershipConfig(enable_sparse=True),
                ),
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                    membership_config=MembershipConfig(enable_sparse=True),
                ),
            ]
        )
        all_sparse_output: Membership = all_sparse_group(input_data)
        self.assertTrue(all_sparse_output.degrees.is_sparse)

    def test_empty_modules_list_defaults(self) -> None:
        """
        FuzzySetGroup() with no modules_list must default to an empty ModuleList rather
        than raising, deferring the "empty" error to forward()/attribute access instead.

        Returns:
            None
        """
        group = FuzzySetGroup()
        self.assertEqual(len(group.modules_list), 0)

    def test_forward_on_empty_group_raises(self) -> None:
        """
        Returns:
            None
        """
        group = FuzzySetGroup()
        with self.assertRaises(ValueError):
            group(torch.zeros(1, 1, device=AVAILABLE_DEVICE))

    def test_attribute_access_on_empty_group_raises(self) -> None:
        """
        Returns:
            None
        """
        group = FuzzySetGroup()
        with self.assertRaises(ValueError):
            _ = group.centers

    def test_multi_module_attribute_caching(self) -> None:
        """
        Coverage/regression test: accessing .centers/.widths/.mask on a group with more
        than one module was never exercised by any existing test - test_save_grouped_
        fuzzy_sets iterates get_object_attributes(), which does not surface these
        __getattribute__-intercepted names. Confirms both the cache-miss (first access)
        and cache-hit (repeated access, unchanged) paths.

        Returns:
            None
        """
        first = self.grouped_fuzzy_sets.centers
        second = self.grouped_fuzzy_sets.centers
        self.assertIs(first, second)  # cached
        self.assertEqual(first.shape[-1], 6)  # 2 modules x 3 terms

        widths = self.grouped_fuzzy_sets.widths
        mask = self.grouped_fuzzy_sets.mask
        self.assertEqual(widths.shape[-1], 6)
        self.assertEqual(mask.shape[-1], 6)

        # mutating a submodule's centers must invalidate the cached
        # concatenation
        with torch.no_grad():
            self.grouped_fuzzy_sets.modules_list[0].get_centers().add_(1.0)
        third = self.grouped_fuzzy_sets.centers
        self.assertIsNot(first, third)

    def test_group_with_generic_non_fuzzyset_modules(self) -> None:
        """
        FuzzySetGroup's own docstring says it can hold any torch.nn.Module, not just
        FuzzySet instances - "the same trick... applies to any kind of torch.nn.Module
        object." This exercises _group_parameter_signature()'s fallback to
        get_centers/get_widths/get_mask for a module that has those but not
        parameter_signature (unlike a real FuzzySet).

        Returns:
            None
        """
        centers = torch.zeros(1, 2, 1, device=AVAILABLE_DEVICE)
        widths = torch.ones(1, 2, 1, device=AVAILABLE_DEVICE)
        mask = torch.ones(2, 1, device=AVAILABLE_DEVICE)
        group = FuzzySetGroup(
            modules_list=[
                _GenericCentersWidthsMaskModule(centers, widths, mask),
                _GenericCentersWidthsMaskModule(centers.clone(), widths, mask),
            ]
        )
        observations = torch.rand(3, 2, device=AVAILABLE_DEVICE)
        first = group(observations)
        second = group(observations)
        self.assertIs(first.degrees, second.degrees)  # cache hit

        with torch.no_grad():
            centers.add_(1.0)
        third = group(observations)
        # invalidated by the mutation
        self.assertIsNot(first.degrees, third.degrees)

    def test_group_with_module_missing_getters_skips_caching(self) -> None:
        """
        A module exposing neither parameter_signature() nor the get_centers/get_widths/
        get_mask trio gives FuzzySetGroup nothing safe to key a memoized result on, so
        _group_parameter_signature() returns None and caching is skipped entirely for the
        whole group (not just that module) - exercising both
        _lookup_group_membership's and _store_group_membership's "signature is None"
        branches.

        Returns:
            None
        """
        degrees = torch.rand(3, 1, 1, device=AVAILABLE_DEVICE)
        mask = torch.ones(1, 1, device=AVAILABLE_DEVICE)
        group = FuzzySetGroup(
            modules_list=[
                _ModuleMissingGetters(degrees, mask),
                _ModuleMissingGetters(degrees.clone(), mask),
            ]
        )
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)
        first = group(observations)
        second = group(observations)
        self.assertIsNot(first.degrees, second.degrees)  # never cached
        self.assertEqual(
            len(group._membership_cache), 0
        )  # pylint: disable=protected-access

    def test_lookup_and_store_defensive_when_cache_attribute_absent(self) -> None:
        """
        _lookup_group_membership()/_store_group_membership() are looked up defensively
        (getattr with a None default) so a torch.jit.script'ed copy of this module - which
        does not carry the cache over - simply calculates memberships instead of failing.
        Exercised directly here by removing the attribute, rather than via an actual
        script'ed copy.

        Returns:
            None
        """
        observations = torch.rand(3, 2, device=AVAILABLE_DEVICE)
        membership = self.grouped_fuzzy_sets(observations)

        del (
            self.grouped_fuzzy_sets._membership_cache
        )  # pylint: disable=protected-access
        cached, signature = (
            self.grouped_fuzzy_sets._lookup_group_membership(  # pylint: disable=protected-access
                observations
            )
        )
        self.assertIsNone(cached)
        self.assertIsNone(signature)
        # must not raise, either
        self.grouped_fuzzy_sets._store_group_membership(  # pylint: disable=protected-access
            observations, membership, signature
        )

    def test_save_and_load_preserves_cache_settings(self) -> None:
        """
        Regression test: cache_membership/membership_cache_size were only ever consumed
        into the underscore-prefixed self._membership_cache, so get_object_attributes()
        (which excludes names starting with '_') never saw them, and NestedTorchJitModule
        .save()/.load() silently dropped them - a loaded group always reverted to the
        constructor defaults (cache_membership=True, membership_cache_size=2) regardless
        of what it was originally built with.

        Returns:
            None
        """
        group = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
            ],
            cache_membership=False,
            membership_cache_size=5,
        )
        self.assertEqual(group.cache_membership, False)
        self.assertEqual(group.membership_cache_size, 5)
        self.assertEqual(
            group._membership_cache.enabled, False
        )  # pylint: disable=protected-access
        self.assertEqual(
            group._membership_cache.maxsize, 5
        )  # pylint: disable=protected-access

        group.save(Path("test_group_cache_settings"))
        loaded_group: FuzzySetGroup = FuzzySetGroup.load(
            Path("test_group_cache_settings"), device=AVAILABLE_DEVICE
        )
        try:
            self.assertEqual(loaded_group.cache_membership, False)
            self.assertEqual(loaded_group.membership_cache_size, 5)
            self.assertEqual(
                loaded_group._membership_cache.enabled,
                False,  # pylint: disable=protected-access
            )
            self.assertEqual(
                loaded_group._membership_cache.maxsize,
                5,  # pylint: disable=protected-access
            )
        finally:
            shutil.rmtree(Path("test_group_cache_settings"), ignore_errors=True)

    def test_eq_with_different_lengths_and_different_content(self) -> None:
        """
        Returns:
            None
        """
        one_module_group = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.LINEAR,
                ),
            ]
        )
        self.assertNotEqual(
            self.grouped_fuzzy_sets, one_module_group
        )  # different length

        different_content_group = FuzzySetGroup(
            modules_list=[
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.RANDOM,
                ),
                Gaussian.create(
                    shape=FuzzySetShape(n_variables=2, n_terms=3),
                    device=AVAILABLE_DEVICE,
                    method=FuzzySetInitMethod.RANDOM,
                ),
            ]
        )
        self.assertNotEqual(
            self.grouped_fuzzy_sets, different_content_group
        )  # same length, different values
