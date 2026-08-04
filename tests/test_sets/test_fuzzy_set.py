"""
Test that various continuous fuzzy set implementations are working as intended, such as the
Gaussian fuzzy set (i.e., membership function), and the Triangular fuzzy set (i.e., membership
function).
"""

import inspect
import os
import shutil
import unittest
from pathlib import Path
from typing import Any, MutableMapping
from unittest import mock

import numpy as np
import torch

from fuzzy.sets import Membership
from fuzzy.sets.abstract import (
    DynamicParameterList,
    FuzzySet,
    FuzzySetInitMethod,
    FuzzySetShape,
)
from fuzzy.sets.impl.basic import NoOp
from fuzzy.sets.impl.gauss_variants.cmf import Gaussian

AVAILABLE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class TestFuzzySet(unittest.TestCase):
    """
    Test the abstract FuzzySet class.
    """

    def test_illegal_attempt_to_create(self) -> None:
        """
        Test that an illegal attempt to create a FuzzySet raises an error.

        Returns:
            None
        """
        with self.assertRaises(NotImplementedError):
            FuzzySet.create(
                shape=FuzzySetShape(n_variables=4, n_terms=2),
                device=torch.device("cpu"),
                method=FuzzySetInitMethod.LINEAR,
            )

    def test_save_and_load(self) -> None:
        """
        Test that saving and loading a FuzzySet works as intended.

        Returns:
            None
        """
        for subclass in FuzzySet.__subclasses__():
            if inspect.isabstract(subclass) or subclass == NoOp:
                continue
            membership_func = subclass.create(
                shape=FuzzySetShape(n_variables=4, n_terms=4),
                device=AVAILABLE_DEVICE,
                method=FuzzySetInitMethod.LINEAR,
            )
            state_dict: MutableMapping = membership_func.state_dict()

            # test that the path must be valid
            with self.assertRaises(ValueError):
                membership_func.save(Path(""))
            with self.assertRaises(ValueError):
                membership_func.save(Path("test"))
            with self.assertRaises(ValueError):
                # this file extension is not supported; see error message to
                # learn why
                membership_func.save(Path("test.pth"))

            # test that saving the state dict works
            saved_state_dict: MutableMapping[str, Any] = membership_func.save(
                Path("membership_func.pt")
            )

            # check that the saved state dict is the same as the original state
            # dict
            for key in state_dict.keys():
                assert key in saved_state_dict and torch.allclose(
                    state_dict[key], saved_state_dict[key]
                )
            # except the saved state dict includes additional information not captured by
            # the original state dict, such as the class name and the labels
            assert "class_name" in saved_state_dict.keys() and saved_state_dict[
                "class_name"
            ] in (subclass.__name__ for subclass in FuzzySet.__subclasses__())

            loaded_membership_func = FuzzySet.load(
                Path("membership_func.pt"), device=AVAILABLE_DEVICE
            )
            # check that the parameters and members are the same
            assert membership_func == loaded_membership_func
            assert torch.allclose(
                membership_func.get_centers(),
                loaded_membership_func.get_centers())
            assert torch.allclose(
                membership_func.get_widths(),
                loaded_membership_func.get_widths())
            if isinstance(
                subclass, Gaussian
            ):  # Gaussian has an additional parameter (alias for widths)
                assert torch.allclose(
                    membership_func.sigmas, loaded_membership_func.sigmas
                )
            # check some functionality that it is still working
            assert torch.allclose(
                membership_func.area(),
                loaded_membership_func.area())
            assert torch.allclose(
                membership_func(
                    torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=AVAILABLE_DEVICE)
                ).degrees.to_dense(),
                loaded_membership_func(
                    torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=AVAILABLE_DEVICE)
                ).degrees.to_dense(),
            )
            # delete the file
            os.remove("membership_func.pt")

    def test_save_and_load_of_no_op(self) -> None:
        """
        Test that saving and loading a NoOp (i.e., no operation) works as intended.

        Returns:
            None
        """
        no_op = NoOp(n_elements=10, membership=0.75, device=AVAILABLE_DEVICE)
        membership = no_op(
            torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=AVAILABLE_DEVICE)
        )
        self.save_and_load_of_no_op_helper(membership, no_op)
        # test saving and loading of the NoOp
        tmp_file = Path("tmp.pt")
        no_op.save(tmp_file)
        loaded_no_op = NoOp.load(tmp_file, device=AVAILABLE_DEVICE)
        # delete the temporary file
        tmp_file.unlink()

        self.save_and_load_of_no_op_helper(membership, loaded_no_op)

    def save_and_load_of_no_op_helper(
        self, membership: Membership, no_op: "FuzzySet"
    ) -> None:
        """
        A simple helper method to check the behavior of no operation (i.e., do nothing)
        membership function works as intended both before and after saving/loading.

        Args:
            membership: The membership degrees.
            no_op: The no operation.

        Returns:
            None
        """
        self.assertEqual(no_op.get_centers().size()[0], 10)
        self.assertEqual(membership.degrees.size()[1], 4)
        self.assertNotEqual(
            no_op.get_centers().size()[0],
            membership.degrees.size()[1])
        self.assertAlmostEqual(
            no_op.membership,
            membership.degrees.mean().item())

    def test_hash_eq_contract(self) -> None:
        """
        Regression test: FuzzySet.__hash__ used to hash the centers/widths *tensors*
        themselves, which torch.Tensor hashes by identity (id()) rather than by value. Since
        __eq__ compares by value (torch.equal), two separately constructed but value-equal
        fuzzy sets were '==' yet had different hashes - a violation of Python's hash contract
        (equal objects must report equal hashes) that breaks their use as dict keys or set
        members.

        Returns:
            None
        """
        for subclass in FuzzySet.__subclasses__():
            if inspect.isabstract(subclass) or subclass == NoOp:
                continue
            first = subclass.create(
                shape=FuzzySetShape(n_variables=2, n_terms=3),
                device=AVAILABLE_DEVICE,
                method=FuzzySetInitMethod.LINEAR,
            )
            second = subclass.create(
                shape=FuzzySetShape(n_variables=2, n_terms=3),
                device=AVAILABLE_DEVICE,
                method=FuzzySetInitMethod.LINEAR,
            )
            # two separately constructed fuzzy sets, built the same way, have identical
            # parameter values but are distinct objects (and therefore distinct underlying
            # tensors) - this is exactly the case that must not violate the
            # hash contract
            self.assertEqual(
                first,
                second,
                f"{subclass.__name__} instances with identical parameters should be equal",
            )
            self.assertEqual(
                hash(first),
                hash(second),
                f"{subclass.__name__}.__hash__ violates the hash contract: "
                f"equal instances must have equal hashes",
            )
            # a fuzzy set must also consistently hash the same as itself across
            # repeated calls
            self.assertEqual(hash(first), hash(first))

    def test_dynamic_parameter_list_to_dtype_only(self) -> None:
        """
        Regression test: DynamicParameterList.to() assumed the first positional argument was
        always a device (self._device = args[0] if args else self._device). torch.nn.Module.to()
        also accepts a dtype-only call (e.g. .to(torch.float64)), which used to corrupt
        _device with a dtype object - later breaking add_parameter() and the empty-list branch
        of the 'tensor' property, both of which pass _device to torch.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0]),
            widths=np.array([1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        # resolve the expected device the same way .to() would (e.g. an unindexed "cuda"
        # resolves to a concrete "cuda:0" once actually applied to a tensor), so the
        # comparison below is correct on both CPU-only and CUDA machines
        resolved_device = torch.empty(0, device=AVAILABLE_DEVICE).device

        gaussian_mf.to(torch.float64)

        self.assertEqual(gaussian_mf._centers._device, resolved_device)
        self.assertEqual(gaussian_mf._centers._dtype, torch.float64)
        self.assertEqual(gaussian_mf.get_centers().dtype, torch.float64)

        # adding a parameter afterward must use the up-to-date dtype, not a
        # stale one
        gaussian_mf._centers.add_parameter(np.array([[2.0, 3.0]]))
        self.assertEqual(gaussian_mf._centers.params[-1].dtype, torch.float64)

        # an empty DynamicParameterList must not raise when .to() is given a
        # dtype only
        empty = DynamicParameterList(
            device=AVAILABLE_DEVICE,
            dtype=torch.float32)
        empty.to(torch.float64)
        self.assertEqual(empty._dtype, torch.float64)
        self.assertEqual(empty.tensor.dtype, torch.float64)

    def test_multi_parameter_tensor_caching(self) -> None:
        """
        Coverage/regression test: DynamicParameterList.tensor's multi-parameter branch
        (_concat_params) was never exercised by any existing test - every existing fuzzy
        set is built with exactly one parameter per centers/widths/mask list. This
        directly tests that with two or more parameters, the concatenation is correctly
        cached and invalidated: a repeated access without changes returns the identical
        cached object, but a subsequent in-place mutation (as an optimizer applies) or
        outright replacement (__setitem__) correctly invalidates it - echoing the
        training-safety fix this session made for the single-parameter case.

        Returns:
            None
        """
        params = DynamicParameterList(
            init_params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )
        self.assertEqual(len(params.params), 2)

        first = params.tensor
        second = params.tensor
        self.assertIs(first, second)  # cached, same object

        with torch.no_grad():
            params.params[0].add_(1.0)  # mutate one parameter in place
        third = params.tensor
        self.assertIsNot(first, third)  # cache correctly invalidated
        self.assertTrue(
            torch.allclose(
                third,
                torch.tensor([2.0, 3.0, 3.0, 4.0], device=AVAILABLE_DEVICE),
            )
        )

        # __setitem__ requires a torch.nn.Parameter
        with self.assertRaises(TypeError):
            params[1] = torch.tensor([5.0, 6.0], device=AVAILABLE_DEVICE)

        # replacing a parameter outright must also invalidate the cache
        params[1] = torch.nn.Parameter(
            torch.tensor([5.0, 6.0], device=AVAILABLE_DEVICE)
        )
        fourth = params.tensor
        self.assertIsNot(third, fourth)
        self.assertTrue(
            torch.allclose(
                fourth,
                torch.tensor([2.0, 3.0, 5.0, 6.0], device=AVAILABLE_DEVICE),
            )
        )

    def test_multi_parameter_tensor_requires_grad_toggle(self) -> None:
        """
        Regression test: _concat_params's cache is keyed via signature_of(), which used to
        omit requires_grad entirely - the same root-cause bug as
        TestMembershipCache.test_requires_grad_toggle_invalidates_cache, but for the
        multi-parameter concatenation cache instead of the membership cache. Freezing one
        parameter, concatenating (caching the result), then unfreezing it must not hand
        back the frozen-graph concatenation: the parameter's gradient would otherwise stay
        None forever even though it is trainable again.

        Returns:
            None
        """
        params = DynamicParameterList(
            init_params=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )
        params.params[0].requires_grad_(False)
        first = params.tensor

        params.params[0].requires_grad_(True)
        second = params.tensor
        self.assertIsNot(first, second)

        second.sum().backward()
        self.assertIsNotNone(params.params[0].grad)

    def test_init_method_unsupported_raises(self) -> None:
        """
        FuzzySetInitMethod.initialize() must raise ValueError for a value it does not
        recognize (a defensive branch for any future enum member added without updating
        this method).

        Returns:
            None
        """

        class _FakeMethod:  # pylint: disable=too-few-public-methods
            pass

        with self.assertRaises(ValueError):
            FuzzySetInitMethod.initialize(
                _FakeMethod(), shape=FuzzySetShape(n_variables=1, n_terms=2)
            )

    def test_render_and_latex_formula(self) -> None:
        """
        render_formula()/latex_formula() were never exercised by any existing test.

        Returns:
            None
        """
        for subclass in FuzzySet.__subclasses__():
            if inspect.isabstract(subclass) or subclass == NoOp:
                continue
            # some subclasses' sympy_formula() is unimplemented (returns None); this is
            # pre-existing behavior, so just confirm neither call raises
            subclass.render_formula()
            latex = subclass.latex_formula()
            self.assertIsInstance(latex, str)

    def test_extend(self) -> None:
        """
        extend() was never exercised by any existing test. Confirm it grows the
        parameters correctly in both 'horizontal' (more terms) and 'vertical' (more
        variables) modes, correctly invalidates any previously memoized membership, and
        rejects an unrecognized mode.

        Returns:
            None
        """
        horizontal_mf = Gaussian(
            centers=np.array([[0.0, 1.0]]),
            widths=np.array([[1.0, 1.0]]),
            device=AVAILABLE_DEVICE,
        )
        observations = torch.tensor([[0.5]], device=AVAILABLE_DEVICE)
        first = horizontal_mf(observations)
        horizontal_mf.extend(
            centers=torch.tensor([[4.0]], device=AVAILABLE_DEVICE),
            widths=torch.tensor([[1.0]], device=AVAILABLE_DEVICE),
            mode="horizontal",
        )
        self.assertEqual(tuple(horizontal_mf.get_centers().shape), (1, 3))
        second = horizontal_mf(observations)
        self.assertIsNot(first.degrees, second.degrees)  # cache invalidated
        self.assertEqual(second.degrees.shape[-1], 3)

        vertical_mf = Gaussian(
            centers=np.array([[0.0, 1.0]]),
            widths=np.array([[1.0, 1.0]]),
            device=AVAILABLE_DEVICE,
        )
        vertical_mf.extend(
            centers=torch.tensor([[4.0, 5.0]], device=AVAILABLE_DEVICE),
            widths=torch.tensor([[1.0, 1.0]], device=AVAILABLE_DEVICE),
            mode="vertical",
        )
        self.assertEqual(tuple(vertical_mf.get_centers().shape), (2, 2))

        with self.assertRaises(ValueError):
            vertical_mf.extend(
                centers=torch.tensor([[1.0]], device=AVAILABLE_DEVICE),
                widths=torch.tensor([[1.0]], device=AVAILABLE_DEVICE),
                mode="not_a_real_mode",
            )

    def test_area_forces_zero_for_missing_term(self) -> None:
        """
        _area_helper() must force a "missing" term's (width <= 0) computed area to
        exactly 0.0. Gaussian's formula squares the width, so a negative width produces
        the same nonzero integral as its positive counterpart, which is exactly the case
        this override exists to correct.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([[0.0, 1.0]]),
            widths=np.array([[1.0, -1.0]]),
            device=AVAILABLE_DEVICE,
        )
        area = gaussian_mf.area()
        self.assertEqual(area[0, 1].item(), 0.0)

    def test_plot_single_variable_with_missing_term_and_highlight(
            self) -> None:
        """
        Covers plot() branches that the general cross-subclass plot test
        (test_impl.py::test_plot) does not reach: a single-variable fuzzy set (the
        ndim==1/shape[0]==1 branch, whose broadcasted memberships also come out 2D and
        need an unsqueeze), a "missing" (mask == 0) term being skipped, and a selected
        term being highlighted via fill_between.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0, 2.0]),
            widths=np.array([1.0, 1.0, -1.0]),
            device=AVAILABLE_DEVICE,
        )
        output_dir = Path(__file__).parent / "plots_single_var"
        try:
            figures, axes = gaussian_mf.plot(
                output_dir=output_dir, selected_terms=[(0, 0)]
            )
            self.assertIsNotNone(axes)
            self.assertTrue(output_dir.exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_stack_eval_mode(self) -> None:
        """
        FuzzySet.stack()'s eval-mode branch (padding missing terms with NaN instead of
        0.0/-1.0) was never exercised by any existing test.

        Returns:
            None
        """
        gaussian_a = Gaussian(
            centers=np.array([0.0, 1.0]),
            widths=np.array([1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        gaussian_b = Gaussian(
            centers=np.array([0.0, 1.0, 2.0]),
            widths=np.array([1.0, 1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        gaussian_a.eval()
        gaussian_b.eval()
        stacked = FuzzySet.stack([gaussian_a, gaussian_b])
        self.assertTrue(bool(torch.isnan(stacked.get_centers()[0, -1])))

    def test_nan_observation_still_produces_nan_degree(self) -> None:
        """
        _calculate_membership_nan_safe() must not change any VALUE calculate_membership()
        would have produced; only the gradient path changes. A NaN observation must still
        yield a NaN degree, matching the "NaN in -> NaN out" contract that
        NAryRelation.nan_replacement (a downstream consumer) relies on to detect missing
        data.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0]),
            widths=np.array([1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        observations = torch.tensor(
            [[0.3], [float("nan")], [0.9]], device=AVAILABLE_DEVICE
        )
        degrees = gaussian_mf(observations).degrees.to_dense()
        self.assertTrue(bool(degrees[1].isnan().all()))
        self.assertFalse(bool(degrees[0].isnan().any()))
        self.assertFalse(bool(degrees[2].isnan().any()))

        # and it must match calculate_membership() exactly for the non-NaN rows
        direct = gaussian_mf.calculate_membership(observations.unsqueeze(-1))
        self.assertTrue(torch.allclose(degrees[0], direct[0]))
        self.assertTrue(torch.allclose(degrees[2], direct[2]))

    def test_nan_observation_does_not_corrupt_shared_parameter_gradient(
            self) -> None:
        """
        Regression test: a membership formula such as Gaussian's
        exp(-((x-c)^2)/(2w^2)) has a local derivative with respect to its parameters that
        is itself NaN whenever x is NaN. Since centers/widths are shared across the whole
        batch, PyTorch's chain rule used to multiply that NaN local derivative through
        (0 * NaN = NaN under IEEE754) and silently corrupt the gradient for the entire
        training step - not just the one missing observation. This must no longer happen:
        one NaN observation among several must not poison the others' contribution to the
        shared parameters' gradient.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([[0.0, 1.0]]),
            widths=np.array([[1.0, 1.0]]),
            device=AVAILABLE_DEVICE,
        )
        observations = torch.tensor(
            [[0.3], [float("nan")], [0.9]], device=AVAILABLE_DEVICE
        )
        degrees = gaussian_mf(observations).degrees.to_dense()
        # nan_to_num *per element*, before summing - summing first would make the whole
        # scalar loss NaN, and nan_to_num on that would zero the gradient entirely rather
        # than excluding only the missing row's contribution
        degrees.nan_to_num(0.0).sum().backward()

        self.assertFalse(bool(gaussian_mf.get_centers().grad.isnan().any()))
        self.assertFalse(bool(gaussian_mf.get_widths().grad.isnan().any()))
        # the valid samples must still have contributed a genuine (nonzero)
        # gradient
        self.assertTrue(bool((gaussian_mf.get_centers().grad != 0).any()))

    def test_no_nan_fast_path_matches_calculate_membership(self) -> None:
        """
        When no observation is NaN, _calculate_membership_nan_safe() must return
        exactly what calculate_membership() would, regardless of which of its two
        size-dependent strategies is used internally (see
        test_no_nan_large_tensor_skips_safety_machinery and
        test_no_nan_small_tensor_always_uses_safe_path for those specifically) - a
        NaN-free torch.where is a value no-op either way.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0]),
            widths=np.array([1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        observations = torch.tensor(
            [[0.3], [0.6], [0.9]], device=AVAILABLE_DEVICE)
        via_forward = gaussian_mf(observations).degrees.to_dense()
        direct = gaussian_mf.calculate_membership(observations.unsqueeze(-1))
        self.assertTrue(torch.equal(via_forward, direct))

    def test_no_nan_large_tensor_skips_safety_machinery(self) -> None:
        """
        Regression/performance test: _calculate_membership_nan_safe used to always
        sync (bool(nan_mask.any())) to decide whether the NaN-safe substitution was
        needed. Calibration showed that sync costs ~400-750us in practice (dominated by
        the reduction kernel and blocking scalar readback, not the sync primitive
        itself), while the substitution it decides whether to skip only costs that much
        once the resulting degrees tensor is large - see
        FuzzySet._nan_safe_sync_threshold_numel. Above that threshold, with no NaN
        present, the substitution (implemented via torch.where) must still be skipped
        entirely.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0, 0.5, 0.25, 0.75]),
            widths=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        # degrees numel == batch_size * n_terms(5); comfortably exceeds the threshold
        batch_size = gaussian_mf._nan_safe_sync_threshold_numel // 5 + 10  # pylint: disable=protected-access
        observations = torch.rand(batch_size, 1, device=AVAILABLE_DEVICE)

        with mock.patch("torch.where", autospec=True) as mocked_where:
            gaussian_mf(observations)
        mocked_where.assert_not_called()

    def test_no_nan_small_tensor_always_uses_safe_path(self) -> None:
        """
        Below FuzzySet._nan_safe_sync_threshold_numel, the sync needed to decide
        whether the NaN-safe substitution is necessary costs more than just always
        performing it -
        so below the threshold, the substitution (torch.where, called twice: once to
        build the NaN-free input, once to re-inject NaN into the output) must always
        run, even when there is no NaN present (a harmless no-op at this size).

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0]),
            widths=np.array([1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        observations = torch.tensor(
            [[0.3], [0.6], [0.9]], device=AVAILABLE_DEVICE
        )  # tiny; no NaN anywhere

        with mock.patch(
            "torch.where", autospec=True, side_effect=torch.where
        ) as mocked_where:
            gaussian_mf(observations)
        self.assertEqual(mocked_where.call_count, 2)

    def test_nan_observation_still_produces_nan_degree_large_tensor(self) -> None:
        """
        The large-tensor, sync-gated branch must still produce the documented
        "NaN observation -> NaN degree" contract when a NaN is actually present, not
        just in the small-tensor always-safe branch (see
        test_nan_observation_still_produces_nan_degree for that one).

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([0.0, 1.0, 0.5, 0.25, 0.75]),
            widths=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        batch_size = gaussian_mf._nan_safe_sync_threshold_numel // 5 + 10  # pylint: disable=protected-access
        observations = torch.rand(batch_size, 1, device=AVAILABLE_DEVICE)
        observations[0, 0] = float("nan")

        degrees = gaussian_mf(observations).degrees.to_dense()
        self.assertTrue(bool(degrees[0].isnan().all()))
        self.assertFalse(bool(degrees[1:].isnan().any()))
