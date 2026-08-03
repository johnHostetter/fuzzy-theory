"""
Test the membership cache (see fuzzy.sets.cache): repeated calls to a fuzzy set with the same
observations, while its parameters are unchanged, must return the previously computed result
verbatim - without breaking gradients, without ever handing back a result whose autograd graph
was already consumed by backward(), and without silently going stale once an optimizer step
changes the underlying parameters.
"""

import gc
import unittest
import weakref

import numpy as np
import torch

from fuzzy.sets.group import FuzzySetGroup
from fuzzy.sets.impl import Gaussian, Trapezoidal

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_gaussian(cache_membership: bool = True) -> Gaussian:
    """
    Build a small, single-variable, two-term Gaussian fuzzy set for testing the cache.
    """
    return Gaussian(
        centers=np.array([0.0, 1.0]),
        widths=np.array([1.0, 1.0]),
        device=AVAILABLE_DEVICE,
        cache_membership=cache_membership,
    )


def make_gaussian_pair() -> "tuple[Gaussian, Gaussian]":
    """
    Build two Gaussian fuzzy sets with identical parameters, one cached and one not, so their
    outputs and gradients can be compared directly.
    """
    centers = np.array([0.2, 0.6, 0.9])
    widths = np.array([0.5, 0.4, 0.6])
    cached_mf = Gaussian(
        centers=centers.copy(),
        widths=widths.copy(),
        device=AVAILABLE_DEVICE,
        cache_membership=True,
    )
    uncached_mf = Gaussian(
        centers=centers.copy(),
        widths=widths.copy(),
        device=AVAILABLE_DEVICE,
        cache_membership=False,
    )
    return cached_mf, uncached_mf


class TestMembershipCache(unittest.TestCase):
    """
    Test that FuzzySet.forward's membership cache is correct under training, and that it
    behaves safely around the autograd edge cases it exists to handle.
    """

    def test_training_updates_centers_and_loss(self) -> None:
        """
        Regression test for the bug that made this cache necessary in the first place:
        DynamicParameterList.tensor used to snapshot the concatenated parameters once and
        never notice an optimizer's in-place update, so get_centers() (and therefore the
        loss) stayed frozen across an entire training loop.

        Returns:
            None
        """
        torch.manual_seed(0)
        gaussian_mf = make_gaussian()
        optimizer = torch.optim.SGD(gaussian_mf.parameters(), lr=0.5)
        observations = torch.tensor([[2.0]], device=AVAILABLE_DEVICE)

        losses = []
        centers_snapshots = []
        for _ in range(3):
            optimizer.zero_grad()
            degrees = gaussian_mf(observations).degrees.to_dense()
            loss = degrees.sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            centers_snapshots.append(
                gaussian_mf.get_centers().detach().clone())

        self.assertFalse(
            losses[0] == losses[1] == losses[2],
            "The loss never changed; training against the cache appears to be frozen.",
        )
        self.assertFalse(
            torch.equal(centers_snapshots[0], centers_snapshots[-1]),
            "get_centers() did not track the optimizer's updates.",
        )

    def test_cache_hit_returns_identical_object(self) -> None:
        """
        A repeated call with the same observations tensor, and unchanged parameters, must
        return the exact same degrees tensor object rather than recomputing it.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(4, 1, device=AVAILABLE_DEVICE)
        first = gaussian_mf(observations)
        second = gaussian_mf(observations)
        self.assertIs(first.degrees, second.degrees)

    def test_cached_matches_uncached_values(self) -> None:
        """
        A cached result must be numerically identical to what a cache-disabled fuzzy set with
        the same parameters computes, and must still carry a gradient function.

        Returns:
            None
        """
        cached_mf, uncached_mf = make_gaussian_pair()
        observations = torch.rand(5, 1, device=AVAILABLE_DEVICE)

        cached_out = cached_mf(observations).degrees.to_dense()
        uncached_out = uncached_mf(observations).degrees.to_dense()

        self.assertTrue(torch.allclose(cached_out, uncached_out))
        self.assertIsNotNone(cached_out.grad_fn)

    def test_gradients_match_uncached(self) -> None:
        """
        Gradients computed through a cached result must match gradients computed through an
        equivalent cache-disabled fuzzy set exactly.

        Returns:
            None
        """
        cached_mf, uncached_mf = make_gaussian_pair()
        observations = torch.rand(5, 1, device=AVAILABLE_DEVICE)

        cached_out = cached_mf(observations).degrees.to_dense()
        uncached_out = uncached_mf(observations).degrees.to_dense()

        (cached_grad,) = torch.autograd.grad(
            cached_out.sum(), cached_mf.get_centers(), retain_graph=True
        )
        (uncached_grad,) = torch.autograd.grad(
            uncached_out.sum(), uncached_mf.get_centers(), retain_graph=True
        )
        self.assertTrue(torch.allclose(cached_grad, uncached_grad))

    def test_forward_backward_step_forward(self) -> None:
        """
        A full forward -> backward -> step -> forward loop must run without ever trying to
        backward through an already-consumed graph, and the second forward's result must
        differ from the first, since the parameters moved in between.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        optimizer = torch.optim.SGD(gaussian_mf.parameters(), lr=0.5)
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        optimizer.zero_grad()
        first = gaussian_mf(observations)
        first.degrees.to_dense().sum().backward()
        optimizer.step()

        second = gaussian_mf(observations)
        self.assertIsNot(second.degrees, first.degrees)

        # if the now-stale entry had been served, this backward() would raise "Trying to
        # backward through the graph a second time"
        optimizer.zero_grad()
        second.degrees.to_dense().sum().backward()
        optimizer.step()

    def test_entry_invalidated_by_parameter_update(self) -> None:
        """
        An in-place update to a parameter (as an optimizer performs, without necessarily
        calling backward() again first) must invalidate any cached entry keyed on it.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        first = gaussian_mf(observations)
        with torch.no_grad():
            gaussian_mf.get_centers().add_(1.0)
        second = gaussian_mf(observations)

        self.assertIsNot(first.degrees, second.degrees)
        self.assertFalse(
            torch.allclose(
                first.degrees.to_dense().detach(),
                second.degrees.to_dense().detach()))

    def test_no_grad_entry_not_served_to_grad_enabled_caller(self) -> None:
        """
        A membership calculated with gradients disabled must never be handed back to a
        subsequent call that needs gradients (or vice versa).

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        with torch.no_grad():
            no_grad_result = gaussian_mf(observations)
        self.assertIsNone(no_grad_result.degrees.grad_fn)

        grad_result = gaussian_mf(observations)
        self.assertIsNotNone(grad_result.degrees.grad_fn)
        self.assertIsNot(grad_result.degrees, no_grad_result.degrees)

    def test_requires_grad_toggle_invalidates_cache(self) -> None:
        """
        Regression test: signature_of() used to key a cache entry on only (id, version) of
        each parameter, missing requires_grad entirely. Freezing a parameter
        (requires_grad_(False)), computing a membership, then unfreezing it
        (requires_grad_(True)) and reusing the same observations tensor used to silently
        serve the frozen-graph result - the parameter's gradient would then stay None
        forever afterward, even though it is trainable again, because neither identity,
        version, nor the *global* torch.is_grad_enabled() flag change when only a single
        parameter's requires_grad is toggled. This is a very standard "freeze, train other
        things, unfreeze" research pattern, so it must not silently break.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        gaussian_mf.get_centers().requires_grad_(False)
        first = gaussian_mf(observations)

        gaussian_mf.get_centers().requires_grad_(True)
        second = gaussian_mf(observations)
        self.assertIsNot(
            first.degrees,
            second.degrees,
            "the cache served a result computed while centers were frozen, after "
            "centers were unfrozen",
        )

        second.degrees.to_dense().sum().backward()
        self.assertIsNotNone(gaussian_mf.get_centers().grad)
        self.assertFalse(bool((gaussian_mf.get_centers().grad == 0).all()))

    def test_observations_requires_grad_toggle_invalidates_cache(self) -> None:
        """
        The same requires_grad blind spot applies to the observations tensor itself, not
        just the fuzzy set's own parameters: toggling requires_grad on the *same*
        observations object between calls (e.g. enabling input-gradient tracking for a
        saliency/adversarial-gradient computation) must not be served a cached result
        computed before that toggle.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        first = gaussian_mf(observations)

        observations.requires_grad_(True)
        second = gaussian_mf(observations)
        self.assertIsNot(first.degrees, second.degrees)

        second.degrees.to_dense().sum().backward()
        self.assertIsNotNone(observations.grad)

    def test_inference_mode_does_not_cache(self) -> None:
        """
        Nothing should be memoized while torch.inference_mode() is active.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        with torch.inference_mode():
            first = gaussian_mf(observations)
            second = gaussian_mf(observations)

        self.assertIsNot(first.degrees, second.degrees)
        self.assertEqual(len(gaussian_mf._membership_cache), 0)

    def test_cache_membership_false_disables_caching(self) -> None:
        """
        With cache_membership=False, repeated calls must always recompute.

        Returns:
            None
        """
        gaussian_mf = make_gaussian(cache_membership=False)
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        first = gaussian_mf(observations)
        second = gaussian_mf(observations)
        self.assertIsNot(first.degrees, second.degrees)

    def test_clear_membership_cache(self) -> None:
        """
        clear_membership_cache() must force the next call to recompute.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        first = gaussian_mf(observations)
        gaussian_mf.clear_membership_cache()
        second = gaussian_mf(observations)
        self.assertIsNot(first.degrees, second.degrees)

    def test_jit_script_matches_eager(self) -> None:
        """
        torch.jit.script must still succeed on a fuzzy set with caching enabled, and produce
        the same values as the eager module.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        eager_out = gaussian_mf(observations).degrees.to_dense()
        scripted = torch.jit.script(gaussian_mf)
        scripted_out = scripted(observations).degrees.to_dense()
        self.assertTrue(torch.allclose(eager_out, scripted_out))

    def test_cache_does_not_keep_dead_observations_alive(self) -> None:
        """
        A cache entry only holds a weak reference to the observations it was computed for, so
        it must not be the reason a batch of observations stays alive after the caller drops
        its own reference. This is tested under plain no_grad() rather than inference_mode(),
        since with gradients enabled the autograd graph itself - not the cache - would keep
        the observations alive via its saved tensors, which would defeat the point of the test.

        Returns:
            None
        """
        gaussian_mf = make_gaussian()
        observations = torch.rand(3, 1, device=AVAILABLE_DEVICE)

        with torch.no_grad():
            gaussian_mf(observations)
        self.assertEqual(len(gaussian_mf._membership_cache), 1)

        observations_ref = weakref.ref(observations)
        del observations
        gc.collect()
        self.assertIsNone(observations_ref())


class TestFuzzySetGroupMembershipCache(unittest.TestCase):
    """
    Test that FuzzySetGroup.forward's memoized concatenation is subject to the same
    correctness rules as a single fuzzy set's cache.
    """

    def _make_group(self) -> FuzzySetGroup:
        return FuzzySetGroup(
            modules_list=[
                Gaussian(
                    centers=np.array([0.0, 1.0]),
                    widths=np.array([1.0, 1.0]),
                    device=AVAILABLE_DEVICE,
                ),
                Gaussian(
                    centers=np.array([0.5, 1.5]),
                    widths=np.array([1.0, 1.0]),
                    device=AVAILABLE_DEVICE,
                ),
            ]
        )

    def test_cache_hit_returns_identical_object(self) -> None:
        """
        Returns:
            None
        """
        group = self._make_group()
        observations = torch.rand(4, 1, device=AVAILABLE_DEVICE)
        first = group(observations)
        second = group(observations)
        self.assertIs(first.degrees, second.degrees)

    def test_training_updates_group_output(self) -> None:
        """
        Returns:
            None
        """
        group = self._make_group()
        optimizer = torch.optim.SGD(group.parameters(), lr=0.5)
        observations = torch.tensor([[2.0]], device=AVAILABLE_DEVICE)

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            degrees = group(observations).degrees.to_dense()
            loss = degrees.sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertFalse(losses[0] == losses[1] == losses[2])

    def test_requires_grad_toggle_invalidates_group_cache(self) -> None:
        """
        Regression test: the same requires_grad blind spot that could affect a single
        FuzzySet's cache (see TestMembershipCache) applies to FuzzySetGroup's own
        group-level cache too, since both share signature_of(). Freezing then unfreezing
        a submodule's parameter, while reusing the same observations, must not serve a
        group-level result whose graph was built while that parameter was frozen.

        Returns:
            None
        """
        group = self._make_group()
        observations = torch.rand(4, 1, device=AVAILABLE_DEVICE)

        first_module = group.modules_list[0]
        first_module.get_centers().requires_grad_(False)
        first = group(observations)

        first_module.get_centers().requires_grad_(True)
        second = group(observations)
        self.assertIsNot(first.degrees, second.degrees)

        second.degrees.to_dense().sum().backward()
        self.assertIsNotNone(first_module.get_centers().grad)

    def test_submodule_extra_parameter_invalidates_group_cache(self) -> None:
        """
        Regression test: FuzzySetGroup._group_parameter_signature() used to hardcode
        get_centers()/get_widths()/get_mask() for every submodule instead of deferring to
        each submodule's own parameter_signature(). A submodule with an extra learnable
        parameter beyond those three - such as Trapezoidal's plateaus - could then have that
        parameter change without the group-level cache ever noticing, serving a stale
        concatenated Membership even though the submodule's own individual cache was correct.

        Returns:
            None
        """
        group = FuzzySetGroup(
            modules_list=[
                Gaussian(
                    centers=np.array([0.0, 1.0]),
                    widths=np.array([1.0, 1.0]),
                    device=AVAILABLE_DEVICE,
                ),
                Trapezoidal(
                    centers=np.array([0.0, 1.0]),
                    widths=np.array([1.0, 1.0]),
                    plateaus=np.array([0.2, 0.2]),
                    device=AVAILABLE_DEVICE,
                ),
            ]
        )
        observations = torch.rand(4, 1, device=AVAILABLE_DEVICE)

        first = group(observations)
        second = group(observations)
        self.assertIs(
            first.degrees,
            second.degrees,
            "expected a cache hit when nothing has changed",
        )

        trapezoidal_submodule = group.modules_list[1]
        with torch.no_grad():
            trapezoidal_submodule.get_plateaus().add_(0.3)  # mutate plateaus ONLY

        third = group(observations)
        self.assertIsNot(
            first.degrees,
            third.degrees,
            "the group cache served a stale result after a submodule's plateaus changed",
        )
        self.assertFalse(
            torch.allclose(
                first.degrees.to_dense().detach(),
                third.degrees.to_dense().detach()),
            "the group output did not reflect the updated plateaus",
        )


if __name__ == "__main__":
    unittest.main()
