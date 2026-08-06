"""
Common helpers for the unit testing of individual n-ary relations (Minimum, Product,
Compound, etc.), shared to avoid repeating comparison boilerplate across the split-out
test_n_ary_*.py modules.
"""

from typing import Any, List, Union

import torch

from fuzzy.relations.n_ary import NAryRelation
from fuzzy.sets import Membership


def assert_matches_expected(
    actual: torch.Tensor,
    expected_values: Union[List[Any], List[List[Any]]],
    device: torch.device,
    atol: float = 1e-8,
) -> None:
    """
    Assert an n-ary relation's output tensor matches an expected (possibly nested) list
    of values, built into a torch.float32 tensor on the given device.

    Args:
        actual: The tensor produced by the n-ary relation under test.
        expected_values: The expected values, as a (possibly nested) list.
        device: The device to build the expected tensor on.
        atol: The absolute tolerance to use.

    Returns:
        None
    """
    expected = torch.tensor(
        expected_values,
        dtype=torch.float32,
        device=device)
    assert torch.allclose(actual, expected, atol=atol)


def assert_apply_mask_matches_expected(
    n_ary: NAryRelation,
    membership: Membership,
    device: torch.device,
) -> torch.Tensor:
    """
    Apply an n-ary relation's mask to TestNAryRelation.test_gaussian_membership()'s
    fixed fixture membership (built over indices (0, 1), (1, 0)) and assert the
    result matches the one, fixed expected value. apply_mask() is inherited, generic
    NAryRelation behavior - not specific to which t-norm subclass (Minimum, Product,
    etc.) is under test - so the expected value is a genuine constant here, not
    something that varies per caller; shared rather than repeated verbatim (down to
    the literal tensor values) in each relation-specific test module.

    Args:
        n_ary: The n-ary relation to apply the mask with.
        membership: The membership to apply the mask to.
        device: The device to build the expected tensor on.

    Returns:
        The masked result, for the caller to continue using (e.g. as input to a
        relation's forward()).
    """
    after_mask = n_ary.apply_mask(membership=membership)
    assert_matches_expected(
        after_mask,
        [
            [[2.5514542e-04], [7.4245834e-01], [1.0000000e00], [1.0000000e00]],
            [[9.6005607e-01], [8.4526926e-01], [1.0000000e00], [1.0000000e00]],
            [[5.7408627e-04], [9.9679035e-01], [1.0000000e00], [1.0000000e00]],
        ],
        device,
    )
    return after_mask
