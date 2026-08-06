"""
Common functions for the unit testing of the continuous fuzzy sets.
"""

from typing import Optional

import numpy as np
import torch


def get_test_elements(device: torch.device) -> torch.Tensor:
    """
    Get test elements for the unit testing of the continuous fuzzy sets.

    Args:
        device: The device to use.

    Returns:
        The test elements.
    """
    return torch.tensor(
        [[0.41737163], [0.78705574], [0.40919196], [0.72005216]],
        device=device,
    )


def assert_membership_matches_numpy(
    mu_pytorch: torch.Tensor,
    mu_numpy: np.ndarray,
    squeeze_dim: Optional[int] = None,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> None:
    """
    Assert a PyTorch-computed membership tensor matches its NumPy reference
    implementation, within tolerance. Shared by the Gaussian/Triangular/Trapezoidal
    tests to avoid repeating this comparison boilerplate. squeeze_dim mirrors the
    extra batch/variable dimension PyTorch's version commonly carries that the plain
    NumPy formula does not.

    Args:
        mu_pytorch: The membership degrees computed by the PyTorch fuzzy set.
        mu_numpy: The membership degrees computed by the NumPy reference formula.
        squeeze_dim: The dimension to squeeze out of mu_pytorch before comparing, if
            any.
        rtol: The relative tolerance to use.
        atol: The absolute tolerance to use.

    Returns:
        None
    """
    pytorch_values = mu_pytorch
    if squeeze_dim is not None:
        pytorch_values = pytorch_values.squeeze(dim=squeeze_dim)
    assert np.allclose(
        pytorch_values.cpu().detach().numpy(), mu_numpy, rtol=rtol, atol=atol
    )


def assert_jit_script_matches_eager(
    fuzzy_set: torch.nn.Module,
    observations: torch.Tensor,
    eager_degrees: torch.Tensor,
) -> None:
    """
    Assert that torch.jit.script-compiling a fuzzy set produces the same membership
    degrees as the already-computed eager result. Shared by the Gaussian/Triangular/
    Trapezoidal tests to avoid repeating this comparison boilerplate.

    Args:
        fuzzy_set: The (eager) fuzzy set to script and compare against.
        observations: The observations to evaluate the scripted fuzzy set on.
        eager_degrees: The already-computed eager membership degrees to compare
            against.

    Returns:
        None
    """
    scripted = torch.jit.script(fuzzy_set)
    assert torch.allclose(
        scripted(observations).degrees.to_dense(),
        eager_degrees)
