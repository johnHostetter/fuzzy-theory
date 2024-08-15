"""
Common functions for the unit testing of the continuous fuzzy sets.
"""

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
