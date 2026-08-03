"""
This script contains various mechanisms to enable the concept of confidence in rules when
performing model inference.
"""

from pathlib import Path
from typing import Any, MutableMapping

import torch


class CertaintyFactors(torch.nn.Module):
    """
    A class that provides access to certainty factors, which may be used in the calculation of the
    rules' firing levels by reflecting on the model's corresponding confidence in them.
    """

    def __init__(self, weights: torch.Tensor, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tensor.to() is not in-place; the previous version discarded its return value,
        # so weights were silently left on their original device whenever it differed
        # from the requested one
        weights = weights.to(device)
        self.weights = torch.nn.Parameter(
            weights,
            requires_grad=True,
        )

    @staticmethod
    def create_default(n_features: int, device, *args,
                       **kwargs) -> "CertaintyFactors":
        """
        A protocol to create an instance with default settings.

        Args:
            n_features: The number of features in the model.
            device: The device on which the certainty factors should be created.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A CertaintyFactors object.
        """
        weights = torch.ones([n_features], dtype=torch.float32, device=device)
        return CertaintyFactors(weights, device, *args, **kwargs)

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the current certainty factors to a directory.

        Args:
            path: An instance of Path that describes a directory path.

        Returns:
            The mutable mapping, or state dictionary, that was saved to file.
        """
        path.mkdir(parents=True, exist_ok=True)
        state_dict: MutableMapping[str, Any] = self.state_dict()
        torch.save(state_dict, path / "state_dict.pt")
        return state_dict

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "CertaintyFactors":
        """
        Load the certainty factors from a directory and put it on the specified device.

        Args:
            path: An instance of Path that describes a directory path containing "state_dict.pt".
            device: The device on which the certainty factors should be loaded.

        Returns:
            A CertaintyFactors object.
        """
        file_path = path / "state_dict.pt"
        if file_path.is_file():
            # map_location=device avoids two failure modes: torch.load() otherwise
            # deserializes tensors onto whatever device they were *saved* from (which
            # raises outright if that device, e.g. a CUDA GPU, isn't available on this
            # machine), and even when it succeeds, __init__ would then need to move
            # them again - map_location does it once, correctly, up front
            state_dict: MutableMapping = torch.load(
                file_path, map_location=device, weights_only=False
            )
            weights = state_dict.pop("weights")
            return CertaintyFactors(weights=weights, device=device)
        raise ValueError(f"Invalid path: {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the certainty factors to a tensor x.

        Args:
            x: The tensor to apply the certainty factors to, which is likely an intermediate
            value during the calculation of rules' firing levels.

        Returns:
            The result of the certainty factors applied to the tensor x.
        """
        return x * torch.nn.functional.tanh(self.weights)
