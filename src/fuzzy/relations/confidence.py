from pathlib import Path
from typing import Any, MutableMapping

import torch


class CertaintyFactors(torch.nn.Module):
    def __init__(self, weights: torch.Tensor, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weights.to(device)
        self.weights = torch.nn.Parameter(
            weights,
            requires_grad=True,
        )

    @staticmethod
    def create_default(n_features: int, device, *args, **kwargs) -> "CertaintyFactors":
        weights = torch.ones([n_features], dtype=torch.float32, device=device)
        return CertaintyFactors(weights, device, *args, **kwargs)

    def save(self, path: Path) -> MutableMapping[str, Any]:
        state_dict: MutableMapping[str, Any] = self.state_dict()
        torch.save(state_dict, path / "state_dict.pt")
        return state_dict

    @classmethod
    # @log_classmethod
    def load(cls, path: Path, device: torch.device) -> "CertaintyFactors":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Returns:
            None
        """
        if path.is_file() and path.suffix == ".pt":
            state_dict: MutableMapping = torch.load(path, weights_only=False)
            weights = state_dict.pop("weights")
            return CertaintyFactors(weights=weights, device=device)
        raise ValueError(f"Invalid path: {path}")

    def forward(self, x):
        return x * torch.nn.functional.tanh(self.weights)
