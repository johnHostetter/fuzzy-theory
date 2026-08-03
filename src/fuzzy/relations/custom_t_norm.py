"""
This script contains various classes that allow the customization of any t-norm operation for the
purposes of enabling rapid research exploration.
"""

from pathlib import Path
from typing import Any, MutableMapping

import scienceplots  # noqa # pylint: disable=unused-import
import torch

from fuzzy.relations.confidence import CertaintyFactors
from fuzzy.utils.options.impl.impl_options import (InferenceConfig,
                                                   PremiseActivation,
                                                   PremiseAggregation,
                                                   RuleElevationEnum,
                                                   RuleWeightsEnum)


class TNormPipeline(torch.nn.Module):
    """
    A generic sequential process that outlines a convenient interface to customize t-norm
    operations in order to enable rapid research exploration.
    """

    def __init__(
        self,
        configuration: InferenceConfig,
        n_relations: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__()
        self.n_relations = n_relations
        self._agg = PremiseAggregation.func(configuration.premise.aggregation)
        self._act = PremiseActivation.func(
            transform=configuration.premise.activation,
            bound=configuration.premise.bound,
        )
        self.layer_norm = None
        if "layer_norm" in kwargs and isinstance(
            kwargs["layer_norm"], torch.nn.LayerNorm
        ):
            self.layer_norm = kwargs["layer_norm"]
        elif configuration.rule.elevation == RuleElevationEnum.LAYER_NORMALIZATION:
            self.layer_norm = torch.nn.LayerNorm(
                [self.n_relations], device=device)

        self.certainty = None
        if "certainty" in kwargs and isinstance(
                kwargs["certainty"], CertaintyFactors):
            self.certainty = kwargs["certainty"]
        elif configuration.rule.weights == RuleWeightsEnum.CERTAINTY_FACTORS:
            self.certainty = CertaintyFactors.create_default(
                n_features=self.n_relations, device=device
            )

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the custom n-ary relation to a dictionary given a path.

        Args:
            path: The (requested) path to save the n-ary relation. This path must be a directory.

        Returns:
            The dictionary representation of the custom n-ary relation.
        """
        state_dict: MutableMapping[str, Any] = self.state_dict()
        state_dict["n_relations"] = self.n_relations
        path.mkdir(parents=True, exist_ok=True)
        if self.certainty is not None:
            (path / "certainty").mkdir(parents=True, exist_ok=True)
            self.certainty.save(path=path / "certainty")
        torch.save(state_dict, path / "state_dict.pt")
        return state_dict

    @classmethod
    # @log_classmethod
    def load(cls, path: Path, device: torch.device) -> "TNormPipeline":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Returns:
            None
        """
        if path.is_dir():
            state_dict: MutableMapping = torch.load(
                path / "state_dict.pt", weights_only=False
            )
            n_relations: int = state_dict.pop("n_relations")
            configuration = InferenceConfig.load(
                path=path.parent / "configuration.yaml"
            )

            t_norm_pipeline = TNormPipeline(
                configuration=configuration,
                n_relations=n_relations,
                device=device)
            t_norm_pipeline.load_state_dict(state_dict)
            return t_norm_pipeline

        raise ValueError(f"Invalid path: {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the custom n-ary t-norm operation.

        Args:
            x: The input to the n-ary t-norm operation, likely a tensor representing membership
            degrees that are to be manipulated.

        Returns:
            Membership degrees determined based on the custom n-ary t-norm operation.
        """
        # x = self._agg(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.certainty is not None:
            x = self.certainty(x)
        x = x - x.amax(dim=-1, keepdim=True)
        x = self._act(x, dim=-1)
        return x
