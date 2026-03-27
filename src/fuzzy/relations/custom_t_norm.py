from pathlib import Path
from typing import Any, MutableMapping

import scienceplots  # noqa # pylint: disable=unused-import
import torch

from fuzzy.relations.confidence import CertaintyFactors
from fuzzy.utils.options.abstract.primitive import GroupedOptions
from fuzzy.utils.options.impl.impl_options import (
    NeuroFuzzyNetworkHyperparameters,
    PremiseConfig,
    RuleConfig,
    RuleElevationEnum,
    RuleWeightsEnum,
)


class CustomTNormOptions(GroupedOptions):
    def __init__(
        self, hyperparameters: NeuroFuzzyNetworkHyperparameters = None, **kwargs
    ):
        super().__init__(**kwargs)
        if hyperparameters is None:
            self.premise = PremiseConfig()
            self.rule = RuleConfig()
        else:
            self.premise = hyperparameters.premise
            self.rule = hyperparameters.rule


class TNormPipeline(torch.nn.Module):
    def __init__(
        self,
        configuration: CustomTNormOptions,
        n_relations: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__()
        self.n_relations = n_relations
        self._agg = configuration.premise.aggregation.func()
        self._act = configuration.premise.activation.func()
        self.layer_norm = None
        if "layer_norm" in kwargs and isinstance(
            kwargs["layer_norm"], torch.nn.LayerNorm
        ):
            self.layer_norm = kwargs["layer_norm"]
        elif (
            configuration.rule.elevation.selection
            == RuleElevationEnum.LAYER_NORMALIZATION
        ):
            self.layer_norm = torch.nn.LayerNorm([self.n_relations], device=device)

        self.certainty = None
        if "certainty" in kwargs and isinstance(kwargs["certainty"], CertaintyFactors):
            self.certainty = kwargs["certainty"]
        elif configuration.rule.weights.selection == RuleWeightsEnum.CERTAINTY_FACTORS:
            self.certainty = CertaintyFactors.create_default(
                n_features=self.n_relations, device=device
            )

    def save(self, path: Path) -> MutableMapping[str, Any]:
        state_dict: MutableMapping[str, Any] = self.state_dict()
        state_dict["n_relations"] = self.n_relations
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
            configuration = CustomTNormOptions.load(path=path.parent / "configuration")

            if isinstance(configuration, GroupedOptions):
                t_norm_pipeline = TNormPipeline(
                    configuration=configuration, n_relations=n_relations, device=device
                )
                t_norm_pipeline.load_state_dict(state_dict)
                return t_norm_pipeline

            raise ValueError(
                f"Expected instance of CustomTNormOptions, but got: {type(configuration)}"
            )
        raise ValueError(f"Invalid path: {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self._agg(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.certainty is not None:
            x = self.certainty(x)
        x = x - x.amax(dim=-1, keepdim=True)
        x = self._act(x, dim=-1)
        return x
