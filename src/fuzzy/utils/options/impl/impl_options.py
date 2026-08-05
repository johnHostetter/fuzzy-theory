"""
This script contains the design options that are available. It extends from Enum classes which
outline the names of each option by allowing a selection to coincide along all possible options.
This functionality is ideal for outlining what is permitted for the neural architecture, while
allowing the end-user (or optuna) to select (and store) their selection alongside those.
Furthermore, some of these classes will also store the accompanying function for cohesion (e.g.,
they inherit from torch.nn.Module and can be used accordingly).
"""

from dataclasses import Field, field
from dataclasses import fields as dataclasses_fields
from pathlib import Path
from typing import Callable, List, Union

import scipy.stats
import torch
import torch.nn.functional as F
import yaml
from entmax import entmax15, entmax_bisect
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
from torch.nn import Module

from fuzzy.utils.options.abstract.meta import EnumPromoter
from fuzzy.utils.options.abstract.primitive import CategoricalOptions
from fuzzy.utils.options.impl.impl_enums import (
    BoundAlphaEntmaxEnum,
    DefuzzificationMethodEnum,
    NeurogenesisEnum,
    PremiseActivationEnum,
    PremiseAggregationEnum,
    PremiseEliminationEnum,
    RuleElevationEnum,
    RuleEliminationEnum,
    RuleWeightsEnum,
    SamplingEnum,
)

_64_BIT_INT: int = (2**63) - 1  # max magnitude of a 64-bit integer


@dataclass(frozen=True)
class Range:
    """
    A range of possible values.
    """

    low: Union[float, int]
    high: Union[float, int]
    log: bool = False  # log scale for floats
    # if step size is given, options are discrete
    step: Union[None, float, int] = None

    def to_scipy(self) -> object:
        """
        Conveniently translate this range instance to scipy.

        Returns:
            A scipy.stats.range instance.
        """
        if self.step:
            # scipy.stats.randint(low, high) samples from [low, high) - high excluded -
            # whereas the rest of Range (contains(), to_optuna()) treats high as
            # inclusive, so it must be offset by one to match
            return scipy.stats.randint(int(self.low), int(self.high) + 1)
        if self.log:
            return scipy.stats.loguniform(self.low, self.high)
        return scipy.stats.uniform(self.low, self.high - self.low)

    def to_optuna(self, trial, name: str) -> None:
        """
        Conveniently translate this range instance to Optuna for hyperparameter search.

        Args:
            trial: The instance of Trial.
            name: The reference name.

        Returns:
            None
        """
        if self.step:
            return trial.suggest_int(name, self.low, self.high, step=self.step)
        if self.log:
            return trial.suggest_float(name, self.low, self.high, log=True)
        return trial.suggest_float(name, self.low, self.high)

    def contains(self, val: Union[int, float]) -> bool:
        """
        A method to check if a value is in the range.

        Args:
            val: The value of interest.

        Returns:
            Whether the value is in the range.
        """
        return self.low <= val <= self.high

    def to_dict(self) -> dict:
        """
        Convert the instance of Range to a dictionary.

        Returns:
            The dictionary representing the Range object.
        """
        entry = {"low": self.low, "high": self.high}
        if self.log:
            entry["log"] = self.log
        if self.step is not None:
            entry["step"] = self.step
        return entry

    @classmethod
    def from_dict(cls, entry: dict) -> "Range":
        """
        An implementation that allows the conversion of a dictionary entry to an instance of Range.

        Args:
            entry: The dictionary entry.

        Returns:
            An instance of Range.
        """
        return cls(
            low=entry["low"],
            high=entry["high"],
            log=entry.get("log", False),
            step=entry.get("step", None),
        )

    def __repr__(self):
        parts = [f"low={self.low}", f"high={self.high}"]
        if self.log:
            parts.append("log=true")
        if self.step:
            parts.append(f"step={self.step}")
        return f"Range({', '.join(parts)})"


class PremiseAggregation(
    CategoricalOptions, EnumPromoter, enum_cls=PremiseAggregationEnum
):
    """
    Outlines the available premise aggregation strategies and their implementations.
    """

    # type hints for Pylint - not required for functionality but only static
    # code analysis
    SUM: PremiseAggregationEnum
    MEAN: PremiseAggregationEnum

    _fn = {
        PremiseAggregationEnum.SUM: lambda x: -1 * x.sum(dim=1),
        PremiseAggregationEnum.MEAN: lambda x: -1 * x.mean(dim=1),
    }

    def __init__(self):
        super().__init__(*self.options)

    @classmethod
    def func(
        cls, selection: PremiseAggregationEnum
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Obtain the appropriate premise aggregation function based on the stored selection.

        Args:
            selection: A selection made from the set of options in PremiseAggregationEnum.

        Returns:
            A callable function that expects a torch.Tensor and will return a torch.Tensor.
        """
        return cls._fn[selection]


class BoundAlphaEntmax(torch.nn.Module):
    """
    Outlines the available strategies for bounding the alpha value used in the entmax_bisect and
    their implementations.
    """

    def __init__(
        self,
        bounding_strategy: BoundAlphaEntmaxEnum,
        *args,
        device=None,
        dim: int = -1,
        alpha: Union[None, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bounding_strategy = bounding_strategy
        self.device = torch.get_default_device() if device is None else device
        self.dim = dim
        if alpha is None:
            alpha = torch.empty(1, dtype=torch.float32, device=self.device)
            torch.nn.init.normal_(alpha)
        self.alpha = torch.nn.Parameter(alpha, requires_grad=True)

    def bound_alpha(self, alpha: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """
        Bound the alpha parameter to abide by the constraint that it must exist within (1, 2) so it
        does not devolve to softmax or sparsemax, respectively.

        Returns:
            The bounded alpha within the appropriate range according to the selected bounding
            strategy outlined in the configuration settings.
        """
        if alpha is None:
            alpha = self.alpha
        if self.bounding_strategy == BoundAlphaEntmaxEnum.SIGMOID:
            return 1.0 + torch.sigmoid(alpha)
        if self.bounding_strategy == BoundAlphaEntmaxEnum.TANH:
            return 1.5 + (0.5 * torch.tanh(alpha))
        if self.bounding_strategy == BoundAlphaEntmaxEnum.HARD_TANH:
            return 1.5 + (0.5 * torch.nn.functional.hardtanh(alpha))
        if self.bounding_strategy == BoundAlphaEntmaxEnum.SOFTPLUS:
            softplus_alpha = F.softplus(alpha)  # pylint: disable=not-callable
            return 1.0 + (softplus_alpha / (1 + softplus_alpha))

        raise ValueError(
            "The only implemented bounding strategies are Sigmoid representation "
            "(sigmoid_reparameterization), Scaled tanh (scaled_tanh), and "
            "Softplus + Shift (softplus_add_shift)"
        )

    def forward(
        self, tensor: torch.Tensor, dim: Union[None, int] = None
    ) -> torch.Tensor:
        """
        Apply entmax_bisect to the given tensor using the bounded alpha parameter.

        Args:
            tensor: The tensor to apply entmax_bisect to.
            dim: The dimension to apply entmax_bisect along; defaults to the
                dimension given at construction (self.dim) if not provided.

        Returns:
            The result of entmax_bisect applied to the tensor.
        """
        if dim is None:
            dim = self.dim  # use internal referenced dim for the forward
        bounded_alpha = self.bound_alpha()
        if tensor.device != bounded_alpha.device:
            bounded_alpha = bounded_alpha.to(tensor.device)
        return entmax_bisect(tensor, alpha=bounded_alpha, dim=dim)


class PremiseActivation(
    CategoricalOptions, torch.nn.Module, EnumPromoter, enum_cls=PremiseActivationEnum
):
    """
    Outlines the available premise activation strategies and their implementations.
    """

    _fn = {
        PremiseActivationEnum.SOFTMAX: torch.nn.functional.softmax,  # default TSK behavior
        PremiseActivationEnum.ENTMAX15: entmax15,
    }

    def __init__(
        self,
        *args,
        device=None,
        dim: int = -1,
        # alpha: Union[None, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*self.options)
        self.dim = dim
        Module.__init__(self, *args, **kwargs)
        self.device = torch.get_default_device() if device is None else device
        # self.bound_alpha = (
        #     BoundAlphaEntmax(device=self.device)
        #     if alpha is None
        #     else (BoundAlphaEntmax(device=self.device, alpha=alpha))
        # )

    @classmethod
    def func(
        cls, transform: PremiseActivationEnum, bound: Union[None, BoundAlphaEntmaxEnum]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Obtain the appropriate premise activation function based on the selected transform.

        Args:
            transform: A selected transform from the set of options in PremiseActivationEnum.
            bound: A selected bounding strategy from the set of options in BoundAlphaEntmaxEnum;
            ignored if transform is NOT PremiseActivationEnum.ENTMAX_BISECT.

        Returns:
            A callable function that expects a torch.Tensor and will return a torch.Tensor.
        """
        if transform == PremiseActivationEnum.ENTMAX_BISECT:
            assert isinstance(bound, BoundAlphaEntmaxEnum), (
                "You must select a bounding strategy to limit the range of alpha when using "
                "an adaptable entmax."
            )
            # build a copy since it has params
            return BoundAlphaEntmax(bounding_strategy=bound)
        return cls._fn[transform]

    # def forward(self, input):
    #     if self.selection == PremiseActivationEnum.ENTMAX15:
    #         return entmax15(input, dim=self.dim)
    #     if self.selection == PremiseActivationEnum.SOFTMAX:
    #         return torch.nn.functional.softmax(input, dim=self.dim)
    #     # if self.selection == PremiseActivationEnum.ENTMAX_BISECT:
    #     #     if input.device != self.bound_alpha.alpha.device:
    #     #         self.bound_alpha.alpha.to(input.device)
    #     #     return entmax_bisect(input, alpha=self.bound_alpha(), dim=self.dim)
    #     raise ValueError(
    #         f"The premise_activation must be either 'entmax15', 'entmax_bisect', "
    #         f"or 'softmax'. Instead, received the selection of '{self.selection}'."
    #     )


class YAMLConfig:
    """
    Allows a convenient interface to save and load dataclass configurations with pydantic.
    """

    @staticmethod
    def __path_validation(path: Path) -> None:
        """
        Ensure the given path is a valid YAML file.

        Args:
            path: The path to validate. It must end with a *.yaml extension.

        Returns:
            None
        """
        assert not path.is_dir(), "The given path must be designated as a file."
        assert path.name.endswith(
            ".yaml"
        ), "Only paths that end with '.yaml' are allowed."

    def save(self, path: Path) -> None:
        """
        Save the current configuration to a *.yaml file.

        Args:
            path: The path to use. It must end with a *.yaml extension.

        Returns:
            None
        """
        self.__path_validation(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        adapter = TypeAdapter(type=type(self))
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(adapter.dump_python(self, mode="json"), f)

    @classmethod
    def load(cls, path: Path) -> "YAMLConfig":
        """
        Load the dataclass configuration from a *.yaml file and instantiate its appropriate class.

        Args:
            path: The path to use. It must end with a *.yaml extension.

        Returns:
            An instance of YAMLConfig class.
        """
        cls.__path_validation(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TypeAdapter(cls).validate_python(data)


@dataclass
class PremiseConfig(YAMLConfig):
    """
    How to set up the premise terms, whether we should use standard implementation (e.g.,
    sum and softmax) or try something more experimental (e.g., mean and 1.5-entmax), as well as
    whether to eliminate any premise terms. Default is SUM for aggregation (standard), NONE for
    elimination, and SOFTMAX for activation (standard if Gaussian fuzzy sets are used, but no exp
    has been applied to them yet).
    """

    aggregation: PremiseAggregationEnum = field(
        default=PremiseAggregationEnum.SUM,
    )
    elimination: PremiseEliminationEnum = field(
        default=PremiseEliminationEnum.NONE,
    )
    activation: PremiseActivationEnum = field(
        default=PremiseActivationEnum.SOFTMAX,
    )
    bound: BoundAlphaEntmaxEnum = field(default=BoundAlphaEntmaxEnum.SIGMOID)


@dataclass
class RuleConfig(YAMLConfig):
    """
    How to set up the fuzzy logic rules, whether they should be weighed (e.g., certainty factors),
    whether to eliminate any, and/or whether to elevate their firing levels. Default is NONE for
    all.
    """

    weights: RuleWeightsEnum = field(
        default=RuleWeightsEnum.NONE,
    )
    elimination: RuleEliminationEnum = field(
        default=RuleEliminationEnum.NONE,
    )
    elevation: RuleElevationEnum = field(
        default=RuleElevationEnum.NONE,
    )


@dataclass
class DefuzzificationConfig(YAMLConfig):
    """
    How to conduct defuzzification after the fuzzy logic rules are activated given the stimuli. For
    instance, whether to use zero-order TSK, TSK, Mamdani, or other experimental methods.
    """

    method: DefuzzificationMethodEnum = field(default=DefuzzificationMethodEnum.TSK)
    n_latent_space_dim: int = field(
        default=32,
        metadata={
            "help": "The dimensionality of the latent space to utilize, if applicable.",
            "range": Range(low=1, high=float("inf")),
            "search": Range(low=32, high=128, step=32),
        },
    )


@dataclass
class InferenceConfig(YAMLConfig):
    """
    A configuration for fuzzy logic rule inference.
    """

    premise: PremiseConfig = field(
        default_factory=PremiseConfig,
        metadata={
            "help": "How premises should be aggregated, eliminated and elevated.",
        },
    )
    rule: RuleConfig = field(
        default_factory=RuleConfig,
        metadata={
            "help": "How rules should be aggregated, eliminated, and elevated.",
        },
    )
    defuzzification: DefuzzificationConfig = field(
        default_factory=DefuzzificationConfig,
        metadata={"help": "Determines how to proceed with defuzzification."},
    )


@dataclass
class GumbelConfig(YAMLConfig):
    """
    How to set up the Gumbel Softmax, whether it should be constrained, how long to delay
    resampling the Gumbel noise, and what temperature to use for the Gumbel distribution.
    Default options and values replicate those explored in Hostetter's dissertation.
    """

    sampling: SamplingEnum = field(
        default=SamplingEnum.ST_GUMBEL_SOFTMAX,
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "How much temperature should be used for the Gumbel distribution.",
            "range": Range(low=1, high=float("inf")),
            "search": Range(low=0.25, high=1.25),
        },
    )
    epsilon_filter: float = field(
        default=0.0,
        metadata={
            "help": "Whether to constrain the Straight-Through Gumbel Softmax Estimator.",
            "range": Range(low=0, high=1.0),
            "choices": [0.0, 0.1],
        },
    )
    noise_delay: int = field(
        default=1,
        metadata={
            "help": "The number of forward passes to process before resampling noise from the "
            "Gumbel distribution.",
            "range": Range(low=1, high=_64_BIT_INT, step=1),
            "choices": [
                1,  # no delay in updating noise (since it's about modulo)
                32,
                64,
                128,
                256,
            ],
        },
    )


@dataclass
class NeurogenesisConfig(YAMLConfig):
    """
    How to set up neurogenesis, such as whether it should be delayed and how it should be
    triggered. Default options and values replicate those explored in Hostetter's dissertation.
    """

    neurogenesis: NeurogenesisEnum = field(
        default=NeurogenesisEnum.MODIFIED_DELAYED_WELFORD,
    )
    epsilon: float = field(
        default=0.5,
        metadata={
            "help": "The desired membership degree that should be satisfied by all elements to "
            "achieve epsilon-completeness.",
            "range": Range(low=0, high=1.0),
            "search": Range(low=0.1, high=0.5),
        },
    )
    add_premise_delay: int = field(
        default=1,
        metadata={
            "help": "How long to wait before adding new premises to the neuro-fuzzy network.",
            # 1 = no delay to add fuzzy sets (since it's about modulo)
            "range": Range(low=1, high=_64_BIT_INT, step=1),
            "search": Range(low=1, high=5, step=2),
        },
    )


@dataclass
class EvolutionConfig(YAMLConfig):
    """
    Dictates how the neuro-fuzzy network will evolve.
    """

    premise: NeurogenesisConfig = field(
        default_factory=NeurogenesisConfig,
        metadata={
            "help": "A modified and delayed version of Welford's method for computing variance.",
        },
    )
    rule: GumbelConfig = field(
        default_factory=GumbelConfig,
        metadata={
            "help": "How to set up the Straight-Through Gumbel Softmax Estimator."
        },
    )


@dataclass
class ParameterConfig(YAMLConfig):
    """
    Determines how to initialize parameters of the neuro-fuzzy network (e.g., premise, weights,
    consequences, etc.).
    """

    @dataclass
    class PremiseParameterConfig(YAMLConfig):
        """
        Determines how to initialize parameters of the neuro-fuzzy network with respect to the
        premise layer.
        """

        init_width: float = field(
            default=0.5,
            metadata={
                "help": "Initial width of fuzzy sets upon initialization.",
                "range": Range(low=0, high=float("inf"), step=1),
            },
        )

    @dataclass
    class RuleParameterConfig(YAMLConfig):
        """
        Determines how to initialize parameters of the neuro-fuzzy network with respect to the
        rule layer, if applicable (i.e., assuming rule sampling is implemented).
        """

    @dataclass
    class ConsequenceParameterConfig(YAMLConfig):
        """
        Determines how to initialize parameters of the neuro-fuzzy network with respect to the
        consequence layer.
        """

    premise: PremiseParameterConfig = field(
        default_factory=PremiseParameterConfig,
        metadata={"help": "How to initialize the premise layer's parameters."},
    )
    rule: RuleParameterConfig = field(
        default_factory=RuleParameterConfig,
        metadata={
            "help": "How to initialize the rule layer's parameters (if applicable)."
        },
    )
    consequence: ConsequenceParameterConfig = field(
        default_factory=ConsequenceParameterConfig,
        metadata={"help": "How to initialize the consequence layer's parameters."},
    )


@dataclass
class StructureConfig(YAMLConfig):
    """
    Determines how to initialize structure of the neuro-fuzzy network (e.g., rule count).
    """

    @dataclass
    class PremiseStructureConfig(YAMLConfig):
        """
        Determines how to initialize structure of the neuro-fuzzy network with respect to the
        premise layer.
        """

    @dataclass
    class RuleStructureConfig(YAMLConfig):
        """
        Determines how to initialize structure of the neuro-fuzzy network with respect to the
        rule layer, if applicable.
        """

        n_rules: int = field(
            default=128,
            metadata={
                "help": "The number of non-unique rules available to the neuro-fuzzy network.",
                "range": Range(low=0, high=10000, step=1),
                "search": Range(low=64, high=256, step=64),
            },
        )

    @dataclass
    class ConsequenceStructureConfig(YAMLConfig):
        """
        Determines how to initialize structure of the neuro-fuzzy network with respect to the
        consequence layer.
        """

    premise: PremiseStructureConfig = field(
        default_factory=PremiseStructureConfig,
        metadata={"help": "How to initialize the premise layer's structure."},
    )
    rule: RuleStructureConfig = field(
        default_factory=RuleStructureConfig,
        metadata={
            "help": "How to initialize the rule layer's structure (if applicable)."
        },
    )
    consequence: ConsequenceStructureConfig = field(
        default_factory=ConsequenceStructureConfig,
        metadata={"help": "How to initialize the consequence layer's structure."},
    )


@dataclass
class ApproximatorHyperparameters(YAMLConfig):
    """
    A standard data class format with attributes that are expected throughout PySoft optuna
    experiments.
    """

    display_name: str = field(
        init=False, default=""
    )  # the display name of the approximator
    # an abbreviated name of the approximator
    abbrev_name: str = field(init=False, default="")


@dataclass
class NeuroFuzzyNetworkHyperparameters(ApproximatorHyperparameters):
    """
    An all-encompassing class for exposing all available hyperparameters or design-choices of
    neuro-fuzzy networks.
    """

    structure: StructureConfig = field(
        default_factory=StructureConfig,
        metadata={
            "help": "How to design the initial structure of the neuro-fuzzy networks.",
        },
    )
    parameter: ParameterConfig = field(
        default_factory=ParameterConfig,
        metadata={
            "help": "How to initialize the parameters of the neuro-fuzzy network."
        },
    )
    inference: InferenceConfig = field(
        default_factory=InferenceConfig,
        metadata={
            "help": "Inference configuration for neuro-fuzzy networks.",
        },
    )
    evolution: EvolutionConfig = field(
        default_factory=EvolutionConfig,
        metadata={
            "help": "Evolution configuration for neuro-fuzzy networks.",
        },
    )

    def __post_init__(self):
        self.display_name = "Concurrent Optimization of Fuzzy Inference Systems"
        self.abbrev_name = "CO-FIS"

        # disable the constraint for every instance, not just the first one built in
        # the process - a ClassVar-gated "only the first time" guard used to make this
        # silently stop applying after the first instantiation anywhere in the
        # process
        self.evolution.rule.epsilon_filter = 0.0

    # noinspection PyTypeChecker
    @property
    def fields(self) -> List[Field]:
        """
        Return only the fields that are unique to this subclass.

        Returns:
            A list of fields that are unique to this subclass.
        """
        return [
            hyperparameter
            for hyperparameter in dataclasses_fields(self)
            if hyperparameter not in dataclasses_fields(ApproximatorHyperparameters)
        ]
