"""
This script contains the design options that are available. It extends from Enum classes which
outline the names of each option by allowing a selection to coincide along all possible options.
This functionality is ideal for outlining what is permitted for the neural architecture, while
allowing the end-user (or optuna) to select (and store) their selection alongside those.
Furthermore, some of these classes will also store the accompanying function for cohesion (e.g.,
they inherit from torch.nn.Module and can be used accordingly).
"""

from dataclasses import dataclass, field, Field, fields
from enum import Enum
from typing import Callable, Tuple, Type, Union, ClassVar, List

import torch
import torch.nn.functional as F
from entmax import entmax15  # , entmax_bisect
from torch.nn import Module

from fuzzy.utils.options.abstract.meta import EnumPromoter, Options
from fuzzy.utils.options.abstract.primitive import (
    CategoricalEnumOptions,
    CategoricalOptions,
    GroupedOptions,
)
from fuzzy.utils.options.impl.impl_enums import (
    BoundAlphaEntmaxEnum,
    PremiseActivationEnum,
    PremiseAggregationEnum,
    PremiseEliminationEnum,
    RuleElevationEnum,
    RuleEliminationEnum,
    RuleWeightsEnum,
    SamplingEnum,
    NeurogenesisEnum,
)


from scipy.stats import loguniform, uniform, randint

_64_BIT_INT: int = 2^63 - 1  # max magnitude of a 64-bit integer

@dataclass(frozen=True)
class Range:
    low:  Union[float, int]
    high: Union[float, int]
    log:  bool = False      # log scale for floats
    step: Union[None, int]  = None       # for integers with a step size

    def to_scipy(self):
        if self.step:
            return randint(self.low, self.high)
        if self.log:
            return loguniform(self.low, self.high)
        return uniform(self.low, self.high - self.low)

    def to_optuna(self, trial, name: str):
        if self.step:
            return trial.suggest_int(name, self.low, self.high, step=self.step)
        if self.log:
            return trial.suggest_float(name, self.low, self.high, log=True)
        return trial.suggest_float(name, self.low, self.high)

    def contains(self, val) -> bool:
        return self.low <= val <= self.high

    def to_dict(self) -> dict:
        d = {"low": self.low, "high": self.high}
        if self.log:
            d["log"] = self.log
        if self.step is not None:
            d["step"] = self.step
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Range":
        return cls(
            low=d["low"],
            high=d["high"],
            log=d.get("log", False),
            step=d.get("step", None),
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
        PremiseAggregationEnum.SUM.value: lambda x: -1 * x.sum(dim=1),
        PremiseAggregationEnum.MEAN.value: lambda x: -1 * x.mean(dim=1),
    }

    def __init__(self):
        super().__init__(*self.options)

    # @property
    def func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Obtain the appropriate premise aggregation function based on the stored selection.

        Returns:
            A callable function that expects a torch.Tensor and will return a torch.Tensor.
        """
        if self.assignable:
            raise ValueError("A selection has not yet been made.")
        if isinstance(self.selection, Enum):
            return self._fn[self.selection.value]
        # self.selection is 'str' if optuna assigns
        return self._fn[self.selection]


class PremiseElimination(
    CategoricalEnumOptions, EnumPromoter, enum_cls=PremiseEliminationEnum
):
    """
    Outlines the available premise elimination strategies.
    """

    # type hints for Pylint - not required for functionality but only static
    # code analysis
    NONE: PremiseEliminationEnum
    NO_OP: PremiseEliminationEnum


class BoundAlphaEntmax(
    CategoricalOptions, torch.nn.Module, EnumPromoter, enum_cls=BoundAlphaEntmaxEnum
):
    """
    Outlines the available strategies for bounding the alpha value used in the entmax_bisect and
    their implementations.
    """

    def __init__(
        self, *args, device=None, alpha: Union[None, torch.Tensor] = None, **kwargs
    ):
        super().__init__(*self.options)
        Module.__init__(self, *args, **kwargs)
        self.device = torch.get_default_device() if device is None else device
        if alpha is None:
            alpha = torch.empty(1, dtype=torch.float32, device=self.device)
            torch.nn.init.normal_(alpha)
        self.alpha = torch.nn.Parameter(alpha, requires_grad=True)

    def forward(self, alpha: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """
        Bound the alpha parameter to abide by the constraint that it must exist within (1, 2) so it
        does not devolve to softmax or sparsemax, respectively.

        Returns:
            The bounded alpha within the appropriate range according to the selected bounding
            strategy outlined in the configuration settings.
        """
        if alpha is None:
            alpha = self.alpha
        if self.selection == BoundAlphaEntmaxEnum.SIGMOID:
            return 1.0 + torch.sigmoid(alpha)
        if self.selection == BoundAlphaEntmaxEnum.TANH:
            return 1.5 + (0.5 * torch.tanh(alpha))
        if self.selection == BoundAlphaEntmaxEnum.SOFTPLUS:
            softplus_alpha = F.softplus(alpha)  # pylint: disable=not-callable
            return 1.0 + (softplus_alpha / (1 + softplus_alpha))

        raise ValueError(
            "The only implemented bounding strategies are Sigmoid representation "
            "(sigmoid_reparameterization), Scaled tanh (scaled_tanh), and "
            "Softplus + Shift (softplus_add_shift)"
        )


class PremiseActivation(
    CategoricalOptions, torch.nn.Module, EnumPromoter, enum_cls=PremiseActivationEnum
):
    """
    Outlines the available premise activation strategies and their implementations.
    """

    _fn = {
        PremiseActivationEnum.ENTMAX15.value: entmax15,
        PremiseActivationEnum.SOFTMAX.value: torch.nn.functional.softmax,
    }

    def __init__(
        self,
        *args,
        device=None,
        dim: int = -1,
        alpha: Union[None, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*self.options)
        self.dim = dim
        Module.__init__(self, *args, **kwargs)
        self.device = torch.get_default_device() if device is None else device
        self.bound_alpha = (
            BoundAlphaEntmax(device=self.device)
            if alpha is None
            else (BoundAlphaEntmax(device=self.device, alpha=alpha))
        )

    # @property
    def func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Obtain the appropriate premise aggregation function based on the stored selection.

        Returns:
            A callable function that expects a torch.Tensor and will return a torch.Tensor.
        """
        if self.assignable:
            raise ValueError("A selection has not yet been made.")
        if isinstance(self.selection, Enum):
            return self._fn[self.selection.value]
        # self.selection is 'str' if optuna assigns
        return self._fn[self.selection]

    def assign(self, trial, name) -> Tuple[str, str]:
        super_assignment = super().assign(trial=trial, name=name)
        bound_alpha_assignment = self.bound_alpha.assign(
            trial=trial, name=f"{name}.bound_alpha"
        )
        return super_assignment, bound_alpha_assignment

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


class Sampling(CategoricalEnumOptions):
    """
    Outlines the available sampling strategies.
    """

    enum_cls = SamplingEnum


class RuleWeights(CategoricalEnumOptions):
    """
    Outlines the available rule weighing strategies.
    """

    enum_cls = RuleWeightsEnum


class RuleElimination(CategoricalEnumOptions):
    """
    Outlines the available rule elimination strategies.
    """

    enum_cls = RuleEliminationEnum


class RuleElevation(CategoricalEnumOptions):
    """
    Outlines the available rule elevation strategies.
    """

    enum_cls = RuleElevationEnum


@dataclass
class PremiseConfig(GroupedOptions):
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

    @property
    def selection(self):
        return (
            option.selection
            for _, option in vars(self).items()
            if isinstance(option, Options)
        )

    def default(self) -> None:
        """
        Select the 'default' settings for a TSK neuro-fuzzy network with respect to premises.

        Returns:
            None
        """
        self.select(
            aggregation=PremiseAggregation.SUM,
            elimination=PremiseElimination.NONE,
            activation=PremiseActivation.SOFTMAX,
        )

    def select(
        self,
        aggregation: Type[PremiseAggregation],
        elimination: Type[PremiseElimination],
        activation: Type[PremiseActivation],
    ) -> None:
        """
        Select the assignments based on the given arguments.

        Args:
            aggregation: A premise aggregation enum member.
            elimination: A premise elimination enum member.
            activation: A premise activation enum member.

        Returns:
            None
        """
        self.aggregation.selection = aggregation
        self.elimination.selection = elimination
        self.activation.selection = activation


@dataclass
class RuleConfig(GroupedOptions):
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

    @property
    def selection(self):
        return (
            option.selection
            for _, option in vars(self).items()
            if isinstance(option, Options)
        )

    def default(self) -> None:
        """
        Select the 'default' settings for a TSK neuro-fuzzy network with respect to rules.

        Returns:
            None
        """
        self.select(
            weights=RuleWeightsEnum.NONE,
            elimination=RuleEliminationEnum.NONE,
            elevation=RuleElevationEnum.NONE,
        )

    def select(
        self,
        weights: Type[RuleWeights],
        elimination: Type[RuleElimination],
        elevation: Type[RuleElevation],
    ):
        """
        Select the assignments based on the given arguments.

        Args:
            weights: A rule weights enum member.
            elimination: A rule elimination enum member.
            elevation: A rule elevation enum member.

        Returns:
            None
        """
        self.weights.selection = weights
        self.elimination.selection = elimination
        self.elevation.selection = elevation


@dataclass
class InferenceConfig:
    premise: PremiseConfig = field(
        default_factory=PremiseConfig,
        metadata={
            "help": "How premises should be aggregated, eliminated and elevated.",
        }
    )
    rule: RuleConfig = field(
        default_factory=RuleConfig,
        metadata={
            "help": "How rules should be aggregated, eliminated, and elevated.",
        }
    )


@dataclass
class GumbelConfig(GroupedOptions):
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
        }
    )
    epsilon_filter: float = field(
        default=0.0,
        metadata={
            "help": "Whether to constrain the Straight-Through Gumbel Softmax Estimator.",
            "range": Range(low=0, high=1.0),
            "choices": [0.0, 0.1],
        }
    )
    noise_delay: int = field(
        default=1,
        metadata={
            "help": "The number of forward passes to process before resampling noise from the "
                    "Gumbel distribution.",
            "range": Range(low=1, high=_64_BIT_INT, step=1),
            "choices": [
                1, # no delay in updating noise (since it's about modulo)
                32,
                64,
                128,
                256,
            ]
        }
    )

    @property
    def selection(self):
        return (
            option.selection
            for _, option in vars(self).items()
            if isinstance(option, Options)
        )

    def default(self) -> None:
        """
        Select 'default' settings for a neuro-fuzzy network with respect to the Gumbel Softmax.

        Returns:
            None
        """
        self.select(
            temperature=1.0,
            epsilon_filter=0.0,
            noise_delay=1,
        )

    def select(
        self,
        temperature: float,
        epsilon_filter: float,
        noise_delay: int,
    ):
        """
        Select the assignments based on the given arguments.

        Args:
            temperature: The temperature of the Gumbel distribution.
            epsilon_filter: Whether to constrain the Gumbel Softmax and by how much.
            noise_delay: Whether to delay resampling of the Gumbel Softmax noise, and for how long.

        Returns:
            None
        """
        self.temperature.selection = temperature
        self.epsilon_filter.selection = epsilon_filter
        self.noise_delay.selection = noise_delay


@dataclass
class NeurogenesisConfig(GroupedOptions):
    """
    How to set up neurogenesis, such as whether it should be delayed and how it should be
    triggered. Default options and values replicate those explored in Hostetter's dissertation.
    """
    neurogenesis: NeurogenesisEnum = field(
        default=NeurogenesisEnum.NONE,
    )
    epsilon: float = field(
        default=0.5,
        metadata={
            "help": "The desired membership degree that should be satisfied by all elements to "
                    "achieve epsilon-completeness.",
            "range": Range(low=0, high=1.0),
            "search": Range(low=0.1, high=0.5)
        }
    )
    add_premise_delay: int = field(
        default=1,
        metadata={
            "help": "How long to wait before adding new premises to the neuro-fuzzy network.",
            # 1 = no delay to add fuzzy sets (since it's about modulo)
            "range": Range(low=1, high=_64_BIT_INT, step=1),
            "search": Range(low=1, high=5, step=2)
        }
    )

    @property
    def selection(self):
        return (
            option.selection
            for _, option in vars(self).items()
            if isinstance(option, Options)
        )

    def default(self) -> None:
        """
        Select 'default' settings for a neuro-fuzzy network with respect to the Gumbel Softmax.

        Returns:
            None
        """
        self.select(
            epsilon=0.5,
            add_premise_delay=1,
        )

    def select(
        self,
        epsilon: float,
        add_premise_delay: int,
    ):
        """
        Select the assignments based on the given arguments.

        Args:
            epsilon: The minimum membership degree that must be achieved; otherwise, neurogenesis
            will be triggered.
            add_premise_delay: Whether to delay adding a new premise term, and for how long.

        Returns:
            None
        """
        self.epsilon.selection = epsilon
        self.add_premise_delay.selection = add_premise_delay


@dataclass
class EvolutionConfig:
    """
    Dictates how the neuro-fuzzy network will evolve.
    """
    premise: NeurogenesisConfig = field(
        default_factory=NeurogenesisConfig,
        metadata={
            "help": "A modified and delayed version of Welford's method for computing variance.",
        }
    )
    rule: GumbelConfig = field(
        default_factory=GumbelConfig,
        metadata={
            "help": "How to set up the Straight-Through Gumbel Softmax Estimator."
        }
    )

@dataclass
class ParameterConfig:
    """
    Determines how to initialize parameters of the neuro-fuzzy network (e.g., premise, weights,
    consequences, etc.).
    """

    @dataclass
    class PremiseParameterConfig:
        """
        Determines how to initialize parameters of the neuro-fuzzy network with respect to the
        premise layer.
        """
        init_width: float = field(
            default=0.5,
            metadata={
                "help": "Initial width of fuzzy sets upon initialization.",
                "range": Range(low=0, high=float("inf"), step=1),
            }
        )

    @dataclass
    class RuleParameterConfig:
        """
        Determines how to initialize parameters of the neuro-fuzzy network with respect to the
        rule layer, if applicable (i.e., assuming rule sampling is implemented).
        """

    @dataclass
    class ConsequenceParameterConfig:
        """
        Determines how to initialize parameters of the neuro-fuzzy network with respect to the
        consequence layer.
        """

    premise: PremiseParameterConfig = field(
        default_factory=PremiseParameterConfig,
        metadata={
            "help": "How to initialize the premise layer's parameters."
        }
    )
    rule: RuleParameterConfig = field(
        default_factory=RuleParameterConfig,
        metadata={
            "help": "How to initialize the rule layer's parameters (if applicable)."
        }
    )
    consequence: ConsequenceParameterConfig = field(
        default_factory=ConsequenceParameterConfig,
        metadata={
            "help": "How to initialize the consequence layer's parameters."
        }
    )

@dataclass
class StructureConfig:
    """
    Determines how to initialize structure of the neuro-fuzzy network (e.g., rule count).
    """

    @dataclass
    class PremiseStructureConfig:
        """
        Determines how to initialize structure of the neuro-fuzzy network with respect to the
        premise layer.
        """

    @dataclass
    class RuleStructureConfig:
        """
        Determines how to initialize structure of the neuro-fuzzy network with respect to the
        rule layer, if applicable.
        """
        n_rules: int = field(
            default=128,
            metadata={
                "help": "The number of non-unique rules available to the neuro-fuzzy network.",
                "range": Range(low=0, high=10000, step=1),
                "search": Range(low=64, high=256, step=64)
            }
        )

    @dataclass
    class ConsequenceStructureConfig:
        """
        Determines how to initialize structure of the neuro-fuzzy network with respect to the
        consequence layer.
        """

    premise: PremiseStructureConfig = field(
        default_factory=PremiseStructureConfig,
        metadata={
            "help": "How to initialize the premise layer's structure."
        }
    )
    rule: RuleStructureConfig = field(
        default_factory=RuleStructureConfig,
        metadata={
            "help": "How to initialize the rule layer's structure (if applicable)."
        }
    )
    consequence: ConsequenceStructureConfig = field(
        default_factory=ConsequenceStructureConfig,
        metadata={
            "help": "How to initialize the consequence layer's structure."
        }
    )


@dataclass
class ApproximatorHyperparameters:
    """
    A standard data class format with attributes that are expected throughout PySoft optuna
    experiments.
    """
    display_name: str = field(init=False, default="")  # the display name of the approximator
    abbrev_name: str  = field(init=False, default="")  # an abbreviated name of the approximator


@dataclass
class NeuroFuzzyNetworkHyperparameters(ApproximatorHyperparameters):
    """
    An all-encompassing class for exposing all available hyperparameters or design-choices of
    neuro-fuzzy networks.
    """
    _initialized: ClassVar[bool] = False
    structure: StructureConfig = field(
        default_factory=StructureConfig,
        metadata={
            "help": "How to design the initial structure of the neuro-fuzzy networks.",
        }
    )
    parameter: ParameterConfig = field(
        default_factory=ParameterConfig,
        metadata={
            "help": "How to initialize the parameters of the neuro-fuzzy network."
        }
    )
    inference: InferenceConfig = field(
        default_factory=InferenceConfig,
        metadata={
            "help": "Inference configuration for neuro-fuzzy networks.",
        }
    )
    evolution: EvolutionConfig = field(
        default_factory=EvolutionConfig,
        metadata={
            "help": "Evolution configuration for neuro-fuzzy networks.",
        }
    )

    def __post_init__(self):
        self.display_name = "Concurrent Optimization of Fuzzy Inference Systems"
        self.abbrev_name = "CO-FIS"

        if not NeuroFuzzyNetworkHyperparameters._initialized:
            self.evolution.rule.epsilon_filter = 0.0  # disable the constraint
            NeuroFuzzyNetworkHyperparameters._initialized = True

    @property
    def fields(self) -> List[Field]:
        return [
            hyperparameter for hyperparameter in fields(self)
            if hyperparameter not in fields(ApproximatorHyperparameters)
        ]