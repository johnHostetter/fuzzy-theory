"""
This script contains the design options that are available. It extends from Enum classes which
outline the names of each option by allowing a selection to coincide along all possible options.
This functionality is ideal for outlining what is permitted for the neural architecture, while
allowing the end-user (or optuna) to select (and store) their selection alongside those.
Furthermore, some of these classes will also store the accompanying function for cohesion (e.g.,
they inherit from torch.nn.Module and can be used accordingly).
"""

from typing import Callable, Tuple, Type, Union

import torch
from entmax import entmax15  # , entmax_bisect
from torch.nn import Module

from fuzzy.utils.options.abstract.ext_enum import CategoricalEnumOptions
from fuzzy.utils.options.abstract.meta import EnumPromoter, Options
from fuzzy.utils.options.abstract.primitive import (
    CategoricalOptions,
    FloatOptions,
    GroupedOptions,
    IntOptions,
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
)


class PremiseAggregation(CategoricalEnumOptions):
    """
    Outlines the available premise aggregation strategies and their implementations.
    """

    enum_cls = PremiseAggregationEnum
    _fn = {
        PremiseAggregationEnum.SUM.value: lambda x: -1 * x.sum(dim=1),
        PremiseAggregationEnum.MEAN.value: lambda x: -1 * x.mean(dim=1),
    }

    @property
    def func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Obtain the appropriate premise aggregation function based on the stored selection.

        Returns:
            A callable function that expects a torch.Tensor and will return a torch.Tensor.
        """
        if self.assignable:
            raise ValueError("A selection has not yet been made.")
        return self._fn[self.selection.value]


class PremiseElimination(CategoricalEnumOptions):
    """
    Outlines the available premise elimination strategies.
    """

    enum_cls = PremiseEliminationEnum


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
            softplus_alpha = torch.nn.functional.softplus(alpha)
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

    @property
    def func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Obtain the appropriate premise aggregation function based on the stored selection.

        Returns:
            A callable function that expects a torch.Tensor and will return a torch.Tensor.
        """
        if self.assignable:
            raise ValueError("A selection has not yet been made.")
        return self._fn[self.selection.value]

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


# @dataclass
class PremiseConfig(GroupedOptions):
    """
    How to set up the premise terms, whether we should use standard implementation (e.g.,
    sum and softmax) or try something more experimental (e.g., mean and 1.5-entmax), as well as
    whether to eliminate any premise terms. Default is SUM for aggregation (standard), NONE for
    elimination, and SOFTMAX for activation (standard if Gaussian fuzzy sets are used, but no exp
    has been applied to them yet).
    """

    def __init__(
        self,
        aggregation: Union[None, PremiseAggregation] = None,
        elimination: Union[None, PremiseElimination] = None,
        activation: Union[None, PremiseActivation] = None,
    ):
        super().__init__()
        self.aggregation: PremiseAggregation = (
            PremiseAggregation() if aggregation is None else aggregation
        )
        self.elimination: PremiseElimination = (
            PremiseElimination() if elimination is None else elimination
        )
        self.activation: PremiseActivation = (
            PremiseActivation() if activation is None else activation
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


# @dataclass
class RuleConfig(GroupedOptions):
    """
    How to set up the fuzzy logic rules, whether they should be weighed (e.g., certainty factors),
    whether to eliminate any, and/or whether to elevate their firing levels. Default is NONE for
    all.
    """

    def __init__(
        self,
        weights: Union[None, RuleWeights] = None,
        elimination: Union[None, RuleElimination] = None,
        elevation: Union[None, RuleElevation] = None,
    ):
        super().__init__()
        self.weights: RuleWeights = RuleWeights() if weights is None else weights
        self.elimination: RuleElimination = (
            RuleElimination() if elimination is None else elimination
        )
        self.elevation: RuleElevation = (
            RuleElevation() if elevation is None else elevation
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


class ApproximatorHyperparameters:
    """
    A standard data class format with attributes that are expected throughout PySoft optuna
    experiments.
    """

    display_name: str  # the display name of the approximator
    abbrev_name: str  # an abbreviated name of the approximator

    def __init__(self, display_name, abbrev_name):
        self.display_name = display_name
        self.abbrev_name = abbrev_name


class NeuroFuzzyNetworkHyperparameters(ApproximatorHyperparameters):
    """
    An all-ecompassing class for exposing all available hyperparameters or design-choices of
    neuro-fuzzy networks.
    """

    def __init__(self):
        super().__init__(
            display_name="Concurrent Optimization of Fuzzy Inference Systems",
            abbrev_name="CO-FIS",
        )
        self.n_rules: IntOptions = IntOptions(
            64, 256, 64
        )  # int range [64, 256] w/ step=64
        self.init_width: float = 0.5  # the initial width for the fuzzy sets
        self.premise: PremiseConfig = PremiseConfig()
        self.premise_sampling: Sampling = Sampling()  # how to sample premises
        self.rule: RuleConfig = RuleConfig()
        self.gumbel_temperature: FloatOptions = FloatOptions(
            0.25,
            1.25,
        )  # what temperature to use for Gumbel-Softmax
        self.epsilon_filter: CategoricalOptions = CategoricalOptions(
            0.0, 0.1
        )  # epsilon-filtering for premise sampling
        self.epsilon_filter.selection = 0.0  # disable it
        self.noise_delay: CategoricalOptions = CategoricalOptions(
            1,
            # means no delay in updating the GMT noise (since it's about
            # modulo)
            32,
            64,
            128,
            256,
        )  # how much to delay updating the GMT noise
        self.epsilon: CategoricalOptions = CategoricalOptions(
            0.1, 0.5
        )  # epsilon-completeness
        self.add_premise_delay: IntOptions = IntOptions(
            1,  # means no delay in adding fuzzy sets (since it's about modulo)
            5,
            2,
        )  # how much to delay adding new premises; int range [1, 5] w/ step=2
