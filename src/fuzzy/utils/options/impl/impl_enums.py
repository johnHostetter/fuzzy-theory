"""
This script contain classes that inherit from str as well as Enum to outline the different names
of various design options for their accompanying neural architecture. These are ideal for a
consistent interface to reference the selection of a particular feature inside a neural
architecture. If interested in selecting from these options, see design_options.py
"""

from enum import Enum


class DimensionEnum(str, Enum):
    """
    Provides consistent access to frequently used dimensions and their associated names.
    """

    BATCH = "batch"
    SEQ = "seq"  # use to refer to temporal dimension
    CHANNEL = "channel"  # use to refer to image channel
    VARIABLE = "variable"  # use when unsure of direction
    TERM = "term"  # use when unsure of direction
    INPUT_VARIABLE = "input_variable"
    INPUT_TERM = "input_term"
    RULE = "rule"
    OUT_TERM = "output_term"
    OUT_VARIABLE = "output_variable"


class PremiseAggregationEnum(str, Enum):
    """
    The aggregation method for premise activations.
    """

    SUM = "sum"
    MEAN = "mean"


class PremiseEliminationEnum(str, Enum):
    """
    Elimination, or drop out, of premise terms from fuzzy logic rules.
    """

    NONE = "none"  # no premise elimination
    # NO_OP = "no_op"  # use the no-op procedure where it is possible for a fake fuzzy set to be
    # used that has no impact or influence (represents "don't care" membership
    # function)


class BoundAlphaEntmaxEnum(str, Enum):
    """
    What technique we should use to keep the alpha within (1, 2) when using entmax_bisect.
    """
    # NONE = "none"  # only valid if no learnable alpha is used for alpha-entmax
    SIGMOID = "sigmoid_reparameterization"
    TANH = "scaled_tanh"
    HARD_TANH = "hard_tanh"
    SOFTPLUS = "softplus_add_shift"


class PremiseActivationEnum(str, Enum):
    """
    Completes calculating the membership degrees of each premise term assuming Gaussian functions
    were used without their exp.
    """

    SOFTMAX = "softmax"  # (alpha=1) default -- no modification
    ENTMAX15 = "entmax15"  # (alpha=1.5) non-tunable -- some sparsity
    # SPARSEMAX = "sparsemax"  # (alpha=2) included for completeness, but strongly advise against it
    ENTMAX_BISECT = "entmax_bisect"  # differentiable w.r.t. X & alpha
    # SPARSEMAX_BISECT = "sparsemax_bisect",  # only normalizes along last dim (not implemented)
    # NORMMAX_BISECT = "normmax_bisect",  # differentiable w.r.t. X (not implemented)
    # BUDGET_BISECT = "budget_bisect",  # differentiable w.r.t. X (not
    # implemented)


class NeurogenesisEnum(str, Enum):
    """
    How to create and add new fuzzy sets to the neuro-fuzzy network.
    """

    # NONE = "none"  # static premise layer
    # Hostetter's 2025 dissertation
    MODIFIED_DELAYED_WELFORD = "modified_delayed_welford"


class SamplingEnum(str, Enum):
    """
    How to complete the one-hot differentiable sampling of a discrete stochastic process.
    """

    # w/ noise ~ Gumbel(0, 1)
    ST_GUMBEL_SOFTMAX = "straight_through_gumbel_softmax"
    # ST = "straight_through"  # no noise


class RuleWeightsEnum(str, Enum):
    """
    This allows for the disabling/enabling of rule weights (e.g., certainty factors).
    """

    NONE = "none"
    CERTAINTY_FACTORS = "certainty_factors"


class RuleEliminationEnum(str, Enum):
    """
    This allows for the disabling/enabling of rule elimination in an attempt to improve
    interpretability of fuzzy inference.
    """

    NONE = "none"  # no rule elimination
    # NO_OP = (
    #     "no_op"  # use the no-op procedure where it is possible for a fuzzy logic rule's
    # )
    # consequence to be ineffective (essentially dropping it from influence)


class RuleElevationEnum(str, Enum):
    """
    This allows for the disabling/enabling of layer normalization in an attempt to alleviate the
    curse of dimensionality's impact on fuzzy logic rule activation. In essence, it elevates the
    firing strengths as well as preserves the overall shape of the firing levels.
    """

    NONE = "none"
    LAYER_NORMALIZATION = "layer_normalization"


class DefuzzificationMethodEnum(str, Enum):
    """
    This allows for the selection of various implemented defuzzification methods in fuzzy inference.
    """

    TSK = "tsk"
    ZERO_ORDER_TSK = "zero_order_tsk"
    MAMDANI = "mamdani"
    CP_DECOMPOSED_TSK = "cp_decomposed_tsk"
