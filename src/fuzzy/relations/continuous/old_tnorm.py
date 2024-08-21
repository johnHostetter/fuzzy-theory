"""
Implements the t-norm fuzzy relations.
"""

from enum import Enum

from fuzzy.relations.continuous.t_norm import Product, Minimum, SoftmaxSum


class TNorm(Enum):
    """
    Enumerates the types of t-norms.
    """

    PRODUCT = Product  # i.e., algebraic product
    MINIMUM = Minimum
    ACZEL_ALSINA = "aczel_alsina"  # not yet implemented
    SOFTMAX_SUM = SoftmaxSum  # not yet implemented
    SOFTMAX_MEAN = "softmax_mean"
    LUKASIEWICZ = "generalized_lukasiewicz"
    # the following are to be implemented
    DRASTIC = "drastic"
    NILPOTENT = "nilpotent"
    HAMACHER = "hamacher"
    EINSTEIN = "einstein"
    YAGER = "yager"
    DUBOIS = "dubois"
    DIF = "dif"
