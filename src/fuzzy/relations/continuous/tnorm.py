"""
Implements the t-norm fuzzy relations.
"""

from enum import Enum

import torch


class TNorm(Enum):
    """
    Enumerates the types of t-norms.
    """

    PRODUCT = "product"  # i.e., algebraic product
    MINIMUM = "minimum"
    ACZEL_ALSINA = "aczel_alsina"  # not yet implemented
    SOFTMAX_SUM = "softmax_sum"
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


class AlgebraicProduct(torch.nn.Module):  # TODO: remove this class
    """
    Implementation of the Algebraic Product t-norm (Fuzzy AND).
    """

    def __init__(self, in_features=None, importance=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            importance is initialized to a one vector by default
        """
        super().__init__()
        self.in_features = in_features

        # initialize antecedent importance
        if importance is None:
            self.importance = torch.nn.parameter.Parameter(torch.tensor(1.0))
            self.importance.requires_grad = False
        else:
            if not isinstance(importance, torch.Tensor):
                importance = torch.Tensor(importance)
            self.importance = torch.nn.parameter.Parameter(
                torch.abs(importance)
            )  # importance can only be [0, 1]
            self.importance.requires_grad = True

    def forward(self, elements):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        self.importance = torch.nn.parameter.Parameter(
            torch.abs(self.importance)
        )  # importance can only be [0, 1]
        return torch.prod(torch.mul(elements, self.importance))
