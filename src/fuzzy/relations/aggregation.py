"""
Implements aggregation operators in fuzzy theory.
"""

import torch

from fuzzy.utils.classes import Loggable

# from fuzzy.utils.functions import log_method


class OrderedWeightedAveraging(torch.nn.Module, Loggable):
    """
    Yager's On Ordered Weighted Averaging Aggregation Operators in
    Multicriteria Decisionmaking (1988)

    An operator that lies between the 'anding' or the 'oring' of multiple criteria.
    The weight vector allows us to easily adjust the degree of 'anding' and 'oring'
    implicit in the aggregation.
    """

    def __init__(self, in_features, weights):
        super().__init__()
        self.in_features = in_features
        if self.in_features != len(weights):
            raise AttributeError(
                "The number of input features expected in the Ordered Weighted Averaging operator "
                "is expected to equal the number of elements in the weight vector."
            )
        with torch.no_grad():
            # validate the value that is actually stored (post-abs()), not the raw input -
            # otherwise a vector with negative entries that happens to sum to 1.0 (e.g.
            # [2.0, -1.0]) would pass validation here but be stored as [2.0, 1.0], which
            # sums to 3.0 and silently violates this class's own invariant
            abs_weights = torch.abs(weights)
            if abs_weights.sum() == 1.0:
                self.weights = torch.nn.parameter.Parameter(abs_weights)
            else:
                raise AttributeError(
                    "The weight vector of the Ordered Weighted Averaging operator must sum to 1.0."
                )

    # @log_method
    def orness(self):
        """
        A degree of 1 means the OWA operator is the 'or' operator,
        and this occurs when the first element of the weight vector is equal to 1
        and all other elements in the weight vector are zero.

        Returns:
            The degree to which the Ordered Weighted Averaging operator is an 'or' operator.
        """
        return (1 / (self.in_features - 1)) * torch.tensor(
            [
                (self.in_features - i) * self.weights[i - 1]
                for i in range(1, self.in_features + 1)
            ]
        ).sum()

    # @log_method
    def dispersion(self):
        """
        The measure of dispersion; essentially, it is a measure of entropy that is related to the
        Shannon information concept. The more disperse the weight vector, the more information
        is being used in the aggregation of the aggregate value.

        Returns:
            The amount of dispersion in the weight vector.
        """
        # there is exactly one entry where it is equal to one
        if len(torch.where(self.weights == 1.0)[0]) == 1:
            return torch.zeros(1)
        # torch.xlogy(0, 0) is defined as 0 (the conventional entropy limit), unlike
        # 0 * torch.log(0) which is IEEE754 NaN (0 * -inf) and would poison the sum
        # for any weight vector containing a zero outside the single-1.0 case
        # above
        return -1 * torch.xlogy(self.weights, self.weights).sum()

    # @log_method
    def forward(self, input_observation):
        """
        Applies the Ordered Weighted Averaging operator. First, it will sort the argument
        in descending order, then multiply by the weight vector, and finally sum over the entries.

        Args:
            input_observation: Argument vector, unordered.

        Returns:
            The aggregation of the ordered argument vector with the weight vector.
        """
        # namedtuple with 'values' and 'indices' properties
        ordered_argument_vector = torch.sort(input_observation, descending=True)
        return (self.weights * ordered_argument_vector.values).sum()
