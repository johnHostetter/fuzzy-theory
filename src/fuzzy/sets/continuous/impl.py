"""
Implements various membership functions by inheriting from ContinuousFuzzySet.
"""

from typing import Union

import sympy
import torch

from .membership import Membership
from .abstract import ContinuousFuzzySet


class LogGaussian(ContinuousFuzzySet):
    """
    Implementation of the Log Gaussian membership function, written in PyTorch.
    This is a modified version that helps when the dimensionality is high,
    and TSK product inference engine will be used.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        width_multiplier: float = 1.0,  # in fuzzy logic, convention is usually 1.0, but can be 2.0
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(centers=centers, widths=widths, device=device)
        self.width_multiplier = width_multiplier
        assert int(self.width_multiplier) in [1, 2]

    # @property
    # @torch.jit.ignore
    # def sigmas(self) -> torch.Tensor:
    #     """
    #     Gets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.
    #
    #     Returns:
    #         torch.Tensor
    #     """
    #     return self.widths
    #
    # @sigmas.setter
    # @torch.jit.ignore
    # def sigmas(self, sigmas) -> None:
    #     """
    #     Sets the sigma for the Gaussian fuzzy set; alias for the 'widths' parameter.
    #
    #     Returns:
    #         None
    #     """
    #     self.widths = sigmas

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        width_multiplier: float,
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Log Gaussian fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            centers: The centers of the Log Gaussian fuzzy set.
            widths: The widths of the Log Gaussian fuzzy set.
            width_multiplier: The width multiplier of the Log Gaussian fuzzy set.

        Returns:
            The membership degrees of the observations for the Log Gaussian fuzzy set.
        """
        return -1.0 * (
            torch.pow(
                observations - centers,
                2,
            )
            / (width_multiplier * torch.pow(widths, 2) + 1e-32)
        )

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        # centers (c), widths (sigma) and observations (x)
        center_symbol = sympy.Symbol("c")
        width_symbol = sympy.Symbol("sigma")
        input_symbol = sympy.Symbol("x")
        return sympy.sympify(
            f"-1.0 * pow(({input_symbol} - {center_symbol}), 2) / (2.0 * pow({width_symbol}, 2))"
        )

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Log Gaussian fuzzy set.

        Args:
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the Log Gaussian fuzzy set.
        """
        return LogGaussian.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            widths=self.get_widths(),
            width_multiplier=self.width_multiplier,
        )

    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        # we do not need torch.float64 for observations
        degrees: torch.Tensor = self.calculate_membership(observations.float())

        # assert (
        #     not degrees.isnan().any()
        # ), "NaN values detected in the membership degrees."
        # assert (
        #     not degrees.isinf().any()
        # ), "Infinite values detected in the membership degrees."

        return Membership(
            elements=observations.squeeze(dim=-1),  # remove the last dimension
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )


class Gaussian(LogGaussian):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        width_multiplier: float = 1.0,  # in fuzzy logic, convention is usually 1.0, but can be 2.0
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Gaussian fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            centers: The centers of the Gaussian fuzzy set.
            widths: The widths of the Gaussian fuzzy set.
            width_multiplier: The width multiplier of the Gaussian fuzzy set.

        Returns:
            The membership degrees of the observations for the Gaussian fuzzy set.
        """
        return torch.exp(
            -1.0
            * (
                torch.pow(
                    observations - centers,
                    2,
                )
                / (width_multiplier * torch.pow(widths, 2) + 1e-32)
            )
        )
        # return LogGaussian.internal_calculate_membership(
        #     centers=centers,
        #     widths=widths,
        #     width_multiplier=width_multiplier,
        #     observations=observations,
        # ).exp()

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        return sympy.exp(LogGaussian.sympy_formula())

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        return Gaussian.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            widths=self.get_widths(),
            width_multiplier=1.0,
        )

    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        # we do not need torch.float64 for observations
        degrees: torch.Tensor = self.calculate_membership(observations.float())

        # assert (
        #     not degrees.isnan().any()
        # ), "NaN values detected in the membership degrees."
        # assert (
        #     not degrees.isinf().any()
        # ), "Infinite values detected in the membership degrees."

        return Membership(
            elements=observations.squeeze(dim=-1),  # remove the last dimension
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )


class Lorentzian(ContinuousFuzzySet):
    """
    Implementation of the Lorentzian membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(centers=centers, widths=widths, device=device)

    @property
    @torch.jit.ignore
    def sigmas(self) -> torch.Tensor:
        """
        Gets the sigma for the Lorentzian fuzzy set; alias for the 'widths' parameter.

        Returns:
            torch.Tensor
        """
        return self.widths

    @sigmas.setter
    @torch.jit.ignore
    def sigmas(self, sigmas) -> None:
        """
        Sets the sigma for the Lorentzian fuzzy set; alias for the 'widths' parameter.

        Returns:
            None
        """
        self.widths = sigmas

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Lorentzian fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            centers: The centers of the Lorentzian fuzzy set.
            widths: The widths of the Lorentzian fuzzy set.

        Returns:
            The membership degrees of the observations for the Lorentzian fuzzy set.
        """
        return 1 / (1 + torch.pow((centers - observations) / (0.5 * widths), 2))

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        # centers (c), widths (sigma) and observations (x)
        center_symbol = sympy.Symbol("c")
        width_symbol = sympy.Symbol("sigma")
        input_symbol = sympy.Symbol("x")
        return sympy.sympify(
            f"1 / (1 + pow(({center_symbol} - {input_symbol}) / (0.5 * {width_symbol}), 2))"
        )

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Lorentzian fuzzy set.

        Args:
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the Lorentzian fuzzy set.
        """
        return Lorentzian.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            widths=self.get_widths(),
        )

    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        # we do not need torch.float64 for observations
        degrees: torch.Tensor = self.calculate_membership(observations.float())

        assert (
            not degrees.isnan().any()
        ), "NaN values detected in the membership degrees."
        assert (
            not degrees.isinf().any()
        ), "Infinite values detected in the membership degrees."

        return Membership(
            elements=observations.squeeze(dim=-1),  # remove the last dimension
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )


class LogisticCurve(torch.nn.Module):
    """
    A generic torch.nn.Module class that implements a logistic curve, which allows us to
    tune the midpoint, and growth of the curve, with a fixed supremum (the supremum is
    the maximum value of the curve).
    """

    def __init__(
        self,
        midpoint: float,
        growth: float,
        supremum: float,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device
        self.midpoint = torch.nn.Parameter(
            torch.as_tensor(midpoint, dtype=torch.float16, device=self.device),
            requires_grad=True,  # explicitly set to True for clarity
        )
        self.growth = torch.nn.Parameter(
            torch.as_tensor(growth, dtype=torch.float16, device=self.device),
            requires_grad=True,  # explicitly set to True for clarity
        )
        self.supremum = torch.nn.Parameter(
            torch.as_tensor(supremum, dtype=torch.float16, device=self.device),
            requires_grad=False,  # not a parameter, so we don't want to track it
        )

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Calculate the value of the logistic curve at the given point.

        Args:
            tensors:

        Returns:

        """
        return self.supremum / (
            1 + torch.exp(-1.0 * self.growth * (tensors - self.midpoint))
        )


class Triangular(ContinuousFuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers=None,
        widths=None,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(centers=centers, widths=widths, device=device)

    @staticmethod
    def internal_calculate_membership(
        centers: torch.Tensor, widths: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Triangular fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            centers: The centers of the Triangular fuzzy set.
            widths: The widths of the Triangular fuzzy set.
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        return torch.max(
            1.0 - (1.0 / widths) * torch.abs(observations - centers),
            torch.tensor(0.0),
        )

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        # centers (c), widths (w) and observations (x)
        center_symbol = sympy.Symbol("c")
        width_symbol = sympy.Symbol("w")
        input_symbol = sympy.Symbol("x")
        return sympy.sympify(
            f"max(1.0 - (1.0 / {width_symbol}) * abs({input_symbol} - {center_symbol}), 0.0)"
        )

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Triangular fuzzy set.
        """
        return Triangular.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            widths=self.get_widths(),
        )

    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        # we do not need torch.float64 for observations
        degrees: torch.Tensor = self.calculate_membership(observations.float())

        assert (
            not degrees.isnan().any()
        ), "NaN values detected in the membership degrees."
        assert (
            not degrees.isinf().any()
        ), "Infinite values detected in the membership degrees."

        return Membership(
            elements=observations.squeeze(dim=-1),  # remove the last dimension
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )
