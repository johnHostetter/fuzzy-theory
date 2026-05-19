"""
Implements various  conventional membership functions (CMFs) by inheriting from FuzzySet,
such as no operation (i.e., do nothing) and triangular fuzzy sets.
"""

from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
import sympy
import torch

from ...utils import check_path_to_save_torch_module
from ...utils.classes import Loggable
from ..abstract import FuzzySet
from ..membership import Membership


class NoOp(FuzzySet):
    """
    Implementation of the NoOp membership function, written in PyTorch.
    """

    def __init__(
        self, n_elements: int, membership: float, device: torch.device, **kwargs
    ):
        centers = np.zeros(n_elements, dtype=np.float32)[:, np.newaxis]
        widths = np.zeros(n_elements, dtype=np.float32)[:, np.newaxis]
        self.membership = membership  # the flat membership degree of the NoOp fuzzy set
        self.n_elements = n_elements
        super().__init__(centers=centers, widths=widths, device=device, **kwargs)

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the fuzzy set to a file.

        Note: This does not preserve the ParameterList structure, but rather concatenates the
        parameters into a single tensor, which is then saved to a file.

        Returns:
            A dictionary containing the state of the fuzzy set.
        """
        check_path_to_save_torch_module(path)
        state_dict: MutableMapping = self.state_dict()
        state_dict["class_name"] = self.__class__.__name__
        state_dict["n_elements"] = self.n_elements
        state_dict["membership"] = self.membership
        torch.save(state_dict, path)
        return state_dict

    @classmethod
    # @log_classmethod
    def load(cls, path: Path, device: torch.device) -> "FuzzySet":
        """
        Load the fuzzy set from a file and put it on the specified device.

        Returns:
            None
        """
        state_dict: MutableMapping = torch.load(path, weights_only=False)
        n_elements = state_dict.pop("n_elements")
        membership = state_dict.pop("membership")
        return NoOp(
            n_elements=n_elements,
            membership=membership,
            device=device,
        )

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        membership_degree: float,
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the NoOp fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            membership_degree: The membership degree of the NoOp fuzzy set.

        Returns:
            The membership degrees of the observations for the NoOp fuzzy set.
        """
        return (
            torch.ones(
                1, device=observations.device, dtype=observations.dtype
            ).expand_as(observations)
            * membership_degree
        )

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        # centers (c), widths (sigma) and observations (x)
        pass

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership of the observations to the NoOp fuzzy set.

        Args:
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the NoOp fuzzy set.
        """
        return NoOp.internal_calculate_membership(
            observations=observations,
            membership_degree=self.membership,
        )

    # pylint: disable=duplicate-code
    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        degrees: torch.Tensor = self.calculate_membership(observations)

        # assert (
        #     not degrees.isnan().any()
        # ), "NaN values detected in the membership degrees."
        # assert (
        #     not degrees.isinf().any()
        # ), "Infinite values detected in the membership degrees."

        return Membership(
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )

    # pylint: enable=duplicate-code


class Lorentzian(FuzzySet):
    """
    Implementation of the Lorentzian membership function, written in PyTorch.
    """

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

    # pylint: disable=duplicate-code
    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        degrees: torch.Tensor = self.calculate_membership(observations)

        assert (
            not degrees.isnan().any()
        ), "NaN values detected in the membership degrees."
        assert (
            not degrees.isinf().any()
        ), "Infinite values detected in the membership degrees."

        return Membership(
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )

    # pylint: enable=duplicate-code


class LogisticCurve(torch.nn.Module, Loggable):
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
        device: torch.device,
    ):
        super().__init__()
        self.device = device
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


class Triangular(FuzzySet):
    """
    Implementation of the Triangular membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers,
        widths,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(centers=centers, widths=widths, device=device, **kwargs)

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

    # pylint: disable=duplicate-code
    def forward(self, observations) -> Membership:
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)
        degrees: torch.Tensor = self.calculate_membership(observations)

        # assert (
        #     not degrees.isnan().any()
        # ), "NaN values detected in the membership degrees."
        # assert (
        #     not degrees.isinf().any()
        # ), "Infinite values detected in the membership degrees."

        return Membership(
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )

    # pylint: enable=duplicate-code
