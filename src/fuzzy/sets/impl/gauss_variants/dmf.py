"""
Implements dimension-dependent membership functions as proposed in:

    High-Dimensional Fuzzy Inference Systems

by Guangdong Xue; Jian Wang; Kai Zhang; Nikhil R. Pal

Citation:

    G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy Inference Systems,"
    in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 1, pp. 507-519,
    Jan. 2024, doi: 10.1109/TSMC.2023.3311475.

The code was modified from the public repository of the authors:

    https://github.com/Eandon/HDFIS/

to make it compatible with this fuzzy-theory library.
"""

import abc
from typing import Union

import numpy as np
import sympy
import torch

from fuzzy.sets.abstract import FuzzySet
from fuzzy.sets.impl.gauss_variants.cmf import Gaussian, LogGaussian


class DimensionDependent(FuzzySet, abc.ABC):
    """
    This class represents a Dimension-Dependent fuzzy set. It is an abstract base class that
    provides the basic structure for Dimension-Dependent fuzzy sets, such as Gaussian DMF and
    Gaussian No-Exp DMF. It inherits from the FuzzySet class and implements the
    calculate_membership method, which is used to calculate the membership of observations
    to the fuzzy set based on the centers, widths, and rho parameters.
    """

    def __init__(
        self,
        centers: np.ndarray,
        widths: np.ndarray,
        device: torch.device,
        rho: Union[None, torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(centers=centers, widths=widths, device=device, **kwargs)
        self.n_inputs: torch.Tensor = torch.tensor(
            [centers.shape[0]], dtype=torch.int8, device=device
        )  # count of input variables/features/dimensions
        self.rho: torch.Tensor = (
            self._calculate_rho(n_inputs=self.n_inputs, device=device)
            if rho is None
            else torch.tensor([rho], dtype=torch.float32, device=device)
        )

    @staticmethod
    def _calculate_rho(
            n_inputs: torch.Tensor,
            device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            return (
                torch.ones(1, device=device)
                - torch.tensor([745], device=device).log() / n_inputs.log()
            )


class GaussianNoExpDMF(DimensionDependent):
    """
    This class represents the Gaussian w/ No-Exp Dimension-Dependent fuzzy set. This is a special
    case of the Dimension-Dependent fuzzy set where the Gaussian membership function is assumed.
    """

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        n_inputs: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Dimension-Dependent fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            centers: The centers of the Gaussian DMF fuzzy set.
            widths: The widths of the Gaussian DMF fuzzy set.
            n_inputs: The widths of the Gaussian DMF fuzzy set.
            rho: The scale parameter for the Gaussian DMF fuzzy set.

        Returns:
            The membership degrees of the observations for the Gaussian DMF fuzzy set.
        """

        return LogGaussian.internal_calculate_membership(
            observations=observations,
            centers=centers,
            widths=widths,
            width_multiplier=torch.pow(n_inputs, rho).item(),
        )

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        # centers (c), widths (sigma), observations (x), dimensions (N) and rho (rho)
        center_symbol = sympy.Symbol("c")
        width_symbol = sympy.Symbol("sigma")
        input_symbol = sympy.Symbol("x")
        dim_symbol = sympy.Symbol("N")
        rho_symbol = sympy.Symbol("rho")
        return sympy.sympify(
            f"-1.0 * pow(({input_symbol} - {center_symbol}), 2) / "
            f"(pow({dim_symbol}, {rho_symbol}) + pow({width_symbol}, 2))"
        )

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Log Gaussian fuzzy set.

        Args:
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the Log Gaussian fuzzy set.
        """
        return GaussianNoExpDMF.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            widths=self.get_widths(),
            n_inputs=self.n_inputs,
            rho=self.rho,
        )

    def prepare_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.float()


class GaussianDMF(DimensionDependent):
    """
    This class represents the Gaussian Dimension-Dependent fuzzy set. This is a special case of the
    Dimension-Dependent fuzzy set where the Gaussian membership function is assumed.
    """

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        n_inputs: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Dimension-Dependent fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            centers: The centers of the Gaussian DMF fuzzy set.
            widths: The widths of the Gaussian DMF fuzzy set.
            n_inputs: The widths of the Gaussian DMF fuzzy set.
            rho: The scale parameter for the Gaussian DMF fuzzy set.

        Returns:
            The membership degrees of the observations for the Gaussian DMF fuzzy set.
        """

        return Gaussian.internal_calculate_membership(
            observations=observations,
            centers=centers,
            widths=widths,
            width_multiplier=torch.pow(n_inputs, rho).item(),
        )
        # ORIGINAL:
        # return torch.exp(
        #     -1.0
        #     * (
        #         torch.pow(
        #             observations - centers,
        #             2,
        #         )
        #         / (torch.pow(n_inputs, rho) + torch.pow(widths, 2) + 1e-32)
        #     )
        # )

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        return sympy.exp(GaussianNoExpDMF.sympy_formula())

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Log Gaussian fuzzy set.

        Args:
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the Log Gaussian fuzzy set.
        """
        return GaussianDMF.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            widths=self.get_widths(),
            n_inputs=self.n_inputs,
            rho=self.rho,
        )

    def prepare_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.float()
