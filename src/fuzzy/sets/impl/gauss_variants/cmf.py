"""
Implements conventional membership functions (CMFs) by inheriting from FuzzySet that are
primarily variations of the Gaussian formula.
"""

from dataclasses import dataclass

import numpy as np
import sympy
import torch

from ...abstract import FuzzySet


@dataclass(frozen=False)
class GaussianKernel:
    """
    A dataclass containing static settings which may adjust or influence Gaussian related functions.
    """

    width_multiplier: float = (
        2.0  # in fuzzy logic, convention is usually 1.0, but can be 2.0
    )
    slope_multiplier: float = 1.0


class GeneralizedGuassian(FuzzySet):
    """
    Implementation of the Generalized Gaussian membership function, written in PyTorch.
    """

    def __init__(
        self,
        centers,
        widths,
        device: torch.device,
        gaussian_kernel: GaussianKernel = GaussianKernel(),
        **kwargs,
    ):
        super().__init__(centers=centers, widths=widths, device=device, **kwargs)
        self._gaussian_kernel = gaussian_kernel
        if self._gaussian_kernel.width_multiplier < 0.0:
            raise ValueError(
                f"The width multiplier must be > 0, but got"
                f" {self._gaussian_kernel.width_multiplier}."
            )
        self._width_multiplier = torch.nn.ParameterList([self.make_parameter(
            self._gaussian_kernel.width_multiplier * np.ones_like(centers))])
        self._slope_multiplier = torch.nn.ParameterList([self.make_parameter(
            self._gaussian_kernel.slope_multiplier * np.ones_like(centers))])

    def get_width_multiplier(self) -> torch.Tensor:
        """
        Get the concatenated width multipliers of the fuzzy set from its
        corresponding ParameterList.

        Returns:
            The concatenated width multipliers of the fuzzy set.
        """
        return torch.cat(list(self._width_multiplier), dim=-1)

    def get_slope_multiplier(self) -> torch.Tensor:
        """
        Get the concatenated slope multipliers of the fuzzy set from its
        corresponding ParameterList.

        Returns:
            The concatenated slope multipliers of the fuzzy set.
        """
        return torch.cat(list(self._slope_multiplier), dim=-1)

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        width_multiplier: torch.Tensor,
        slope_multiplier: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Generalized Gaussian fuzzy set.
        This is a static method, so it can be called without instantiating the class.
        This static method is particularly useful when animating the membership function.

        Warning: This method is not meant to be called directly, as it does not take into account
        the mask that likely should exist. Use the calculate_membership method instead.

        Args:
            observations: The observations to calculate the membership for.
            centers: The centers of the Generalized Gaussian fuzzy set.
            width_multiplier: The width multiplier of the Generalized Gaussian fuzzy set.
            slope_multiplier: The slope multiplier of the Generalized Gaussian fuzzy set.

        Returns:
            The membership degrees of the observations for the Generalized Gaussian fuzzy set.
        """
        vals = -1.0 * torch.pow((torch.pow(observations - centers, 2) /
                                 torch.pow(width_multiplier, 2)), slope_multiplier, )
        # this works pretty well -- but does cause NaNs later on
        # vals = (
        #     -1.0 * torch.pow(observations - centers, 2) / torch.pow(width_multiplier, 2)
        # )
        return vals

    @classmethod
    @torch.jit.ignore
    def sympy_formula(cls) -> sympy.Expr:
        # centers (c), widths (sigma) and observations (x)
        pass

    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership of the observations to the Log Gaussian fuzzy set.

        Args:
            observations: The observations to calculate the membership for.

        Returns:
            The membership degrees of the observations for the Log Gaussian fuzzy set.
        """
        return GeneralizedGuassian.internal_calculate_membership(
            observations=observations,
            centers=self.get_centers(),
            width_multiplier=self.get_width_multiplier(),
            slope_multiplier=self.get_slope_multiplier(),
        )


class LogGaussian(FuzzySet):
    """
    Implementation of the Log Gaussian membership function, written in PyTorch.
    This is a modified version that helps when the dimensionality is high,
    and TSK product inference engine will be used.
    """

    def __init__(
        self,
        centers,
        widths,
        device: torch.device,
        gaussian_kernel: GaussianKernel = GaussianKernel(),
        **kwargs,
    ):
        super().__init__(centers=centers, widths=widths, device=device, **kwargs)
        self._gaussian_kernel = gaussian_kernel
        self.width_multiplier = self._gaussian_kernel.width_multiplier
        if int(self.width_multiplier) not in [1, 2]:
            raise ValueError(
                "The width multiplier must be either 1.0 or 2.0, but got {self.width_multiplier}."
            )

    @staticmethod
    @torch.jit.script
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        width_multiplier: float,
        # buffer: torch.Tensor,
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
        # return (
        #     -1.0
        #     * (
        #         torch.pow(
        #             observations - centers,
        #             2,
        #         )
        #         / (width_multiplier * torch.pow(widths, 2) + 1e-32)
        #     )
        # ).clamp(
        #     min=-10, max=0  # was -50 for visualization
        # )  # force values very close to zero to be zero

        # pre-allocate output
        # batch, features = observations.shape[0], observations.shape[1]
        # terms = centers.shape[-1]
        # out = torch.empty(batch, features, terms, device=observations.device,
        #                   dtype=observations.dtype)

        # inv_sigma2 = widths.mul(widths)  # widths^2
        # inv_sigma2.mul_(width_multiplier)  # *= width_multiplier
        # inv_sigma2.add_(1e-32)  # += epsilon
        # inv_sigma2.reciprocal_()  # 1/x (in-place)
        #
        # # Suppose buffer is already allocated with the right shape
        # buffer[:] = observations  # copy data into buffer
        # buffer.sub(centers)  # in-place subtraction (autograd-safe)
        # buffer.mul_(buffer)  # square in-place
        # buffer.mul_(inv_sigma2)  # scale in-place
        # return buffer

        #
        # buffer = torch.sub(observations, centers)
        # buffer.mul_(buffer)
        # buffer.mul_(inv_sigma2)
        # buffer.neg_()
        # buffer.clamp_(min=-10.0, max=0.0)
        # return buffer

        # step 1: observations - centers
        diff = observations - centers  # (batch, features, terms)

        # step 2: square diff in-place (safe, diff not used elsewhere)
        # autograd-safe because diff is a view, not a leaf requiring grad
        diff.pow_(2)

        # step 3: denominator
        denom = width_multiplier * widths.pow(2) + 1e-32  # (features, terms)

        # step 4: division and multiply by -1, store directly in pre-allocated output
        # torch.div(diff, denom, out=out)
        out = diff / denom
        out.mul_(-1.0)

        # step 5: clamp in-place (autograd-safe)
        out.clamp_(min=-10, max=0)
        return out

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


class Gaussian(LogGaussian):
    """
    Implementation of the Gaussian membership function, written in PyTorch.
    """

    @staticmethod
    def internal_calculate_membership(
        observations: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        width_multiplier: float = 1.0,
        # in fuzzy logic, convention is usually 1.0, but can be 2.0
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
