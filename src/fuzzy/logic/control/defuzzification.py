"""
Implements the various versions of the defuzzification process within a fuzzy inference engine.
"""

import abc
from typing import Union

import torch
import numpy as np

from fuzzy.logic.rulebase import RuleBase
from fuzzy.logic.control.configurations import Shape
from fuzzy.sets.membership import Membership
from fuzzy.sets.group import FuzzySetGroup


class Defuzzification(torch.nn.Module):
    """
    Implements the defuzzification process for a fuzzy inference engine.
    """

    def __init__(
        self,
        shape: Shape,
        source: Union[None, np.ndarray, torch.nn.Sequential, FuzzySetGroup],
        device: torch.device,
        rule_base: Union[None, RuleBase],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.source = source
        self.device = device
        self.rule_base: Union[None, RuleBase] = (
            rule_base  # currently only used for Mamdani
        )

    def to(self, device: torch.device, *args, **kwargs) -> "Defuzzification":
        """
        Move the defuzzification process to a device.

        Args:
            device: The device to move the defuzzification process to.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The defuzzification process.
        """
        super().to(device, *args, **kwargs)
        self.device = device
        if hasattr(self.source, "to"):
            self.source.to(device)
        return self

    @abc.abstractmethod
    def forward(self, rule_activations: Membership) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the
        fuzzy logic controller.

        Args:
            rule_activations: The rule activations, or firing levels.

        Returns:
            The defuzzified output of the fuzzy logic controller.
        """


class ZeroOrder(Defuzzification):
    """
    Implements the zero-order (TSK) fuzzy inference; this is also Mamdani fuzzy inference too
    but with fuzzy singleton values as the consequences.
    """

    def __init__(
        self,
        shape: Shape,
        source: Union[None, np.ndarray],
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(shape=shape, source=source, device=device, *args, **kwargs)
        if source is None:
            consequences = torch.empty(
                self.shape.n_rules, self.shape.n_outputs, device=self.device
            )
            # pylint: disable=fixme
            # TODO: Add support for different initialization methods
            torch.nn.init.xavier_normal_(consequences)
        else:
            consequences = torch.as_tensor(source, device=self.device)
        self.consequences = torch.nn.Parameter(consequences)

    def to(self, device: torch.device, *args, **kwargs) -> "ZeroOrder":
        """
        Move the defuzzification process to a device.

        Args:
            device: The device to move the defuzzification process to.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The defuzzification process.
        """
        super().to(device, *args, **kwargs)
        self.consequences.to(device)
        return self

    def forward(self, rule_activations: Membership) -> torch.Tensor:
        # if self.training:
        #     assert not rule_activations.isnan().any(), "Rule activations are NaN!"
        #     assert not rule_activations.isinf().any(), "Rule activations are infinite!"
        #     # assert (
        #     #     rule_activations.sum() > 0
        #     # ), "The sum of all rule activations is zero!"

        # if self.consequences.shape[-1] == 1:  # Multi-Input-Single-Output (MISO)
        #     numerator = (rule_activations * self.consequences.T[0]).sum(dim=1)
        #     denominator = rule_activations.sum(dim=1)
        #     denominator += 1e-32
        #     # the dim=1 takes product across ALL terms, now shape (num of observations,
        #     # num of rules), MISO
        #     return (numerator / denominator)

        # Multi-Input-Multi-Output (MIMO)
        # try:
        #     consequences = torch.mm(self.gg.cuda(), self.consequences.cuda())
        # except AttributeError:
        #     consequences = self.consequences
        # numerator = torch.matmul(rule_activations, consequences)
        # rule_links = self.intermediate_calculation_modules(antecedents_memberships)
        # rule_weight_matrix = self.intermediate_calculation_modules.grouped_links(
        #     antecedents_memberships.elements
        # )
        # curr_device = antecedents_memberships.elements.device
        # rule_activations = (
        #     antecedents_memberships.elements.unsqueeze(-1).to(curr_device) * (
        #     rule_links.transpose(0, 1).to(curr_device) * rule_weight_matrix.to(curr_device)
        #     ).sum(dim=-1).to(
        #             curr_device
        #     )
        # )
        # t = (
        #     antecedents_memberships.elements.unsqueeze(dim=-1)
        #     * self.consequences_matrix
        # )
        numerator = (
            # rule_activations * (t.sum(dim=1).unsqueeze(dim=-1) + self.consequences)
            rule_activations.degrees.unsqueeze(dim=-1)
            * self.consequences
        ).sum(dim=1)

        # pylint: disable=fixme
        # TODO: get this to work properly
        # if "softmax" in self.specs["defuzzification"]:
        #     # the high-dimensional TSK trick with Softmax requires only the numerator
        #     return numerator

        # unsqueeze must be there with or without confidences
        denominator = (rule_activations.degrees).sum(dim=1, keepdim=True)
        denominator += (
            1e-32  # an offset to help with potential near-zero values in denominator
        )
        # shape is (num of observations, num of actions), MIMO
        defuzzification = numerator / denominator

        if self.training:
            assert not defuzzification.isnan().any(), "Defuzzification is NaN!"

        return defuzzification


class TSK(Defuzzification):
    """
    Implements the TSK fuzzy inference (where inputs influence the consequence calculation).
    """

    def __init__(
        self,
        shape: Shape,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(shape=shape, device=device, *args, **kwargs)
        self.consequences = torch.nn.Sequential(
            torch.nn.Linear(
                self.shape.n_rules, self.shape.n_outputs, device=self.device
            ),
        )

    def to(self, device: torch.device, *args, **kwargs) -> "TSK":
        """
        Move the defuzzification process to a device.

        Args:
            device: The device to move the defuzzification process to.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The defuzzification process.
        """
        super().to(device, *args, **kwargs)
        self.consequences.to(device)
        return self

    def forward(self, rule_activations: Membership) -> torch.Tensor:
        return self.consequences(rule_activations.degrees.squeeze(dim=-1))


class Mamdani(Defuzzification):
    """
    Implements Mamdani fuzzy inference.
    """

    def __init__(
        self,
        shape: Shape,
        source: FuzzySetGroup,
        device: torch.device,
        rule_base: RuleBase,
        *args,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            source=source,
            device=device,
            rule_base=rule_base,
            *args,
            **kwargs,
        )
        # this is used for Mamdani inference, but not for TSK inference
        self.output_links: Union[None, torch.Tensor] = torch.as_tensor(
            self.rule_base.consequences.applied_mask.permute(dims=(2, 0, 1)),
            dtype=torch.int8,
            device=self.device,
        )
        self.consequences: FuzzySetGroup = source

    def to(self, device: torch.device, *args, **kwargs) -> "Mamdani":
        """
        Move the defuzzification process to a device.

        Args:
            device: The device to move the defuzzification process to.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The defuzzification process.
        """
        super().to(device, *args, **kwargs)
        self.output_links.to(device)
        self.consequences.to(device)
        return self

    def forward(self, rule_activations: Membership) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the Mamdani FLC.

        Args:
            rule_activations: The rule activations, or firing levels.

        Returns:
            The defuzzified output of a Mamdani FLC.
        """
        numerator = (
            self.output_links * self.consequences.centers * self.consequences.widths
        )
        denominator = self.output_links * self.consequences.widths

        # the below commented out is a Work in Progress

        # gumbel_dist = torch.distributions.Gumbel(0, 1)
        # gumbel_noise = gumbel_dist.sample(self.output_logits.shape)
        # gumbel_softmax = torch.nn.functional.gumbel_softmax(
        #     (self.output_logits + gumbel_noise), dim=-1, hard=True
        # )
        # try:
        #     numerator = (
        #         self.output_links
        #         * self.consequences.centers
        #         * torch.exp(self.consequences.log_widths())
        #     )
        #     denominator = self.output_links * torch.exp(
        #         self.consequences.log_widths()
        #     )
        # except TypeError:
        #     numerator = (
        #         gumbel_softmax
        #         * self.consequences.centers
        #         * torch.exp(self.consequences.widths)
        #     )
        #     denominator = gumbel_softmax * torch.exp(self.consequences.widths)
        return (
            rule_activations.degrees.unsqueeze(dim=-1)
            * (
                torch.nan_to_num(numerator).sum(-1)
                / torch.nan_to_num(denominator).sum(-1)
            )
        ).sum(dim=1)
