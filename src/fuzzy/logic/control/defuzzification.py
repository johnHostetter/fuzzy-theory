"""
Implements the various versions of the defuzzification process within a fuzzy inference engine.
"""

import abc
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from fuzzy.logic.control.configurations.data import Shape
from fuzzy.logic.rulebase import RuleBase
from fuzzy.sets.group import FuzzySetGroup
from fuzzy.sets.membership import Membership
from fuzzy.utils import TorchJitModule
from fuzzy.utils.options.impl.impl_enums import ConsequenceInitEnum


class Defuzzification(TorchJitModule, abc.ABC):
    """
    Implements the defuzzification process for a fuzzy inference engine.
    """

    def __init__(
        self,
        *args,
        shape: Shape,
        source: Union[None, np.ndarray, torch.nn.Sequential, FuzzySetGroup],
        device: torch.device,
        rule_base: Union[None, RuleBase],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.source = source
        self.device = device
        self.rule_base: Union[None, RuleBase] = (
            rule_base  # currently only used for Mamdani
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "FuzzySet":
        """
        Load the fuzzy set from a file and put it on the specified device.

        Returns:
            None
        """
        try:
            state_dict: MutableMapping = torch.load(path, weights_only=False)
        except UnicodeDecodeError:
            # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xde in position 157881408:
            # invalid continuation byte
            state_dict = torch.load(path, weights_only=False, encoding="latin1")

        shape: Shape = Shape(*state_dict.pop("shape"))
        class_name: str = state_dict.pop("class_name")

        source: Union[None, np.ndarray, torch.nn.Sequential, FuzzySetGroup] = (
            state_dict.pop("source") if "source" in state_dict else None
        )
        rule_base: Union[None, RuleBase] = (
            state_dict.pop("rule_base") if "rule_base" in state_dict else None
        )
        defuzzification = cls.get_subclass(class_name)(
            shape=shape,
            source=source,
            device=device,
            rule_base=rule_base,
        )
        # load the remaining parameters
        defuzzification.load_state_dict(state_dict)
        return defuzzification

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
    def forward(
        self,
        rule_activations: Membership,
        observations: Union[None, torch.Tensor],
    ) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the
        fuzzy logic controller.

        Args:
            rule_activations: The rule activations, or firing levels.
            observations: The observations that activated the rules (does nothing here).

        Returns:
            The defuzzified output of the fuzzy logic controller.
        """


class ZeroOrder(Defuzzification):
    """
    Implements the zero-order (TSK) fuzzy inference; this is also Mamdani fuzzy inference too
    but with fuzzy singleton values as the consequences.
    """

    _INIT_METHODS = {
        ConsequenceInitEnum.XAVIER_NORMAL: torch.nn.init.xavier_normal_,
        ConsequenceInitEnum.XAVIER_UNIFORM: torch.nn.init.xavier_uniform_,
        ConsequenceInitEnum.KAIMING_NORMAL: torch.nn.init.kaiming_normal_,
        ConsequenceInitEnum.KAIMING_UNIFORM: torch.nn.init.kaiming_uniform_,
        ConsequenceInitEnum.ZEROS: torch.nn.init.zeros_,
        ConsequenceInitEnum.NORMAL: torch.nn.init.normal_,
        ConsequenceInitEnum.UNIFORM: torch.nn.init.uniform_,
    }

    def __init__(
        self,
        shape: Shape,
        source: Union[None, np.ndarray, FuzzySetGroup],
        device: torch.device,
        *args,
        init_method: ConsequenceInitEnum = ConsequenceInitEnum.XAVIER_NORMAL,
        **kwargs,
    ):
        super().__init__(shape=shape, source=source, device=device, *args, **kwargs)
        if source is None:
            consequences = torch.empty(
                self.shape.n_rules, self.shape.n_outputs, device=self.device
            )
            self._INIT_METHODS[init_method](consequences)
        elif isinstance(source, FuzzySetGroup):
            consequences = torch.as_tensor(source.centers, device=self.device)
        else:
            consequences = torch.as_tensor(source, device=self.device)
        assert consequences.shape[0] == self.shape.n_rules
        assert consequences.shape[1] == self.shape.n_outputs
        self.consequences = torch.nn.Parameter(consequences)

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the defuzzification process to a directory.

        Args:
            path: The directory path to save the defuzzification process to.

        Returns:
            The state dictionary of the defuzzification process that was saved.
        """
        state_dict: MutableMapping = self.state_dict()
        state_dict["class_name"] = self.__class__.__name__
        # convert to tuple for serialization
        state_dict["shape"] = tuple(self.shape)
        state_dict["source"] = (
            self.consequences.detach().cpu().numpy()  # pylint: disable=not-callable
        )
        torch.save(state_dict, path)
        return state_dict

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

    def forward(
        self,
        rule_activations: Membership,
        observations: Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the
        fuzzy logic controller.

        Args:
            rule_activations: The rule activations, or firing levels.
            observations: The observations that activated the rules (does nothing here).

        Returns:
            The defuzzified output of the fuzzy logic controller.
        """
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

        return (rule_activations.degrees.unsqueeze(dim=-1) * self.consequences).sum(
            dim=1
        )


class NormalizedZeroOrder(ZeroOrder):
    """
    Implements the normalized zero-order (TSK) fuzzy inference; this is also Mamdani fuzzy inference
    but with fuzzy singleton values as the consequences, and the output is normalized by the sum of
    the rule activations.

    This is useful for cases where the sum of the rule activations is not equal to
    one, and we want to ensure that the output is normalized.
    """

    def forward(
        self,
        rule_activations: Membership,
        observations: Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the
        fuzzy logic controller.

        Args:
            rule_activations: The rule activations, or firing levels.
            observations: The observations that activated the rules (does nothing here).

        Returns:
            The defuzzified output of the fuzzy logic controller.
        """
        numerator: torch.Tensor = super().forward(
            rule_activations,
        )
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
        source: Union[None, np.ndarray],
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(shape=shape, source=source, device=device, *args, **kwargs)
        if source is None:
            consequences = torch.randn(  # used to be torch.zero until April 30, 2026
                [shape.n_outputs, shape.n_rules, shape.n_inputs + 1],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            consequences = torch.as_tensor(source, device=self.device)

        # self.weights = torch.nn.Parameter(
        #     torch.ones([shape.n_rules], dtype=torch.float32), requires_grad=True
        # )
        # Split weights and bias once
        shape = consequences[:, :, 1:].shape
        # Flatten (r, o) into one big linear projection dimension
        self.r, self.o, self.f = shape[0], shape[1], shape[2]
        self.weights = torch.nn.Parameter(
            consequences[:, :, 1:].reshape(self.r * self.o, self.f).T.contiguous(),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            consequences[:, :, 0].contiguous().unsqueeze(0), requires_grad=True
        )

    @property
    def consequences(self):
        """
        Obtain the consequences of the fuzzy logic controller, which for this particular type of
        defuzzification, include both biases and weights.

        Returns:
            The biases and weights of the fuzzy logic controller's decisions.
        """
        return torch.cat(
            [
                self.bias.squeeze(0).unsqueeze(-1),
                # self.weights is stored transposed (see __init__: reshape(r*o, f).T)
                # for the linear-projection matmul in forward() - undo that transpose
                # before reshaping back to (r, o, f), or reshape() reinterprets the
                # wrong bytes and silently scrambles the values
                self.weights.T.reshape(self.r, self.o, self.f),
            ],
            dim=-1,
        )

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the defuzzification process to a directory.

        Args:
            path: The directory path to save the defuzzification process to.

        Returns:
            The state dictionary of the defuzzification process that was saved.
        """
        state_dict: MutableMapping = self.state_dict()
        state_dict["class_name"] = self.__class__.__name__
        # convert to tuple for serialization
        state_dict["shape"] = tuple(self.shape)
        state_dict["source"] = self.consequences.detach().cpu().numpy()
        torch.save(state_dict, path)
        return state_dict

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
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)
        return self

    def forward(
        self,
        rule_activations: Membership,
        observations: Union[None, torch.Tensor],
    ) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the
        fuzzy logic controller.

        Args:
            rule_activations: The rule activations, or firing levels.
            observations: The observations that activated the rules (required).

        Returns:
            The defuzzified output of the fuzzy logic controller.
        """
        # print("w", self.consequences[0].state_dict()['weight'][0][0])
        # print("b", self.consequences[0].state_dict()['bias'][0])
        # old_rule_output = (
        #     self.consequences[:, :, 1:] @ observations.T
        # ).T + self.consequences[:, :, 0].T

        # Compute batch matrix multiplication without transposes
        # Using einsum: 'r o f, b f -> b r o'
        # if self.weights.device != observations.device:
        #     self.weights = self.weights.to(observations.device)
        # slow einsum
        # rule_output = torch.einsum("r o f, b f -> b r o", self.weights, observations)
        # fast alt
        # weights: (r, o, f)
        # observations: (b, f)
        # want: (b, r, o)

        rule_output = observations @ self.weights  # (b, r*o)
        rule_output = rule_output.view(
            observations.shape[0], self.r, self.o
        )  # (b, r, o)

        # Add bias with broadcasting
        # if self.bias.device != observations.device:
        #     self.bias = self.bias.to(observations.device)
        # shape: (batch_size, n_rules,
        rule_output.add_(self.bias)
        # n_outputs)
        rule_output = rule_output.transpose(1, 2)

        # assert torch.allclose(rule_output.float(), old_rule_output)

        # a product t-norm's rule strengths commonly underflow to exactly 0.0 in
        # float32 once enough sub-1 factors are multiplied together (e.g. ~64+ input
        # variables is enough for every rule to underflow) - without this offset, a
        # batch element where every rule underflows divides 0.0 by 0.0, producing NaN
        # that poisons the entire output (and gradient) for that batch, not just the
        # degenerate element. Mirrors the same offset already used by the sibling
        # Defuzzification.forward() (the "standard"/Mamdani path) for exactly this
        # reason.
        fir_str_bar = rule_activations.degrees / (
            torch.sum(rule_activations.degrees, 1).unsqueeze(1) + 1e-32
        )  # [num_sam,num_rule]
        model_output = torch.einsum(
            "NRC,NR->NC", rule_output, fir_str_bar  # * self.weights
        )  # [num_sam,out_dim]
        # print(model_output.max().item())
        return model_output


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
            self.rule_base.consequences.get_mask().permute(dims=(2, 0, 1)),
            dtype=torch.int8,
            device=self.device,
        )
        self.consequences: FuzzySetGroup = source

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the defuzzification process to a directory.

        Args:
            path: The directory path to save the defuzzification process to.

        Returns:
            The state dictionary of the defuzzification process that was saved.
        """
        raise NotImplementedError(
            "Mamdani defuzzification does not support saving yet."
        )

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
        # output_links is a plain tensor, not a Parameter or a registered buffer, so
        # nn.Module.to() (called above) does not know to move it - Tensor.to() is not
        # in-place, so its return value must be reassigned or the move silently never
        # happens
        self.output_links = self.output_links.to(device)
        self.consequences.to(device)
        return self

    def forward(
        self,
        rule_activations: Membership,
        observations: Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given the activations of the fuzzy logic rules, calculate the output of the Mamdani FLC.

        Args:
            rule_activations: The rule activations, or firing levels.
            observations: The observations that activated the rules (does nothing here).

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
