"""
This module contains the FuzzySetGroup class, which is a generic and abstract torch.nn.Module
class that contains a torch.nn.ModuleList of FuzzySet objects. The expectation here is
that each FuzzySet may define fuzzy sets of different conventions, such as Gaussian,
Triangular, Trapezoidal, etc. Then, subsequent inference engines can handle these heterogeneously
defined fuzzy sets with no difficulty. Further, this class was specifically designed to incorporate
dynamic addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.
"""

from typing import Union, List, Any

import torch

from .membership import Membership
from ..utils import NestedTorchJitModule


class FuzzySetGroup(NestedTorchJitModule):
    """
    A generic and abstract torch.nn.Module class that contains a torch.nn.ModuleList
    of FuzzySet objects. The expectation here is that each FuzzySet may define fuzzy sets of
    different conventions, such as Gaussian, Triangular, Trapezoidal, etc.
    Then, subsequent inference engines can handle these heterogeneously defined fuzzy sets
    with no difficulty. Further, this class was specifically designed to incorporate dynamic
    addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.

    However, this class does *not* carry out any functionality that is necessarily tied to fuzzy
    sets, it is simply named so as this was its intended purpose - grouping fuzzy sets. In other
    words, the same "trick" of using a torch.nn.ModuleList of torch.nn.Module objects applies to
    any kind of torch.nn.Module object.
    """

    def __init__(
        self,
        *args,
        modules_list: Union[None, List[torch.nn.Module]] = None,
        device: torch.device = None,
        **kwargs,
    ):
        """
        Initialize the FuzzySetGroup object.

        Args:
            *args: Optional positional arguments.
            modules_list: A list of torch.nn.Module objects.
            device: The device to move the FuzzySetGroup object to; if None, the device is
            inferred from the modules_list.
            **kwargs: Optional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if modules_list is None:
            modules_list = []
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.device = device

    def __getattribute__(self, item):
        try:
            if item in ("centers", "widths", "mask"):
                modules_list = self.__dict__["_modules"]["modules_list"]
                if len(modules_list) > 0:
                    module_attributes: List[torch.Tensor] = (
                        []
                    )  # the secondary response denoting module filter
                    for module in modules_list:
                        # get the method for the module and then call it
                        item_method: callable = getattr(module, f"get_{item}")
                        module_attributes.append(item_method())
                    return torch.cat(module_attributes, dim=-1)
                raise ValueError("The torch.nn.ModuleList of FuzzySetGroup is empty.")
            return object.__getattribute__(self, item)
        except AttributeError:
            return self.__getattr__(item)

    def __hash__(self) -> int:
        _hash: str = ""
        for module in self.modules_list:
            _hash += str(hash(module))
        return hash(_hash)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FuzzySetGroup):
            return False
        if len(self.modules_list) != len(other.modules_list):
            return False
        for self_module, other_module in zip(self.modules_list, other.modules_list):
            if not self_module == other_module:
                return False
        return True

    def to(self, device: torch.device, *args, **kwargs) -> "FuzzySetGroup":
        """
        Move the FuzzySetGroup to a different device.

        Args:
            device: The device to move the FuzzySetGroup object to.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The FuzzySetGroup object.
        """
        super().to(device, *args, **kwargs)
        self.device = device
        for module in self.modules_list:
            module.to(device)
        return self

    def calculate_module_responses(self, observations) -> Membership:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of FuzzySetGroup.
        """
        if len(self.modules_list) > 0:
            # modules' responses are membership degrees when modules are FuzzySet
            if len(self.modules_list) == 1:
                # for computational efficiency, return the response from the only module
                return self.modules_list[0](observations)

            # this can be computationally expensive, but it is necessary to calculate the responses
            # from all the modules in the torch.nn.ModuleList of FuzzySetGroup
            # ideally this should be done in parallel, but it is not possible with the current
            # implementation; only use this if the torch.nn.Module objects are different
            module_elements: List[torch.Tensor] = []
            module_memberships: List[torch.Tensor] = (
                []
            )  # the primary response from the module
            module_masks: List[torch.Tensor] = (
                []
            )  # the secondary response denoting module filter
            for module in self.modules_list:
                membership: Membership = module(observations)
                module_elements.append(membership.elements)
                module_memberships.append(membership.degrees)
                module_masks.append(membership.mask)
            return Membership(
                elements=torch.cat(module_elements, dim=-1),
                degrees=torch.cat(module_memberships, dim=-1),
                mask=torch.cat(module_masks, dim=-1),
            )
        raise ValueError("The torch.nn.ModuleList of FuzzySetGroup is empty.")

    def forward(self, observations) -> Membership:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of FuzzySetGroup.
        Expand the FuzzySetGroup if necessary.
        """
        (
            _,  # module_elements
            module_responses,
            module_masks,
        ) = self.calculate_module_responses(observations)

        # TODO: this code does not work for torch.jit.script  # pylint: disable=fixme
        # self.expand(observations, module_responses, module_masks)

        return Membership(
            elements=observations, degrees=module_responses, mask=module_masks
        )
