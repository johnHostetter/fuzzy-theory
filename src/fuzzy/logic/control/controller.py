"""
Contains various classes necessary for Fuzzy Logic Controllers (FLCs) to function properly,
as well as the Fuzzy Logic Controller (FLC) itself.

This Python module also contains functions for extracting information from a knowledge base
(to avoid circular dependency). The functions are used to extract premise terms, consequence terms,
and fuzzy logic rule matrices. These components may then be used to create a fuzzy inference system.
"""

from collections import OrderedDict
from typing import Union, List, Type

import torch
from fuzzy.sets.abstract import FuzzySet
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables

from .defuzzification import Defuzzification
from .configurations import Shape, FuzzySystem, GranulationLayers
from ...relations.t_norm import TNorm


class FuzzyLogicController(torch.nn.Sequential):
    """
    Abstract implementation of the Multiple-Input-Multiple-Output (MIMO)
    Fuzzy Logic Controller (FLC).
    """

    def __init__(
        self,
        source: FuzzySystem,
        inference: Type[Defuzzification],
        device: torch.device,
        disabled_parameters: Union[None, List[str]] = None,
        **kwargs,
    ):
        super().__init__(*[], **kwargs)
        if disabled_parameters is None:
            disabled_parameters = []

        self.source = source
        self.inference_type = inference
        self.device: torch.device = device
        self.disabled_parameters: List[str] = disabled_parameters

        # build or extract the necessary components for the FLC from the source
        granulation_layers: GranulationLayers = source.granulation_layers(
            device=self.device
        )
        engine: TNorm = source.engine(device=self.device)

        defuzzification = self.inference_type(
            shape=self.source.shape,
            source=granulation_layers["output"],
            device=self.device,
            rule_base=(
                self.source.rule_base
                if isinstance(self.source, KnowledgeBase)
                else None
            ),
        )

        # disables certain parameters & prepare fuzzy inference process
        self.disable_parameters_and_build(
            modules=OrderedDict(
                [
                    ("input", granulation_layers["input"]),
                    ("engine", engine),
                    ("defuzzification", defuzzification),
                ]
            )
        )

    @property
    def shape(self) -> Shape:
        """
        Shortcut to the shape of the FLC.

        Returns:
            The shape of the FLC.
        """
        return self.source.shape

    def to(self, *args, **kwargs):
        """
        Move the FLC to a different device. This is an override of the 'to' method in the
        'torch.nn.Module' class. This exists as some modules within the FLC may not be moved
        properly using the 'to' method. For example, modules that have tensors that are not
        torch.nn.Parameters, but are important for fuzzy inference.

        Args:
            *args: The positional arguments.
            **kwargs: The keyword arguments.

        Returns:

        """
        # Call the parent class's `to` method to handle parameters and submodules
        super().to(*args, **kwargs)

        # special handling for the modules with non-parameter tensors, such as mask or links
        for module in self.children():
            if hasattr(module, "to"):
                module.to(*args, **kwargs)
        self.device = self.engine.device  # assuming torch.nn.Sequential is non-empty
        return self

    def disable_parameters_and_build(self, modules: OrderedDict) -> None:
        """
        Disable any selected parameters across the modules (e.g., granulation layers). This is
        useful for stability and convergence. It is also useful for preventing the learning of
        certain parameters. Adds the modules to the FLC.

        Args:
            *modules: The modules to add to the FLC, where some may have parameters disabled.

        Returns:
            None
        """
        for module_name, module in modules.items():  # ignore the name
            if module is not None:
                for param_name, param in module.named_parameters():
                    if "mask" not in param_name and hasattr(param, "requires_grad"):
                        # ignore attribute with "mask" in it; assume it's a non-learnable parameter,
                        # or cannot enable this parameter; this is by design - do not raise an error
                        # examples of such a case are mask parameters, links, and offsets
                        param.requires_grad = param_name not in self.disabled_parameters
                self.add_module(module_name, module)

    def split_granules_by_type(self) -> OrderedDict[str, List[FuzzySet]]:
        """
        Retrieves the granules at a given layer (e.g., premises, consequences) from the Fuzzy Logic
        Controller. Specifically, this operation takes the granulation layer (a more computationally
        efficient representation) and converts the premises back to a list of granules format.
        For example, rather than using a single Gaussian object to represent all Gaussian membership
        functions in the layer space, this function will convert that to a list of Gaussian objects,
        where each Gaussian function is defined and restricted to a single dimension in that layer.

        Returns:
            A nested list of FuzzySet objects, where the length is equal to the number
            of layer's dimensions. Within each element of the outer list, is another list that
            contains all the definitions for FuzzySet within that dimension. For
            example, if the 0'th index has a list equal to [Gaussian(), Trapezoid()], then this
            means in the 0'th dimension there are both membership functions defined using the
            Gaussian formula and the Trapezoid formula.
        """
        results: {str: List[FuzzySet]} = OrderedDict()

        # at each variable index, it is possible to have more than 1 type of module
        for module_name, module in self.named_modules():
            if hasattr(module, "split_by_variables"):
                results[module_name] = module.split_by_variables()
        return results

    def linguistic_variables(self) -> LinguisticVariables:
        """
        Extract the linguistic variables from the FLC. This is useful for extracting the linguistic
        variables for the input and output spaces. This is useful for visualizing the linguistic
        variables in the FLC.

        Returns:
            A list of FuzzySet objects that represent the linguistic variables in the
            given layer.
        """
        results: OrderedDict = self.split_granules_by_type()
        if len(results) < 1 or 2 < len(results):
            raise ValueError(
                f"Expected 1 or 2 granulation layers, but received {len(results)}."
            )
        results_lst: List[List[FuzzySet]] = [
            variables for _, variables in results.items()
        ]  # discard the name of where the variables are from
        return LinguisticVariables(
            inputs=results_lst[0],
            targets=None if len(results_lst) < 2 else results_lst[1],
        )
