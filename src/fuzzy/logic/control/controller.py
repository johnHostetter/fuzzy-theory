"""
Contains various classes necessary for Fuzzy Logic Controllers (FLCs) to function properly,
as well as the Fuzzy Logic Controller (FLC) itself.

This Python module also contains functions for extracting information from a knowledge base
(to avoid circular dependency). The functions are used to extract premise terms, consequence terms,
and fuzzy logic rule matrices. These components may then be used to create a fuzzy inference system.
"""

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, MutableMapping, Type, Union

import torch

from fuzzy.logic.variables import LinguisticVariables
from fuzzy.sets.abstract import FuzzySet

from ...relations.t_norm import TNorm
from ...sets import FuzzySetGroup
from .configurations.abstract import FuzzySystem
from .configurations.data import GranulationLayers, Shape
from .configurations.impl import Defined
from .defuzzification import Defuzzification


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
        self.device: torch.device = device
        self.disabled_parameters: List[str] = disabled_parameters

        # build or extract the necessary components for the FLC from the source
        granulation_layers: GranulationLayers = source.granulation_layers
        engine: TNorm = source.engine
        defuzzification = source.defuzzification(inference, device=self.device)

        # check that the size of the components are compatible
        input_granulation_size: torch.Size = granulation_layers["input"].centers.shape
        engine_size: torch.Size = engine.shape[
            :-1
        ]  # drop the last dimension (# of rules)
        if input_granulation_size != engine_size:
            # raise ValueError(
            #     f"The input granulation layer size {input_granulation_size} "
            #     f"does not match the engine size {engine_size}."
            # )
            pass

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

    def save(self, path: Path) -> None:
        """
        Save the FLC to a directory. This is a custom process to ensure that all the necessary
        components are saved properly.

        Args:
            path: The directory path to save the FLC to.

        Returns:
            None
        """
        # each component is given its own directory to save to for easier access
        # and to avoid the risk of overwriting files
        state_dict: MutableMapping[str, Any] = self.state_dict()
        # cast to tuple for serialization
        state_dict["shape"] = tuple(self.shape)
        # save the FLC state dictionary
        torch.save(state_dict, path / "flc.pt")
        self.input.save(
            path / "input"
        )  # save the input granulation layer (drop the extension)
        self.engine.save(path / "engine.pt")  # save the inference engine
        # TODO: figure out a better way to handle the configuration
        if hasattr(self.engine, "configuration"):
            # pickle the configuration for the engine
            with open(path / "engine_config.pkl", "wb+") as f:
                pickle.dump(
                    self.engine.configuration, f, protocol=pickle.HIGHEST_PROTOCOL
                )
            # save the layer_norm weights
            torch.save(self.engine.layer_norm.state_dict(), path / "layer_norm.pt")

        self.defuzzification.save(
            path / "defuzzification"
        )  # save the defuzzification method

    @staticmethod
    def load(path: Path, device: torch.device) -> "FuzzyLogicController":
        """
        Load the FLC from a directory. This is a custom process to ensure that all the necessary
        components are loaded properly.

        Args:
            path: The directory path to load the FLC from.
            device: The device to load the FLC to.

        Returns:
            The FLC object.
        """
        # load the components from their respective directories
        input_granules = FuzzySetGroup.load(path / "input", device=device)
        engine = TNorm.load(path / "engine", device=device)
        # TODO: figure out a better way to handle the configuration
        if hasattr(engine, "configuration"):
            # pickle the configuration for the engine
            with open(path / "engine_config.pkl", "rb") as f:
                engine.configuration = pickle.load(f)
            # load the layer_norm weights
            engine.layer_norm.load_state_dict(
                torch.load(path / "layer_norm.pt", map_location=device)
            )

        defuzzification = Defuzzification.load(path / "defuzzification", device=device)

        # load the FLC state dictionary for the remaining components
        state_dict: MutableMapping[str, Any] = torch.load(
            path / "flc.pt", map_location=device
        )
        shape: Shape = Shape(*state_dict.pop("shape"))

        defined_fuzzy_system = Defined(
            shape=shape,
            granulation=GranulationLayers(input=input_granules, output=None),
            engine=engine,
            defuzzification=defuzzification,
        )

        return FuzzyLogicController(
            source=defined_fuzzy_system,
            inference=type(defuzzification),
            device=device,
        )

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
        # Call the parent class's `to` method to handle parameters and
        # submodules
        super().to(*args, **kwargs)

        # special handling for the modules with non-parameter tensors, such as
        # mask or links
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
                # for param_name, param in module.named_parameters():
                #     if "mask" not in param_name and hasattr(param, "requires_grad"):
                #         # ignore attribute with "mask" in it; assume it's a non-learnable
                #         # parameter, or cannot enable this parameter; this is by design
                #         # - do not raise an error examples of such a case are
                #         # mask parameters, links, and offsets
                #         param.requires_grad = param_name not in self.disabled_parameters
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

        # at each variable index, it is possible to have more than 1 type of
        # module
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FLC. This is the main method that will be called when the FLC is used
        in a forward pass. This method will perform the fuzzy inference process, which includes
        fuzzification, rule evaluation, and defuzzification.

        Args:
            input: The input (observations) to perform the fuzzy inference on.

        Returns:
            The defuzzified output of the FLC.
        """
        # fuzzification
        granulated_input = self.input(input)

        # rule evaluation
        rule_strengths = self.engine(granulated_input)

        # defuzzification
        try:  # TSK
            return self.defuzzification(input, rule_strengths)
        except TypeError:  # Mamdani, ZeroOrder, etc.
            return self.defuzzification(rule_strengths)
