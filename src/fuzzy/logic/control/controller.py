"""
Contains various classes necessary for Fuzzy Logic Controllers (FLCs) to function properly,
as well as the Fuzzy Logic Controller (FLC) itself.

This Python module also contains functions for extracting information from a knowledge base
(to avoid circular dependency). The functions are used to extract premise terms, consequence terms,
and fuzzy logic rule matrices. These components may then be used to create a fuzzy inference system.
"""

import inspect
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, MutableMapping, Optional, Tuple, Type

import torch
import torch.utils.checkpoint

from fuzzy.logic.variables import LinguisticVariables
from fuzzy.sets.abstract import FuzzySet
from fuzzy.sets.membership import Membership

from ...relations.n_ary import NAryRelation
from ...relations.t_norm import TNorm
from ...sets import FuzzySetGroup
from ...utils import load_module_class, module_class
from .configurations.abstract import FuzzySystem
from .configurations.data import ExecutionOptions, GranulationLayers, Shape
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
        execution: Optional[ExecutionOptions] = None,
        **kwargs,
    ):
        super().__init__(*[], **kwargs)
        if execution is None:
            execution = ExecutionOptions()

        self.source = source
        self.device: torch.device = device
        self.disabled_parameters: List[str] = execution.disabled_parameters
        self.max_batch_chunk: Optional[int] = execution.max_batch_chunk
        self.gradient_checkpointing: bool = execution.gradient_checkpointing

        # A = torch.ones((self.source.configuration["algorithm"].learning.batch.selection,
        #                 self.shape.n_outputs),
        #                device=self.device)
        # self.A = torch.nn.Parameter(A, requires_grad=True)

        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(self.shape.n_inputs, 512),
        #     getattr(torch.nn, "ReLU")(),
        #     torch.nn.Linear(512, self.shape.n_outputs),
        # )

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
                    ("input_granulation", granulation_layers["input"]),
                    ("engine", engine),
                    ("defuzzification", defuzzification),
                ]
            )
        )

        # bind the correct defuzzification call to avoid try/except on every
        # forward. Rather than isinstance-checking against every known TSK-style
        # subclass - which would require importing each one here, including
        # downstream subclasses defined outside fuzzy-theory entirely (e.g. a
        # project's own CP-decomposed TSK variant), creating a backwards
        # dependency - inspect the concrete class's own forward() signature.
        # Defuzzification's contract (see TSK.forward vs.
        # Mamdani/ZeroOrder.forward's docstrings) is that a TSK-style variant
        # declares "observations" with no default, since it's required to
        # compute the consequence, while a Mamdani-style variant gives it a
        # default of None (unused, present only so both call shapes are
        # interchangeable).
        try:
            observations_param = inspect.signature(
                type(defuzzification).forward
            ).parameters["observations"]
        except KeyError as error:
            raise TypeError(
                f"{type(defuzzification).__name__}.forward must declare an "
                "'observations' parameter (required, with no default, for a "
                "TSK-style variant that needs it to compute the consequence; "
                "given a default of None otherwise) so the FLC can determine "
                "how to call it."
            ) from error
        requires_observations = observations_param.default is inspect.Parameter.empty
        self._defuzzify = (
            self._defuzzify_tsk if requires_observations else self._defuzzify_standard
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
        path.mkdir(parents=True, exist_ok=True)
        state_dict: MutableMapping[str, Any] = self.state_dict()
        # cast to tuple for serialization
        state_dict["shape"] = tuple(self.shape)
        # save the FLC state dictionary
        torch.save(state_dict, path / "flc.pt")
        self.input_granulation.save(
            path / "input"
        )  # save the input granulation layer (drop the extension)
        # save the inference engine under a subdirectory named after its full
        # module path (e.g. "fuzzy.relations.t_norm.Product"), not just its bare
        # class name - external tooling (and load(), below) can then discover the
        # engine's concrete type - including a custom TNorm subclass that isn't
        # bundled with fuzzy-theory at all - straight from the directory listing,
        # without needing to unpickle state_dict.pt first
        engine_type: str = module_class(self.engine)
        self.engine.save(path / "engine" / engine_type)
        self.defuzzification.save(
            path / "defuzzification"
        )  # save the defuzzification method

    @staticmethod
    def _locate_engine_save_dir(engine_dir: Path) -> Path:
        """
        Find the single module-path-named subdirectory save() wrote the engine to
        (see save() above), so load() can read from it without needing to already
        know the engine's concrete type.

        Args:
            engine_dir: The "engine" directory directly under an FLC's save path.

        Returns:
            The engine's own save directory, e.g.
            engine_dir / "fuzzy.relations.t_norm.Product".
        """
        subdirs = [entry for entry in engine_dir.iterdir() if entry.is_dir()]
        if len(subdirs) != 1:
            raise ValueError(
                f"Expected exactly one engine-type subdirectory under {engine_dir}, "
                f"but found {[entry.name for entry in subdirs]}."
            )
        return subdirs[0]

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
        # load the components from their respective directories. the engine is
        # loaded via its own concrete type (engine_type.load(...), not
        # NAryRelation.load(...)) rather than relying on NAryRelation.load()'s
        # internal get_subclass() dispatch, so that a custom TNorm subclass that
        # overrides load() with extra behavior actually gets invoked - calling the
        # base class's load() explicitly would silently skip any such override.
        # get_subclass()/TorchJitModule can only find a class Python has already
        # imported, though, so the module is dynamically imported (via
        # load_module_class) first - see _locate_engine_save_dir for where that
        # module path comes from (the engine's own save() directory name, see
        # save() above).
        input_granules = FuzzySetGroup.load(path / "input", device=device)
        engine_dir = FuzzyLogicController._locate_engine_save_dir(path / "engine")
        engine_type: Type[object] = load_module_class(engine_dir.name)
        assert issubclass(
            engine_type, NAryRelation
        ), "The loaded module type must be a subclass of NAryRelation."
        engine: NAryRelation = engine_type.load(engine_dir, device=device)
        assert isinstance(
            engine, TNorm
        ), "The loaded engine must be an instance of TNorm."
        defuzzification = Defuzzification.load(path / "defuzzification", device=device)

        # load the FLC state dictionary for the remaining components
        state_dict: MutableMapping[str, Any] = torch.load(
            path / "flc.pt", map_location=device, weights_only=False
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

    def _defuzzify_tsk(
        self, observations: torch.Tensor, rule_strengths
    ) -> torch.Tensor:
        return self.defuzzification(
            observations=observations, rule_activations=rule_strengths
        )

    def _defuzzify_standard(
        self, _observations: torch.Tensor, rule_strengths
    ) -> torch.Tensor:
        # _observations is intentionally unused here (this is the Mamdani/"standard"
        # path - see Defuzzification.forward's docstring); the parameter exists so
        # this and _defuzzify_tsk share an identical, interchangeable signature (see
        # the self._defuzzify binding below)
        return self.defuzzification(rule_strengths)

    def _forward_impl(
        self, observations: torch.Tensor  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        """
        Core forward pass implementing the fuzzy inference pipeline:
        fuzzification, rule evaluation, and defuzzification.
        """
        granulated_input = self.input_granulation(observations)

        if self.gradient_checkpointing and self.training:
            # torch.utils.checkpoint.checkpoint's pytree-based arg flatten/unflatten
            # and its traced HigherOrderOperator body both require tensors only
            # under torch.compile(fullgraph=True) - a Membership namedtuple's
            # formula field (a str, since FuzzySet.degree_range/Membership.formula
            # were added) violates that on *both* sides: as an input argument
            # (pytree cannot reconstruct a Membership from only its tensor leaves,
            # since "formula" has no field default to fall back on) and as the
            # checkpointed function's return value ("HigherOrderOperator body's
            # output must consist of tensors only"). Passing/returning only the
            # plain tensor fields across the checkpoint boundary, and
            # constructing/reading the full Membership just inside/outside of it
            # (ordinary, non-HigherOrderOperator Dynamo-traced code, unaffected by
            # either restriction), avoids both.
            def _checkpointed_engine(
                degrees: torch.Tensor, mask: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                membership = Membership(
                    degrees=degrees, mask=mask, formula=granulated_input.formula
                )
                result = self.engine(membership)
                return result.degrees, result.mask

            degrees, mask = torch.utils.checkpoint.checkpoint(
                _checkpointed_engine,
                granulated_input.degrees,
                granulated_input.mask,
                use_reentrant=False,
            )
            rule_strengths = Membership(
                degrees=degrees, mask=mask, formula=type(self.engine).__name__
            )
        else:
            rule_strengths = self.engine(granulated_input)

        return self._defuzzify(observations, rule_strengths)

    def forward(  # pylint: disable=arguments-renamed
        self, observations: torch.Tensor  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        """
        Forward pass for the FLC. This is the main method that will be called when the FLC is used
        in a forward pass. This method will perform the fuzzy inference process, which includes
        fuzzification, rule evaluation, and defuzzification.

        Args:
            observations: The observations to perform the fuzzy inference on.

        Returns:
            The defuzzified output of the FLC.
        """
        if (
            self.max_batch_chunk is not None
            and observations.shape[0] > self.max_batch_chunk
        ):
            return torch.cat(
                [
                    self._forward_impl(chunk)
                    for chunk in observations.split(self.max_batch_chunk)
                ],
                dim=0,
            )
        return self._forward_impl(observations=observations)
