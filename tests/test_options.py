"""
Test and validate the primitive options and Enum implementations are behaving as intended.
"""

import os
import shutil
import unittest
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple, Type

from fuzzy.utils.options.abstract.meta import EnumPromoter, Options
from fuzzy.utils.options.abstract.primitive import (CategoricalEnumOptions,
                                                    CategoricalOptions,
                                                    FloatOptions,
                                                    GroupedOptions, IntOptions)
from fuzzy.utils.options.impl.impl_enums import (NeurogenesisEnum,
                                                 PremiseActivationEnum,
                                                 PremiseAggregationEnum,
                                                 PremiseEliminationEnum,
                                                 RuleElevationEnum,
                                                 RuleEliminationEnum,
                                                 RuleWeightsEnum, SamplingEnum)
from fuzzy.utils.options.impl.impl_options import (
    _64_BIT_INT, EvolutionConfig, GumbelConfig,
    NeuroFuzzyNetworkHyperparameters, PremiseActivation, PremiseConfig,
    Range, RuleConfig)


class DemoEnum(Enum):
    """
    A small, module-level Enum used only to exercise CategoricalEnumOptions, which
    promotes an enum_cls' members onto whichever concrete subclass sets it as a class
    attribute (rather than via the enum_cls= class keyword argument).
    """

    A = "a"
    B = "b"


class DemoCategoricalEnumOptions(CategoricalEnumOptions):
    """
    A minimal concrete subclass of CategoricalEnumOptions, set up the way a real user
    of the library would: enum_cls assigned in the class body rather than passed as
    the enum_cls= class keyword argument (the latter is already exercised elsewhere,
    e.g. PremiseAggregation in impl_options.py).
    """

    enum_cls = DemoEnum


class TestOptions(unittest.TestCase):
    """
    A class to test various implementations of the abstract class called Options.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir = Path(os.path.dirname(os.path.abspath(__file__)))

    def test_categorical_options(
            self) -> tuple[CategoricalOptions, CategoricalOptions]:
        """
        Test categorical options satisfies common functionality and can reliably store its
        selected categorical value.

        Returns:
            The instance of categorical options before assignment, and the instance of
            categorical options after assignment.
        """
        options: Tuple[str, str, str] = ("A", "B", "C")
        selection: str = options[1]
        before_assignment, after_assignment = self.check_common_functionality(
            CategoricalOptions,
            options,
            selection,
            path=self.dir / "categorical_options.pickle",
        )
        # type checking
        self.assertIsInstance(before_assignment, CategoricalOptions)
        self.assertIsInstance(after_assignment, CategoricalOptions)
        # options were correctly stored
        self.assertEqual(options, after_assignment.options)
        return before_assignment, after_assignment

    def test_int_options(self) -> Tuple[IntOptions, IntOptions]:
        """
        Test integer options satisfies common functionality and can reliably store its
        selected integer value.

        Returns:
            The instance of integer options before assignment, and the instance of
            integer options after assignment.
        """
        args: Tuple[int, int, int] = (0, 4, 1)
        before_assignment, after_assignment = self.check_common_functionality(
            IntOptions, args, selection=2, path=self.dir / "int_options.pickle"
        )
        # type checking
        self.assertIsInstance(before_assignment, IntOptions)
        self.assertIsInstance(after_assignment, IntOptions)
        attributes: Tuple[str, str, str] = ("start", "end", "step")
        for idx, attribute in enumerate(attributes):
            self.assertEqual(args[idx], getattr(after_assignment, attribute))
        return before_assignment, after_assignment

    def test_float_options(self) -> Tuple[FloatOptions, FloatOptions]:
        """
        Test float options satisfies common functionality and can reliably store its
        selected float value.

        Returns:
            The instance of float options before assignment, and the instance of
            float options after assignment.
        """
        args: Tuple[float, float] = (0.1, 0.9)
        before_assignment, after_assignment = self.check_common_functionality(
            FloatOptions, args, selection=0.75, path=self.dir / "float_options.pickle")
        # type checking
        self.assertIsInstance(before_assignment, FloatOptions)
        self.assertIsInstance(after_assignment, FloatOptions)
        attributes: Tuple[str, str] = ("start", "end")
        for idx, attribute in enumerate(attributes):
            self.assertEqual(args[idx], getattr(after_assignment, attribute))
        return before_assignment, after_assignment

    def test_group_options(self) -> None:
        """
        Test categorical, integer, and float options satisfies common functionality and can
        reliably store their selected values. Then, it tests that a GroupedOptions instance can
        be created from these primitive subclasses of Options as well as validates the
        GroupedOptions behavior works as intended. This test validates both the scenarios where
        assignments have not yet taken place, and after selections have been made.

        Returns:
            None
        """
        categorical_options_before_assignment, categorical_options_after_assignment = (
            self.test_categorical_options())
        int_options_before_assignment, int_options_after_assignment = (
            self.test_int_options()
        )
        float_options_before_assignment, float_options_after_assignment = (
            self.test_float_options()
        )

        path = self.dir / "grouped_options"
        path.mkdir(parents=True, exist_ok=True)
        grouped_options, kwargs = self.create_grouped_options_from_kwargs(
            categorical_options=categorical_options_before_assignment,
            int_options=int_options_before_assignment,
            float_options=float_options_before_assignment,
        )
        self.check_grouped_options_behavior(grouped_options, path, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        grouped_options, kwargs = self.create_grouped_options_from_kwargs(
            categorical_options=categorical_options_after_assignment,
            int_options=int_options_after_assignment,
            float_options=float_options_after_assignment,
        )
        self.check_grouped_options_behavior(grouped_options, path, **kwargs)

    def test_premise_activation(self) -> None:
        """
        Test the default selections for PremiseActivation.

        Returns:
            None
        """
        premise_activation: PremiseActivation = PremiseActivation()
        self.assertEqual(-1, premise_activation.dim)

    def test_premise_config(self) -> None:
        """
        Test the default selections for PremiseConfig.

        Returns:
            None
        """
        premise_config: PremiseConfig = PremiseConfig()
        self.assertIsNotNone(premise_config)
        self.assertEqual(
            PremiseAggregationEnum.SUM,
            premise_config.aggregation)
        self.assertEqual(
            PremiseEliminationEnum.NONE,
            premise_config.elimination)
        self.assertEqual(
            PremiseActivationEnum.SOFTMAX,
            premise_config.activation)

    def test_rule_config(self) -> None:
        """
        Test the default selections for RuleConfig.

        Returns:
            None
        """
        rule_config: RuleConfig = RuleConfig()
        self.assertIsNotNone(rule_config)
        self.assertEqual(RuleWeightsEnum.NONE, rule_config.weights)
        self.assertEqual(RuleEliminationEnum.NONE, rule_config.elimination)
        self.assertEqual(RuleElevationEnum.NONE, rule_config.elevation)

    def test_default_neuro_fuzzy_network_hyperparameters(self) -> None:
        """
        Test that an instance of NeuroFuzzyNetworkHyperparameters has the correct default values.

        Returns:
            None
        """
        nfn = NeuroFuzzyNetworkHyperparameters()
        # default structure settings --- #
        self.assertEqual(nfn.structure.rule.n_rules, 128)

        # default parameter settings --- #
        self.assertEqual(nfn.parameter.premise.init_width, 0.5)

        # --- default evolution settings --- #
        self.assertEqual(
            nfn.evolution.premise.neurogenesis,
            NeurogenesisEnum.NONE)
        self.assertEqual(nfn.evolution.premise.epsilon, 0.5)
        self.assertEqual(nfn.evolution.premise.add_premise_delay, 1)
        self.assertEqual(
            nfn.evolution.rule.sampling,
            SamplingEnum.ST_GUMBEL_SOFTMAX)
        self.assertEqual(nfn.evolution.rule.temperature, 1.0)
        self.assertEqual(nfn.evolution.rule.epsilon_filter, 0.0)
        self.assertEqual(nfn.evolution.rule.noise_delay, 1)

        # --- default inference settings --- #
        self.assertEqual(
            nfn.inference.premise.aggregation,
            PremiseAggregationEnum.SUM)
        self.assertEqual(
            nfn.inference.premise.elimination,
            PremiseEliminationEnum.NONE)
        self.assertEqual(
            nfn.inference.premise.activation, PremiseActivationEnum.SOFTMAX
        )
        self.assertEqual(nfn.inference.rule.weights, RuleWeightsEnum.NONE)
        self.assertEqual(
            nfn.inference.rule.elimination,
            RuleEliminationEnum.NONE)
        self.assertEqual(
            nfn.inference.rule.elevation,
            RuleEliminationEnum.NONE)

    def test_64_bit_int_is_correct_magnitude(self) -> None:
        """
        Regression guard: _64_BIT_INT used to be written as `2 ^ 63 - 1`, where `^` is
        Python's XOR operator (not exponentiation) and binds looser than `-`, silently
        evaluating to 60 instead of the intended max magnitude of a 64-bit integer.

        Returns:
            None
        """
        self.assertEqual(_64_BIT_INT, (2**63) - 1)

    def test_range_to_scipy_high_is_inclusive_with_step(self) -> None:
        """
        Regression guard: Range.to_scipy() used to call scipy.stats.randint(low, high)
        directly, but scipy.stats.randint excludes its upper bound while the rest of
        Range (contains(), to_optuna()) treats high as inclusive - an off-by-one
        between the two search-space representations of the same Range.

        Returns:
            None
        """
        value_range = Range(low=0, high=4, step=1)
        distribution = value_range.to_scipy()
        low_support, high_support = distribution.support()
        self.assertEqual(0, low_support)
        self.assertEqual(4, high_support)

    def test_epsilon_filter_disabled_for_every_instance(self) -> None:
        """
        Regression guard: the "disable the epsilon_filter constraint" override used to
        be gated behind a ClassVar that only let it run for the very first
        NeuroFuzzyNetworkHyperparameters instantiated in the process - any later
        instance, even one built with an explicit non-zero epsilon_filter, silently
        kept it instead of being forced to 0.0.

        Returns:
            None
        """
        # an instance built first, to occupy the "first ever" slot the old ClassVar
        # guard would have consumed
        NeuroFuzzyNetworkHyperparameters()

        custom_evolution = EvolutionConfig(
            rule=GumbelConfig(epsilon_filter=0.1))
        self.assertEqual(0.1, custom_evolution.rule.epsilon_filter)

        nfn = NeuroFuzzyNetworkHyperparameters(evolution=custom_evolution)
        self.assertEqual(0.0, nfn.evolution.rule.epsilon_filter)

    def test_categorical_enum_options_promotes_onto_concrete_subclass(self) -> None:
        """
        Regression guard: CategoricalEnumOptions.__init__ used to call
        EnumPromoter.__init_subclass__(enum_cls=self.enum_cls) directly.
        __init_subclass__ is implicitly a classmethod, so accessing it through the
        base class bound cls=EnumPromoter itself, not the actual concrete subclass -
        every CategoricalEnumOptions subclass would clobber the same shared
        EnumPromoter attributes instead of getting its own.

        Returns:
            None
        """
        options = DemoCategoricalEnumOptions()

        # promoted onto the concrete subclass, as intended
        self.assertIs(DemoCategoricalEnumOptions.A, DemoEnum.A)
        self.assertIs(DemoCategoricalEnumOptions.B, DemoEnum.B)
        self.assertEqual(("a", "b"), DemoCategoricalEnumOptions.options)
        self.assertEqual(("a", "b"), options.options)

        # NOT leaked onto the shared EnumPromoter base class
        self.assertFalse(hasattr(EnumPromoter, "A"))
        self.assertFalse(hasattr(EnumPromoter, "B"))

    @staticmethod
    def create_grouped_options_from_kwargs(
        categorical_options: CategoricalOptions,
        int_options: IntOptions,
        float_options: FloatOptions,
    ) -> Tuple[GroupedOptions, Dict[str, Options]]:
        """
        Create grouped options from keyword arguments.

        Args:
            categorical_options: The categorical options.
            int_options: The integer options.
            float_options: The float options.

        Returns:
            The created grouped options, as well as the keyword arguments used to create it.
        """
        kwargs: Dict[str, Options] = {
            "categorical_options": categorical_options,
            "int_options": int_options,
            "float_options": float_options,
        }
        return GroupedOptions(**kwargs), kwargs

    def check_grouped_options_behavior(
        self, grouped_options: GroupedOptions, path: Path, **kwargs
    ) -> None:
        """
        Check the behavior of grouped options. In particular, ensure that saving and loading
        works as expected.

        Args:
            grouped_options: The grouped options.
            path: The path to use for saving and loading. Depending on the complexity of the
                Options object, a file or directory may be utilized (e.g., IntOptions expects a
                path with a file name and a .pickle extension, whereas GroupedOptions expects a
                directory).
            **kwargs: Keyword arguments that were used to create the grouped options,
            which should exist both before and after saving as well as loading.

        Returns:
            None
        """
        # check that we can save and load it
        loaded_grouped_options = self.check_save_and_load(
            grouped_options, path=path)
        self.assertIsInstance(loaded_grouped_options, GroupedOptions)
        for attr_name in kwargs:
            orig_attr_value = self.check_attr_exists_and_options_are_not_yet_assigned(
                grouped_options, attr_name)
            loaded_attr_value = self.check_attr_exists_and_options_are_not_yet_assigned(
                loaded_grouped_options, attr_name)
            self.assertEqual(type(orig_attr_value), type(loaded_attr_value))

    def check_attr_exists_and_options_are_not_yet_assigned(
        self, grouped_options: GroupedOptions, attr_name: str
    ) -> Any:
        """
        A helper method that may be called recursively to examine attributes exist and that they
        have not yet been assigned a particular selected value.

        Args:
            grouped_options: The grouped options.
            attr_name: The attribute to check.

        Returns:
            The option object for that attribute; it is guaranteed to not have been assigned a
            value yet.
        """
        # check that the keys of kwargs are now attributes in the
        # GroupedOptions object
        self.assertTrue(hasattr(grouped_options, attr_name))
        # check that no assignment has been made for each individual Option
        attr_value = getattr(grouped_options, attr_name)
        if isinstance(attr_value, GroupedOptions):
            for inner_attr_name in vars(attr_value):
                self.check_attr_exists_and_options_are_not_yet_assigned(
                    grouped_options=attr_value, attr_name=inner_attr_name
                )
        elif isinstance(attr_value, Options):
            if attr_value.assignable:
                self.assertRaises(ValueError, lambda: attr_value.selection)
            else:
                self.assertIsNotNone(attr_value.selection)
        return attr_value

    def check_common_functionality(
        self, cls: Type[Options], args: tuple[Any, ...], selection: Any, path: Path
    ) -> Tuple[Options, Options]:
        """
        A helper method that checks saving and loading of Options instances works as expected.
        Additionally, it checks that a value can be selected from the options; lastly,
        it will then ensure it can save and load this Options instance with the selected value.

        Args:
            cls: The class to check, which must inherit from Options.
            args: The arguments to use to create the Options instance.
            selection: The value that should be selected for testing.
            path: The path to use for saving and loading. Depending on the complexity of the
                Options object, a file or directory may be utilized (e.g., IntOptions expects a
                path with a file name and a .pickle extension, whereas GroupedOptions expects a
                directory).

        Returns:
            The instance of options before assignment, and the same instance after assignment.
        """
        options_object = cls(*args)
        # check that it raises an error if you try to access a selection before
        # assignment
        self.assertRaises(ValueError, lambda: options_object.selection)
        # check that we can save it
        loaded_options_object_before_assignment = self.check_save_and_load(
            options_object, path=path
        )
        # check that we can assign to the original
        options_object.selection = selection
        self.assertEqual(selection, options_object.selection)
        # check that it has not changed the absent selection in the loaded copy
        self.assertRaises(
            ValueError,
            lambda: loaded_options_object_before_assignment.selection)
        # check that we can save it with the assigned selection
        loaded_categorical_options = self.check_save_and_load(
            options_object, path=path)
        self.assertEqual(options_object, loaded_categorical_options)
        return loaded_options_object_before_assignment, options_object

    def check_save_and_load(
            self,
            options_object: Options,
            path: Path) -> Options:
        """
        A generic function that checks the class that inherits and implements from Options can
        save and load.

        Args:
            options_object: The object which inherits and implements from Options.
            path: The path to use for saving and loading. Depending on the complexity of the
                Options object, a file or directory may be utilized (e.g., IntOptions expects a
                path with a file name and a .pickle extension, whereas GroupedOptions expects a
                directory).

        Returns:
            The Options object that was recovered from the saved file(s).
        """
        options_object.save(path=path)
        # check that we can load it
        loaded_options_object = type(options_object).load(path)
        self.assertEqual(options_object, loaded_options_object)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)  # clean up; delete the file
        return loaded_options_object
