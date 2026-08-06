"""
Test and validate the primitive options and Enum implementations are behaving as intended.
"""

import os
import shutil
import unittest
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple, Type
from unittest import mock

import scipy.stats
import torch

from fuzzy.utils.options.abstract.meta import EnumPromoter, IterableOptions, Options
from fuzzy.utils.options.abstract.primitive import (
    CategoricalEnumOptions,
    CategoricalOptions,
    FloatOptions,
    GroupedOptions,
    IntOptions,
)
from fuzzy.utils.options.impl.impl_enums import (
    BoundAlphaEntmaxEnum,
    NeurogenesisEnum,
    PremiseActivationEnum,
    PremiseAggregationEnum,
    PremiseEliminationEnum,
    RuleElevationEnum,
    RuleEliminationEnum,
    RuleWeightsEnum,
    SamplingEnum,
)
from fuzzy.utils.options.impl.impl_options import (
    _64_BIT_INT,
    BoundAlphaEntmax,
    EvolutionConfig,
    GumbelConfig,
    NeuroFuzzyNetworkHyperparameters,
    PremiseActivation,
    PremiseAggregation,
    PremiseConfig,
    Range,
    RuleConfig,
)


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

    # type hints for Pylint - not required for functionality but only static
    # code analysis (see PremiseAggregation in impl_options.py); A/B are
    # actually promoted at runtime by EnumPromoter.__init_subclass__
    A: DemoEnum
    B: DemoEnum


class _OptionsTestHelpers(unittest.TestCase):
    """
    Shared, non-test helper methods for exercising Options subclasses' save/load and
    selection-assignment behavior. Defines no test_* methods itself (and its
    underscore-prefixed name keeps pytest/unittest from collecting it directly), so
    it only ever runs as a base class - its methods count as inherited (not
    contributing to any subclass' own public-method count) rather than being
    duplicated across the TestCase classes below that need them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir = Path(os.path.dirname(os.path.abspath(__file__)))

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


class _GroupedOptionsTestHelpers(_OptionsTestHelpers):
    """
    Shared, non-test helper methods for exercising GroupedOptions specifically - see
    _OptionsTestHelpers' docstring for why this is a base class rather than
    duplicated code.
    """

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


class TestPrimitiveOptions(_OptionsTestHelpers):
    """
    Test CategoricalOptions, IntOptions, and FloatOptions, as well as the behavior
    they share via the Options/IterableOptions base classes (equality, iteration,
    optuna Trial delegation via assign()).
    """

    def test_categorical_options(
            self) -> Tuple[CategoricalOptions, CategoricalOptions]:
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

    def test_options_eq_with_non_options_object(self) -> None:
        """
        Coverage/regression test: Options.__eq__()'s "not comparable" branch (an
        Options instance compared against something that is not an Options at all)
        had no test coverage - every existing equality check compares two Options
        instances.

        Returns:
            None
        """
        self.assertFalse(IntOptions(0, 4, 1) == 42)
        self.assertFalse(IntOptions(0, 4, 1) == "not an Options instance")

    def test_iterable_options_iteration_protocol(self) -> None:
        """
        Coverage/regression test: IterableOptions' __iter__/__next__ (the manual
        iterator protocol - as opposed to just reading .options directly, which is
        all any existing test does) had no test coverage at all, including the
        StopIteration/index-reset behavior at the end of a pass.

        Returns:
            None
        """
        int_options = IntOptions(0, 4, 1)
        self.assertIsInstance(int_options, IterableOptions)
        # __iter__ returns an iterator over .options directly (not self)
        self.assertEqual([0, 1, 2, 3], list(iter(int_options)))

        # __next__ (self is its own iterator, driven by self._idx) must yield every
        # option in order, then raise StopIteration and reset back to the start
        self.assertEqual(0, next(int_options))
        self.assertEqual(1, next(int_options))
        self.assertEqual(2, next(int_options))
        self.assertEqual(3, next(int_options))
        with self.assertRaises(StopIteration):
            next(int_options)
        # the reset means a fresh pass works identically afterward
        self.assertEqual(0, next(int_options))

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

    def test_assign_delegates_to_matching_trial_method(self) -> None:
        """
        Coverage/regression test: assign() had no test coverage at all, for any of
        IntOptions/FloatOptions/CategoricalOptions/GroupedOptions - confirm each
        delegates to the matching optuna Trial method with this instance's own
        options/bounds, stores the result as .selection, and (via assign_once)
        refuses a second assignment.

        Returns:
            None
        """
        trial = mock.Mock()
        trial.suggest_int.return_value = 2
        trial.suggest_float.return_value = 0.5
        trial.suggest_categorical.return_value = "B"

        int_options = IntOptions(0, 4, 1)
        self.assertEqual(2, int_options.assign(trial, "int_param"))
        trial.suggest_int.assert_called_once_with("int_param", 0, 4, step=1)
        self.assertEqual(2, int_options.selection)
        with self.assertRaises(ValueError):
            int_options.assign(trial, "int_param")

        float_options = FloatOptions(0.1, 0.9)
        self.assertEqual(0.5, float_options.assign(trial, "float_param"))
        trial.suggest_float.assert_called_once_with("float_param", 0.1, 0.9)
        self.assertEqual(0.5, float_options.selection)

        categorical_options = CategoricalOptions("A", "B", "C")
        self.assertEqual(
            "B", categorical_options.assign(
                trial, "categorical_param"))
        trial.suggest_categorical.assert_called_once_with(
            "categorical_param", ("A", "B", "C")
        )
        self.assertEqual("B", categorical_options.selection)


class TestGroupedOptions(_GroupedOptionsTestHelpers):
    """
    Test GroupedOptions: composing CategoricalOptions/IntOptions/FloatOptions (and
    other GroupedOptions) together, delegating assign() to assignable members, and
    save/load behavior (including nested groups and a group's own selection).
    """

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
            self.check_common_functionality(
                CategoricalOptions,
                ("A", "B", "C"),
                "B",
                path=self.dir / "categorical_options.pickle",
            )
        )
        int_options_before_assignment, int_options_after_assignment = (
            self.check_common_functionality(
                IntOptions, (0, 4, 1), selection=2, path=self.dir / "int_options.pickle"))
        float_options_before_assignment, float_options_after_assignment = (
            self.check_common_functionality(
                FloatOptions,
                (0.1, 0.9),
                selection=0.75,
                path=self.dir / "float_options.pickle",
            )
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

    def test_grouped_options_assign_delegates_to_each_assignable_member(
        self,
    ) -> None:
        """
        Coverage/regression test: GroupedOptions.assign() had no test coverage -
        confirm it calls assign() on every member that is itself still-assignable
        Options (skipping members that already have a selection), namespacing each
        member's trial parameter name as "{group_name}.{member_name}".

        Returns:
            None
        """
        trial = mock.Mock()
        trial.suggest_int.return_value = 2
        trial.suggest_float.return_value = 0.5

        already_assigned = FloatOptions(0.1, 0.9)
        already_assigned.selection = 0.42

        grouped_options = GroupedOptions(
            int_options=IntOptions(0, 4, 1),
            float_options=already_assigned,
        )
        grouped_options.assign(trial=trial, name="group")

        trial.suggest_int.assert_called_once_with(
            "group.int_options", 0, 4, step=1)
        # already_assigned must be left untouched - assign() only touches members
        # that are still assignable
        trial.suggest_float.assert_not_called()
        self.assertEqual(0.42, already_assigned.selection)

    def test_grouped_options_save_creates_missing_directory(self) -> None:
        """
        Coverage/regression test: GroupedOptions.save()'s path.mkdir() branch was
        never exercised - every existing save/load test pre-creates the directory
        itself before calling save().

        Returns:
            None
        """
        path = self.dir / "grouped_options_fresh_directory"
        shutil.rmtree(path, ignore_errors=True)
        self.assertFalse(path.exists())
        try:
            GroupedOptions(int_options=IntOptions(0, 4, 1)).save(path)
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_grouped_options_save_and_load_nested_group(self) -> None:
        """
        Coverage/regression test: GroupedOptions.save()'s recursive branch for a
        member that is itself a GroupedOptions (as opposed to a plain Options leaf)
        was never exercised - every existing test only nests plain Int/Float/
        Categorical options directly.

        Returns:
            None
        """
        path = self.dir / "grouped_options_nested"
        shutil.rmtree(path, ignore_errors=True)
        try:
            inner = GroupedOptions(int_options=IntOptions(0, 4, 1))
            getattr(inner, "int_options").selection = 2
            outer = GroupedOptions(inner_group=inner)

            outer.save(path)
            self.assertTrue((path / "inner_group").is_dir())

            loaded = GroupedOptions.load(path)
            loaded_inner = getattr(loaded, "inner_group")
            self.assertEqual(2, getattr(loaded_inner, "int_options").selection)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_grouped_options_save_and_load_with_its_own_selection(
            self) -> None:
        """
        Coverage/regression test: GroupedOptions.load()'s "restore this group's own
        .selection" branch (as opposed to its members') was never exercised - a
        GroupedOptions instance can have its own selection set directly, distinct
        from any of its member options.

        Returns:
            None
        """
        path = self.dir / "grouped_options_with_own_selection"
        path.mkdir(parents=True, exist_ok=True)
        try:
            grouped_options = GroupedOptions(int_options=IntOptions(0, 4, 1))
            grouped_options.selection = "chosen"
            grouped_options.save(path)

            loaded = GroupedOptions.load(path)
            self.assertEqual("chosen", loaded.selection)
        finally:
            shutil.rmtree(path, ignore_errors=True)


class TestConfigDefaults(unittest.TestCase):
    """
    Test default values for PremiseConfig, RuleConfig, and
    NeuroFuzzyNetworkHyperparameters, plus regressions specific to the latter's
    dataclass field handling.
    """

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

    def test_fields_excludes_base_class_fields(self) -> None:
        """
        Regression guard: NeuroFuzzyNetworkHyperparameters.fields used to pass
        type(ApproximatorHyperparameters) (the metaclass, i.e. `type`) to
        dataclasses.fields() instead of the class itself, which always raised
        TypeError - so `.fields` could never be accessed at all.

        Returns:
            None
        """
        nfn = NeuroFuzzyNetworkHyperparameters()
        field_names = {hyperparameter.name for hyperparameter in nfn.fields}
        # unique to the subclass
        self.assertIn("structure", field_names)
        self.assertIn("evolution", field_names)
        # inherited from ApproximatorHyperparameters, so excluded
        self.assertNotIn("display_name", field_names)
        self.assertNotIn("abbrev_name", field_names)


class TestRange(unittest.TestCase):
    """
    Test the Range dataclass: its scipy/optuna search-space conversions, containment
    check, (de)serialization round trip, and __repr__.
    """

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

    def test_range_to_scipy_log_and_uniform_branches(self) -> None:
        """
        Coverage/regression test: to_scipy()'s log-scale and plain-uniform branches
        (as opposed to the step/discrete branch, already covered above) were never
        exercised.

        Returns:
            None
        """
        log_range = Range(low=1, high=100, log=True)
        log_distribution = log_range.to_scipy()
        self.assertIsInstance(
            log_distribution.dist, type(
                scipy.stats.loguniform))

        plain_range = Range(low=0, high=4)
        plain_distribution = plain_range.to_scipy()
        low_support, high_support = plain_distribution.support()
        self.assertEqual(0, low_support)
        self.assertEqual(4, high_support)

    def test_range_to_optuna_all_branches(self) -> None:
        """
        Coverage/regression test: to_optuna() had no test coverage at all - its three
        branches (step/discrete, log-scale, plain) must call the matching Optuna
        trial method with this Range's bounds.

        Returns:
            None
        """
        trial = mock.Mock()

        Range(low=0, high=4, step=1).to_optuna(trial, "step_param")
        trial.suggest_int.assert_called_once_with("step_param", 0, 4, step=1)

        Range(low=1, high=100, log=True).to_optuna(trial, "log_param")
        trial.suggest_float.assert_called_once_with(
            "log_param", 1, 100, log=True)

        Range(low=0, high=4).to_optuna(trial, "plain_param")
        trial.suggest_float.assert_called_with("plain_param", 0, 4)

    def test_range_contains(self) -> None:
        """
        Coverage/regression test: contains() had no test coverage at all.

        Returns:
            None
        """
        value_range = Range(low=0, high=4)
        self.assertTrue(value_range.contains(0))
        self.assertTrue(value_range.contains(4))
        self.assertTrue(value_range.contains(2))
        self.assertFalse(value_range.contains(-1))
        self.assertFalse(value_range.contains(5))

    def test_range_to_dict_and_from_dict_round_trip(self) -> None:
        """
        Coverage/regression test: to_dict()/from_dict() had no test coverage at all.
        to_dict() must omit log/step when they are at their (falsy) defaults, and
        include them otherwise; from_dict() must reconstruct an equal Range either
        way.

        Returns:
            None
        """
        minimal_range = Range(low=0, high=4)
        minimal_dict = minimal_range.to_dict()
        self.assertEqual({"low": 0, "high": 4}, minimal_dict)
        self.assertEqual(minimal_range, Range.from_dict(minimal_dict))

        full_range = Range(low=1, high=100, log=True, step=1)
        full_dict = full_range.to_dict()
        self.assertEqual(
            {"low": 1, "high": 100, "log": True, "step": 1}, full_dict)
        self.assertEqual(full_range, Range.from_dict(full_dict))

    def test_range_repr(self) -> None:
        """
        Coverage/regression test: __repr__() had no test coverage at all.

        Returns:
            None
        """
        self.assertEqual("Range(low=0, high=4)", repr(Range(low=0, high=4)))
        self.assertEqual(
            "Range(low=1, high=100, log=true, step=2)",
            repr(Range(low=1, high=100, log=True, step=2)),
        )


class TestPremiseActivationFunctions(unittest.TestCase):
    """
    Test the function-selection behavior behind premise aggregation/activation:
    PremiseAggregation's own construction, PremiseActivation.func()'s ENTMAX_BISECT
    branch, and the BoundAlphaEntmax module that branch builds.
    """

    def test_premise_aggregation_construction(self) -> None:
        """
        Coverage/regression test: PremiseAggregation's own __init__ (as opposed to
        its .func() classmethod, already exercised elsewhere) had no test coverage.

        Returns:
            None
        """
        premise_aggregation = PremiseAggregation()
        self.assertIsNotNone(premise_aggregation)
        sum_fn = PremiseAggregation.func(PremiseAggregationEnum.SUM)
        mean_fn = PremiseAggregation.func(PremiseAggregationEnum.MEAN)
        degrees = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(sum_fn(degrees), -1 * degrees.sum(dim=1)))
        self.assertTrue(torch.equal(
            mean_fn(degrees), -1 * degrees.mean(dim=1)))

    def test_bound_alpha_entmax(self) -> None:
        """
        Coverage/regression test: BoundAlphaEntmax (construction, every bound_alpha()
        strategy, its ValueError for an unrecognized strategy, and forward()) had no
        test coverage at all.

        Returns:
            None
        """
        alpha = torch.zeros(1)
        for strategy in (
            BoundAlphaEntmaxEnum.SIGMOID,
            BoundAlphaEntmaxEnum.TANH,
            BoundAlphaEntmaxEnum.HARD_TANH,
            BoundAlphaEntmaxEnum.SOFTPLUS,
        ):
            module = BoundAlphaEntmax(
                bounding_strategy=strategy,
                alpha=alpha.clone())
            bounded = module.bound_alpha()
            # every strategy must keep alpha strictly within (1, 2), per the
            # class' documented contract
            self.assertTrue(bool((bounded > 1.0).all()))
            self.assertTrue(bool((bounded < 2.0).all()))

            tensor = torch.rand(3, 4)
            output = module(tensor)
            self.assertEqual(output.shape, tensor.shape)

        # an unrecognized bounding strategy must raise, not silently return
        # None
        module = BoundAlphaEntmax(
            bounding_strategy=BoundAlphaEntmaxEnum.SIGMOID, alpha=alpha.clone()
        )
        module.bounding_strategy = "not_a_real_strategy"
        with self.assertRaises(ValueError):
            module.bound_alpha()

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires a second device (CUDA) to move to"
    )
    def test_bound_alpha_entmax_forward_moves_alpha_to_input_device(
            self) -> None:
        """
        Coverage/regression test: forward() must move its bounded alpha onto the
        input tensor's device when they differ, rather than letting entmax_bisect
        fail on a device mismatch.

        Returns:
            None
        """
        module = BoundAlphaEntmax(
            bounding_strategy=BoundAlphaEntmaxEnum.SIGMOID,
            device=torch.device("cpu"),
        )
        tensor = torch.rand(3, 4, device=torch.device("cuda"))
        output = module(tensor)
        self.assertEqual("cuda", output.device.type)

    def test_premise_activation_func_entmax_bisect(self) -> None:
        """
        Coverage/regression test: PremiseActivation.func()'s ENTMAX_BISECT branch
        (which builds and returns a BoundAlphaEntmax instance, rather than looking up
        a plain function like the other transforms) had no test coverage, nor did its
        assertion that a bounding strategy must be given for it.

        Returns:
            None
        """
        result = PremiseActivation.func(
            transform=PremiseActivationEnum.ENTMAX_BISECT,
            bound=BoundAlphaEntmaxEnum.SIGMOID,
        )
        self.assertIsInstance(result, BoundAlphaEntmax)

        with self.assertRaises(AssertionError):
            PremiseActivation.func(
                transform=PremiseActivationEnum.ENTMAX_BISECT, bound=None
            )


class TestCategoricalEnumOptions(unittest.TestCase):
    """
    Test CategoricalEnumOptions: enum-member promotion onto the concrete subclass
    (not the shared base), and save/load.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir = Path(os.path.dirname(os.path.abspath(__file__)))

    def test_categorical_enum_options_promotes_onto_concrete_subclass(
            self) -> None:
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

    def test_categorical_enum_options_save_and_load(self) -> None:
        """
        Regression guard: CategoricalEnumOptions.load used to be a @staticmethod that
        called zero-arg super() - which, having no enclosing self/cls to bind to
        inside a staticmethod, fell back to treating its first positional argument
        (path, a Path) as the instance, always raising TypeError. It has since become
        a classmethod, called on the concrete subclass being loaded, so this checks
        a real save/load round trip works and reproduces the original instance.

        Returns:
            None
        """
        path = self.dir / "categorical_enum_options.pickle"
        options = DemoCategoricalEnumOptions()
        options.selection = DemoEnum.A.value
        options.save(path)

        loaded = DemoCategoricalEnumOptions.load(path)

        self.assertIsInstance(loaded, DemoCategoricalEnumOptions)
        self.assertEqual(options.options, loaded.options)
        self.assertEqual(options.selection, loaded.selection)
        self.assertEqual(options, loaded)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
