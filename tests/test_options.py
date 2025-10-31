import os
import shutil
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

from fuzzy.utils.options.abstract.meta import Options
from fuzzy.utils.options.abstract.primitive import (
    CategoricalOptions,
    FloatOptions,
    GroupedOptions,
    IntOptions,
)
from fuzzy.utils.options.impl.impl_enums import (
    PremiseActivationEnum,
    PremiseAggregationEnum,
    PremiseEliminationEnum,
    RuleElevationEnum,
    RuleEliminationEnum,
    RuleWeightsEnum,
)
from fuzzy.utils.options.impl.impl_options import (
    NeuroFuzzyNetworkHyperparameters,
    PremiseActivation,
    PremiseConfig,
    RuleConfig,
)


class TestOptions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir = Path(os.path.dirname(os.path.abspath(__file__)))

    def test_categorical_options(self) -> tuple[CategoricalOptions, CategoricalOptions]:
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
        args: Tuple[float, float] = (0.1, 0.9)
        before_assignment, after_assignment = self.check_common_functionality(
            FloatOptions, args, selection=0.75, path=self.dir / "float_options.pickle"
        )
        # type checking
        self.assertIsInstance(before_assignment, FloatOptions)
        self.assertIsInstance(after_assignment, FloatOptions)
        attributes: Tuple[str, str] = ("start", "end")
        for idx, attribute in enumerate(attributes):
            self.assertEqual(args[idx], getattr(after_assignment, attribute))
        return before_assignment, after_assignment

    def test_group_options(self) -> None:
        categorical_options_before_assignment, categorical_options_after_assignment = (
            self.test_categorical_options()
        )
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
        premise_activation: PremiseActivation = PremiseActivation()
        self.assertEqual(-1, premise_activation.dim)

    def test_premise_config(self) -> None:
        premise_config: PremiseConfig = PremiseConfig()
        self.assertIsNotNone(premise_config)
        premise_config.default()
        self.assertEqual(
            PremiseAggregationEnum.SUM, premise_config.aggregation.selection
        )
        self.assertEqual(
            PremiseEliminationEnum.NONE, premise_config.elimination.selection
        )
        self.assertEqual(
            PremiseActivationEnum.SOFTMAX, premise_config.activation.selection
        )

    def test_rule_config(self) -> None:
        rule_config: RuleConfig = RuleConfig()
        self.assertIsNotNone(rule_config)
        rule_config.default()
        self.assertEqual(RuleWeightsEnum.NONE, rule_config.weights.selection)
        self.assertEqual(RuleEliminationEnum.NONE, rule_config.elimination.selection)
        self.assertEqual(RuleElevationEnum.NONE, rule_config.elevation.selection)

    def test_create_neuro_fuzzy_network_hyperparameters(self) -> None:
        self.assertIsNotNone(NeuroFuzzyNetworkHyperparameters())

    @staticmethod
    def create_grouped_options_from_kwargs(
        categorical_options: CategoricalOptions,
        int_options: IntOptions,
        float_options: FloatOptions,
    ) -> Tuple[GroupedOptions, Dict[str, Options]]:
        kwargs: Dict[str, Options] = {
            "categorical_options": categorical_options,
            "int_options": int_options,
            "float_options": float_options,
        }
        return GroupedOptions(**kwargs), kwargs

    def check_grouped_options_behavior(
        self, grouped_options: GroupedOptions, path: Path, **kwargs
    ) -> None:
        # check that we can save and load it
        loaded_grouped_options = self.check_save_and_load(grouped_options, path=path)
        self.assertIsInstance(loaded_grouped_options, GroupedOptions)
        for attr_name in kwargs.keys():
            orig_attr_value = self.check_attr_exists_and_options_are_not_yet_assigned(
                grouped_options, attr_name
            )
            loaded_attr_value = self.check_attr_exists_and_options_are_not_yet_assigned(
                loaded_grouped_options, attr_name
            )
            self.assertEqual(type(orig_attr_value), type(loaded_attr_value))

    def check_attr_exists_and_options_are_not_yet_assigned(
        self, grouped_options: GroupedOptions, attr_name: str
    ) -> Any:
        # check that the keys of kwargs are now attributes in the
        # GroupedOptions object
        self.assertTrue(hasattr(grouped_options, attr_name))
        # check that no assignment has been made for each individual Option
        attr_value = getattr(grouped_options, attr_name)
        if isinstance(attr_value, GroupedOptions):
            for inner_attr_name in vars(attr_value).keys():
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
        self, cls: Options, args: tuple[Any, ...], selection: Any, path: Path
    ) -> Tuple[Options, Options]:
        """

        Args:
            cls:
            args:
            selection:
            path:

        Returns:

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
            ValueError, lambda: loaded_options_object_before_assignment.selection
        )
        # check that we can save it with the assigned selection
        loaded_categorical_options = self.check_save_and_load(options_object, path=path)
        self.assertEqual(options_object, loaded_categorical_options)
        return loaded_options_object_before_assignment, options_object

    def check_save_and_load(self, options_object: Options, path: Path) -> Options:
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
