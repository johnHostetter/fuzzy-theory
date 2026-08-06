"""
Unit tests for fuzzy.utils (both classes.py and functions.py).
"""

import logging
import unittest
from pathlib import Path

import torch

from fuzzy.utils.classes import Loggable
from fuzzy.utils.functions import (
    all_subclasses,
    check_path_to_save_torch_module,
    exp_sum_log,
    get_object_attributes,
    load_module_class,
    log_classmethod,
    log_func,
    log_method,
    module_class,
)
from tests import AVAILABLE_DEVICE


class TestCheckPathToSaveTorchModule(unittest.TestCase):
    """
    Test check_path_to_save_torch_module.
    """

    def test_pt_extension_is_accepted(self) -> None:
        """
        Returns:
            None
        """
        check_path_to_save_torch_module(Path("model.pt"))  # must not raise

    def test_missing_extension_raises(self) -> None:
        """
        Returns:
            None
        """
        with self.assertRaises(ValueError):
            check_path_to_save_torch_module(Path("model"))

    def test_pth_extension_raises_with_dedicated_message(self) -> None:
        """
        Returns:
            None
        """
        with self.assertRaises(ValueError):
            check_path_to_save_torch_module(Path("model.pth"))

    def test_misleading_suffix_is_rejected(self) -> None:
        """
        Regression test: the extension check used to test for '.pt' as a substring of
        the whole file name, so a misleading name like 'model.pt.bak' (whose actual
        suffix is '.bak') incorrectly passed validation. Path.suffix must be checked
        instead.

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            check_path_to_save_torch_module(Path("model.pt.bak"))


class TestAllSubclasses(unittest.TestCase):
    """
    Test all_subclasses.
    """

    def test_all_subclasses_includes_nested_descendants(self) -> None:
        """
        Returns:
            None
        """

        class Base:  # pylint: disable=too-few-public-methods
            """Base class for the test."""

        class Child(Base):  # pylint: disable=too-few-public-methods
            """Direct child of Base."""

        class Grandchild(Child):  # pylint: disable=too-few-public-methods
            """Indirect descendant of Base."""

        self.assertEqual(all_subclasses(Base), {Base, Child, Grandchild})


class TestGetObjectAttributes(unittest.TestCase):
    """
    Test get_object_attributes.
    """

    def test_purely_inherited_attribute_from_second_base_is_excluded(self) -> None:
        """
        Regression test: get_object_attributes used to only inspect
        obj_instance.__class__.__bases__[0] to determine which attributes are
        "inherited" (and therefore excluded), so in a multiple-inheritance scenario an
        attribute purely inherited from a base other than the first was incorrectly
        reported as "local" to the class. The full MRO must be checked instead.

        Returns:
            None
        """

        class FirstBase:  # pylint: disable=too-few-public-methods
            """First base class, contributes 'first_attr'."""

            first_attr = "from_first_base"

        class SecondBase:  # pylint: disable=too-few-public-methods
            """Second base class, contributes 'second_attr'."""

            second_attr = "from_second_base"

        # pylint: disable-next=too-few-public-methods
        class Combined(FirstBase, SecondBase):
            """Combines both bases and adds a genuinely local attribute."""

            def __init__(self):
                self.local_attr = "genuinely_local"

        instance = Combined()
        attributes = get_object_attributes(instance)

        self.assertIn("local_attr", attributes)
        self.assertNotIn("first_attr", attributes)
        self.assertNotIn(
            "second_attr", attributes
        )  # would previously have been (incorrectly) included

    def test_underscore_prefixed_attributes_are_excluded(self) -> None:
        """
        Returns:
            None
        """

        class WithPrivateAttr:  # pylint: disable=too-few-public-methods
            """A plain class with a private attribute."""

            def __init__(self):
                self._private = "hidden"
                self.public = "visible"

        attributes = get_object_attributes(WithPrivateAttr())
        self.assertIn("public", attributes)
        self.assertNotIn("_private", attributes)


class TestModuleClass(unittest.TestCase):
    """
    Test module_class and load_module_class.
    """

    def test_module_class_and_load_module_class_round_trip(self) -> None:
        """
        Returns:
            None
        """
        instance = torch.nn.Linear(1, 1)
        path = module_class(instance)
        self.assertEqual(path, "torch.nn.modules.linear.Linear")
        loaded_class = load_module_class(path)
        self.assertIs(loaded_class, torch.nn.Linear)


class TestExpSumLog(unittest.TestCase):
    """
    Test exp_sum_log.
    """

    def test_exp_sum_log_matches_prod(self) -> None:
        """
        Returns:
            None
        """
        x = torch.tensor([[0.5, 0.25, 0.8], [0.1, 0.9, 0.2]], device=AVAILABLE_DEVICE)
        actual = exp_sum_log(x, dim=-1)
        expected = torch.prod(x, dim=-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_exp_sum_log_handles_zeros_without_underflow(self) -> None:
        """
        Returns:
            None
        """
        x = torch.tensor([[0.0, 0.5]], device=AVAILABLE_DEVICE)
        result = exp_sum_log(x, dim=-1)
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())


class TestLoggingDecorators(unittest.TestCase):
    """
    Test log_method, log_classmethod, and log_func.
    """

    def test_log_method_logs_and_preserves_return_value(self) -> None:
        """
        Returns:
            None
        """

        # pylint: disable-next=too-few-public-methods
        class Greeter:
            """A class whose method is wrapped with log_method."""

            def __init__(self):
                self.logger = logging.getLogger("test_log_method")

            @log_method
            def greet(self, name: str) -> str:
                """Returns a greeting."""
                return f"hello {name}"

        greeter = Greeter()
        with self.assertLogs("test_log_method", level="DEBUG") as captured:
            result = greeter.greet("world")
        self.assertEqual(result, "hello world")
        joined_output = "\n".join(captured.output)
        self.assertIn("Greeter.greet", joined_output)

    def test_log_classmethod_logs_and_preserves_return_value(self) -> None:
        """
        Returns:
            None
        """

        # pylint: disable-next=too-few-public-methods
        class Factory:
            """A class whose classmethod is wrapped with log_classmethod."""

            @classmethod
            @log_classmethod
            def make(cls) -> str:
                """Returns a fixed value."""
                return "made"

        with self.assertLogs(level="DEBUG") as captured:
            result = Factory.make()
        self.assertEqual(result, "made")
        joined_output = "\n".join(captured.output)
        self.assertIn("Factory.make", joined_output)

    def test_log_func_logs_and_preserves_return_value(self) -> None:
        """
        Returns:
            None
        """

        @log_func
        def add(a: int, b: int) -> int:
            """Adds two numbers."""
            return a + b

        with self.assertLogs(level="DEBUG") as captured:
            result = add(1, 2)
        self.assertEqual(result, 3)
        joined_output = "\n".join(captured.output)
        self.assertIn("add", joined_output)


class TestLoggable(unittest.TestCase):
    """
    Test the Loggable class.
    """

    def test_default_logger_is_created_from_class_name(self) -> None:
        """
        Returns:
            None
        """

        class Widget(Loggable):  # pylint: disable=too-few-public-methods
            """A class using Loggable's default logger creation."""

        widget = Widget()
        self.assertEqual(widget.logger.name, "Widget")
        self.assertEqual(widget.logger.level, logging.INFO)

    def test_debug_flag_sets_debug_level(self) -> None:
        """
        Returns:
            None
        """
        loggable = Loggable(debug=True)
        self.assertEqual(loggable.logger.level, logging.DEBUG)

    def test_explicit_logger_is_used_as_is(self) -> None:
        """
        Returns:
            None
        """
        custom_logger = logging.getLogger("my_custom_logger")
        loggable = Loggable(logger=custom_logger)
        self.assertIs(loggable.logger, custom_logger)

    def test_explicit_name_overrides_class_name(self) -> None:
        """
        Returns:
            None
        """
        loggable = Loggable(name="custom_name")
        self.assertEqual(loggable.logger.name, "custom_name")


if __name__ == "__main__":
    unittest.main()
