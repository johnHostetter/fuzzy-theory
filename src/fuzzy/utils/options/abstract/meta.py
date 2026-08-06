"""
Contained within here are abstract classes and mixins to provide a common, reliable interface to
options that may or may not involve enum members.
"""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from optuna import Trial


class Options(ABC):
    """
    An abstract class definition of the concept behind 'options'. This class offers a convenient
    interface to select from various options and retain this selection in perpetuity. In particular,
    this is a useful class for reliably referencing options for branching, and provides
    compatibility with hyperparameter optimization libraries, such as optuna.
    """

    def __init__(self):
        self._value: Any = None

    def __eq__(self, other) -> bool:
        if isinstance(other, Options):
            return vars(self) == vars(other)
        return False

    @property
    def assignable(self) -> bool:
        """
        Determine if this instance that inherited from Options is still assignable (i.e.,
        no selection has been made yet).

        Returns:
            Whether a selection exists
        """
        return self._value is None

    @property
    def selection(self) -> Any:
        """
        The selected option from the available options.

        Returns:
            Any
        """
        if self._value is None:
            raise ValueError(
                "A value has not been assigned to this trial from the available options."
            )
        return self._value

    @selection.setter
    def selection(self, value: Any) -> None:
        @self.assign_once
        def update_selection() -> Any:
            return value

        self._value = update_selection()

    def assign_once(self, func) -> Callable:
        """
        A decorator that restricts the decorated function so that it can only be called once.

        Args:
            func: The function that can only be called once before a ValueError is raised.

        Returns:
            The decorated function.
        """

        def wrapper(*args, **kwargs) -> Any:
            if self._value is not None:
                raise ValueError(
                    "A value has already been assigned to this trial from the available options."
                )

            self._value = func(*args, **kwargs)
            return self._value

        return wrapper

    def save(self, path: Path) -> None:
        """
        Given a path, save the class that inherited from Options as a pickle file.

        Args:
            path: An instance of Path that describes a file path which ends with ".pickle".

        Returns:
            None
        """
        assert path.name.endswith(
            ".pickle"), 'File path should end with ".pickle"'
        with open(path, "wb") as file:
            pickle.dump(vars(self), file)

    @staticmethod
    @abstractmethod
    def load(path: Path) -> "Options":
        """
        An abstract method that outlines the signature for how a class which inherits from
        Options should load and be instantiated.

        Args:
            path: An instance of Path that describes a file path which ends with ".pickle".

        Returns:
            An instance of Options.
        """

    @abstractmethod
    def assign(self, trial: Trial, name: str) -> Any:
        """
        Assign a selection from the available options for this given trial.

        Args:
            trial: An instance of a Trial.
            name: The name that this selection should be assigned to.

        Returns:
            Any
        """


class IterableOptions(Iterator, Options, ABC):
    """
    An abstract class definition of the concept behind 'iterable options'. This class offers a
    convenient interface to select from various options, retain this selection in perpetuity,
    as well as iterate over those options in a consistent manner.
    """

    def __init__(self, values):
        super().__init__()
        self._value: Any = None
        self.options: Any = values
        self._idx: int = 0

    def __iter__(self):
        return iter(self.options)

    def __next__(self):
        self._idx += 1
        try:
            return self.options[self._idx - 1]
        except IndexError as exc:
            self._idx = 0
            raise StopIteration from exc


class EnumPromoter:  # pylint: disable=too-few-public-methods
    """
    This is a mixin that will promote members of a provided Enum (i.e., the enum_cls) onto the
    subclass so that they are accessible as class-level attributes. It happens once at class
    definition time, but not at instance creation, so it directly modifies the class itself.
    """

    def __init_subclass__(cls, enum_cls=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if enum_cls:
            options = []
            for member in enum_cls:
                setattr(cls, member.name, member)
                options.append(member.value)
            cls.options = tuple(options)
