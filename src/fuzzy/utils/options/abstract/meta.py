import sys

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
import pickle
from abc import ABC, ABCMeta, abstractmethod
from enum import EnumMeta
from pathlib import Path
from typing import Any, Callable

from optuna import Trial


class Options(ABC):
    def __init__(self):
        self._value: Any = None

    def __eq__(self, other) -> bool:
        if isinstance(other, Options):
            return vars(self) == vars(other)
        return False

    @property
    def assignable(self) -> bool:
        return self._value is None

    @property
    def selection(self) -> Any:
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
        def wrapper(*args, **kwargs) -> Any:
            if self._value is not None:
                raise ValueError(
                    "A value has already been assigned to this trial from the available options."
                )

            self._value = func(*args, **kwargs)
            return self._value

        return wrapper

    def save(self, path: Path):
        assert path.name.endswith(".pickle"), 'File path should end with ".pickle"'
        with open(path, "wb") as file:
            pickle.dump(vars(self), file)

    @staticmethod
    @abstractmethod
    def load(path: Path):
        """

        Args:
            path:

        Returns:

        """

    @abstractmethod
    def assign(self, trial: Trial, name: str) -> Any:
        """

        Args:
            trial:
            name:

        Returns:

        """


class IterableOptions(Iterator, Options, ABC):
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
        except IndexError:
            self._idx = 0
            raise StopIteration


class EnumPromoter:
    def __init_subclass__(cls, enum_cls=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if enum_cls:
            options = []
            for member in enum_cls:
                setattr(cls, member.name, member)
                options.append(member.value)
            cls.options = tuple(options)


class ExtendedEnumMeta(EnumMeta, ABCMeta):
    pass
