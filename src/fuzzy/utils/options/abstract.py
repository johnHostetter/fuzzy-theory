import collections
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


class IterableOptions(collections.Iterator, Options, ABC):
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


class CombineMeta:
    def __prepare__(self, name, bases, **kwargs):
        namespace = {}
        for metaclass in self._get_most_derived_metaclasses(bases):
            ns = metaclass.__prepare__(name, bases, **kwargs)
            if type(ns) in (dict, type(namespace)):
                namespace.update(ns)
            else:
                if type(namespace) is not dict:
                    raise TypeError(
                        "metaclass conflict: " "multiple custom namespaces defined."
                    )
                ns.update(namespace)
                namespace = ns
        return namespace

    def __call__(self, name, bases, namespace, **kwargs):
        metaclasses = self._get_most_derived_metaclasses(bases)
        if len(metaclasses) > 1:
            merged_name = "__".join(meta.__name__ for meta in metaclasses)
            ns = self.__prepare__(merged_name, metaclasses)
            metaclass = self(merged_name, tuple(metaclasses), ns, **kwargs)
        else:
            (metaclass,) = metaclasses or (type,)
        return metaclass(name, bases, namespace, **kwargs)

    @staticmethod
    def _get_most_derived_metaclasses(bases):
        metaclasses = []
        for metaclass in map(type, bases):
            if metaclass is not type:
                metaclasses = [
                    other for other in metaclasses if not issubclass(metaclass, other)
                ]
                if not any(issubclass(other, metaclass) for other in metaclasses):
                    metaclasses.append(metaclass)
        return metaclasses


class EnumPromoter:
    def __init_subclass__(cls, enum_cls=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if enum_cls:
            original_mapping_proxy = dict(vars(cls))
            options = []
            for member in enum_cls:
                setattr(cls, member.name, member)
                options.append(member.value)
            cls.options = tuple(options)
            # cls.options: Tuple[Any, ...] = tuple(
            #     enum.value for attr, enum in dict(vars(cls)).items()
            #     if attr not in original_mapping_proxy
            # )


class ExtendedEnumMeta(EnumMeta, ABCMeta):
    pass
    # def __new__(mcls, name, bases, namespace, **kwargs):
    #     cls = super().__new__(mcls, name, bases, namespace)
    #     if name.endswith("Enum"):
    #         wrapper_name = name[:-4]
    #         wrapper_cls = type(
    #             wrapper_name,
    #             (CategoricalOptions, EnumPromoter),
    #             {"enum_cls": cls, "__module__": cls.__module__},
    #         )
    #         globals()[wrapper_name] = wrapper_cls
    #     return cls
    # __instancecheck__ = type.__instancecheck__
    # __subclasscheck__ = type.__subclasscheck__
    # @classmethod
    # pass
    # def __subclasscheck__(cls, subclass):
    #     try:
    #         if super().__subclasshook__(subclass):
    #             print("default")
    #             return True
    #     except TypeError:
    #         print("new")
    #         return any(base is cls for base in inspect.getmro(subclass))

    # @classmethod
    # def __instancecheck__(cls, instance):
    #     print("test")
    #     # try:
    #     #     if super().__instancecheck__(instance):
    #     #         print("default")
    #     #         return True
    #     # except TypeError:
    #     #     print("new")
    #     return any(base is cls for base in inspect.getmro(type(instance)))
