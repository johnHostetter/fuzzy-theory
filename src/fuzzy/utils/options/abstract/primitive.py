"""
This script contains various ready-to-use classes that can handle primitive options, where the
values may only involve either categories, integers, or floats. Additionally, it provides the
capability to group options together to form more comprehensive option selection as well as
better organization. The benefit of using this options interface is that code handling complex
branching in neural architectures can rapidly be developed and accessible to external
hyperparameter optimization via libraries such as optuna.
"""

import pickle
from pathlib import Path
from typing import Tuple, Union

from fuzzy.utils.options.abstract.meta import EnumPromoter, IterableOptions, Options


class IntOptions(IterableOptions):
    """
    An abstract class definition of the concept behind 'options' for integer values. This class
    offers a convenient interface to select from various options and retain this selection in
    perpetuity. In particular, this is a useful class to reliably reference integer options for
    hyperparameter optimization libraries, such as optuna.
    """

    def __init__(self, start: int, end: int, step: int = 1):
        super().__init__(range(start, end, step))
        self.start: int = start
        self.end: int = end  # inclusive (values includes this end value)
        self.step: int = step
        self._value: Union[None, int] = None

    @staticmethod
    def load(path: Path) -> "IntOptions":
        with open(path, "rb") as file:
            loaded_dict = pickle.load(file)
        kwargs = {
            key: value
            for key, value in loaded_dict.items()
            if key in ["start", "end", "step"]
        }
        int_options = IntOptions(**kwargs)
        if loaded_dict["_value"] is not None:
            int_options.selection = loaded_dict["_value"]

        for key, value in loaded_dict.items():
            if key not in ["options", "_value"] and kwargs.keys():
                setattr(int_options, key, value)
        return int_options

    def assign(self, trial, name) -> int:
        @self.assign_once
        def suggest():
            return trial.suggest_int(
                name, self.start, self.end, step=self.step)

        return suggest()


class FloatOptions(Options):
    """
    An abstract class definition of the concept behind 'options' for float values. This class
    offers a convenient interface to select from various options and retain this selection in
    perpetuity. In particular, this is a useful class to reliably reference float options for
    hyperparameter optimization libraries, such as optuna.
    """

    def __init__(self, start: float, end: float):
        super().__init__()
        self._value: Union[None, float] = None
        # inclusive (values includes this start value)
        self.start: float = start
        self.end: float = end  # inclusive (values includes this end value)

    @staticmethod
    def load(path: Path) -> "FloatOptions":
        with open(path, "rb") as file:
            loaded_dict = pickle.load(file)
        kwargs = {
            key: value for key,
            value in loaded_dict.items() if key in [
                "start",
                "end"]}
        float_options = FloatOptions(**kwargs)
        if loaded_dict["_value"] is not None:
            float_options.selection = loaded_dict["_value"]

        for key, value in loaded_dict.items():
            if key not in ["options", "_value"] and kwargs.keys():
                setattr(float_options, key, value)
        return float_options

    def assign(self, trial, name) -> float:
        @self.assign_once
        def suggest():
            return trial.suggest_float(name, self.start, self.end)

        return suggest()


class CategoricalOptions(IterableOptions):
    """
    An abstract class definition of the concept behind 'options' for categorical values. This class
    offers a convenient interface to select from various options and retain this selection in
    perpetuity. In particular, this is a useful class to reliably reference categorical options for
    hyperparameter optimization libraries, such as optuna.
    """

    def __init__(self, *values: Union[str, int, float]):
        super().__init__(values)
        self._value: Union[None, str, int, float] = None

    @staticmethod
    def load(path: Path, cls=None) -> "CategoricalOptions":
        if cls is None:
            cls = CategoricalOptions
        with open(path, "rb") as file:
            loaded_dict = pickle.load(file)
        categorical_options = cls(*loaded_dict["options"])
        if loaded_dict["_value"] is not None:
            categorical_options.selection = loaded_dict["_value"]

        for key, value in loaded_dict.items():
            if key not in {"options", "_value"}:
                setattr(categorical_options, key, value)
        return categorical_options

    def assign(self, trial, name) -> Union[str, int, float]:
        @self.assign_once
        def suggest():
            return trial.suggest_categorical(name, self.options)

        return suggest()


class CategoricalEnumOptions(CategoricalOptions, EnumPromoter):
    """
    A class that exposes a set of categorical options that should also be treated as members of
    the CategoricalOptions object (i.e., similar behavior to an Enum).
    """

    enum_cls = None  # required

    def __init__(self, *args, **kwargs):
        # __init_subclass__ is implicitly a classmethod, so accessing it through the
        # base class (EnumPromoter.__init_subclass__) binds cls=EnumPromoter itself,
        # promoting the enum members onto the wrong class - every subclass would then
        # clobber the same shared EnumPromoter attributes instead of getting its own.
        # Accessing it through type(self) binds cls to the actual concrete subclass.
        type(self).__init_subclass__(enum_cls=self.enum_cls)
        # *args is intentionally NOT forwarded here: this class' options always come
        # from enum_cls (just promoted onto self.options above), never from
        # constructor args - forwarding *args as well duplicated the options tuple
        # whenever load() reconstructed an instance via cls(*loaded_dict["options"])
        # (see CategoricalOptions.load), since that already IS self.options.
        super().__init__(*self.options, **kwargs)

    @classmethod
    def load(cls, path: Path) -> "CategoricalEnumOptions":
        """
        Load a CategoricalEnumOptions object. Must be called on the concrete subclass
        that was originally saved (e.g. MySubclass.load(path)), not on
        CategoricalEnumOptions directly - the saved state has no record of which
        subclass produced it (unlike enum_cls, which only exists as a class
        attribute), so reconstructing the right type of object relies on the caller
        already knowing it and invoking .load() on it.

        Args:
            path: The path where the CategoricalEnumOptions object is located.

        Returns:
            An instance of the concrete CategoricalEnumOptions subclass this was
            called on.
        """
        # previously a @staticmethod that ignored its 'cls' argument and always
        # reconstructed the base CategoricalEnumOptions class itself - which has no
        # enum_cls of its own, so __init__ crashed with AttributeError on
        # self.options. A classmethod naturally receives the concrete subclass that
        # .load() was actually called on, which is what CategoricalOptions.load's own
        # cls= parameter needs to build the right kind of object. Since a classmethod
        # can only ever be invoked as CategoricalEnumOptions.load(...) or
        # SomeSubclass.load(...), cls(*args) is always an instance of
        # CategoricalEnumOptions - no isinstance check is needed here.
        return CategoricalOptions.load(path=path, cls=cls)


class GroupedOptions(Options):
    """
    A class that allows related options to be grouped together for greater organization.
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def assign(self, trial, name):
        for attr, options in vars(self).items():
            if hasattr(options, "assign") and options.assignable:
                options.assign(trial=trial, name=f"{name}.{attr}")

    @staticmethod
    def load(path: Path) -> "GroupedOptions":
        assert (
            path.is_dir()
        ), "The path argument to save GroupedOptions needs to be a directory."
        with open(path / "vars.pickle", "rb") as file:
            loaded_dict = pickle.load(file)
        covered_keys: Tuple[str, str] = ("options", "_value")
        kwargs = {
            key: value for key,
            value in loaded_dict.items() if key not in covered_keys}
        grouped_options = GroupedOptions(**kwargs)
        if loaded_dict["_value"] is not None:
            grouped_options.selection = loaded_dict["_value"]

        for key, value in loaded_dict.items():
            if key not in covered_keys and kwargs.keys():
                setattr(grouped_options, key, value)
        return grouped_options

    def save(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        grouped_options_vars = vars(self)
        for attr_name, attr_value in grouped_options_vars.items():
            if isinstance(attr_value, GroupedOptions):
                attr_value.save(path / f"{attr_name}")
            elif isinstance(attr_value, Options):
                attr_value.save(path / f"{attr_name}.pickle")
        super().save(path / "vars.pickle")
