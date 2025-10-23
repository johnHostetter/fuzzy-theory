import pickle
from pathlib import Path
from typing import Tuple, Union

from fuzzy.utils.options.abstract import (
    IterableOptions,
    Options,
)


class CategoricalOptions(IterableOptions, Options):
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
            if key != "options" and key != "_value":
                setattr(categorical_options, key, value)
        return categorical_options

    def assign(self, trial, name) -> Union[str, int, float]:
        @self.assign_once
        def suggest():
            return trial.suggest_categorical(name, self.options)

        return suggest()


class IntOptions(IterableOptions, Options):
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
            return trial.suggest_int(name, self.start, self.end, step=self.step)

        return suggest()


class FloatOptions(Options):
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
            key: value for key, value in loaded_dict.items() if key in ["start", "end"]
        }
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


class GroupedOptions(Options):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def assign(self, trial, name):
        for attr, options in vars(self).items():
            if (
                hasattr(options, "assign") and options.assignable
            ):  # TODO: this is not being reached
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
            key: value for key, value in loaded_dict.items() if key not in covered_keys
        }
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
