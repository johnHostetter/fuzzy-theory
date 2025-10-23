from enum import Enum
from pathlib import Path
from typing import Any

from fuzzy.utils.options.abstract.meta import EnumPromoter, ExtendedEnumMeta
from fuzzy.utils.options.abstract.primitive import CategoricalOptions


class CategoricalEnumOptions(CategoricalOptions, EnumPromoter):
    enum_cls = None  # required

    def __init__(self, *args, **kwargs):
        EnumPromoter.__init_subclass__(enum_cls=self.enum_cls)
        super().__init__(*self.options, *args, **kwargs)

    @classmethod
    def load(cls, path: Path):
        loaded_object = super().load(path=path, cls=cls)
        if isinstance(loaded_object, cls):
            return loaded_object
        raise ValueError(f"Failed to load from: {path}")


class ExtendedEnum(
    CategoricalOptions, str, Enum, metaclass=ExtendedEnumMeta
):  # incl. 'str' so == will work between string value and Enum member

    def __init__(self, _):
        super().__init__()
        self._assigned = {}

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return Enum.__eq__(self, other)

    @classmethod
    def options(cls) -> tuple[Any, ...]:
        return tuple(enum.value for enum in cls)
