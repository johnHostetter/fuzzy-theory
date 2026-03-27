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

    @staticmethod
    def load(path: Path, cls=None) -> "CategoricalEnumOptions":
        """
        The function to load a CategoricalEnumOptions object. The 'cls' argument is ignored but
        kept for consistency with the static load method from 'CategoricalOptions'.
        Args:
            path: The path where the CategoricalEnumOptions object is located.
            cls: An ignored argument; kept for interface consistency.

        Returns:
            An instance of CategoricalEnumOptions.
        """
        loaded_object = super().load(path=path, cls=CategoricalEnumOptions)
        if isinstance(loaded_object, CategoricalEnumOptions):
            return loaded_object
        raise ValueError(f"Failed to load from: {path}")


class ExtendedEnum(
    CategoricalOptions, str, Enum, metaclass=ExtendedEnumMeta
):  # incl. 'str' so == will work between string value and Enum member
    __hash__ = str.__hash__

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
