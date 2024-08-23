"""
Utility functions for fuzzy-theory.
"""

from pathlib import Path
from typing import Set, Any

import torch.nn


def check_path_to_save_torch_module(path: Path) -> None:
    """
    Check if the path to save a PyTorch module has the correct file extension. If it does not,
    raise an error.

    Args:
        path: The path to save the PyTorch module.

    Returns:
        None
    """
    if ".pt" not in path.name and ".pth" not in path.name:
        raise ValueError(
            f"The path to save the fuzzy set must have a file extension of '.pt', "
            f"but got {path.name}"
        )
    if ".pth" in path.name:
        raise ValueError(
            f"The path to save the fuzzy set must have a file extension of '.pt', "
            f"but got {path.name}. Please change the file extension to '.pt' as it is not "
            f"recommended to use '.pth' for PyTorch models, since it conflicts with Python path"
            f"configuration files."
        )


def all_subclasses(cls) -> Set[Any]:
    """
    Get all subclasses of the given class, recursively.

    Returns:
        A set of all subclasses of the given class.
    """
    return {cls}.union(s for c in cls.__subclasses__() for s in all_subclasses(c))


# class FuzzySet:
#     """
#     A fuzzy set is a set that has a degree of membership for each element in the set.
#     """
#
#     def __init__(self, name: str, membership_function=None):
#         """
#         Initialize the fuzzy set.
#
#         Args:
#             name: The name of the fuzzy set.
#             membership_function: The membership function of the fuzzy set.
#
#         Returns:
#             None
#         """
#         self.name = name
#         self.membership_function = membership_function
#
#     def __call__(self, *args, **kwargs):
#         """
#         Call the fuzzy set.
#
#         Returns:
#             The membership function of the fuzzy set.
#         """
#         return self.membership_function(*args, **kwargs)
#
#     def save(self, path: Path) -> None:
#         """
#         Save the fuzzy set.
#
#         Args:
#         """
#         check_path_to_save_torch_module(path)
#         # torch.save(self, path)


class TorchJitModule(torch.nn.Module):
    @classmethod
    @torch.jit.ignore
    def get_subclass(cls, class_name: str) -> "TorchJitModule":
        """
        Get the subclass of TorchJitModule with the given class name.

        Args:
            class_name: The name of the subclass to find.

        Returns:
            A subclass implementation of TorchJitModule with the given class name.
        """
        fuzzy_set_class = None
        for subclass in all_subclasses(cls):
            if subclass.__name__ == class_name:
                fuzzy_set_class = subclass
                break
        if fuzzy_set_class is None:
            raise ValueError(
                f"The class {class_name} was not found in the subclasses of "
                f"{cls}. Please ensure that {class_name} is a subclass of {cls}."
            )
        return fuzzy_set_class
