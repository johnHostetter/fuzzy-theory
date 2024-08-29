"""
Utility functions for fuzzy-theory.
"""

from pathlib import Path
from typing import Set, Any


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
