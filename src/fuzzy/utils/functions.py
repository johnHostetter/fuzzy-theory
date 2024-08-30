"""
Utility functions for fuzzy-theory.
"""

import inspect
from pathlib import Path
from typing import Set, Any, Dict


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


def get_object_attributes(obj_instance) -> Dict[str, Any]:
    """
    Get the attributes of an object instance.
    """
    # get the attributes that are local to the class, but may be inherited from the super class
    local_attributes = inspect.getmembers(
        obj_instance,
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are inherited from (or found within) the super class
    super_attributes = inspect.getmembers(
        obj_instance.__class__.__bases__[0],
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are local to the class, but not inherited from the super class
    return {
        attr: value
        for attr, value in local_attributes
        if (attr, value) not in super_attributes and not attr.startswith("_")
    }
