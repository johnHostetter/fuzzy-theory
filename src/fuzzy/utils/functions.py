"""
Utility functions for fuzzy-theory.
"""

import inspect
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Set

import torch


@torch.jit.script
def exp_sum_log(x: torch.Tensor, dim: int, eps: float = 1e-12):
    return torch.exp(torch.sum(torch.log(torch.clamp_min(x, eps)), dim=dim))


def log_method(method):
    """
    Log the call and completion of an object's method.

    Args:
        method: The method to be logged.

    Returns:
        The wrapped method.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs)
        called_method: str = f"{self.__class__.__name__}.{method.__name__}"
        self.logger.debug(f"<{called_method}>")
        start_time = time.perf_counter()
        result = method(self, *args, **kwargs)
        end_time = time.perf_counter()
        self.logger.debug(f"<perf_counter>{end_time - start_time}</perf_counter>")
        self.logger.debug(f"</{called_method}>")
        return result

    return wrapper


def log_classmethod(classmethod):
    """
    Log the call and completion of a class method using the root logger.

    Args:
        classmethod: The class method to be logged.

    Returns:
        The wrapped class method.
    """

    @wraps(classmethod)
    def wrapper(cls, *args, **kwargs):
        return classmethod(cls, *args, **kwargs)
        called_method: str = f"{cls.__name__}.{classmethod.__name__}"
        logging.debug(f"<{called_method}>")
        start_time = time.perf_counter()
        result = classmethod(cls, *args, **kwargs)
        end_time = time.perf_counter()
        logging.debug(f"<perf_counter>{end_time - start_time}</perf_counter>")
        logging.debug(f"</{called_method}>")
        return result

    return wrapper


def log_func(func):
    """
    Log the call and completion of a function using the root logger.

    Args:
        func: The function to be logged.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
        called_func: str = f"{func.__name__}"
        logging.debug(f"<{called_func}>")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.debug(f"</{called_func}>")
        logging.debug(f"<perf_counter>{end_time - start_time}</perf_counter>")
        return result

    return wrapper


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
    # get the attributes that are local to the class, but may be inherited
    # from the super class
    local_attributes = inspect.getmembers(
        obj_instance,
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are inherited from (or found within) the super
    # class
    super_attributes = inspect.getmembers(
        obj_instance.__class__.__bases__[0],
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are local to the class, but not inherited from
    # the super class
    return {
        attr: value
        for attr, value in local_attributes
        if (attr, value) not in super_attributes and not attr.startswith("_")
    }
