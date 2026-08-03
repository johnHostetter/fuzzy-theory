"""
Utility functions for fuzzy-theory.
"""

import inspect
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Set, Type

import torch


def module_class(instance: object) -> str:
    """
    Given an instance of a class, obtain the name of its class and append it to a string
    representation of its module path.

    Args:
        instance: An instance of a class.

    Returns:
        A string representation of the module path.
    """
    cls_type = type(instance)
    return f"{cls_type.__module__}.{cls_type.__name__}"


def load_module_class(module_path: str) -> Type[object]:
    """
    Given a module path, load the class and return it.

    Args:
        module_path: The module path, which includes the class name; this is usually prepared by
        the function called module_class.

    Returns:
        The loaded class.
    """
    split_file_path = module_path.split(".")
    class_name: str = split_file_path[-1]
    module_path: str = ".".join(split_file_path[:-1])  # drop the Class name
    # https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)


@torch.jit.script
def exp_sum_log(x: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
    """
    A numerically stable product which may offer more time-efficient performance than torch.prod
    while remaining equivalent. Although this is not a simpler operation, it may perform
    better on PyTorch CUDA backend as sum-based reductions are more efficient than
    multiplicative ones.

    Also, if x contains zeros, torch.prod will give zero (which may underflow if chaining
    gradients) so it may be more ideal to use this function instead.

    Overall, this function could perhaps:

    1. Leverage fused add+exp+log kernels (heavily optimized in CUDA)
    2. Exploit tensor core friendly ops (adds and exps are vectorized; multiplications in prod
    are chained and harder to parallelize efficiently)
    3. Benefits from numerical stability with fewer infs/NaNs. Hopefully, fewer slow paths

    Args:
        x: The tensor to operate on.
        dim: The dimension of the given tensor to apply this operation onto.
        eps: A very small numerical offset.

    Returns:
        The product along that dimension of the given tensor.
    """
    # torch.jit.script compiles this body to TorchScript, so it executes outside the
    # CPython interpreter and coverage.py's line tracer never sees it run, regardless
    # of how many tests call this function - see test_exp_sum_log_matches_prod and
    # test_exp_sum_log_handles_zeros_without_underflow in tests/test_utils.py instead
    return torch.exp(torch.sum(torch.log(x.clamp_min(eps)), dim=dim))  # pragma: no cover


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
        called_method: str = f"{self.__class__.__name__}.{method.__name__}"
        self.logger.debug("<%s>", called_method)
        start_time = time.perf_counter()
        result = method(self, *args, **kwargs)
        end_time = time.perf_counter()
        self.logger.debug("<perf_counter>%s</perf_counter>", end_time - start_time)
        self.logger.debug("</%s>", called_method)
        return result

    return wrapper


def log_classmethod(class_method):
    """
    Log the call and completion of a class method using the root logger.

    Args:
        class_method: The class method to be logged.

    Returns:
        The wrapped class method.
    """

    @wraps(class_method)
    def wrapper(cls, *args, **kwargs):
        called_method: str = f"{cls.__name__}.{class_method.__name__}"
        logging.debug("<%s>", called_method)
        start_time = time.perf_counter()
        result = class_method(cls, *args, **kwargs)
        end_time = time.perf_counter()
        logging.debug("<perf_counter>%s</perf_counter>", end_time - start_time)
        logging.debug("</%s>", called_method)
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
        called_func: str = f"{func.__name__}"
        logging.debug("<%s>", called_func)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.debug("<perf_counter>%s</perf_counter>", end_time - start_time)
        logging.debug("</%s>", called_func)
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
    if path.suffix not in (".pt", ".pth"):
        raise ValueError(
            f"The path to save the fuzzy set must have a file extension of '.pt', "
            f"but got {path.name}"
        )
    if path.suffix == ".pth":
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
    # get the attributes that are inherited from (or found within) any of the
    # super classes; using only __bases__[0] would miss attributes purely
    # inherited from other bases in multiple-inheritance scenarios, so the
    # full MRO (excluding the class itself) is checked instead
    super_attributes = [
        attr_pair
        for base in obj_instance.__class__.__mro__[1:]
        for attr_pair in inspect.getmembers(
            base,
            lambda attr: not (inspect.ismethod(attr))
            and not (inspect.isfunction(attr)),
        )
    ]
    # get the attributes that are local to the class, but not inherited from
    # any of the super classes
    return {
        attr: value
        for attr, value in local_attributes
        if (attr, value) not in super_attributes and not attr.startswith("_")
    }
