"""
Utility functions for fuzzy-theory.
"""

import inspect
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Type

import torch

# A signature identifying the tensors a calculation depended upon; see signature_of().
# A plain List rather than a variadic Tuple[..., ...], since the latter's Ellipsis is not a
# type TorchScript's annotation resolver supports, and this is only ever compared with '==' /
# '!=' (never hashed or used as a dict key), so a List loses nothing here. The third element
# of each tuple is the tensor's requires_grad flag at signature time - see
# signature_of().
ParameterSignature = List[Tuple[int, int, bool]]


def version_of(tensor: torch.Tensor) -> int:
    """
    Read the version counter of a tensor, which PyTorch increments whenever the tensor is
    mutated in place (as an optimizer does when it applies an update).

    Inference tensors do not track a version counter at all, so -1 is reported for them; this
    never compares equal to a real version, meaning entries involving inference tensors are
    simply not re-used.

    Args:
        tensor: The tensor to read the version counter of.

    Returns:
        The version counter of the tensor, or -1 if it does not track one.
    """
    try:
        return tensor._version  # pylint: disable=protected-access
    except RuntimeError:
        # "Inference tensors do not track version counter."
        return -1


def signature_of(tensors: List[torch.Tensor]) -> ParameterSignature:
    """
    Summarize the tensors that a calculation depended upon, such that the summary changes if any
    of those tensors is replaced by another object, is mutated in place, or has its
    requires_grad flag toggled.

    The identity of a tensor is safe to use here (rather than a weak reference) because the
    caller - a torch.nn.Module - holds a strong reference to its own parameters for as long as
    the cache entry can be looked up, so an identifier cannot be recycled behind our back.

    requires_grad is included because it changes the autograd graph a calculation produces
    without changing the tensor's identity or bumping its version counter: freezing a
    parameter (requires_grad_(False)), computing with it, then unfreezing it and reusing the
    same observations would otherwise hand back a cached result whose graph was built with
    that parameter detached - so its gradient would silently stay None forever afterward,
    even though it is trainable again.

    Args:
        tensors: The tensors that a calculation depended upon (e.g., centers and widths).

    Returns:
        A hashable and comparable signature of those tensors.
    """
    return [
        (id(tensor), version_of(tensor), tensor.requires_grad) for tensor in tensors
    ]


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
    # test_exp_sum_log_handles_zeros_without_underflow in tests/test_utils.py
    # instead
    return torch.exp(
        torch.sum(torch.log(x.clamp_min(eps)), dim=dim)
    )  # pragma: no cover


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


def _is_read_only_property(cls: type, name: str) -> bool:
    """
    Check whether name is a computed, read-only @property on cls (or any of its
    bases) - i.e. one with no setter, such as FuzzySetGroup.centers/widths/mask.
    Such a property can never be reconstructed via setattr(), so it must never be
    treated as save/load-able state by get_object_attributes() below.

    Args:
        cls: The class to search (its own __dict__ and every base's, via MRO).
        name: The attribute name to check.

    Returns:
        True if name resolves to a property with no setter.
    """
    for klass in cls.__mro__:
        attr = klass.__dict__.get(name)
        if isinstance(attr, property):
            return attr.fset is None
    return False


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
        if (attr, value) not in super_attributes
        and not attr.startswith("_")
        and not _is_read_only_property(obj_instance.__class__, attr)
    }


@contextmanager
def capture(model, layers=None, include_inputs=False):
    """
    Capture intermediate activations from a PyTorch model without modifying it.

    Args:
        model: Any instance of an object that inherits from torch.nn.Module.
        layers: Optional set/list of layer names to capture (None = all).
        include_inputs: If True, store (input, output) tuples instead of only the output.

    Yields:
        OrderedDict mapping layer names to captured tensors (or tuples).
    """
    activations = OrderedDict()
    hooks = []

    def _detach(x):
        if isinstance(x, torch.Tensor):
            return x.detach()
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            # a namedtuple (e.g. Membership): __new__ takes one positional arg per
            # field, not a single iterable, so each detached value must be
            # unpacked
            return type(x)(*(_detach(v) for v in x))
        if isinstance(x, (tuple, list)):
            return type(x)(_detach(v) for v in x)
        if isinstance(x, dict):
            return {k: _detach(v) for k, v in x.items()}
        return x  # non-tensor leaf (None, int, etc.)

    def make_hook(name):
        def hook(module, inp, out):
            try:
                if include_inputs:
                    activations[name] = (_detach(inp), _detach(out))
                else:
                    activations[name] = _detach(out)
            except Exception as e:
                raise RuntimeError(
                    f"Hook failed on layer '{name}' "
                    f"(type={type(module).__name__}, "
                    f"output type={type(out).__name__})"
                ) from e

        return hook

    for name, layer in model.named_modules():
        if name == "":  # skip the top-level module itself
            continue
        if layers is not None and name not in layers:
            continue
        hooks.append(layer.register_forward_hook(make_hook(name)))

    try:
        yield activations
    finally:
        for h in hooks:
            h.remove()
        hooks.clear()
