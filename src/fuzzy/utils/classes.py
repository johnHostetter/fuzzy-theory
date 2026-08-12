"""
This module contains classes that are reserved more for the internal use of the fuzzy package.
"""

import inspect
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (Any, Callable, Dict, List, MutableMapping, Optional, Set,
                    Tuple, Union)

import numpy as np
import torch
from natsort import natsorted
from torch.nn.modules.module import _forward_unimplemented

from fuzzy.utils.functions import (all_subclasses, get_object_attributes,
                                   signature_of)


class Loggable:  # pylint: disable=too-few-public-methods
    """
    Inherit this Loggable class to automatically create a logger to use.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is not None:
            self.logger = logger
            return

        name = name or self.__class__.__name__
        self.logger = logging.getLogger(name)

        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)


class DynamicParameterList(torch.nn.Module):  # pylint: disable=abstract-method
    """
    Wraps a torch.nn.ParameterList and maintains a contiguous cached tensor for fast operations.
    """

    def __init__(
            self,
            init_params=None,
            dtype=None,
            device=None,
            parameters: bool = True):
        super().__init__()
        self.params: Union[torch.nn.ParameterList, List[torch.Tensor]] = (
            torch.nn.ParameterList() if parameters else []
        )
        self._cached_tensor = None
        self._cached_signature = None
        self._device = device
        self._dtype = dtype

        if init_params is not None:
            for p in init_params:
                self.add_parameter(p)

    def __getitem__(self, item):
        return self.params[item]

    def __setitem__(self, idx, value):
        # 1. Enforce that the incoming value is a valid PyTorch Parameter
        if not isinstance(value, torch.nn.Parameter):
            raise TypeError(
                f"Expected a torch.nn.Parameter, but got {type(value)}")

        # 2. Update the internal tracker
        # If using nn.ParameterList, it handles module registration
        # automatically.
        self.params[idx] = value

        # 3. Optional: Give it a unique string key name on the parent module
        # if your custom class relies on named parameters attribute binding.
        # setattr(self, f"param_{idx}", value)

        # 4. Invalidate the cached tensor
        self._invalidate_cache()

    def add_parameter(self, tensor: Union[np.ndarray, torch.Tensor]):
        """
        Add a new Parameter and invalidate cached tensor.
        tensor: torch.Tensor (will be converted to Parameter)
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(
                tensor, dtype=self._dtype, device=self._device)
        if isinstance(self.params, torch.nn.ParameterList):
            param = torch.nn.Parameter(tensor)
        else:
            param = tensor  # leave it unmodified
        if self._device is not None:
            param.data = param.data.to(self._device, dtype=self._dtype)
        self.params.append(param)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """
        Discard the cached concatenation, forcing the next access to rebuild it.

        Returns:
            None
        """
        self._cached_tensor = None
        self._cached_signature = None

    @torch.jit.ignore
    def _concat_params(self, params_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Return a cached concatenation of two or more parameters, rebuilding it if any
        parameter was replaced or mutated in place since it was last built.

        Pulled out of the 'tensor' property (and marked to be skipped by torch.jit.script)
        because comparing two ParameterSignature values with '!=' is not something
        TorchScript's type system supports; this keeps that comparison in ordinary Python
        while leaving the property itself scriptable.

        Args:
            params_list: The parameters to concatenate, materialized as a plain list.

        Returns:
            The concatenation of the parameters along dim=-1.
        """
        signature = signature_of(params_list)
        if self._cached_tensor is None or self._cached_signature != signature:
            self._cached_tensor = torch.cat(params_list, dim=-1).contiguous()
            self._cached_signature = signature
        return self._cached_tensor

    @property
    def tensor(self):
        """
        Returns a contiguous tensor concatenating all parameters along dim=-1.

        In the (overwhelmingly common) case of a single parameter, that parameter is returned
        directly: concatenating one tensor only copies it, and the copy would both waste time
        and - more importantly - go stale, since it does not observe subsequent in-place updates
        to the parameter such as those an optimizer applies.

        For several parameters the concatenation is cached, but the cache is keyed on the
        identity and version counter of each parameter, so that replacing *or* updating any of
        them rebuilds it. Without this, the concatenation would keep reporting the parameter
        values as they were when it was first built.

        Note: `self.params` (a torch.nn.ParameterList) is deliberately materialized into a
        plain list before anything else; torch.jit.script cannot compile a bare `len(...)` or
        indexing call directly against a ParameterList attribute in this position, but has no
        trouble with a plain List[Tensor].
        """
        params_list: List[torch.Tensor] = list(self.params)
        if len(params_list) == 0:
            return torch.tensor([], device=self._device, dtype=self._dtype)

        if len(params_list) == 1:
            # avoid a copy that would immediately be at risk of going stale
            return params_list[0]

        return self._concat_params(params_list)

    def to(self, *args, **kwargs):
        """
        Override to move both ParameterList and cached tensor.
        """
        super().to(*args, **kwargs)
        # torch.nn.Module.to() accepts several call signatures - a device, a dtype, another
        # tensor to match, or a combination - so args[0] is not reliably a device (e.g.
        # .to(torch.float64) is a legitimate dtype-only call, and would otherwise corrupt
        # _device with a dtype object). Replaying the same call against a throwaway tensor
        # seeded with the current device/dtype and reading back what it resolved to avoids
        # re-implementing that parsing here.
        probe = torch.empty(0, dtype=self._dtype, device=self._device).to(
            *args, **kwargs
        )
        self._device = probe.device
        self._dtype = probe.dtype
        # the parameters were moved, so any concatenation of them refers to the old device;
        # rebuild it on next access rather than moving a copy that is about to
        # go stale
        self._invalidate_cache()
        return self


class TimeDistributed(torch.nn.Module):
    """
    A wrapper class for PyTorch modules that allows them to operate on a sequence of data.
    """

    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module: torch.nn.Module, batch_first: bool = False):
        """
        Initialize the TimeDistributed wrapper class.

        Args:
            module: A PyTorch module.
            batch_first: Whether the batch dimension is the first dimension.
        """
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TimeDistributed wrapper class.

        Args:
            input_data: The input data.

        Returns:
            The output of the module on the input sequence of data.
        """
        if len(input_data.size()) <= 2:
            return self.module(input_data)

        # squash samples and timesteps into a single axis
        reshaped_input_data = input_data.contiguous().view(
            -1, input_data.size(-1)
        )  # (samples * timesteps, input_size)

        module_output = self.module(reshaped_input_data)

        # reshape the output back to the original shape; this is independent of
        # batch_first, since unflattening a (dim0 * dim1, ...) tensor always restores
        # the original leading-dimension order, regardless of what those two
        # dimensions semantically represent
        output_dim = 1
        if module_output.ndim == 2:
            output_dim = module_output.size(-1)
        module_output = module_output.contiguous().view(
            input_data.size(0), input_data.size(1), output_dim
        )

        return module_output


class TorchJitModule(torch.nn.Module, ABC):
    """
    A TorchJitModule is a torch.nn.Module that can be saved and loaded to and from a file. It is
    also expected that the class that inherits from TorchJitModule will have subclasses of its own.
    """

    @abstractmethod
    @torch.jit.ignore
    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the torch.nn.Module object to a file.

        Note: This does not preserve ParameterList structures, but rather concatenates the
        parameters into a single tensor, which is then saved to a file.

        Returns:
            A dictionary containing the state of the torch.nn.Module object.
        """

    @classmethod
    @abstractmethod
    @torch.jit.ignore
    def load(cls, path: Path, device: torch.device) -> "TorchJitModule":
        """
        Load the class object from a file and put it on the specified device.

        Returns:
            None
        """

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
                f"{cls}. Please ensure that {class_name} is a subclass of {cls}.")
        return fuzzy_set_class


class NestedTorchJitModule(torch.nn.Module):
    """
    A NestedTorchJitModule is a torch.nn.Module that contains other torch.nn.Module objects as
    attributes. This class is used to save and load the torch.nn.Module object to and from a
    directory, respectively.
    """

    forward: Callable[..., Any] = (
        _forward_unimplemented  # unsure of forward signature yet
    )

    def save(self, path: Path) -> None:
        """
        Save the torch.nn.Module object to a directory.

        Note: This does not preserve ParameterList structures, but rather concatenates the
        parameters into a single tensor, which is then saved to a file.

        Args:
            path: The path to save the NestedTorchJitModule to; it must be a directory.

        Returns:
            None
        """
        # get the attributes that are local to the class, but not inherited
        # from the super class
        local_attributes_only = get_object_attributes(self)

        # save a reference to the attributes (and their values) so that when iterating over them,
        # we do not modify the dictionary while iterating over it (which would cause an error)
        # we modify the dictionary by removing attributes that have a value of torch.nn.ModuleList
        # because we want to save the modules in the torch.nn.ModuleList
        # separately
        local_attributes_only_items: List[Tuple[str, Any]] = list(
            local_attributes_only.items()
        )
        for attr, value in local_attributes_only_items:
            if isinstance(
                value, torch.nn.ModuleList
            ):  # e.g., attr may be self.modules_list
                for idx, module in enumerate(value):
                    subdirectory = path / attr / str(idx)
                    subdirectory.mkdir(parents=True, exist_ok=True)
                    if isinstance(module, TorchJitModule):
                        # save the fuzzy set using the fuzzy set's special
                        # protocol
                        module.save(
                            path / attr / str(idx) / f"{module.__class__.__name__}.pt")
                    else:
                        # unknown and unrecognized module, but attempt to save
                        # the module
                        torch.save(
                            module,
                            path /
                            attr /
                            str(idx) /
                            f"{module.__class__.__name__}.pt",
                        )
                # remove the torch.nn.ModuleList from the local attributes
                del local_attributes_only[attr]

        # save the remaining attributes
        with open(path / f"{self.__class__.__name__}.pickle", "wb") as handle:
            pickle.dump(
                local_attributes_only,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, device: torch.device,
             **kwargs) -> "NestedTorchJitModule":
        """
        Load the torch.nn.Module from the given path.

        Args:
            path: The path to load the NestedTorchJitModule from.
            device: The device to load the NestedTorchJitModule to.
            **kwargs:

        Returns:
            The loaded NestedTorchJitModule.
        """
        local_attributes_only: Dict[str, Any] = {}
        for file_path in path.iterdir():
            if ".pickle" in file_path.name:
                # load the remaining attributes
                with open(file_path, "rb") as handle:
                    local_attributes_only.update(pickle.load(handle))
            elif file_path.is_dir():
                local_attributes_only[file_path.name] = (
                    cls.find_load_and_return_modules(
                        path=file_path, device=device, kwargs=kwargs
                    )
                )

        # of the remaining attributes, we must determine which are shared between the
        # super class and the local class, otherwise we will get an error when trying to
        # initialize the local class (more specifically, the torch.nn.Module __init__ method
        # requires self.call_super_init to be set to True, but then the attribute would exist
        # as a super class attribute, and not a local class attribute)
        shared_args: Set[str] = set(
            inspect.signature(cls).parameters.keys()
        ).intersection(local_attributes_only.keys())

        # create the GroupedFuzzySet object with the shared arguments
        # (e.g., modules_list, expandable)
        grouped_fuzzy_set: NestedTorchJitModule = cls(
            **{
                key: value
                for key, value in local_attributes_only.items()
                if key in shared_args
            }
        )

        # determine the remaining attributes
        remaining_args: Dict[str, Any] = {
            key: value
            for key, value in local_attributes_only.items()
            if key not in shared_args
        }

        # set the remaining attributes
        for attr, value in remaining_args.items():
            try:
                setattr(grouped_fuzzy_set, attr, value)
            except AttributeError:
                # the attribute is not a valid attribute of the class (e.g.,
                # property)
                continue
        return grouped_fuzzy_set

    @classmethod
    def find_load_and_return_modules(
        cls,
        path: Path,
        device: torch.device,
        kwargs: dict[str, Any],
    ) -> List[Union[TorchJitModule, torch.nn.Module]]:
        """

        Args:
            path: The path to where the modules are stored.
            device: The device to which the modules should be loaded.
            kwargs: The keyword arguments to pass to the modules.

        Returns:
            A list of the discovered modules.
        """
        modules_list: List[Union[TorchJitModule, torch.nn.Module]] = []
        for subdirectory in natsorted(path.iterdir()):
            if subdirectory.is_dir():
                module_path: Path = list(subdirectory.glob("*.pt"))[0]
                # load the fuzzy set using the fuzzy set's special
                # protocol
                class_name: str = module_path.name.split(".pt")[0]
                try:
                    modules_list.append(
                        TorchJitModule.get_subclass(class_name).load(
                            module_path, device=device, **kwargs
                        )
                    )
                except ValueError:
                    # unknown and unrecognized module, but attempt to
                    # load the module
                    modules_list.append(
                        torch.load(
                            module_path,
                            weights_only=False))
            else:
                pass  # Unexpected file found (might be a *.yaml)
        return modules_list
