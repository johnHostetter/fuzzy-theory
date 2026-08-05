"""
Implements an abstract class called FuzzySet using PyTorch. All fuzzy sets defined over
a continuous domain are derived from this class. Further, the Membership class is defined within,
which contains a helpful interface understanding membership degrees.
"""

import abc
import inspect

# import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, MutableMapping, NoReturn, Optional, Tuple, Type, Union

import numpy as np
import sympy
import torch
import torchquad
from numpy import ndarray
from numpy._typing import _64Bit
from torchquad.utils.set_up_backend import set_up_backend

from ..utils import TorchJitModule, check_path_to_save_torch_module

# DynamicParameterList re-exported here for backward compatibility (it used to be
# defined in this module)
from ..utils.classes import DynamicParameterList, Loggable  # noqa: F401
from ..utils.functions import ParameterSignature, signature_of

# from ..utils.functions import log_classmethod, log_func, log_method
from .cache import MembershipCache
from .membership import Membership

# FuzzySetShape/FuzzySetInitResult/FuzzySetInitMethod re-exported here for backward
# compatibility (they used to be defined in this module)
from .shape import FuzzySetInitMethod, FuzzySetInitResult, FuzzySetShape  # noqa: F401


class FuzzySet(TorchJitModule, Loggable, metaclass=abc.ABCMeta):
    """
    A generic and abstract torch.nn.Module class that implements continuous fuzzy sets.

    This is the most important Python class regarding fuzzy sets within this Soft Computing library.

    Defined here are most of the common methods made available to all fuzzy sets. Fuzzy sets that
    will later be used in other features such as neuro-fuzzy networks are expected to abide by the
    conventions outlined within. For example, parameters 'centers' and 'widths' are often expected,
    but inference engines (should) only rely on the fuzzy set membership degrees.

    However, for convenience, some aspects of the SelfOrganize code may search for vertices that
    have attributes of type 'FuzzySet'. Thus, if it is pertinent that a vertex within
    the KnowledgeBase is recognized as a fuzzy set, it is very likely one might be interested in
    inheriting or extending from FuzzySet.
    """

    # Subclasses may set this to True to check the calculated membership degrees for NaN and
    # infinite values; this costs a synchronization per call, so it is off by
    # default.
    _validate_degrees: bool = False

    # See NAryRelation's GATHER_APPLY_MASK_SYNC_THRESHOLD_NUMEL (module-level there,
    # since NAryRelation is not currently torch.jit.script-compatible) for the
    # underlying calibration this mirrors: on a CUDA tensor, a sync to ask "is there
    # any NaN at all?" (~400-750us in practice, dominated by the reduction kernel and
    # blocking scalar readback, not the sync primitive itself) costs more than just
    # always doing the NaN-safe substitution below this many (estimated) elements in
    # the resulting degrees tensor, and less above it. A class attribute (not a module
    # global) because torch.jit.script rejects references to arbitrary Python module
    # globals from a scripted method; it accepts instance attributes assigned in
    # __init__ instead (see _validate_degrees just above). Threshold is an
    # empirically-calibrated heuristic (measured on one GPU), not a theoretically
    # derived constant.
    _nan_safe_sync_threshold_numel: int = 5_000_000

    def __init__(
        self,
        centers: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]],
        widths: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]],
        device: torch.device,
        use_sparse_tensor: bool = False,
        cache_membership: bool = True,
        membership_cache_size: int = 2,
        # debug: bool = False,
    ):
        super().__init__()
        self.device = device
        # self.create_logger(self.__class__.__name__, debug=debug)
        self.__check_args(centers, widths)

        self._centers = None
        self._widths = None
        self._mask = None
        self.__alloc_members(centers, use_sparse_tensor, widths)

        # memoizes membership calculations; see fuzzy.sets.cache for when a
        # result is re-used
        self._membership_cache = MembershipCache(
            maxsize=membership_cache_size, enabled=cache_membership
        )

        # torch.jit.script only picks up attributes assigned in __init__, so the class-level
        # default declared above is re-assigned here as an instance attribute; this also lets
        # a subclass such as Lorentzian's class-level override take effect
        # under scripting
        self._validate_degrees: bool = self._validate_degrees
        self._nan_safe_sync_threshold_numel: int = self._nan_safe_sync_threshold_numel

    # @log_method
    @staticmethod
    def __check_args(centers: np.ndarray, widths: np.ndarray) -> None:
        """
        Check that the provided argument values are accepted variable types and that their
        dimensionality is correct. This function will raise a ValueError if the argument values
        are not compatible with FuzzySet.

        Args:
            centers: The data describing the centers of the fuzzy sets.
            widths: The data describing the widths of the fuzzy sets.

        Returns:
            None
        """
        if not isinstance(centers, np.ndarray):
            # ensure that the centers are a numpy array (done for consistency)
            # specifically, we want to internally control the dtype and device
            # of the centers
            raise ValueError(
                f"The centers of a FuzzySet must be a numpy array, "
                f"but got {type(centers)}"
            )
        if not isinstance(widths, np.ndarray):
            # ensure that the widths are a numpy array (done for consistency)
            # specifically, we want to internally control the dtype and device
            # of the widths
            raise ValueError(
                f"The widths of a FuzzySet must be a numpy array, but got {type(widths)}"
            )

        if centers.ndim != widths.ndim:
            raise ValueError(
                f"The number of dimensions for the centers ({centers.ndim}) and widths "
                f"({widths.ndim}) must be the same.")

        if centers.ndim == 0 or widths.ndim == 0:
            raise ValueError(
                f"The centers and widths of a FuzzySet must have at least one dimension. "
                f"Centers has {centers.ndim} dimensions and widths has {widths.ndim} dimensions.")

    # @log_method
    def __alloc_members(
        self, centers: np.ndarray, use_sparse_tensor: bool, widths: np.ndarray
    ) -> None:
        if centers.ndim == 1 and widths.ndim == 1:
            # assuming that the array is a single linguistic variable
            centers, widths = centers[None, :], widths[None, :]

        # avoid allocating new memory for the centers and widths
        # use torch.float32 to save memory and speed up computations
        self._centers = DynamicParameterList(
            init_params=[centers], dtype=torch.float32, device=self.device
        )
        self._widths = DynamicParameterList(
            init_params=[widths], dtype=torch.float32, device=self.device
        )
        self.use_sparse_tensor = use_sparse_tensor
        self._mask = DynamicParameterList(
            init_params=[self.make_mask(widths)],
            dtype=torch.uint8,
            device=self.device,
            parameters=False,
        )

    # @log_method
    def to(self, *args, **kwargs):
        """
        Move the FuzzySet to a new device.

        Returns:
            None
        """
        # Call the parent class's `to` method to handle parameters and
        # submodules
        super().to(*args, **kwargs)

        # special handling for DynamicParameterList
        self._centers = self._centers.to(*args, **kwargs)
        self._widths = self._widths.to(*args, **kwargs)

        # special handling for the non-parameter tensors, such as mask
        # self._mask = [mask.to(*args, **kwargs) for mask in self._mask]
        self._mask = self._mask.to(*args, **kwargs)
        self.device = self._centers[0].device
        # memberships were calculated on the previous device, and moving parameters replaces
        # their data without bumping a version counter, so the memo cannot be
        # trusted
        self.clear_membership_cache()
        # self.logger.debug(f"Moved {self.__class__.__name__} to {self.device} device")
        return self

    # @log_method
    def make_parameter(self, parameter: np.ndarray) -> torch.nn.Parameter:
        """
        Create a torch.nn.Parameter from a numpy array, with the appropriate dtype and device.

        Args:
            parameter: The numpy array to convert to a torch.nn.Parameter (e.g., centers or widths).

        Returns:
            A torch.nn.Parameter object.
        """
        return torch.nn.Parameter(
            torch.as_tensor(
                parameter,
                dtype=torch.float32,
                device=self.device),
            # requires_grad=True,  # explicitly set to True
        )

    # @log_method
    def make_mask(self, widths: np.ndarray) -> torch.Tensor:
        """
        Create a mask for the fuzzy set, where the mask is used to filter out fuzzy sets that are
        not real. This is particularly useful when the fuzzy set is not fully defined, and some
        fuzzy sets are missing. The mask is a binary tensor that is used to filter out fuzzy sets
        that are not real. If the mask is 0, then the fuzzy set is not real; otherwise, it is real.

        Args:
            widths: The widths of the fuzzy set.

        Returns:
            A torch.Tensor object.
        """
        return torch.as_tensor(
            widths > 0.0,
            dtype=torch.uint8,
            device=self.device)
        # return torch.nn.Parameter(
        #     torch.as_tensor(widths > 0.0, dtype=torch.int8, device=self.device),
        #     requires_grad=False,  # explicitly set to False (mask is not trainable)
        # )

    @classmethod
    # @log_classmethod
    def create(
        cls,
        shape: FuzzySetShape,
        device: torch.device,
        method: FuzzySetInitMethod,
        init_width: float = 0.5,
        **kwargs,
    ) -> Union[NoReturn, "FuzzySet"]:
        """
        Create a fuzzy set with the given number of variables and terms, where each variable
        has the same number of terms. For example, if we have two variables, then we might have
        three terms for each variable, such as "low", "medium", and "high". This would result in
        a total of nine fuzzy sets. The centers and widths are initialized randomly.

        Args:
            shape: The shape of these fuzzy sets.
            device: The device to use.
            method: The method to use for creating the fuzzy set (e.g., "random" or "linear").
            init_width: The initial width of the fuzzy set (for "linear" method).

        Returns:
            A FuzzySet object, or a NotImplementedError if the method is not implemented.
        """
        if inspect.isabstract(cls):
            # this error is thrown if the class is abstract, such as FuzzySet, but
            # the method is not implemented (e.g., self.calculate_membership)
            raise NotImplementedError(
                "The FuzzySet has no defined membership function. Please create a class "
                "and inherit from FuzzySet, or use a predefined class, such as Gaussian.")

        init_result: FuzzySetInitResult = method.initialize(
            shape=shape,
            init_width=init_width,
        )

        fuzzy_sets = cls(
            centers=init_result.centers,
            widths=init_result.widths,
            device=device,
            **kwargs,
        )
        # logging.debug(f"New fuzzy set(s) created with the %s method.", method)
        return fuzzy_sets

    # @log_method
    def __hash__(self):
        """
        Hash the fuzzy set.

        This must agree with __eq__, which compares centers/widths by *value* (torch.equal).
        Hashing the tensors themselves would hash by identity instead (torch.Tensor keeps
        the default id()-based __hash__), so two separately constructed but value-equal fuzzy
        sets would violate Python's hash contract - equal objects reporting different hashes -
        which breaks their use as dict keys or set members. Hashing the flattened values (as
        plain Python numbers, not tensors) keeps this consistent with __eq__.

        Returns:
            The hash of the fuzzy set.
        """
        return hash(
            (
                type(self),
                tuple(self.get_centers().flatten().tolist()),
                tuple(self.get_widths().flatten().tolist()),
            )
        )

    # @log_method
    def __eq__(self, other: Any) -> bool:
        """
        Check if the fuzzy set is equal to another fuzzy set.

        Args:
            other: The other fuzzy set to compare to.

        Returns:
            True if the fuzzy sets are equal, False otherwise.
        """
        return (
            isinstance(other, type(self))
            and torch.equal(self.get_centers(), other.get_centers())
            and torch.equal(self.get_widths(), other.get_widths())
        )

    # @log_method
    def get_centers(self) -> torch.Tensor:
        """
        Get the concatenated centers of the fuzzy set from its corresponding ParameterList.

        Returns:
            The concatenated centers of the fuzzy set.
        """
        return self._centers.tensor

    # @log_method
    def get_widths(self) -> torch.Tensor:
        """
        Get the concatenated widths of the fuzzy set from its corresponding ParameterList.

        Returns:
            The concatenated widths of the fuzzy set.
        """
        return self._widths.tensor

    # @log_method
    def get_mask(self) -> torch.Tensor:
        """
        Get the concatenated mask of the fuzzy set from its corresponding ParameterList.

        Returns:
            The concatenated mask of the fuzzy set.
        """
        return self._mask.tensor

    def _extra_save_state(self) -> MutableMapping[str, Any]:
        """
        Hook for subclasses with additional learnable parameters beyond centers/widths
        (e.g. Trapezoidal's plateaus) to contribute extra entries to the state_dict that
        save() builds - avoiding the need to re-implement save() just to add one
        more field.

        Returns:
            Additional entries to merge into the state_dict save() builds. Empty by
            default (no extra parameters).
        """
        return {}

    @classmethod
    def _extra_load_kwargs(
        cls, _state_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """
        Hook for subclasses to pop() their extra _extra_save_state() entries back out of
        the loaded state_dict and turn them into constructor kwargs - the counterpart to
        _extra_save_state(), avoiding the need to re-implement load() just to
        restore one more field.

        Args:
            state_dict: The loaded state_dict, after "centers"/"widths"/"class_name"
                have already been popped by load(). Subclasses should pop() their own
                extra keys from this (mutating it) rather than merely reading them.

        Returns:
            Additional keyword arguments to pass to the subclass' constructor. Empty by
            default (no extra parameters).
        """
        return {}

    # @log_method
    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the fuzzy set to a file.

        Note: This does not preserve the ParameterList structure, but rather concatenates the
        parameters into a single tensor, which is then saved to a file.

        Returns:
            A dictionary containing the state of the fuzzy set.
        """
        check_path_to_save_torch_module(path)
        state_dict: MutableMapping = self.state_dict()
        state_dict["class_name"] = self.__class__.__name__
        state_dict["centers"] = self.get_centers()  # concatenate the centers
        state_dict["widths"] = self.get_widths()  # concatenate the widths
        state_dict["mask"] = self.get_mask()  # currently not used
        state_dict.update(self._extra_save_state())
        torch.save(state_dict, path)
        return state_dict

    @classmethod
    # @log_classmethod
    def load(cls, path: Path, device: torch.device) -> "FuzzySet":
        """
        Load the fuzzy set from a file and put it on the specified device.

        Returns:
            None
        """
        state_dict: MutableMapping = torch.load(path, weights_only=False)
        centers = state_dict.pop("centers")
        widths = state_dict.pop("widths")
        class_name = state_dict.pop("class_name")
        subclass = cls.get_subclass(class_name)
        return subclass(
            centers=centers.cpu().detach().numpy(),
            widths=widths.cpu().detach().numpy(),
            device=device,
            **subclass._extra_load_kwargs(  # pylint: disable=protected-access
                state_dict
            ),
        )

    # @log_method
    def extend(self, centers: torch.Tensor, widths: torch.Tensor, mode: str):
        """
        Given additional parameters, centers and widths, extend the existing self.centers and
        self.widths, respectively. Additionally, update the necessary backend logic.

        Args:
            centers: The centers of new fuzzy sets.
            widths: The widths of new fuzzy sets.

        Returns:
            None
        """
        if mode == "vertical":
            method_of_extension: callable = torch.cat
        elif mode == "horizontal":
            method_of_extension: callable = torch.hstack
        else:
            raise ValueError(
                f"The mode must be either 'horizontal' or 'vertical', but got '{mode}'."
            )
        with torch.no_grad():
            self._centers[0] = torch.nn.Parameter(
                method_of_extension([self._centers[0], centers])
            )
            self._widths[0] = torch.nn.Parameter(
                method_of_extension([self._widths[0], widths])
            )
        # the fuzzy set now describes more terms, so previously calculated membership degrees
        # no longer have the right shape
        self.clear_membership_cache()

    # @log_method
    def area(self) -> torch.Tensor:
        """
        Calculate the area beneath the fuzzy curve (i.e., membership function) using torchquad.

        This is a slightly expensive operation, but it is used for approximating the Mamdani fuzzy
        inference with arbitrary continuous fuzzy sets.

        Typically, the results will be cached somewhere, so that the area value can be reused.

        Splits the fuzzy set (if representing a fuzzy variable) into individual fuzzy sets (the
        fuzzy variable's possible fuzzy terms), and does so recursively until the base case is
        reached. Once the base case is reached (i.e., a single fuzzy set), the area under its
        curve within the integration_domain is calculated.

        Returns:
            torch.Tensor
        """
        all_areas: List[List[float]] = []
        for variable_params in zip(self.get_centers(), self.get_widths()):
            variable_centers, variable_widths = variable_params[0], variable_params[1]
            variable_areas = []
            for term_params in zip(variable_centers, variable_widths):
                centers, widths = term_params[0].item(), term_params[1].item()
                # has to be "cpu" device for torchquad.Simpson to work
                fuzzy_set = self.__class__(
                    centers=np.array([centers]),
                    widths=np.array([widths]),
                    device=self.device,
                )

                # Enable GPU support if available and set the floating point
                # precision
                set_up_backend("torch", data_type="float32")

                simpson_method = torchquad.Simpson()
                area: float = simpson_method.integrate(
                    fuzzy_set.calculate_membership,
                    dim=1,
                    N=101,
                    integration_domain=[
                        [
                            fuzzy_set.get_centers().item()
                            - fuzzy_set.get_widths().item(),
                            fuzzy_set.get_centers().item()
                            + fuzzy_set.get_widths().item(),
                        ]
                    ],
                    backend="torch",
                ).item()
                if fuzzy_set.get_widths().item() <= 0 and area != 0.0:
                    # if the width of a fuzzy set is negative or zero, it is a special flag that
                    # the fuzzy set does not exist; thus, the calculated area of a fuzzy set w/ a
                    # width <= 0 should be zero. However, in the case this does not occur,
                    # a zero will substitute to be sure that this issue does
                    # not affect results
                    area = 0.0
                variable_areas.append(area)
            all_areas.append(variable_areas)
        return torch.tensor(all_areas, device=self.device, dtype=torch.float32)

    # @log_method
    def split_by_variables(self) -> Union[list, List[Type["FuzzySet"]]]:
        """
        This operation takes the FuzzySet and converts it to a list of FuzzySet
        objects, if applicable. For example, rather than using a single Gaussian object to represent
        all Gaussian membership functions in the input space, this function will convert that to a
        list of Gaussian objects, where each Gaussian function is defined and restricted to a single
        input dimension. This is particularly helpful when modifying along a specific dimension.

        Returns:
            A list of FuzzySet objects, where the length is equal to the number
            of input dimensions.
        """
        variables = []
        for centers, widths in zip(self.get_centers(), self.get_widths()):
            centers = centers.cpu().detach().tolist()
            widths = widths.cpu().detach().tolist()

            # the centers and widths must be trimmed to remove missing fuzzy
            # set placeholders
            trimmed_centers, trimmed_widths = [], []
            for center, width in zip(centers, widths):
                if width > 0:
                    # if an input dimension has less fuzzy sets than another,
                    # then it is possible for the width entry to have '-1' as a
                    # placeholder indicating so
                    trimmed_centers.append(center)
                    trimmed_widths.append(width)

            variables.append(
                type(self)(
                    centers=np.array(trimmed_centers),
                    widths=np.array(trimmed_widths),
                    device=self.device,
                )
            )

        return variables

    # @log_method
    def plot(
        self, output_dir: Path, selected_terms: List[Tuple[int, int]] = None
    ) -> Tuple[List[Any], Any]:
        """
        Plot the fuzzy set.

        A thin convenience wrapper: data preparation and matplotlib rendering live in
        fuzzy.sets.visualization.FuzzySetPlot - imported lazily here so that importing
        this module (and therefore FuzzySet itself) does not require
        matplotlib/scienceplots unless plotting is actually used.

        Args:
            output_dir: The path to the directory where to save the plot(s).
            selected_terms: The terms to highlight in the plot.

        Returns:
            A 2-tuple containing the figures and axes of the plot for each variable (e.g., 0th
            index contains the figure and axes for the 0th variable).
        """
        # pylint: disable-next=import-outside-toplevel
        from .visualization import FuzzySetPlot

        plot = FuzzySetPlot(self)
        variable_plots = plot.build(selected_terms)
        return plot.render(variable_plots, output_dir)

    @staticmethod
    # @log_func
    def count_granule_terms(granules: List["FuzzySet"]) -> np.ndarray:
        """
        Count the number of granules that occur in each dimension.

        Args:
            granules: A list of granules, where each granule is a FuzzySet object.

        Returns:
            A Numpy array with shape (len(granules), ) and the data type is integer.
        """
        return np.array(
            [
                (
                    params.get_centers().size(dim=-1)
                    if params.get_centers().dim() > 0
                    else 0
                )
                for params in granules
            ],
            dtype=np.int8,
        )

    @staticmethod
    # @log_func
    def stack(
        granules: List["FuzzySet"],
    ) -> "FuzzySet":
        """
        Create a condensed and stacked representation of the given granules.

        Args:
            granules: A list of granules, where each granule is a FuzzySet object.

        Returns:
            A FuzzySet object.
        """
        if list(granules)[0].training:
            missing_center, missing_width = 0.0, -1.0
        else:
            missing_center = missing_width = torch.nan

        centers = torch.vstack(
            [
                (
                    torch.nn.functional.pad(
                        params.get_centers(),
                        pad=(
                            0,
                            FuzzySet.count_granule_terms(granules).max()
                            - params.get_centers().shape[-1],
                        ),
                        mode="constant",
                        value=missing_center,
                    )
                    if params.get_centers().dim() > 0
                    else torch.tensor(missing_center).repeat(
                        FuzzySet.count_granule_terms(granules).max()
                    )
                )
                for params in granules
            ]
        )
        widths = torch.vstack(
            [
                (
                    torch.nn.functional.pad(
                        params.get_widths(),
                        pad=(
                            0,
                            FuzzySet.count_granule_terms(granules).max()
                            - params.get_widths().shape[-1],
                        ),
                        mode="constant",
                        value=missing_width,
                    )
                    if params.get_centers().dim() > 0
                    else torch.tensor(missing_center).repeat(
                        FuzzySet.count_granule_terms(granules).max()
                    )
                )
                for params in granules
            ]
        )

        # prepare a condensed and stacked representation of the granules
        mf_type = type(granules[0])
        return mf_type(
            centers=centers.cpu().detach().numpy(),
            widths=widths.cpu().detach().numpy(),
            device=centers.device,
        )

    @classmethod
    # @log_classmethod
    @abstractmethod
    def sympy_formula(cls) -> sympy.Expr:
        """
        The abstract method that defines the membership function of the fuzzy set using sympy.

        Returns:
            A sympy.Expr object that represents the membership function of the fuzzy set.
        """

    @abc.abstractmethod
    def calculate_membership(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Calculate the membership degrees of the observations for this fuzzy set.

        Implementations are expected to be a pure function of the observations and this fuzzy
        set's parameters, as the result may be memoized (see fuzzy.sets.cache).

        Args:
            observations: The observations to calculate the membership degrees for.

        Returns:
            The membership degrees of the observations for this fuzzy set.
        """

    def parameter_signature(self) -> ParameterSignature:
        """
        Summarize the parameters that a membership calculation depends upon, so that the
        membership cache can tell when a memoized result has been outdated by a parameter
        changing (as happens on every optimizer step).

        Subclasses with additional parameters of their own should extend this; otherwise their
        memberships would be re-used after those parameters were updated.

        Returns:
            A signature of this fuzzy set's parameters.
        """
        return signature_of(
            [self.get_centers(), self.get_widths(), self.get_mask()])

    @torch.jit.ignore
    def clear_membership_cache(self) -> None:
        """
        Discard any memoized membership degrees, forcing the next call to recalculate them.

        Returns:
            None
        """
        cache: Union[None, MembershipCache] = getattr(
            self, "_membership_cache", None)
        if cache is not None:
            cache.clear()

    @torch.jit.ignore
    @torch.compiler.disable
    def _lookup_membership(
            self,
            observations: torch.Tensor) -> Optional[Membership]:
        """
        Retrieve memoized membership degrees for the given observations, if they are still valid.

        The cache is looked up defensively so that a torch.jit.script'ed copy of this module -
        which does not carry the cache over - simply calculates the membership degrees instead
        of failing.

        @torch.compiler.disable mirrors the @torch.jit.ignore above, for the same reason:
        this method's parameter_signature() call (see below) reads id(tensor), which
        torch.compile's Dynamo tracer cannot trace through (identity is not a
        graph-representable concept) - telling Dynamo not to trace into this method at
        all, running it eagerly instead, is both correct (the cache is a pure Python-side
        optimization, redundant once the surrounding model is already compiled) and
        necessary (without it, Dynamo hits the untraceable id() call and breaks the
        graph anyway, just less predictably).

        Args:
            observations: The observations that membership degrees are wanted for.

        Returns:
            The memoized Membership, or None if it has to be calculated.
        """
        cache: Union[None, MembershipCache] = getattr(
            self, "_membership_cache", None)
        if cache is None or not cache.enabled:
            # skip computing parameter_signature() (which reads id(tensor) per
            # parameter) when there is no cache to serve the lookup anyway or it is
            # disabled - cache.lookup() would discard the signature unused, but as a
            # function-call argument it would already have been computed by the time
            # cache.lookup() runs to discard it
            return None
        return cache.lookup(observations, self.parameter_signature())

    @torch.jit.ignore
    @torch.compiler.disable
    def _store_membership(
        self, observations: torch.Tensor, membership: Membership
    ) -> None:
        """
        Memoize the membership degrees calculated for the given observations.

        See _lookup_membership's docstring for why this is also marked
        @torch.compiler.disable and skips parameter_signature() when there is nothing
        to store it into.

        Args:
            observations: The observations the membership degrees were calculated for.
            membership: The calculated membership degrees and mask.

        Returns:
            None
        """
        cache: Union[None, MembershipCache] = getattr(
            self, "_membership_cache", None)
        if cache is not None and cache.enabled:
            cache.store(observations, self.parameter_signature(), membership)

    def _calculate_membership_nan_safe(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate membership degrees without letting a NaN observation (representing a
        missing value - see e.g. NAryRelation's nan_replacement) corrupt the gradient of
        this fuzzy set's parameters for every OTHER, valid observation that shares them.

        A membership formula such as Gaussian's exp(-((x-c)^2)/(2w^2)) has a local
        derivative with respect to its parameters that is itself NaN whenever x is NaN,
        regardless of what the downstream loss actually needs from that observation.
        Since centers/widths are shared across the whole batch, PyTorch's chain rule
        would otherwise multiply that NaN local derivative through - 0 * NaN = NaN under
        IEEE754 - silently corrupting the gradient for the entire training step, not just
        the one missing observation. Calculating on a NaN-free substitute, then
        re-injecting NaN into the *output* via a constant (gradient-disconnected)
        substitution, keeps the "NaN observation -> NaN degree" contract callers already
        rely on, while torch.where's backward pass correctly contributes zero gradient at
        exactly those positions instead of NaN.

        Args:
            observations: The (already-prepared) observations to calculate membership for.

        Returns:
            The membership degrees, identical in value to calculate_membership(observations),
            but safe to backpropagate through even when some observations are NaN.
        """
        nan_mask = torch.isnan(observations)
        # a free (no GPU sync), crude estimate of what the resulting degrees tensor's
        # size will be, used only to decide whether syncing to ask "is there any NaN
        # at all?" is worth its cost - see _nan_safe_sync_threshold_numel
        estimated_degrees_numel = observations.numel() * \
            self.get_centers().shape[-1]
        if estimated_degrees_numel > self._nan_safe_sync_threshold_numel and not bool(
                nan_mask.any()):
            return self.calculate_membership(observations)

        safe_observations = torch.where(
            nan_mask, torch.zeros_like(observations), observations
        )
        degrees = self.calculate_membership(safe_observations)
        return torch.where(
            nan_mask.expand_as(degrees),
            torch.full_like(
                degrees,
                float("nan")),
            degrees)

    # @log_method
    def forward(self, observations: torch.Tensor) -> Membership:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        The membership degrees are memoized; calling this again with the same observations, while
        this fuzzy set's parameters are unchanged, returns the very same result rather than
        recalculating it. Gradients are unaffected, as the returned tensor carries the autograd
        graph it was originally built with. Note that a repeated call therefore hands back the
        same tensor object, which must not be modified in place. Memoization can be turned off
        per fuzzy set with cache_membership=False, or reset with clear_membership_cache().

        Args:
            observations: Two-dimensional matrix of observations, where a row is a single
            observation and each column is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for this fuzzy set.
        """
        cached: Optional[Membership] = self._lookup_membership(observations)
        if cached is not None:
            return cached

        original_observations: torch.Tensor = observations
        if observations.ndim == self.get_centers().ndim:
            observations = observations.unsqueeze(dim=-1)

        degrees: torch.Tensor = self._calculate_membership_nan_safe(
            observations)

        if self._validate_degrees:
            assert (
                not degrees.isnan().any()
            ), "NaN values detected in the membership degrees."
            assert (
                not degrees.isinf().any()
            ), "Infinite values detected in the membership degrees."

        membership = Membership(
            degrees=degrees.to_sparse() if self.use_sparse_tensor else degrees,
            mask=self.get_mask(),
        )
        self._store_membership(original_observations, membership)
        return membership
