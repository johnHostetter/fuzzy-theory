"""
This module contains the GroupedFuzzySets class, which is a generic and abstract torch.nn.Module
class that contains a torch.nn.ModuleList of ContinuousFuzzySet objects. The expectation here is
that each ContinuousFuzzySet may define fuzzy sets of different conventions, such as Gaussian,
Triangular, Trapezoidal, etc. Then, subsequent inference engines can handle these heterogeneously
defined fuzzy sets with no difficulty. Further, this class was specifically designed to incorporate
dynamic addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.
"""

import pickle
import inspect
from pathlib import Path
from typing import List, Tuple, Any, Dict, Set, Type, Union

import torch
from natsort import natsorted

from .membership import Membership
from .abstract import ContinuousFuzzySet
from .utils import get_object_attributes, find_widths


class GroupedFuzzySets(torch.nn.Module):
    """
    A generic and abstract torch.nn.Module class that contains a torch.nn.ModuleList
    of ContinuousFuzzySet objects. The expectation here is that each ContinuousFuzzySet
    may define fuzzy sets of different conventions, such as Gaussian, Triangular, Trapezoidal, etc.
    Then, subsequent inference engines can handle these heterogeneously defined fuzzy sets
    with no difficulty. Further, this class was specifically designed to incorporate dynamic
    addition of new fuzzy sets in the construction of neuro-fuzzy networks via network morphism.

    However, this class does *not* carry out any functionality that is necessarily tied to fuzzy
    sets, it is simply named so as this was its intended purpose - grouping fuzzy sets. In other
    words, the same "trick" of using a torch.nn.ModuleList of torch.nn.Module objects applies to
    any kind of torch.nn.Module object.
    """

    def __init__(self, *args, modules_list=None, expandable=False, **kwargs):
        super().__init__(*args, **kwargs)
        if modules_list is None:
            modules_list = []
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.expandable = expandable
        self.epsilon = 1.5  # epsilon-completeness
        # keep track of minimums and maximums if for fuzzy set width calculation
        self.minimums: torch.Tensor = torch.empty(0, 0)
        self.maximums: torch.Tensor = torch.empty(0, 0)
        # store data that we have seen to later add new fuzzy sets
        self.data_seen: torch.Tensor = torch.empty(0, 0)
        # after we see this many data points, we will update the fuzzy sets
        self.data_limit_until_update: int = 64

    def __getattribute__(self, item):
        try:
            if item in ("centers", "widths", "mask"):
                modules_list = self.__dict__["_modules"]["modules_list"]
                if len(modules_list) > 0:
                    module_attributes: List[torch.Tensor] = (
                        []
                    )  # the secondary response denoting module filter
                    for module in modules_list:
                        # get the method for the module and then call it
                        item_method: callable = getattr(module, f"get_{item}")
                        module_attributes.append(item_method())
                    return torch.cat(module_attributes, dim=-1)
                raise ValueError(
                    "The torch.nn.ModuleList of GroupedFuzzySets is empty."
                )
            return object.__getattribute__(self, item)
        except AttributeError:
            return self.__getattr__(item)

    def save(self, path: Path) -> None:
        """
        Save the model to the given path.

        Args:
            path: The path to save the GroupedFuzzySet to.

        Returns:
            None
        """
        # get the attributes that are local to the class, but not inherited from the super class
        local_attributes_only = get_object_attributes(self)

        # save a reference to the attributes (and their values) so that when iterating over them,
        # we do not modify the dictionary while iterating over it (which would cause an error)
        # we modify the dictionary by removing attributes that have a value of torch.nn.ModuleList
        # because we want to save the modules in the torch.nn.ModuleList separately
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
                    if isinstance(module, ContinuousFuzzySet):
                        # save the fuzzy set using the fuzzy set's special protocol
                        module.save(
                            path / attr / str(idx) / f"{module.__class__.__name__}.pt"
                        )
                    else:
                        # unknown and unrecognized module, but attempt to save the module
                        torch.save(
                            module,
                            path / attr / str(idx) / f"{module.__class__.__name__}.pt",
                        )
                # remove the torch.nn.ModuleList from the local attributes
                del local_attributes_only[attr]

        # save the remaining attributes
        with open(path / f"{self.__class__.__name__}.pickle", "wb") as handle:
            pickle.dump(local_attributes_only, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, device: Union[str, torch.device]) -> "GroupedFuzzySets":
        """
        Load the model from the given path.

        Args:
            path: The path to load the GroupedFuzzySet from.
            device: The device to load the GroupedFuzzySet to.

        Returns:
            The loaded GroupedFuzzySet.
        """
        if isinstance(device, str):
            device = torch.device(device)
        modules_list = []
        local_attributes_only: Dict[str, Any] = {}
        for file_path in path.iterdir():
            if ".pickle" in file_path.name:
                # load the remaining attributes
                with open(file_path, "rb") as handle:
                    local_attributes_only.update(pickle.load(handle))
            elif file_path.is_dir():
                for subdirectory in natsorted(file_path.iterdir()):
                    if subdirectory.is_dir():
                        module_path: Path = list(subdirectory.glob("*.pt"))[0]
                        # load the fuzzy set using the fuzzy set's special protocol
                        class_name: str = module_path.name.split(".pt")[0]
                        try:
                            modules_list.append(
                                ContinuousFuzzySet.get_subclass(class_name).load(
                                    module_path, device=device
                                )
                            )
                        except ValueError:
                            # unknown and unrecognized module, but attempt to load the module
                            modules_list.append(torch.load(module_path))
                    else:
                        raise UserWarning(
                            f"Unexpected file found in {file_path}: {module_path}"
                        )
                local_attributes_only[file_path.name] = modules_list

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
        grouped_fuzzy_set: GroupedFuzzySets = cls(
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
            setattr(grouped_fuzzy_set, attr, value)

        return grouped_fuzzy_set

    def calculate_module_responses(self, observations) -> Membership:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of GroupedFuzzySets.
        """
        if len(self.modules_list) > 0:
            # modules' responses are membership degrees when modules are ContinuousFuzzySet
            if len(self.modules_list) == 1:
                # for computational efficiency, return the response from the only module
                return self.modules_list[0](observations)

            # this can be computationally expensive, but it is necessary to calculate the responses
            # from all the modules in the torch.nn.ModuleList of GroupedFuzzySets
            # ideally this should be done in parallel, but it is not possible with the current
            # implementation; only use this if the torch.nn.Module objects are different
            module_elements: List[torch.Tensor] = []
            module_memberships: List[torch.Tensor] = (
                []
            )  # the primary response from the module
            module_masks: List[torch.Tensor] = (
                []
            )  # the secondary response denoting module filter
            for module in self.modules_list:
                membership: Membership = module(observations)
                module_elements.append(membership.elements)
                module_memberships.append(membership.degrees)
                module_masks.append(membership.mask)
            return Membership(
                elements=torch.cat(module_elements, dim=-1),
                degrees=torch.cat(module_memberships, dim=-1),
                mask=torch.cat(module_masks, dim=-1),
            )
        raise ValueError("The torch.nn.ModuleList of GroupedFuzzySets is empty.")

    def expand(
        self, observations, module_responses, module_masks
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expand the GroupedFuzzySets if necessary.
        """
        if self.expandable and self.training:
            # save the data that we have seen
            if self.data_seen.shape[0] == 0:  # buffer is empty, shape of (0, 0)
                self.data_seen = observations
            else:
                self.data_seen = torch.cat([self.data_seen, observations], dim=0)

            if self.data_seen.shape[0] % self.data_limit_until_update == 0:
                # keep a running tally of mins and maxs of the domain
                minimums = self.data_seen.min(dim=0).values
                maximums = self.data_seen.max(dim=0).values

                if (
                    self.minimums.shape[0] == 0 and self.maximums.shape[0] == 0
                ):  # first time
                    self.minimums = minimums
                    self.maximums = maximums
                else:
                    self.minimums = torch.min(minimums, self.minimums).detach()
                    self.maximums = torch.max(maximums, self.maximums).detach()

                # find where the new centers should be added, if any
                # LogGaussian was used, then use following to check for real membership degrees:
                # for module in self.modules_list:
                #     if isinstance(module, LogGaussian) and not isinstance(
                #             module, Gaussian
                #     ):
                #         with torch.no_grad():
                #             assert (
                #                    module_responses.exp() * module_masks
                #             ).max().item() <= 1.0, "Memberships are not in the range [0, 1]."

                exemplars: List[torch.Tensor] = []

                max_peaks: int = 3
                for var_idx in range(self.data_seen.shape[-1]):
                    discovered_exemplars: torch.Tensor = self.evenly_spaced_exemplars(
                        self.data_seen[:, var_idx], max_peaks
                    )
                    if discovered_exemplars.ndim == 1:
                        discovered_exemplars = discovered_exemplars[:, None]

                    num_of_exemplars_found = discovered_exemplars.shape[0]
                    if num_of_exemplars_found < max_peaks:
                        # pad the exemplars with torch.nan if there are not enough exemplars
                        discovered_exemplars = torch.nn.functional.pad(
                            discovered_exemplars,
                            pad=(0, 0, 0, max_peaks - num_of_exemplars_found),
                            value=torch.nan,
                        )

                    exemplars.append(discovered_exemplars.transpose(0, 1))

                if len(exemplars) == 0:
                    # no exemplars found in any dimension
                    return observations, module_responses, module_masks

                exemplars: torch.Tensor = torch.vstack(exemplars).transpose(0, 1)

                # Create a new matrix with nan values
                new_centers = torch.full_like(exemplars, float("nan"))

                # Use torch.where to update values that satisfy the condition
                new_centers = torch.where(
                    self.calculate_module_responses(exemplars)
                    .degrees.exp()
                    .max(
                        dim=-1
                    )  # TODO: assume LogGaussian is used (exp)  # pylint: disable=fixme
                    .values
                    < self.epsilon,
                    exemplars,
                    new_centers,
                )

                if not new_centers.isnan().all():  # add new centers
                    # TODO: find_centers_and_widths call is problematic  # pylint: disable=fixme
                    new_widths: torch.Tensor = find_widths(
                        data_point=new_centers.nan_to_num(0.0).mean(dim=0),
                        minimums=self.minimums,
                        maximums=self.maximums,
                        alpha=0.3,
                    )

                    # new_widths = torch.tensor(
                    #     [term["widths"] for term in terms], device=self.data_seen.device
                    # )

                    # assert new_widths.isnan().any() is False

                    # create the widths for the new centers
                    new_widths = (
                        # only keep the widths for the entries that are not torch.nan
                        ~torch.isnan(new_centers)
                        * new_widths
                    ) + (torch.isnan(new_centers) * -1)

                    # above result is tensor that contains new centers, but also contains torch.nan
                    # in the places where a new center is not needed

                    new_centers = (
                        new_centers.nan_to_num(0.0)
                        .transpose(0, 1)
                        .max(dim=-1, keepdim=True)
                        .values
                    )
                    new_widths = (
                        new_widths.transpose(0, 1).max(dim=-1, keepdim=True).values
                    )

                    # TODO: this code does not work for torch.jit.script  # pylint: disable=fixme
                    # the following assumes only the first module is to be expanded
                    module = self.modules_list[0]
                    module._centers.append(  # pylint: disable=protected-access
                        module.make_parameter(parameter=new_centers)
                    )
                    module._widths.append(  # pylint: disable=protected-access
                        module.make_parameter(parameter=new_widths)
                    )
                    module._mask.append(  # pylint: disable=protected-access
                        module.make_mask(widths=new_widths)
                    )

                    # TODO: this code does not work for torch.jit.script  # pylint: disable=fixme
                    # the following assumes an entire new module is to be added
                    # module_type = type(self.modules_list[0])  # cannot call type
                    # if issubclass(module_type, ContinuousFuzzySet):
                    #     # cannot call .get_subclass
                    #     granule = ContinuousFuzzySet.get_subclass(module_type.__name__)(
                    #         centers=new_centers,
                    #         widths=new_widths,
                    #     )  # cannot dynamically create a PyTorch module in torch.jit.script
                    # else:
                    #     raise ValueError(
                    #         "The module type is not ContinuousFuzzySet, and therefore cannot "
                    #         "be used for dynamic expansion."
                    #     )

                    # granule = LogGaussian(
                    #     centers=new_centers.nan_to_num(0.0)
                    #     .transpose(0, 1)
                    #     .max(dim=-1, keepdim=True)
                    #     .values,
                    #     widths=new_widths.transpose(0, 1)
                    #     .max(dim=-1, keepdim=True)
                    #     .values,
                    #     device=self.data_seen.device
                    # )
                    # print(
                    #     f"add {granule.centers.shape}; modules already: {len(self.modules_list)}"
                    # )
                    # print(f"to dimensions: {set(range(len(sets))) - set(empty_sets)}")
                    # self.modules_list.add_module(str(len(self.modules_list)), granule)

                    # clear the history
                    self.data_seen = torch.empty(0, 0)

                    # reduce the number of torch.nn.Modules in the list for computational efficiency
                    # (this is not necessary, but it is a good idea)
                    # self.prune(module_type)

            (
                _,
                module_responses,
                module_masks,
            ) = self.calculate_module_responses(observations)
        return observations, module_responses, module_masks

    @staticmethod
    def evenly_spaced_exemplars(data: torch.Tensor, max_peaks: int) -> torch.Tensor:
        """
        Find the peaks in the data and return the peaks, or a subset of the peaks if there are
        more than max_peaks.

        Args:
            data: The data to find the peaks in.
            max_peaks: The maximum number of peaks to return.

        Returns:
            The peaks, or a subset of the peaks if there are more than max_peaks.
        """
        # Find the peaks in a 1D tensor
        peaks = (data[1:-1] > data[:-2]) & (data[1:-1] > data[2:])
        peak_indices = torch.nonzero(peaks).squeeze() + 1
        if peak_indices.ndim > 0 and len(peak_indices) <= max_peaks:
            sampled_peak_values = data[peak_indices][
                :, None
            ]  # return the peaks' values
        else:
            sampled_peaks_indices = torch.linspace(
                0, len(peak_indices) - 1, max_peaks, dtype=torch.int
            )
            sampled_peaks = peak_indices[sampled_peaks_indices]
            sampled_peak_values = data[sampled_peaks][:, None]
        return torch.as_tensor(
            sampled_peak_values, dtype=torch.float16, device=data.device
        )

    def prune(self, module_type: Type[ContinuousFuzzySet]) -> None:
        """
        Prune the torch.nn.ModuleList of GroupedFuzzySets by keeping the first module, but
        collapsing the rest of the modules into a single module. This is done to reduce the
        number of torch.nn.Modules in the list for computational efficiency.
        """
        if len(self.modules_list) > 5:
            centers, widths = [], []
            for module in self.modules_list[1:]:
                if module.centers.shape[-1] > 1:
                    centers.append(module.centers.mean(dim=-1, keepdim=True))
                    widths.append(module.widths.max(dim=-1, keepdim=True).values)
                else:
                    centers.append(module.centers)
                    widths.append(module.widths)
            if issubclass(module_type, ContinuousFuzzySet):
                module = ContinuousFuzzySet.get_subclass(module_type.__name__)(
                    centers=torch.cat(centers, dim=-1),
                    widths=torch.cat(widths, dim=-1),
                )
                print(module.centers.shape)
            else:
                raise ValueError(
                    "The module type is not ContinuousFuzzySet, and therefore cannot "
                    "be used for dynamic expansion."
                )
            self.modules_list = torch.nn.ModuleList([self.modules_list[0], module])

    def forward(self, observations) -> Membership:
        """
        Calculate the responses from the modules in the torch.nn.ModuleList of GroupedFuzzySets.
        Expand the GroupedFuzzySets if necessary.
        """
        (
            _,  # module_elements
            module_responses,
            module_masks,
        ) = self.calculate_module_responses(observations)

        # TODO: this code does not work for torch.jit.script  # pylint: disable=fixme
        # self.expand(observations, module_responses, module_masks)

        return Membership(
            elements=observations, degrees=module_responses, mask=module_masks
        )
