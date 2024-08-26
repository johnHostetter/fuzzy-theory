"""
This file helps support the linkage between relations necessary for fuzzy logic inference engines.
"""

from pathlib import Path
from typing import List, Union, MutableMapping, Any  # , Callable, Tuple

import torch
import numpy as np
from torch._C import Size

from fuzzy.sets.continuous.membership import Membership
from fuzzy.utils import check_path_to_save_torch_module, NestedTorchJitModule


class BinaryLinks(torch.nn.Module):
    """
    This class will implement the 'standard' neuro-fuzzy network definition, where connections or
    edges between layers of a neuro-fuzzy network can only have a value of either 0 or 1.

    Disclaimer: This is useful when the neuro-fuzzy network architecture has been defined a priori
    to training, such as through the SelfOrganize functionality, but is *not* to be used when
    performing network morphism.
    """

    def __init__(self, links: np.ndarray, device: torch.device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.links: torch.Tensor = torch.tensor(links, dtype=torch.int8, device=device)
        self.device: torch.device = device

    def __hash__(self) -> int:
        return hash(self.links)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BinaryLinks) and torch.equal(self.links, other.links)

    @property
    def shape(self) -> Size:
        """
        Get the shape of the binary links.
        """
        return self.links.shape

    def save(self, path: Path) -> MutableMapping[str, Any]:
        """
        Save the n-ary relation to a dictionary.

        Args:
            The path to save the n-ary relation.

        Returns:
            The dictionary representation of the n-ary relation.
        """
        check_path_to_save_torch_module(path)
        state_dict: MutableMapping = self.state_dict()
        state_dict["links"] = self.links.cpu().numpy()
        torch.save(state_dict, path)
        return state_dict

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "BinaryLinks":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Returns:
            None
        """
        state_dict: MutableMapping = torch.load(path, weights_only=False)
        links = state_dict.pop("links")
        return cls(links, device, **state_dict)

    def to(self, *args, **kwargs) -> "BinaryLinks":
        """
        Move the BinaryLinks to a new device.

        Returns:
            The BinaryLinks object.
        """
        # Call the parent class's `to` method to handle parameters and submodules
        super().to(*args, **kwargs)

        # special handling for the non-parameter tensors, such as mask
        self.links = self.links.to(*args, **kwargs)
        return self

    def forward(self, *_) -> torch.Tensor:
        """
        Apply the defined binary linkage to the given membership degrees.

        Note: This 'forward' function is primarily intended for use with membership degrees
        describing the relationship to the premise layer.

        Returns:
            The membership degrees having been appropriately unsqueezed and applied to their
            respective dimensions for later use in inferring rule activation.
        """
        return self.links


class GroupedLinks(NestedTorchJitModule):
    """
    This class is a container for the various LogitLinks or BinaryLinks that are used to
    probabilistically sample from the fuzzy sets along some dimension. This class is defined as a
    torch.nn.Module for compatibility with torch.nn.ModuleList among other helpful necessities. More
    specifically, this class enables us to use other code that may expect torch.nn.Module
    functionality, such as GumbelSoftmax.
    """

    def __init__(self, modules_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if modules_list is None:
            modules_list = []
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.membership_dimension = -1

    @property
    def shape(self) -> Size:
        """
        Get the shape of links.
        """
        shape = list(self.modules_list[0].shape)
        for module in self.modules_list[1:]:
            shape[self.membership_dimension] += module.shape[self.membership_dimension]
        return torch.Size(shape)

    def to(self, *args, **kwargs):
        """
        Move the GroupedLinks to a new device.

        Returns:
            None
        """
        # Call the parent class's `to` method to handle parameters and submodules
        super().to(*args, **kwargs)

        # special handling for the non-parameter tensors, such as mask
        # don't use self.modules_list.to(*args, **kwargs) as it will not work
        self.modules_list = torch.nn.ModuleList(
            [module.to(*args, **kwargs) for module in self.modules_list]
        )
        return self

    # pylint: disable=fixme
    # TODO: add this back in
    # def expand_logits_if_necessary(
    #     self,
    #     membership_degrees: torch.Tensor,
    #     values: str = "random",
    #     callback: Callable[[Tuple[int, int, int]], torch.nn.Module] = None,
    #     membership_dimension: int = -1,
    # ) -> bool:
    #     """
    #     Determine if there is a mismatch between the incoming membership degrees and
    #     the currently
    #     stored logits. In other words, if it appears in the membership degrees that a fuzzy set
    #     has been introduced or created, then the logits probabilistically sampling from those
    #     fuzzy
    #     sets along some dimension will not be 'aware' of this new fuzzy set yet. As such, we
    #     need to
    #     expand the defined logits to account for this newly created fuzzy set.
    #
    #     Args:
    #         membership_degrees: The membership degrees to some fuzzy sets.
    #         values: Whether the new logits should follow a real-number value convention
    #         or binary
    #         value convention. If values is 'random', then real-number values will be used;
    #         this is
    #         useful for when Gumbel Softmax trick is being applied. If values is 'zero', then
    #         binary values will be used, but all the values are initialized as zero. This is
    #         helpful
    #         when we are using predefined rule premises, but a new fuzzy set has been added in the
    #         premise layer that we must accommodate for. However, the network will *not* use the
    #         newly added premise (as there is no connection to it, hence the zeroes).
    #         membership_dimension: The membership dimension under consideration. For example,
    #         whether
    #         to perform this operation along the number of inputs dimension. This is typically the
    #         desired behavior, but which dimension refers to number of inputs may change from
    #         code to
    #         code.
    #
    #     Returns:
    #         None
    #     """
    #     existing_logits: torch.Tensor = self(None)
    #     difference_between_shapes: int = (
    #         membership_degrees.shape[-1] - existing_logits.shape[membership_dimension]
    #     )
    #     if difference_between_shapes > 0:
    #         print("expanding the logits")
    #         new_shape = list(existing_logits.shape)
    #         new_shape[membership_dimension] = difference_between_shapes
    #         if "random" in values.lower():
    #             new_logits = LogitLinks(
    #                 logits=torch.ones(new_shape, device=existing_logits.device).abs()
    #                 * existing_logits.max(),
    #                 # temperature=self.temperature,
    #             )
    #         elif "zero" in values.lower():
    #             new_logits = BinaryLinks(
    #                 links=torch.zeros(new_shape),
    #                 device=existing_logits.device,
    #                 # temperature=self.temperature,
    #             )
    #         else:
    #             raise ValueError(f"The given values is not recognized: {values}")
    #         new_idx: int = len(self.modules_list)
    #         self.modules_list.add_module(str(new_idx), new_logits)
    #     elif difference_between_shapes < 0:
    #         print("cutting down the logits")
    #         # cut down the logits to match the membership degrees
    #         new_logits = existing_logits[:, :, 0 : membership_degrees.shape[-1]]
    #         self.modules_list = torch.nn.ModuleList(
    #             [LogitLinks(logits=new_logits).cuda()]
    #         )
    #         assert membership_degrees.shape[-1] == existing_logits.shape[-1], (
    #             "The membership degrees have a larger shape than the logits. "
    #             "This should not be possible, as the logits should be truncated to "
    #             "account for any newly removed fuzzy sets."
    #         )
    #     return difference_between_shapes > 0

    def forward(self, membership: Membership) -> torch.Tensor:
        """
        Fetch the links for later use.
        """
        all_links: List[Union[torch.Tensor, torch.nn.Parameter]] = []
        for links in self.modules_list:
            all_links.append(links(membership))
        return torch.cat(all_links, dim=self.membership_dimension)
