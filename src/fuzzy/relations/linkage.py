"""
This file helps support the linkage between relations necessary for fuzzy logic inference engines.
"""

from pathlib import Path
from typing import Any, Callable, List, MutableMapping, Optional, Union

import numpy as np
import torch
from torch._C import Size

from fuzzy.sets.membership import Membership
from fuzzy.utils import NestedTorchJitModule, check_path_to_save_torch_module
from fuzzy.utils.classes import Loggable
from fuzzy.utils.functions import log_classmethod, log_method


class BinaryLinks(torch.nn.Module, Loggable):
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
        # indices: torch.Tensor = torch.tensor(links.nonzero(), device=device)
        # self.links = torch.sparse_coo_tensor(
        #     indices=indices, values=torch.ones(indices.shape[1], dtype=torch.bool, device=device),
        #     size=links.shape,
        # )
        # the below is VALID but NOT compatible w/ autograd
        # store term selections as integers for major memory savings; each variable -> selected term
        # self.original_links: np.ndarray = links
        # cast_links = links.astype(dtype=float)
        # cast_links[cast_links == 0] = 'nan'
        # self.memory_efficient_links: np.ndarray = np.nanargmax(
        #     links, axis=1
        # )  # 2D shape: (n_inputs, n_relations)
        # self.links: torch.Tensor = torch.tensor(
        #     self.memory_efficient_links, dtype=torch.int8, device=device
        # )
        self.device: torch.device = device

    # @log_method
    def __hash__(self) -> int:
        return hash(self.links)

    # @log_method
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BinaryLinks) and torch.equal(
            self.links.to_dense(), other.links.to_dense()
        )

    @property
    # @log_method
    def shape(self) -> Size:
        """
        Get the shape of the binary links.
        """
        return self.links.shape

    # @log_method
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
    # @log_classmethod
    def load(cls, path: Path, device: torch.device) -> "BinaryLinks":
        """
        Load the n-ary relation from a file and put it on the specified device.

        Returns:
            None
        """
        state_dict: MutableMapping = torch.load(path, weights_only=False)
        links = state_dict.pop("links")
        return cls(links, device, **state_dict)

    # @log_method
    def to(self, *args, **kwargs) -> "BinaryLinks":
        """
        Move the BinaryLinks to a new device.

        Returns:
            The BinaryLinks object.
        """
        # Call the parent class's `to` method to handle parameters and
        # submodules
        super().to(*args, **kwargs)

        # special handling for the non-parameter tensors, such as mask
        self.links = self.links.to(*args, **kwargs)
        self.device = self.links.device
        return self

    # @log_method
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


class GroupedLinks(NestedTorchJitModule, Loggable):
    """
    This class is a container for the various LogitLinks or BinaryLinks that are used to
    probabilistically sample from the fuzzy sets along some dimension. This class is defined as a
    torch.nn.Module for compatibility with torch.nn.ModuleList among other helpful necessities. More
    specifically, this class enables us to use other code that may expect torch.nn.Module
    functionality, such as GumbelSoftmax.
    """

    def __init__(
        self,
        *args,
        modules_list: Union[None, List[torch.nn.Module]],
        debug: bool = False,
        callback: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.create_logger(name=self.__class__.__name__, debug=debug)
        if modules_list is None:
            modules_list = []
        self.modules_list = torch.nn.ModuleList(modules_list)
        self.membership_dimension: int = 1
        self.callback = callback
        self._compute_shape_cache()
        # self.streams = [torch.cuda.Stream() for _ in range(len(self.modules_list))]

    @property
    def shape(self):
        return self._cached_shape

    def append(self, module: torch.nn.Module):
        self.modules_list.append(module)
        self._compute_shape_cache()

    def extend(self, modules: List[torch.nn.Module]):
        self.modules_list.extend(modules)
        self._compute_shape_cache()

    def _compute_shape_cache(self) -> Size:
        base_shape = self.modules_list[0].shape
        # this is efficient and works if modules inside self.modules_list are only added when
        # neurogenesis is enabled and self.membership_dimension = 1, but if you need it to work with
        # splitting up logits/links for the purpose of intra-GPU parallelism, then you need to
        # change self.membership_dimension = 2
        # self.membership_dimension = 2
        dim_sum = sum(
            module.shape[self.membership_dimension] for module in self.modules_list[1:]
        )
        shape = tuple(
            s + dim_sum if i == self.membership_dimension else s
            for i, s in enumerate(base_shape)
        )
        self._cached_shape = torch.Size(shape)
        if self.callback:
            self.callback()
        return self._cached_shape

    # @log_method
    def to(self, *args, **kwargs):
        """
        Move the GroupedLinks to a new device.

        Returns:
            None
        """
        # Call the parent class's `to` method to handle parameters and
        # submodules
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

    # @log_method
    def forward(self, membership: Membership) -> torch.Tensor:
        """
        Fetch the links for later use.
        """
        # if torch.is_grad_enabled():
        #     assert (
        #         membership.degrees.grad_fn is not None
        #     ), "The membership degrees must have a grad_fn."
        if len(self.modules_list) == 1:
            return self.modules_list[0](membership)
        all_links: List[Union[torch.Tensor, torch.nn.Parameter]] = []
        # out_tensor = torch.empty(*self.shape, device=self.modules_list[0].logits.device)
        # for idx, links in enumerate(self.modules_list):
        #     with torch.cuda.stream(self.streams[idx]):
        #         start = idx * 64
        #         end =  ((idx + 1) * 64)
        # out_tensor[:, :, start:end] = links(membership)  # (option 1) for
        # stream

        for links in self.modules_list:
            # out_tensor[:, start:end, :] = links(membership)  # (option 2) for
            # expanding mu
            all_links.append(links(membership))
        # torch.cuda.synchronize(device=self.modules_list[0].logits.device)
        # return out_tensor
        return torch.cat(all_links, dim=self.membership_dimension)
