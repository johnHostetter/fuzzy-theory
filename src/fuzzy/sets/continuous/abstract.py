"""
Implements an abstract class called ContinuousFuzzySet using PyTorch. All fuzzy sets defined over
a continuous domain are derived from this class. Further, the Membership class is defined within,
which contains a helpful interface understanding membership degrees.
"""

import abc
import inspect
from pathlib import Path
from abc import abstractmethod
from typing import List, NoReturn, Union, MutableMapping, Any, Type, Tuple

import sympy
import torch
import torchquad
import numpy as np

# import scienceplots is used via plt.style.context(["science", "no-latex", "high-contrast"])
import scienceplots  # noqa # pylint: disable=unused-import
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchquad.utils.set_up_backend import set_up_backend

from .membership import Membership
from ...utils import check_path_to_save_torch_module, TorchJitModule


class ContinuousFuzzySet(TorchJitModule, metaclass=abc.ABCMeta):
    """
    A generic and abstract torch.nn.Module class that implements continuous fuzzy sets.

    This is the most important Python class regarding fuzzy sets within this Soft Computing library.

    Defined here are most of the common methods made available to all fuzzy sets. Fuzzy sets that
    will later be used in other features such as neuro-fuzzy networks are expected to abide by the
    conventions outlined within. For example, parameters 'centers' and 'widths' are often expected,
    but inference engines (should) only rely on the fuzzy set membership degrees.

    However, for convenience, some aspects of the SelfOrganize code may search for vertices that
    have attributes of type 'ContinuousFuzzySet'. Thus, if it is pertinent that a vertex within
    the KnowledgeBase is recognized as a fuzzy set, it is very likely one might be interested in
    inheriting or extending from ContinuousFuzzySet.
    """

    def __init__(
        self,
        centers: np.ndarray,
        widths: np.ndarray,
        device: torch.device,
        use_sparse_tensor=False,
    ):
        super().__init__()
        self.device = device
        if not isinstance(centers, np.ndarray):
            # ensure that the centers are a numpy array (done for consistency)
            # specifically, we want to internally control the dtype and device of the centers
            raise ValueError(
                f"The centers of a ContinuousFuzzySet must be a numpy array, "
                f"but got {type(centers)}"
            )
        if not isinstance(widths, np.ndarray):
            # ensure that the widths are a numpy array (done for consistency)
            # specifically, we want to internally control the dtype and device of the widths
            raise ValueError(
                f"The widths of a ContinuousFuzzySet must be a numpy array, but got {type(widths)}"
            )

        if centers.ndim != widths.ndim:
            raise ValueError(
                f"The number of dimensions for the centers ({centers.ndim}) and widths "
                f"({widths.ndim}) must be the same."
            )

        if centers.ndim == 0 or widths.ndim == 0:
            raise ValueError(
                f"The centers and widths of a ContinuousFuzzySet must have at least one dimension. "
                f"Centers has {centers.ndim} dimensions and widths has {widths.ndim} dimensions."
            )

        if centers.ndim == 1:
            # assuming that the array is a single linguistic variable
            centers = centers[None, :]
        if widths.ndim == 1:
            # assuming that the array is a single linguistic variable
            widths = widths[None, :]

        # avoid allocating new memory for the centers and widths
        # use torch.float32 to save memory and speed up computations
        self._centers = torch.nn.ParameterList([self.make_parameter(centers)])
        self._widths = torch.nn.ParameterList([self.make_parameter(widths)])
        self.use_sparse_tensor = use_sparse_tensor
        # self._mask = torch.nn.ParameterList(
        #     [
        #         self.make_mask(widths)
        #     ]
        # )
        self._mask = [self.make_mask(widths)]

    def to(self, *args, **kwargs):
        """
        Move the ContinuousFuzzySet to a new device.

        Returns:
            None
        """
        # Call the parent class's `to` method to handle parameters and submodules
        super().to(*args, **kwargs)

        # special handling for the non-parameter tensors, such as mask
        self._mask = [mask.to(*args, **kwargs) for mask in self._mask]
        return self

    def make_parameter(self, parameter: np.ndarray) -> torch.nn.Parameter:
        """
        Create a torch.nn.Parameter from a numpy array, with the appropriate dtype and device.

        Args:
            parameter: The numpy array to convert to a torch.nn.Parameter (e.g., centers or widths).

        Returns:
            A torch.nn.Parameter object.
        """
        return torch.nn.Parameter(
            torch.as_tensor(parameter, dtype=torch.float32, device=self.device),
            requires_grad=True,  # explicitly set to True
        )

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
        return torch.as_tensor(widths > 0.0, dtype=torch.int8, device=self.device)
        # return torch.nn.Parameter(
        #     torch.as_tensor(widths > 0.0, dtype=torch.int8, device=self.device),
        #     requires_grad=False,  # explicitly set to False (mask is not trainable)
        # )

    @classmethod
    def create(
        cls,
        n_variables: int,
        n_terms: int,
        device: torch.device,
        **kwargs,
    ) -> Union[NoReturn, "ContinuousFuzzySet"]:
        """
        Create a fuzzy set with the given number of variables and terms, where each variable
        has the same number of terms. For example, if we have two variables, then we might have
        three terms for each variable, such as "low", "medium", and "high". This would result in
        a total of nine fuzzy sets. The centers and widths are initialized randomly.

        Args:
            n_variables: The number of variables.
            n_terms: The number of terms.
            device: The device to use.

        Returns:
            A ContinuousFuzzySet object, or a NotImplementedError if the method is not implemented.
        """
        if inspect.isabstract(cls):
            # this error is thrown if the class is abstract, such as ContinuousFuzzySet, but
            # the method is not implemented (e.g., self.calculate_membership)
            raise NotImplementedError(
                "The ContinuousFuzzySet has no defined membership function. Please create a class "
                "and inherit from ContinuousFuzzySet, or use a predefined class, such as Gaussian."
            )

        if isinstance(device, str):
            device = torch.device(device)

        centers: np.ndarray = np.random.randn(n_variables, n_terms)
        widths: np.ndarray = np.random.randn(n_variables, n_terms)
        return cls(centers=centers, widths=widths, device=device, **kwargs)

    def __hash__(self):
        """
        Hash the fuzzy set.

        Returns:
            The hash of the fuzzy set.
        """
        return hash((type(self), self.get_centers(), self.get_widths()))

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

    def get_centers(self) -> torch.Tensor:
        """
        Get the concatenated centers of the fuzzy set from its corresponding ParameterList.

        Returns:
            The concatenated centers of the fuzzy set.
        """
        return torch.cat(list(self._centers), dim=-1)

    def get_widths(self) -> torch.Tensor:
        """
        Get the concatenated widths of the fuzzy set from its corresponding ParameterList.

        Returns:
            The concatenated widths of the fuzzy set.
        """
        return torch.cat(list(self._widths), dim=-1)

    def get_mask(self) -> torch.Tensor:
        """
        Get the concatenated mask of the fuzzy set from its corresponding ParameterList.

        Returns:
            The concatenated mask of the fuzzy set.
        """
        return torch.cat(list(self._mask), dim=-1)

    @classmethod
    def render_formula(cls) -> sympy.Expr:
        """
        Render of the fuzzy set's membership function.

        Note: This is more beneficial for Python Console or Jupyter Notebook usage.

        Returns:
            Render of the fuzzy set's membership function.
        """
        sympy.init_printing(use_unicode=True)
        return cls.sympy_formula()

    @classmethod
    def latex_formula(cls) -> str:
        """
        String LaTeX representation of the fuzzy set's membership function.

        Note: This is more beneficial for animations or LaTeX documents.

        Returns:
            The LaTeX representation of the fuzzy set's membership function.
        """
        return sympy.latex(cls.sympy_formula())

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
        state_dict["centers"] = self.get_centers()  # concatenate the centers
        state_dict["widths"] = self.get_widths()  # concatenate the widths
        state_dict["mask"] = self.get_mask()  # currently not used
        state_dict["class_name"] = self.__class__.__name__
        torch.save(state_dict, path)
        return state_dict

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "ContinuousFuzzySet":
        """
        Load the fuzzy set from a file and put it on the specified device.

        Returns:
            None
        """
        state_dict: MutableMapping = torch.load(path, weights_only=False)
        centers = state_dict.pop("centers")
        widths = state_dict.pop("widths")
        class_name = state_dict.pop("class_name")
        return cls.get_subclass(class_name)(
            centers=centers.cpu().detach().numpy(),
            widths=widths.cpu().detach().numpy(),
            device=device,
        )

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
                f"The mode must be either 'horizontal' or 'vertical', but got {mode}"
            )
        with torch.no_grad():
            self._centers[0] = torch.nn.Parameter(
                method_of_extension([self._centers[0], centers])
            )
            self._widths[0] = torch.nn.Parameter(
                method_of_extension([self._widths[0], widths])
            )

    def _area_helper(self, fuzzy_sets) -> List[List[float]]:
        """
        Splits the fuzzy set (if representing a fuzzy variable) into individual fuzzy sets (the
        fuzzy variable's possible fuzzy terms), and does so recursively until the base case is
        reached. Once the base case is reached (i.e., a single fuzzy set), the area under its
        curve within the integration_domain is calculated. The result is a

        Args:
            fuzzy_sets: The fuzzy set to split into smaller fuzzy sets.

        Returns:
            A list of floats.
        """
        all_areas: List[List[float]] = []
        for variable_params in zip(fuzzy_sets.get_centers(), fuzzy_sets.get_widths()):
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

                # Enable GPU support if available and set the floating point precision
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
                    # a zero will substitute to be sure that this issue does not affect results
                    area = 0.0
                variable_areas.append(area)
            all_areas.append(variable_areas)
        return all_areas

    def area(self) -> torch.Tensor:
        """
        Calculate the area beneath the fuzzy curve (i.e., membership function) using torchquad.

        This is a slightly expensive operation, but it is used for approximating the Mamdani fuzzy
        inference with arbitrary continuous fuzzy sets.

        Typically, the results will be cached somewhere, so that the area value can be reused.

        Returns:
            torch.Tensor
        """
        return torch.tensor(
            self._area_helper(self), device=self.device, dtype=torch.float32
        )

    def split_by_variables(self) -> Union[list, List[Type["ContinuousFuzzySet"]]]:
        """
        This operation takes the ContinuousFuzzySet and converts it to a list of ContinuousFuzzySet
        objects, if applicable. For example, rather than using a single Gaussian object to represent
        all Gaussian membership functions in the input space, this function will convert that to a
        list of Gaussian objects, where each Gaussian function is defined and restricted to a single
        input dimension. This is particularly helpful when modifying along a specific dimension.

        Returns:
            A list of ContinuousFuzzySet objects, where the length is equal to the number
            of input dimensions.
        """
        variables = []
        for centers, widths in zip(self.get_centers(), self.get_widths()):
            centers = centers.cpu().detach().tolist()
            widths = widths.cpu().detach().tolist()

            # the centers and widths must be trimmed to remove missing fuzzy set placeholders
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

    def plot(
        self, output_dir: Path, selected_terms: List[Tuple[int, int]] = None
    ) -> None:
        """
        Plot the fuzzy set.

        Args:
            output_dir: The path to the directory where to save the plot(s).
            selected_terms: The terms to highlight in the plot.

        Returns:
            None
        """
        if selected_terms is None:
            selected_terms = []

        with plt.style.context(["science", "no-latex", "high-contrast"]):
            for variable_idx in range(self.get_centers().shape[0]):
                _, ax = plt.subplots(1, figsize=(6, 4), dpi=100)
                mpl.rcParams["figure.figsize"] = (6, 4)
                mpl.rcParams["figure.dpi"] = 100
                mpl.rcParams["savefig.dpi"] = 100
                mpl.rcParams["font.size"] = 20
                mpl.rcParams["legend.fontsize"] = "medium"
                mpl.rcParams["figure.titlesize"] = "medium"
                mpl.rcParams["lines.linewidth"] = 2
                ax.tick_params(width=2, length=6)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                real_centers: List[float] = [
                    self.get_centers()[variable_idx, term_idx].item()
                    for term_idx, mask_value in enumerate(self.get_mask()[variable_idx])
                    if mask_value == 1
                ]
                real_widths: List[float] = [
                    self.get_widths()[variable_idx, term_idx].item()
                    for term_idx, mask_value in enumerate(self.get_mask()[variable_idx])
                    if mask_value == 1
                ]
                x_values = torch.linspace(
                    min(real_centers) - 2 * max(real_widths),
                    max(real_centers) + 2 * max(real_widths),
                    steps=1000,
                    device=self.device,
                )

                if self.get_centers().ndim == 1 or self.get_centers().shape[0] == 1:
                    x_values = x_values[:, None]
                elif self.get_centers().ndim == 2 or self.get_centers().shape[0] > 1:
                    x_values = x_values[:, None, None]

                memberships: torch.Tensor = self.calculate_membership(x_values)

                if memberships.ndim == 2:
                    memberships = memberships.unsqueeze(
                        dim=1
                    )  # add a temporary dimension for the variable

                memberships = memberships.cpu().detach().numpy()
                x_values = x_values.squeeze().cpu().detach().numpy()

                for term_idx in range(memberships.shape[-1]):
                    if self.get_mask()[variable_idx, term_idx] == 0:
                        continue  # not a real fuzzy set
                    y_values = memberships[:, variable_idx, term_idx]
                    label: str = (
                        r"$\mu_{"
                        + str(variable_idx + 1)
                        + ","
                        + str(term_idx + 1)
                        + "}$"
                    )
                    if (variable_idx, term_idx) in selected_terms:
                        # edgecolor="#0bafa9"  # beautiful with facecolor=None  (AAMAS 2023)
                        plt.fill_between(
                            x_values, y_values, alpha=0.5, hatch="///", label=label
                        )
                    else:
                        plt.plot(x_values, y_values, alpha=0.5, label=label)
                plt.legend(
                    bbox_to_anchor=(0.5, -0.2),
                    loc="upper center",
                    ncol=len(real_centers),
                )
                plt.subplots_adjust(bottom=0.3, wspace=0.33)
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f"mu_{variable_idx}.png")
                plt.clf()

    @staticmethod
    def count_granule_terms(granules: List["ContinuousFuzzySet"]) -> np.ndarray:
        """
        Count the number of granules that occur in each dimension.

        Args:
            granules: A list of granules, where each granule is a ContinuousFuzzySet object.

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
    def stack(
        granules: List["ContinuousFuzzySet"],
    ) -> "ContinuousFuzzySet":
        """
        Create a condensed and stacked representation of the given granules.

        Args:
            granules: A list of granules, where each granule is a ContinuousFuzzySet object.

        Returns:
            A ContinuousFuzzySet object.
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
                            ContinuousFuzzySet.count_granule_terms(granules).max()
                            - params.get_centers().shape[-1],
                        ),
                        mode="constant",
                        value=missing_center,
                    )
                    if params.get_centers().dim() > 0
                    else torch.tensor(missing_center).repeat(
                        ContinuousFuzzySet.count_granule_terms(granules).max()
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
                            ContinuousFuzzySet.count_granule_terms(granules).max()
                            - params.get_widths().shape[-1],
                        ),
                        mode="constant",
                        value=missing_width,
                    )
                    if params.get_centers().dim() > 0
                    else torch.tensor(missing_center).repeat(
                        ContinuousFuzzySet.count_granule_terms(granules).max()
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
    @abstractmethod
    def sympy_formula(cls) -> sympy.Expr:
        """
        The abstract method that defines the membership function of the fuzzy set using sympy.

        Returns:
            A sympy.Expr object that represents the membership function of the fuzzy set.
        """
        raise NotImplementedError("The sympy_formula method must be implemented.")

    @abc.abstractmethod
    def forward(self, observations) -> Membership:
        """
        Forward pass of the function. Applies the function to the input elementwise.

        Args:
            observations: Two-dimensional matrix of observations,
            where a row is a single observation and each column
            is related to an attribute measured during that observation.

        Returns:
            The membership degrees of the observations for the Gaussian fuzzy set.
        """
        raise NotImplementedError(
            "The ContinuousFuzzySet has no defined forward function. Please create a class and "
            "inherit from ContinuousFuzzySet, or use a predefined class, such as Gaussian."
        )
