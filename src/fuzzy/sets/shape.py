"""
Describes the shape of homogeneous fuzzy sets and the built-in strategies for
initializing their centers and widths from that shape.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Union

import numpy as np
from numpy import ndarray
from numpy._typing import _64Bit


@dataclass(frozen=True)
class FuzzySetShape:
    """
    A dataclass containing information about the shape of homogeneous fuzzy sets.
    """

    n_variables: int
    n_terms: int


@dataclass
class FuzzySetInitResult:
    """
    A dataclass representing an initialization result concerning the centers and widths of fuzzy
    set(s).
    """

    centers: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]]
    widths: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]]


@dataclass
class MembershipConfig:
    """
    This dataclass assists in configuring how to handle membership calculations, such as whether
    to represent them with PyTorch's sparse tensors and/or cache them for follow-up retrieval.
    """

    enable_sparse: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable sparse tensors to represent membership degrees."
        },
    )
    cache_membership: bool = field(
        default=False, metadata={
            "help": "Whether to cache the membership of the fuzzy sets."}, )
    membership_cache_size: int = field(
        default=2, metadata={
            "help": "The size of membership cache for fuzzy sets."})


class FuzzySetInitMethod(Enum):
    """
    An extended Enum that offers various built-in functionality for basic fuzzy set initialization
    methods.
    """

    RANDOM = auto()
    LINEAR = auto()

    def initialize(
        self,
        shape: FuzzySetShape,
        init_width: float = 1.0,
    ) -> FuzzySetInitResult:
        """
        Perform a basic initialization of parameters for fuzzy sets.

        Args:
            shape: The shape of these fuzzy sets.
            init_width: The initial width to use for each fuzzy set.

        Returns:
            An initialization result containing parameters, such as centers and widths.
        """
        if self is FuzzySetInitMethod.RANDOM:
            centers: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]] = (
                np.random.randn(shape.n_variables, shape.n_terms))
            widths: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]] = np.abs(
                np.random.randn(shape.n_variables, shape.n_terms)).clip(min=0.1)

        elif self is FuzzySetInitMethod.LINEAR:
            base = np.linspace(0.0, 1.0, num=shape.n_terms)
            centers: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]] = (
                np.repeat(base[None, :], repeats=shape.n_variables, axis=0))
            widths: Union[float, ndarray[Any, np.dtype[np.floating[_64Bit]]]] = np.full(
                (shape.n_variables, shape.n_terms), init_width, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported method: {self}")

        return FuzzySetInitResult(centers, widths)
