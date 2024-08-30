"""
This file contains classes and functions that are used to contain information or specifications
about the Fuzzy Logic Controller (FLC) or Neuro-Fuzzy Network (NFN) to build.
"""

import abc
from typing import TypedDict, NamedTuple

import torch

from fuzzy.relations.t_norm import TNorm
from fuzzy.sets.group import FuzzySetGroup


class Shape(NamedTuple):
    """
    The shape that a Fuzzy Logic Controller (FLC) or Neuro-Fuzzy Network (NFN) should follow in
    their calculations. This is a named tuple that contains the (number of input variables, number
    of input variable terms, number of fuzzy logic rules, number of output variable, number of
    output variable terms) in that exact order.

    This is used to ensure that the FLC or NFN is built correctly and that the KnowledgeBase
    contains the correct number of fuzzy sets. The choice to put variables first and then terms
    comes from this is how fuzzy sets operate in the library, so this applies even for the output
    variable, though it might be more accurate for the output term layer to occur before the output
    variable layer.

    Example:
    ```
    shape = Shape(2, 3, 100, 1, 5)
    ```

    This shape represents a FLC with 2 input variables, each with 3 terms, 100 rules, 1 output
    variable with 5 terms.
    """

    n_inputs: int
    n_input_terms: int
    n_rules: int
    n_outputs: int
    n_output_terms: int


class GranulationLayers(TypedDict):
    """
    A dictionary that contains the input and output granulation layers. The input granulation
    layer is a FuzzySetGroup object that contains the input granules. The output granulation
    layer is a FuzzySetGroup object that contains the output granules. If the layer is None,
    then it is not defined and will be created during the construction of the FLC by searching
    the KnowledgeBase for the appropriate granules.
    """

    input: FuzzySetGroup
    output: FuzzySetGroup


class FuzzySystem(abc.ABC):
    """
    The abstract class that defines the interface for a Fuzzy System. This is used to ensure that
    the FLC and NFN classes can be used interchangeably in the library. This is useful for
    constructing the FLC or NFN in a similar way, as well as for defining the inference engine
    that is used to make predictions.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Shape:
        """
        Get the shape of the Fuzzy System. This is used to ensure that the Fuzzy System is built
        correctly and that the KnowledgeBase contains the correct number of fuzzy sets.
        """

    @abc.abstractmethod
    def granulation_layers(self, device: torch.device) -> GranulationLayers:
        """
        Create the granulation layers and the inference engine for the Fuzzy System.

        Args:
            device: The device to use.

        Returns:
            The granulation layers (e.g., premise, consequence).
        """

    @abc.abstractmethod
    def engine(self, device: torch.device) -> TNorm:
        """
        Create the inference engine for the Fuzzy System.

        Args:
            device: The device to use.

        Returns:
            The inference engine for the Fuzzy System.
        """
