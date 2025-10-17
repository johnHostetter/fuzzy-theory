"""
This file contains classes and functions that are used to contain information or specifications
about the Fuzzy Logic Controller (FLC) or Neuro-Fuzzy Network (NFN) to build.
"""

import abc

import torch

from fuzzy.logic.control.defuzzification import Defuzzification
from fuzzy.relations.t_norm import TNorm

from .data import GranulationLayers, Shape


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

    @property
    @abc.abstractmethod
    def granulation_layers(self) -> GranulationLayers:
        """
        Create the granulation layers and the inference engine for the Fuzzy System.

        Returns:
            The granulation layers (e.g., premise, consequence).
        """

    @property
    @abc.abstractmethod
    def engine(self) -> TNorm:
        """
        Fetch the inference engine for the Fuzzy System.

        Returns:
            The inference engine for the Fuzzy System.
        """

    def defuzzification(self, cls_type, device: torch.device) -> Defuzzification:
        """
        Create the defuzzification engine for the Fuzzy System.

        Args:
            cls_type: The type of defuzzification engine to use.
            device: The device to use.

        Returns:
            The defuzzification engine for the Fuzzy System.
        """
        granulation_layers: GranulationLayers = self.granulation_layers
        return cls_type(
            shape=self.shape,
            source=granulation_layers["output"],
            device=device,
            rule_base=self.rule_base if hasattr(self, "rule_base") else None,
        )

    @shape.setter
    def shape(self, value):
        self._shape = value
