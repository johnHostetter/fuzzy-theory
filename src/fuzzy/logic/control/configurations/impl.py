"""
This file contains classes and functions that are used to contain information or specifications
about the Fuzzy Logic Controller (FLC) or Neuro-Fuzzy Network (NFN) to build.
"""

import torch

from fuzzy.relations.t_norm import TNorm

from ..defuzzification import Defuzzification
from .abstract import FuzzySystem
from .data import GranulationLayers, Shape


class Defined(FuzzySystem):
    """
    A defined Fuzzy System. This is helpful for when the Fuzzy System is already defined and
    the components are known. We can use this class to create the Fuzzy System without having
    to use a KnowledgeBase to search for the necessary components.
    """

    def __init__(
        self,
        shape: Shape,
        granulation: GranulationLayers,
        engine: TNorm,
        defuzzification: Defuzzification,
    ):
        self._shape = shape
        self._granulation = granulation
        self._engine = engine
        self._defuzzification = defuzzification

    @property
    def shape(self) -> Shape:
        """
        Get the shape of the Fuzzy System. This is used to ensure that the Fuzzy System is built
        correctly and that the KnowledgeBase contains the correct number of fuzzy sets.
        """
        return self._shape

    @property
    def granulation_layers(self) -> GranulationLayers:
        """
        Create the granulation layers and the inference engine for the Fuzzy System.

        Returns:
            The granulation layers (e.g., premise, consequence).
        """
        return self._granulation

    @property
    def engine(self) -> TNorm:
        """
        Create the inference engine for the Fuzzy System.

        Returns:
            The inference engine for the Fuzzy System.
        """
        return self._engine

    def defuzzification(self, cls_type, device: torch.device) -> Defuzzification:
        return self._defuzzification
