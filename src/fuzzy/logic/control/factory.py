"""
Contains various classes necessary for Fuzzy Logic Controllers (FLCs) to function properly,
as well as the Fuzzy Logic Controller (FLC) itself.

This Python module also contains functions for extracting information from a knowledge base
(to avoid circular dependency). The functions are used to extract premise terms, consequence terms,
and fuzzy logic rule matrices. These components may then be used to create a fuzzy inference system.
"""

from typing import Union, Tuple, Any, Type

import torch
import igraph

from fuzzy.relations.continuous.t_norm import TNorm
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.sets.continuous.group import GroupedFuzzySets
from .configurations import (
    Specifications,
    GranulationLayers,
)
from .. import RuleBase, Defuzzification


class ControlFactory:
    """
    A factory to create Fuzzy Logic Controllers (FLCs) depending on the source of the information.
    """

    def __init__(self, source: Any, *args, **kwargs):
        """
        Initialize the ControlFactory object.

        Args:
            source: The source of the information (e.g., a KnowledgeBase)
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.source: Any = source

    @staticmethod
    def from_knowledge_base(
        knowledge_base: KnowledgeBase,
        defuzzification: Type[Defuzzification],
        device: torch.device,
    ) -> Tuple[GranulationLayers, TNorm, Defuzzification]:
        """
        Create the components needed for a Fuzzy Logic Controller (FLC) from a KnowledgeBase.

        Args:
            knowledge_base: The knowledge base to extract from.
            defuzzification: The defuzzification strategy to use.
            device: The device to use.

        Returns:
            The necessary components for a Fuzzy Logic Controller (FLC).
        """
        premise_vertices: igraph.VertexSeq = knowledge_base.select_by_tags(
            tags={"premise", "group"}
        )
        consequence_vertices: igraph.VertexSeq = knowledge_base.select_by_tags(
            tags={"consequence", "group"}
        )
        if len(premise_vertices) == 0:
            premises: None = None
        elif len(premise_vertices) == 1:
            premises: GroupedFuzzySets = premise_vertices[0]["item"].to(device)
        else:
            raise ValueError("Ambiguous selection of premise group.")
        if len(consequence_vertices) == 0:
            consequences: None = None
        elif len(consequence_vertices) == 1:
            consequences: GroupedFuzzySets = consequence_vertices[0]["item"].to(device)
        else:
            raise ValueError("Ambiguous selection of consequence group.")
        granulation_layers: GranulationLayers = GranulationLayers(
            input=premises,
            output=consequences,
        )
        engine = defuzzification(
            shape=knowledge_base.shape,
            source=granulation_layers["output"],
            device=device,
            rule_base=knowledge_base.rule_base,
        )

        return granulation_layers, knowledge_base.rule_base.premises, engine
