"""
Implements the vital KnowledgeBase class.
"""

import ast
import pickle
import warnings
import importlib.util
from pathlib import Path
from typing import Union, Set, List, Any

import torch
import numpy as np
import pandas as pd
import igraph as ig

from rough.decisions import RoughDecisions

from fuzzy.logic.rule import Rule
from fuzzy.logic.rulebase import RuleBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.control.configurations import Shape, GranulationLayers, FuzzySystem
from fuzzy.relations.t_norm import TNorm
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.sets.group import FuzzySetGroup
from fuzzy.sets.abstract import FuzzySet


class KnowledgeBase(RoughDecisions, FuzzySystem):
    """
    The KnowledgeBase class is a significant component to this Soft Computing library.

    As the name suggests, it implements the concept of a Knowledge Base in the
    field of Artificial Intelligence.

    Its implementation is abstract and generic, using a graph structure (i.e., self.graph)
    to store facts, data, functions, implications, relations, etc. in the form
    of vertices or edges, whichever is most appropriate.

    Typically, facts, data, and functions are vertices, whereas implications and
    relations are implemented with edges in the graph structure.

    The distinction between what each vertex represents is made by the
    vertex's attributes, such as 'tags'.

    The same can be said for edges.

    For example, fuzzy sets may exist in the KnowledgeBase's graph (i.e., self.graph)
    as vertices, and how those fuzzy sets are used is represented with an edge. At the time of
    writing (4/14/2023), these fuzzy sets are connected to another vertex representing the t-norm
    in which they are aggregated together (e.g., hot weather and cold soup) to offer flexibility
    in the choice of t-norm. In other words, it is possible to construct a KnowledgeBase where
    each fuzzy logic rule has its own independent t-norm definition. Then, this vertex representing
    the compound relation between the fuzzy sets can be interpreted as an antecedent,
    if a directed edge is connected from this compound relation to another compound relation
    such that (source_vertex, target_vertex) represent antecedents and consequences, respectively.

    Naturally, this allows for fuzzy sets to be dynamically added or removed from fuzzy rules,
    or fuzzy rules to be dynamically added or removed. Allowing for the process of
    self-organizing with various methods (e.g., genetic algorithms, unsupervised learning,
    rough set theory) to be somewhat easily incorporated.
    """

    @property
    def shape(self) -> Shape:
        return self.rule_base.shape

    @property
    def rules(self) -> List[Rule]:
        """
        Get a list of fuzzy logic rules, where each element in the list is a Rule object.

        Note: This retrieves the rules from the KnowledgeBase's graph, as opposed to simply
        retrieving the rules from an attribute. This is because the KnowledgeBase's graph
        can be analyzed, modified, and used to create new rules.

        Returns:
            A list of fuzzy logic rules (as instances of Rule), ordered by their creation
            (i.e., rule's id attribute).
        """
        rule_vertices: ig.VertexSeq = self.select_by_tags(tags="rule")
        return sorted(
            [rule["item"] for rule in rule_vertices], key=lambda rule: rule.id
        )

    @property
    def rule_base(self) -> RuleBase:
        """
        Get the RuleBase object from the KnowledgeBase; automatically determine the device.

        Returns:
            A RuleBase object.
        """
        return RuleBase(rules=self.rules, device=None)

    def granulation_layers(self, device: torch.device) -> GranulationLayers:
        layers = {"input": None, "output": None}
        for attr, layer in zip(["input", "output"], ["premise", "consequence"]):
            group_vertices: ig.VertexSeq = self.select_by_tags(tags={layer, "group"})
            layer: Union[None, FuzzySetGroup] = None  # default to None if no granules
            if len(group_vertices) == 1:
                layer: FuzzySetGroup = group_vertices[0]["item"].to(device)
            elif len(group_vertices) > 1:
                raise ValueError(f"Ambiguous selection of {layer} group.")

            layers[attr] = layer

        return GranulationLayers(**layers)

    def engine(self, device: torch.device) -> TNorm:
        """
        Create the fuzzy logic inference engine from the KnowledgeBase.

        Args:
            device: The device to use.

        Returns:
            The fuzzy logic inference engine.
        """
        return self.rule_base.premises

    def get_granules(self, tags: Union[str, Set[str]]) -> ig.VertexSeq:
        """
        Given a set of tags, find the granules in the KnowledgeBase.

        Args:
            tags: The tags to filter the granules by.

        Returns:
            The granules in the KnowledgeBase (their vertex sequence).
        """
        if isinstance(tags, str):
            tags = {tags}
        all_matching_vertices: ig.VertexSeq = self.select_by_tags(
            tags=tags | {"granule"}
        )
        # drop the FuzzySetGroup, as they are not part of the granules
        return all_matching_vertices.select(
            lambda vertex: not isinstance(vertex["item"], FuzzySetGroup)
        )

    def intra_dimensions(self, tags: Union[str, Set[str]]) -> np.ndarray:
        """
        Calculate the dimensionality within dimensions. For example, calculate the
        number of linguistic terms available for the first input variable, and the
        second input variable, and so on. The result is a calculation for each variable,
        stored as a 1-dimensional Numpy array.

        Args:
            tags: A string or a set of strings that are used to filter the vertices
            in the graph to find the granules.

        Returns:
            A 1-dimensional Numpy array where each element is a count of
            the possible terms for their respective dimension
            (e.g., the 0th element may say 3, meaning that
            there are 3 possible terms/sets within the first dimension).
        """
        # fetch the created vertex by stack_granules, get its input vertices, and finally its names
        # granules = self.graph.vs.find(source_eq=add_stacked_granule.__name__)['input']['type']
        granule_vertices = self.get_granules(tags)

        return np.array(
            [
                (
                    params.get_centers().size(dim=-1)
                    if params.get_centers().dim() > 0
                    else 0
                )
                for params in granule_vertices["item"]
            ],
            dtype=np.int32,
        )

    def attributes(self, element: Union[int, str]) -> dict:
        """
        Fetch the attributes and their values for the given element.

        Args:
            element: An element or object in the universe of discourse.

        Returns:
            A dictionary where each key is the attribute type (or relation)
            and the value associated with the key is the value of that attribute
            (or unique auto-generated id for relations).
        """
        equivalence_classes = self[element]
        # dictionary to store the attribute (key) to its value (values) for element
        attributes_values = {}
        for attribute_name, equivalences in equivalence_classes.items():
            attributes_values[attribute_name] = self.attribute_table[
                frozenset({attribute_name, equivalences})
            ]
        return attributes_values

    def save(self, path: Path) -> None:
        """
        Save this Knowledgebase object for later use.

        Args:
            path: The path to the directory that this KnowledgeBase should be saved at.

        Returns:
            None
        """
        path.mkdir(parents=True, exist_ok=True)

        # save the attribute table
        path_to_attribute_table: Path = path / "attribute_table.pickle"
        with open(path_to_attribute_table, "wb") as handle:
            pickle.dump(self.attribute_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # backup the graph's attributes
        deepcopy_graph = self.graph.copy()

        # save the FuzzySet objects
        self.save_granules(path, FuzzySet, extension=".pt")
        # save the FuzzySetGroup objects (hypercubes)
        self.save_granules(path, FuzzySetGroup, extension="")
        # save the Rule objects
        self.save_granules(path, Rule, extension="")
        # save the NAryRelation objects
        self.save_granules(path, NAryRelation, extension=".pt")

        path_to_graph = path / "graph"
        path_to_graph.mkdir(parents=True, exist_ok=True)

        # save the graph vertices and edges
        vertices_df = self.graph.get_vertex_dataframe()
        vertices_df.to_csv(f"{path_to_graph / 'vertices'}.csv", index=False)
        edges_df = self.graph.get_edge_dataframe()
        edges_df.to_csv(f"{path_to_graph / 'edges'}.csv", index=False)

        self.graph.save(f"{path_to_graph / 'network'}.graphml")

        # restore the graph's attributes
        self.graph = deepcopy_graph

    def save_granules(self, path: Path, class_type: Any, extension: str) -> None:
        """
        Save the granules to the given path.

        Args:
            path: The path to the directory that the granules should be saved at.
            class_type: The type of the granules.
            extension: The file extension to save the granules as.

        Returns:
            None
        """
        # find any granules that are instances of the class type
        matched_objects: List[ig.Vertex] = [
            vertex for vertex in self.graph.vs if isinstance(vertex["item"], class_type)
        ]  # the index refers to the KnowledgeBase's graph's index of the vertex

        # save the granules
        for granule_vertex in matched_objects:
            granule = granule_vertex["item"]  # granule is the object instance
            # create the directory to save the granule in
            subdirectory = path / class_type.__name__ / str(granule_vertex.index)
            subdirectory.mkdir(parents=True, exist_ok=True)
            actual_path = subdirectory / f"{granule.__class__.__name__}{extension}"
            _ = granule.save(actual_path)  # ignore the return value
            granule_vertex["file"] = (
                actual_path  # store the path to the granule to load from
            )
            # remove object instance from the graph
            granule_vertex["item"] = f"{granule.__module__}.{type(granule).__name__}"

    def add_hypercube(self, tags: Union[str, Set[str]]) -> Union[None, ig.Vertex]:
        """
        Constructs the granules for this mapping in a format that will be compatible with the
        expected functionality. Also, modifies a Knowledgebase such that the condensed
        (i.e., stacked) version of the granules is readily available.

        Args:
            tags: The tags that indicate which granules to stack.

        Returns:
            A single membership function that combines all the granules for faster computation; if
            there is an uneven number of granules in each dimension, then the values (e.g.,
            centers, widths) have been padded with torch.nan (if not trainable) or 0 for centers
            and -1 for widths (if trainable).
        """
        if isinstance(tags, str):
            tags = {tags}
        # get the granules from the KnowledgeBase
        granule_vertices: ig.seq.VertexSeq = self.select_by_tags(tags)
        if len(granule_vertices) > 0:
            # from the igraph.VertexSeq object, extract the granules, stored in the "type" attribute
            granules: List[FuzzySet] = granule_vertices["item"]
            # create the efficient granule module
            stacked_granules: FuzzySet = FuzzySet.stack(granules)
            hypercube: FuzzySetGroup = FuzzySetGroup(modules_list=[stacked_granules])
            # store the efficient granule module in the KnowledgeBase
            target_vertex: Union[None, ig.Vertex] = self.graph.add_vertex(
                # source=add_stacked_granule,
                item=hypercube,
                tags=tags | {"group"},  # add the same tags as the granules w/ "group"
                # input=list(granule_vertices.indices),  # store the vertex indices
            )
            if target_vertex is not None:
                # add edges that point from the granules to the stacked granule representation
                edges = set()
                for granule in granule_vertices:
                    edges.add((granule.index, target_vertex.index))
                self.graph.add_edges(edges)
            return target_vertex
        return None

    @staticmethod
    def create(
        linguistic_variables: LinguisticVariables,
        rules: List[Rule],
    ) -> "KnowledgeBase":
        """
        A convenient helper method for the expert-design procedure of a Knowledge Base.
        The linguistic variable's inputs member contains the premises and the targets
        member contains the consequences.

        Args:
            linguistic_variables: The linguistic variables involved and their terms.
            rules: The fuzzy logic rules; must be non-empty.

        Returns:
            A KnowledgeBase object, with the linguistic_variables (i.e., antecedents, consequents),
            and fuzzy logic rules added.
        """
        if len(rules) == 0:
            warnings.warn("No fuzzy logic rules were provided.")
        # gather the premises' graphs
        rule_graphs: List[ig.Graph] = []
        for rule in rules:
            rule_graph: List[ig.Graph] = []
            for component in ["premise", "consequence"]:
                rule_component: NAryRelation = getattr(rule, component)
                rule_component.create_igraph()  # re-create the graph to prevent malformed names
                if len(getattr(rule, component).indices) > 1:
                    raise ValueError(
                        f"The rule's {component} must be a single relation."
                    )
                for tag in ["relation", "anchor"]:
                    # add the premise tag to the matched vertices
                    rule_component.graph.vs.select(tags_eq={tag})["tags"] = {
                        component,
                        tag,
                    }
                # avoid duplicate names in the graph to prevent igraph from merging vertices
                rule_component.graph.vs["name"] = [
                    f"{component}:{name}" for name in rule_component.graph.vs["name"]
                ]
                rule_graph.append(rule_component.graph)
            rule_graphs.append(ig.disjoint_union(rule_graph))

            premise_relation_vertex = rule_graphs[-1].vs.find(
                tags_eq={"premise", "relation"}
            )
            consequence_relation_vertex = rule_graphs[-1].vs.find(
                tags_eq={"consequence", "relation"}
            )
            rule_vertex = rule_graphs[-1].add_vertex(
                name=str(hash(rule)),
                item=rule,
                tags={"rule"},
            )
            # point the premise relation to the rule node
            rule_graphs[-1].add_edge(premise_relation_vertex.index, rule_vertex.index)
            # point the consequence relation to the rule node
            rule_graphs[-1].add_edge(
                rule_vertex.index, consequence_relation_vertex.index
            )

        knowledge_base = KnowledgeBase()
        # add premises' graphs to KnowledgeBase (order matters, must occur before .set_granules())
        if len(rule_graphs) > 0:
            knowledge_base.graph = ig.union(rule_graphs, byname=True)
            # drop the name attribute to avoid malformed name identifiers
            del knowledge_base.graph.vs["name"]

        for attribute, granule_layer in zip(
            ["inputs", "targets"], ["premise", "consequence"]
        ):
            knowledge_base.set_granules(
                items=getattr(linguistic_variables, attribute),
                tags={granule_layer, "granule"},
            )
            knowledge_base.add_hypercube(
                tags={granule_layer, "granule"}
            )  # merge this layer's granules for efficiency

        return knowledge_base

    @staticmethod
    def load(path: Path, device: torch.device) -> "KnowledgeBase":
        """
        Given a path to a directory, load the saved KnowledgeBase object at that location.

        Args:
            path: The path to the directory that the KnowledgeBase was saved at.
            device: The device to use.

        Returns:
            KnowledgeBase
        """
        # the path_to_graph stores the directory that contains the graph + additional files
        # Note: path_to_graph / 'network' stores the actual graph file, but is not required here
        path_to_graph: Path = path / "graph"

        vertices_df: pd.DataFrame = pd.read_csv(f"{path_to_graph / 'vertices'}.csv")
        vertices_df.replace({np.nan: None}, inplace=True)  # convert np.nan to Nan

        KnowledgeBase.parse_vertex_attributes(vertices_df, device=device)
        knowledge_base = KnowledgeBase()

        # load the attribute table
        path_to_attribute_table: Path = path / "attribute_table.pickle"
        with open(path_to_attribute_table, "rb") as handle:
            knowledge_base.attribute_table = pickle.load(handle)

        vertices_df: pd.DataFrame = vertices_df.replace(np.nan, None).replace(
            -1, None
        )  # undo temporary -1
        # remove the 'file' column from the DataFrame
        vertices_df.drop(columns=["file"], inplace=True)
        knowledge_base.graph = ig.Graph.DataFrame(
            # convert np.nan to None since igraph uses None and return the DataFrame
            pd.read_csv(f"{path_to_graph / 'edges'}.csv").replace(np.nan, None),
            directed=True,
            vertices=vertices_df,
        )

        return knowledge_base

    @staticmethod
    def parse_vertex_attributes(
        vertices_df: pd.DataFrame, device: torch.device
    ) -> None:
        """
        Parse the vertex attributes from the vertices_df and modify them in-place.

        Args:
            vertices_df: The DataFrame containing the vertex attributes.
            device: The device to use.

        Returns:
            None
        """
        vertices_df.replace({np.nan: None}, inplace=True)
        for _, vertex_row in vertices_df.iterrows():  # ignore row index
            if vertex_row["file"] is not None:
                tokens = vertex_row["item"].split(".")
                module_path: str = ".".join(tokens[:-1])  # e.g., fuzzy.sets.impl
                class_name: str = tokens[-1]  # e.g., Gaussian
                module = getattr(importlib.import_module(module_path), class_name)
                # load the vertex 'type' information from the file
                vertex_row["item"] = module.load(
                    Path(vertex_row["file"]), device=device
                )  # must be passed Path for module's that require Path.iterdir()
            for attribute in vertices_df.columns:
                if attribute == "file":
                    continue
                # convert the string representation of the attribute to the actual attribute
                # e.g., '(0, 1)' to (0, 1)
                attr_val = vertex_row[attribute]
                if attr_val:
                    try:
                        vertex_row[attribute] = ast.literal_eval(attr_val)
                    except ValueError:
                        continue  # unrecognized value - malformed node or string (e.g., 'A0')
