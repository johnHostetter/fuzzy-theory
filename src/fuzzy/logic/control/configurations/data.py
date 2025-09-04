"""
This module contains the data structures used to define the shape of a Fuzzy Logic Controller (FLC)
and the granulation layers used in the FLC.
"""

from typing import TypedDict, NamedTuple, Union

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
    output: Union[None, FuzzySetGroup]
