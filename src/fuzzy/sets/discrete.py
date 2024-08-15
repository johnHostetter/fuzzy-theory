"""
Implements the discrete fuzzy sets.
"""

from abc import ABC, abstractmethod
from typing import Union, List, NoReturn, Tuple

import sympy
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# https://docs.sympy.org/latest/modules/integrals/integrals.html
# https://docs.sympy.org/latest/modules/sets.html
# https://numpydoc.readthedocs.io/en/latest/example.html


class BaseDiscreteFuzzySet(ABC):
    """
    A parent class for all fuzzy sets to inherit. Allows the user to visualize the fuzzy set.
    """

    def __init__(self, formulas: list, name: str):
        self.formulas = formulas
        self.name = name

    @abstractmethod
    def degree(self, element) -> NoReturn:
        """
        Calculates degree of membership for the provided "element" where element is a(n) int/float.

        Args:
            element: The element is from the universe of discourse.

        Returns:
            NotImplementedError as this is an abstract method.
        """
        raise NotImplementedError(
            "Attempted to call an abstract method from BaseDiscreteFuzzySet."
        )

    def fetch(self, element: Union[int, float]):
        """
        Fetch the corresponding formula for the provided element where element is a(n) int/float.

        Parameters
        ----------
        element : 'float'
            The element is from the universe of discourse X.

        Returns
        -------
        formula : 'tuple'/'None'
            Returns the tuple containing the formula and corresponding Interval.
            Returns None if a formula for the element could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(
                element
            ):  # check the formula's interval to see if it contains element
                return formula
        return None

    def plot(
        self, lower: float = 0, upper: float = 100, samples: int = 100
    ) -> (Figure, Axes):
        """
        Graphs the fuzzy set in the universe of elements.

        Parameters
        ----------
        lower : 'float', optional
            Default value is 0. Specifies the infimum value for the graph.
        upper : 'float', optional
            Default value is 100. Specifies the supremum value for the graph.
        samples : 'int', optional
            Default value is 100. Specifies the number of values to test in the domain
            to approximate the graph. A higher sample value will yield a higher resolution
            of the graph, but large values will lead to performance issues.
        """
        fig, axes = plt.subplots()
        x_list = np.linspace(lower, upper, samples)
        y_list = []
        for x_value in x_list:
            y_list.append(self.degree(x_value))
        if self.name is not None:
            plt.title(f"{self.name} Fuzzy Set")
        else:
            plt.title("Unnamed Fuzzy Set")

        axes.set_xlim([lower, upper])
        axes.set_ylim([0, 1.1])
        axes.set_xlabel("Elements of Universe")
        axes.set_ylabel("Degree of Membership")
        axes.plot(x_list, y_list, color="grey", label="mu")
        axes.legend()
        return fig, axes


class DiscreteFuzzySet(BaseDiscreteFuzzySet):
    """
    An ordinary fuzzy set that is of type 1 and level 1.
    """

    def __init__(self, formulas: List[tuple], name: str = None):
        """
        Parameters
        ----------
        formulas : 'list'
            A list of 2-tuples. The first element in the tuple at index 0 is the formula
            equal to f(x) and the second element in the tuple at index 1 is the Interval
            where the formula in the tuple is valid.

            Warning: Formulas should be organized in the list such that the formulas and
            their corresponding intervals are specified from the smallest possible x values
            to the largest possible x values.

            The list of formulas provided constitutes the piece-wise function of the
            fuzzy set's membership function.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        BaseDiscreteFuzzySet.__init__(self, formulas, name)

    def degree(self, element: Union[int, float]):
        """
        Calculates degree of membership for the provided element where element is a(n) int/float.

        Parameters
        ----------
        element : 'float'
            The element is from the universe of discourse X.

        Returns
        -------
        mu : 'float'
            The degree of membership for the element.
        """
        formula = self.fetch(element)[0]
        try:
            membership = float(formula.subs(Symbol("x"), element))
        except AttributeError:
            membership = formula
        return membership

    def height(self) -> float:
        """
        Calculates the height of the fuzzy set.

        Returns:
            The height, or supremum, of the fuzzy set.
        """
        heights = []
        for formula in self.formulas:
            if isinstance(formula[0], sympy.Expr):
                inf_x = formula[1].inf
                sup_x = formula[1].sup
                if formula[1].left_open:
                    inf_x += 1e-8
                if formula[1].right_open:
                    sup_x -= 1e-8
                inf_y = formula[0].subs(Symbol("x"), inf_x)
                sup_y = formula[0].subs(Symbol("x"), sup_x)
                heights.append(inf_y)
                heights.append(sup_y)
            else:
                heights.append(formula[0])
        return max(heights)


class FuzzyVariable(BaseDiscreteFuzzySet):
    """
    A fuzzy variable, or linguistic variable, that contains fuzzy sets.
    """

    def __init__(self, fuzzy_sets: List[BaseDiscreteFuzzySet], name=None):
        """
        Parameters
        ----------
        fuzzy_sets : 'list'
            A list of elements each of type OrdinaryDiscreteFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        BaseDiscreteFuzzySet.__init__(self, fuzzy_sets, name)
        self.fuzzy_sets = fuzzy_sets

    def degree(self, element: Union[int, float]) -> Tuple[float]:
        """
        Calculates the degree of membership for the provided element value
        where element is a(n) int/float.

        Args:
            element: The element from the universe of discourse.

        Returns:
            The degree of membership for the element.
        """
        degrees = []
        for fuzzy_set in self.formulas:
            degrees.append(float(fuzzy_set.degree(element)))
        return tuple(degrees)

    def plot(
        self, lower: float = 0, upper: float = 100, samples: int = 100
    ) -> (Figure, Axes):
        """
        Graphs the fuzzy set in the universe of elements.

        Args:
            lower: Default value is 0. Specifies the infimum value for the graph.
            upper: Default value is 100. Specifies the supremum value for the graph.
            samples: Default value is 100. Specifies the number of values to test in the domain
            to approximate the graph. A higher sample value will yield a higher resolution
            of the graph, but large values will lead to performance issues.

        Returns:
            None
        """
        fig, axes = plt.subplots()
        for fuzzy_set in self.fuzzy_sets:
            x_list = np.linspace(lower, upper, samples)
            y_list = []
            for x_value in x_list:
                y_list.append(fuzzy_set.degree(x_value))
            axes.plot(
                x_list,
                y_list,
                color=np.random.rand(
                    3,
                ),
                label=fuzzy_set.name,
            )

        if self.name is not None:
            axes.set_title(f"{self.name} Fuzzy Variable")
        else:
            axes.set_title("Unnamed Fuzzy Variable")

        axes.set_xlim([lower, upper])
        axes.set_ylim([0, 1.1])
        axes.set_xlabel("Elements of Universe")
        axes.set_ylabel("Degree of Membership")
        axes.legend()
        return fig, axes
