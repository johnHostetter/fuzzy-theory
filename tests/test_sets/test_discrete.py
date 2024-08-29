"""
Test the DiscreteFuzzySet and FuzzyVariable classes.
"""

import unittest

from fuzzy.sets.discrete import (
    BaseDiscreteFuzzySet,
    DiscreteFuzzySet,
    FuzzyVariable,
)

from examples.discrete.student import (
    unknown,
    known,
    unsatisfactory_unknown,
    learned,
)


class TestDiscreteFuzzySet(unittest.TestCase):
    """
    Test the DiscreteFuzzySet (i.e., Linguistic Term) class.
    """

    def test_abstract_methods(self):
        """
        Test that the abstract methods correctly throw a NotImplementedError.

        Returns:
            None
        """
        self.assertRaises(
            NotImplementedError, BaseDiscreteFuzzySet.degree, None, element=0
        )

    def test_empty_discrete_fuzzy_set(self) -> None:
        """
        Test that an empty DiscreteFuzzySet object can be created.

        Returns:
            None
        """
        discrete_fuzzy_set = DiscreteFuzzySet(formulas=[], name="")
        assert len(discrete_fuzzy_set.formulas) == 0
        assert len(discrete_fuzzy_set.name) == 0
        assert discrete_fuzzy_set.fetch(element=0.0) is None

    def test_create_discrete_fuzzy_set(self) -> None:
        """
        Test that an empty DiscreteFuzzySet object can be created.

        Returns:
            None
        """
        discrete_fuzzy_set = known()
        assert len(discrete_fuzzy_set.formulas) == 5
        assert discrete_fuzzy_set.name == "Known"

    def test_discrete_fuzzy_set_height(self) -> None:
        """
        Test that the height of the DiscreteFuzzySet is correct.

        Returns:
            None
        """
        discrete_fuzzy_set = known()
        assert discrete_fuzzy_set.height() == 1

    def test_discrete_fuzzy_set_plot(self) -> None:
        """
        Test that the plot method for the DiscreteFuzzySet works as intended.

        Returns:
            None
        """
        discrete_fuzzy_set = known()
        _, axes = discrete_fuzzy_set.plot(
            lower=0, upper=100, samples=100
        )  # ignore figure
        assert axes.get_title() == "Known Fuzzy Set"
        assert axes.get_xlabel() == "Elements of Universe"
        assert axes.get_ylabel() == "Degree of Membership"
        assert axes.get_xlim() == (0, 100)
        assert axes.get_ylim() == (0, 1.1)

        # now checking that removing the name changes it to an "Unnamed" DiscreteFuzzySet
        discrete_fuzzy_set.name = None
        _, axes = discrete_fuzzy_set.plot(
            lower=0, upper=100, samples=100
        )  # ignore figure
        assert axes.get_title() == "Unnamed Fuzzy Set"


class TestFuzzyVariable(unittest.TestCase):
    """
    Test the FuzzyVariable (i.e., Linguistic Variable) class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terms = [unknown(), known(), unsatisfactory_unknown(), learned()]
        self.fuzzy_variable = FuzzyVariable(
            fuzzy_sets=self.terms, name="Student Knowledge"
        )

    def test_create_fuzzy_variable(self) -> None:
        """
        Test that a FuzzyVariable object can be created.

        Returns:
            None
        """
        assert len(self.fuzzy_variable.fuzzy_sets) == len(self.terms)
        assert self.fuzzy_variable.name == "Student Knowledge"

    def test_fuzzy_variable_membership(self) -> None:
        """
        Test the degree of membership for the FuzzyVariable is correct.

        Returns:
            None
        """
        actual_membership = self.fuzzy_variable.degree(element=73)
        expected_membership = (0.0, 0.6, 0.4, 0.0)
        assert actual_membership == expected_membership

    def test_fuzzy_variable_plot(self) -> None:
        """
        Test the plot method for the FuzzyVariable.

        Returns:
            None
        """
        _, axes = self.fuzzy_variable.plot(
            lower=0, upper=100, samples=100
        )  # ignore figure
        assert axes.get_title() == "Student Knowledge Fuzzy Variable"
        assert axes.get_xlabel() == "Elements of Universe"
        assert axes.get_ylabel() == "Degree of Membership"
        assert axes.get_xlim() == (0, 100)
        assert axes.get_ylim() == (0, 1.1)

        # now checking that removing the name changes it to an "Unnamed" FuzzyVariable
        self.fuzzy_variable.name = None
        _, axes = self.fuzzy_variable.plot(
            lower=0, upper=100, samples=100
        )  # ignore figure
        assert axes.get_title() == "Unnamed Fuzzy Variable"
