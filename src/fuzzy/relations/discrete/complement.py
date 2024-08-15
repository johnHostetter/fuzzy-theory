"""
Implements the various fuzzy complement definitions for the discrete fuzzy sets.
"""

from fuzzy.sets.discrete import BaseDiscreteFuzzySet


def standard_complement(fuzzy_set):
    """
    Obtains the standard complement of a fuzzy set as defined by Lotfi A. Zadeh.

    Returns True if successful, else returns False.

    Parameters
    ----------
    fuzzy_set : 'OrdinaryDiscreteFuzzySet'

    Returns
    -------
    success : 'bool'
    """

    if isinstance(fuzzy_set, BaseDiscreteFuzzySet):
        formulas = []
        for formula in fuzzy_set.formulas:
            formula = list(formula)
            formula[0] = 1 - formula[0]
            formula = tuple(formula)
            formulas.append(formula)
        fuzzy_set.formulas = formulas
        return True
    return False
