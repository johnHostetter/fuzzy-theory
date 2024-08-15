"""
Demo of working with discrete fuzzy sets for a toy task regarding knowledge of material.
"""

from sympy import Symbol, Interval, oo  # oo is infinity

from fuzzy.relations.discrete.tnorm import StandardIntersection
from fuzzy.relations.discrete.extension import AlphaCut, SpecialFuzzySet
from fuzzy.sets.discrete import DiscreteFuzzySet, FuzzyVariable


# https://www-sciencedirect-com.prox.lib.ncsu.edu/science/article/pii/S0957417412008056


def unknown():
    """
    Create a fuzzy set for the linguistic term 'unknown'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol("x")
    formulas.append((1, Interval.Lopen(-oo, 55)))
    formulas.append((1 - (element - 55) / 5, Interval.open(55, 60)))
    formulas.append((0, Interval.Ropen(60, oo)))
    return DiscreteFuzzySet(formulas, "Unknown")


def known():
    """
    Create a fuzzy set for the linguistic term 'known'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol("x")
    formulas.append(((element - 70) / 5, Interval.open(70, 75)))
    formulas.append((1, Interval(75, 85)))
    formulas.append((1 - (element - 85) / 5, Interval.open(85, 90)))
    formulas.append((0, Interval.Lopen(-oo, 70)))
    formulas.append((0, Interval.Ropen(90, oo)))
    return DiscreteFuzzySet(formulas, "Known")


def unsatisfactory_unknown():
    """
    Create a fuzzy set for the linguistic term 'unsatisfactory unknown'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol("x")
    formulas.append(((element - 55) / 5, Interval.open(55, 60)))
    formulas.append((1, Interval(60, 70)))
    formulas.append((1 - (element - 70) / 5, Interval.open(70, 75)))
    formulas.append((0, Interval.Lopen(-oo, 55)))
    formulas.append((0, Interval.Ropen(75, oo)))
    return DiscreteFuzzySet(formulas, "Unsatisfactory Unknown")


def learned():
    """
    Create a fuzzy set for the linguistic term 'learned'.

    Returns:
        OrdinaryDiscreteFuzzySet
    """
    formulas = []
    element = Symbol("x")
    formulas.append(((element - 85) / 5, Interval.open(85, 90)))
    formulas.append((1, Interval(90, 100)))
    formulas.append((0, Interval.Lopen(-oo, 85)))
    return DiscreteFuzzySet(formulas, "Learned")


if __name__ == "__main__":
    terms = [unknown(), known(), unsatisfactory_unknown(), learned()]

    fuzzy_variable = FuzzyVariable(fuzzy_sets=terms, name="Student Knowledge")
    fig, _ = fuzzy_variable.plot(samples=150)
    fig.show()

    # --- DEMO --- Classify 'element'

    example_element_membership = fuzzy_variable.degree(element=73)

    alpha_cut = AlphaCut(known(), 0.6, "AlphaCut")
    fig, _ = alpha_cut.plot(samples=250)
    fig.show()
    special_fuzzy_set = SpecialFuzzySet(known(), 0.5, "Special")
    fig, _ = special_fuzzy_set.plot()
    fig.show()

    alpha_cuts = []
    idx, MAX_HEIGHT, IDX_OF_MAX = 0, 0, 0
    for idx, membership_to_fuzzy_term in enumerate(example_element_membership):
        if example_element_membership[idx] > 0:
            special_fuzzy_set = SpecialFuzzySet(
                terms[idx], membership_to_fuzzy_term, terms[idx].name
            )
            alpha_cuts.append(
                StandardIntersection(
                    [special_fuzzy_set, terms[idx]], name=f"A{idx + 1}"
                )
            )

            # maximum membership principle
            if membership_to_fuzzy_term > MAX_HEIGHT:
                MAX_HEIGHT, IDX_OF_MAX = membership_to_fuzzy_term, idx

    confluence: FuzzyVariable = FuzzyVariable(fuzzy_sets=alpha_cuts, name="Confluence")
    fig, _ = confluence.plot()
    fig.show()

    # maximum membership principle
    print(f"Maximum Membership Principle: {terms[idx].name}")
