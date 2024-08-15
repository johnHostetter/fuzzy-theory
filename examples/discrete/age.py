"""
Demo of working with discrete fuzzy sets for a toy task regarding age.
"""

from sympy import Symbol, Interval, oo  # oo is infinity

from fuzzy.relations.discrete.snorm import StandardUnion
from fuzzy.relations.discrete.tnorm import StandardIntersection
from fuzzy.relations.discrete.complement import standard_complement
from fuzzy.sets.discrete import DiscreteFuzzySet, FuzzyVariable


def a_1():
    """
    A sample construction of a Fuzzy Set called 'A1'.
    """
    formulas = []
    element = Symbol("x")
    formulas.append((1, Interval.Lopen(-oo, 20)))
    formulas.append(((35 - element) / 15, Interval.open(20, 35)))
    formulas.append((0, Interval.Ropen(35, oo)))
    return DiscreteFuzzySet(formulas, "A1")


def a_2():
    """
    A sample construction of a Fuzzy Set called 'A2'.
    """
    formulas = []
    element = Symbol("x")
    formulas.append((0, Interval.Lopen(-oo, 20)))
    formulas.append(((element - 20) / 15, Interval.open(20, 35)))
    formulas.append((1, Interval(35, 45)))
    formulas.append(((60 - element) / 15, Interval.open(45, 60)))
    formulas.append((0, Interval.Ropen(60, oo)))
    return DiscreteFuzzySet(formulas, "A2")


def a_3():
    """
    A sample construction of a Fuzzy Set called 'A3'.
    """
    formulas = []
    element = Symbol("x")
    formulas.append((0, Interval.Lopen(-oo, 45)))
    formulas.append(((element - 45) / 15, Interval.open(45, 60)))
    formulas.append((1, Interval.Ropen(60, oo)))
    return DiscreteFuzzySet(formulas, "A3")


a1 = a_1()
a2 = a_2()
a3 = a_3()

FuzzyVariable([a1, a2, a3], "Age").plot(0, 80)
b = StandardIntersection([a1, a2], "B")
b.plot(0, 80)
c = StandardIntersection([a2, a3], "C")
c.plot(0, 80)
StandardUnion([b, c], "B Union C").plot(0, 80)
standard_complement(a1)
a1.plot(0, 80)
standard_complement(a3)
a3.plot(0, 80)
StandardIntersection([a1, a3], "Not(A1) Intersection Not(A3)").plot(0, 80)
standard_complement(a1)
standard_complement(a3)
# doesn't work yet
# StandardComplement(StandardUnion([b, c], 'Not (B Union C)')).graph(0, 80)
