"""Independent Python reference semantics for the 90 challenge primitives.

This module intentionally does not import Elisa-generated code.  It is used by
the validation harness to catch drift between evaluator semantics and a second
implementation, and to produce reproducible seed manifests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

KINDS = (
    "ArithmeticChain", "InequalityChain", "Parity", "Sequence", "GraphReachability",
    "GraphDistance", "TemporalOrder", "Membership", "BooleanFormula", "ConstraintAssignment",
    "MinimumValue", "MaximumValue", "AbsoluteDifference", "SetIntersection", "GraphCycle",
    "TopologicalOrder", "CalendarOffset", "XorFormula", "LinearEquation", "DigitCount",
    "ThreePremiseChain", "FourPremiseChain", "DistractorCheck", "ContradictionCheck", "Biconditional",
    "UnlessRule", "OnlyIfRule", "QuantifierAll", "QuantifierSome", "ExclusiveOr", "TransitiveRelation",
    "SymmetricRelation", "DoubleNegation", "EarliestEvent", "LatestEvent", "PartialOrder", "GridDistance",
    "ManhattanPath", "Rotation", "Mirror", "SumValues", "ProductValues", "AverageFloor", "ValueRange",
    "SignCheck", "Divisibility", "ModuloCheck", "PowerOfTwo", "CountTrue", "Majority", "AtLeast",
    "ExactlyOne", "ImplicationChain", "NandFormula", "NorFormula", "MinimumIndex", "MaximumIndex",
    "StableSortCheck", "PermutationParity", "UniqueValue", "AllDistinct", "PairwiseOrder", "MedianValue",
    "SecondMinimum", "SecondMaximum", "CountLess", "CountGreater", "WeightedSum", "LinearInequality",
    "IntervalMembership", "TemporalGap", "ScheduleConflict", "ReachTwoSteps", "DegreeCount", "BipartiteCheck",
    "PathParity", "CoordinateQuadrant", "CompassDirection", "Rotation3D", "SpaceTimeOrder", "AllOf", "AnyOf",
    "NoneOf", "ExactlyK", "SyllogismMood", "AnalogyRelation", "SortingKey", "PermutationCheck",
    "ConstraintCount", "CompositeProof",
)

@dataclass(frozen=True)
class Challenge:
    kind: str
    values: tuple[int, ...]
    claim: int
    answer: int
    # Curriculum metadata is public to the generator/evaluator tooling but is
    # not used to compute the answer.  Defaults preserve the four-field API
    # used by the native cross-check and independent solver.
    nesting_depth: int = 1
    distractor_count: int = 0
    interference_permille: int = 0

def _sgn(v: int) -> int:
    return 1 if v > 0 else (0 if v == 0 else -1)

def solve(c: Challenge) -> int:
    """Solve one challenge using only its public values and kind."""
    v = c.values
    k = c.kind
    if k in ("ArithmeticChain", "SumValues"): return sum(v)
    if k in ("InequalityChain", "TopologicalOrder", "StableSortCheck"):
        return int(all(a < b for a, b in zip(v, v[1:]))) if k != "StableSortCheck" else int(all(a <= b for a, b in zip(v, v[1:])))
    if k == "Parity": return abs(v[0]) % 2
    if k == "Sequence": return v[-1] + (v[2] - v[1])
    if k == "GraphReachability": return int(v[0] == v[1] or v[2] == v[1])
    if k == "GraphDistance": return sum(x >= 0 for x in v[2:])
    if k == "TemporalOrder": return int(all(a <= b for a, b in zip(v, v[1:])))
    if k == "Membership": return int(v[0] in v[1:])
    if k == "BooleanFormula": return int(bool(v[0]) and not bool(v[1]))
    if k == "ConstraintAssignment": return abs(v[0])
    if k in ("MinimumValue", "EarliestEvent"): return min(v)
    if k in ("MaximumValue", "LatestEvent"): return max(v)
    if k == "AbsoluteDifference": return abs(v[0] - v[1])
    if k == "SetIntersection": return int(any(x == v[0] for x in v[1:]))
    if k == "GraphCycle": return int(v[0] == v[-1])
    if k == "CalendarOffset": return (v[0] + v[1]) % 7
    if k in ("XorFormula", "ExclusiveOr"): return int(bool(v[0]) != bool(v[1]))
    if k == "LinearEquation":
        numerator = v[2] - v[1]
        denominator = v[0]
        # Elisa's integer division truncates toward zero (unlike Python //).
        return abs(numerator) // abs(denominator) * (-1 if (numerator < 0) != (denominator < 0) else 1)
    if k == "DigitCount": return len(str(abs(v[0])))
    if k == "ThreePremiseChain": return int(v[0] < v[1] < v[2])
    if k == "FourPremiseChain": return int(v[0] < v[1] < v[2] < v[3])
    if k == "DistractorCheck": return int(v[0] == v[1])
    if k == "ContradictionCheck": return int(v[0] == -v[1])
    if k == "Biconditional": return int(bool(v[0]) == bool(v[1]))
    if k == "UnlessRule": return int(bool(v[0]) or not bool(v[1]))
    if k == "OnlyIfRule": return int((not bool(v[0])) or bool(v[1]))
    if k == "QuantifierAll": return int(all(v))
    if k == "QuantifierSome": return int(any(v))
    if k == "TransitiveRelation": return int(v[0] < v[1] and v[1] < v[2] and v[0] < v[2])
    if k == "SymmetricRelation": return int(v[0] == v[1])
    if k == "DoubleNegation": return v[0]
    if k == "PartialOrder": return int(v[0] <= v[1])
    if k == "GridDistance": return abs(v[0] - v[2]) + abs(v[1] - v[3])
    if k == "ManhattanPath": return abs(v[0]) + abs(v[1])
    if k == "Rotation": return (v[0] + v[1]) % 4
    if k == "Mirror": return -v[0]
    if k == "ProductValues":
        out = 1
        for x in v: out *= x
        return out
    if k == "AverageFloor":
        total = sum(v)
        return abs(total) // len(v) * (-1 if total < 0 else 1)
    if k == "ValueRange": return max(v) - min(v)
    if k == "SignCheck": return _sgn(v[0])
    if k == "Divisibility": return int(v[0] % v[1] == 0)
    if k == "ModuloCheck": return v[0] % v[1]
    if k == "PowerOfTwo": return int(v[0] > 0 and v[0] & (v[0] - 1) == 0)
    if k == "CountTrue": return sum(bool(x) for x in v)
    if k == "Majority": return int(sum(bool(x) for x in v) * 2 > len(v))
    if k == "AtLeast": return int(sum(bool(x) for x in v[1:]) >= v[0])
    if k == "ExactlyOne": return int(sum(bool(x) for x in v) == 1)
    if k == "ImplicationChain": return int(not bool(v[0]) or bool(v[1]))
    if k == "NandFormula": return int(not (bool(v[0]) and bool(v[1])))
    if k == "NorFormula": return int(not (bool(v[0]) or bool(v[1])))
    if k == "MinimumIndex": return min(range(len(v)), key=v.__getitem__)
    if k == "MaximumIndex": return max(range(len(v)), key=v.__getitem__)
    if k == "PermutationParity": return sum(v[i] > v[j] for i in range(len(v)) for j in range(i + 1, len(v))) % 2
    if k in ("UniqueValue", "AllDistinct"): return int(len(set(v)) == len(v))
    if k == "PairwiseOrder": return int(v[0] <= v[1] <= v[2])
    if k == "MedianValue": return sorted(v[:3])[1]
    if k == "SecondMinimum": return sorted(v)[1]
    if k == "SecondMaximum": return sorted(v, reverse=True)[1]
    if k == "CountLess": return sum(x < v[0] for x in v[1:])
    if k == "CountGreater": return sum(x > v[0] for x in v[1:])
    if k == "WeightedSum": return v[0] + 2 * v[1] + 3 * v[2]
    if k == "LinearInequality": return int(v[0] * v[1] <= v[2])
    if k == "IntervalMembership": return int(v[1] <= v[0] <= v[2])
    if k == "TemporalGap": return abs(v[1] - v[0])
    if k == "ScheduleConflict": return int(v[0] < v[1] < v[2])
    if k == "ReachTwoSteps": return int(v[0] == v[2] or v[1] == v[2])
    if k == "DegreeCount": return sum(x == v[0] for x in v[1:])
    if k == "BipartiteCheck": return int(v[0] % 2 != v[1] % 2)
    if k == "PathParity": return (v[0] + v[1] + v[2]) % 2
    if k == "CoordinateQuadrant": return 1 if v[0] >= 0 and v[1] >= 0 else (2 if v[0] < 0 and v[1] >= 0 else (3 if v[0] < 0 else 4))
    if k == "CompassDirection": return 0 if v[0] == 0 and v[1] == 0 else (1 if v[1] > 0 else (2 if v[0] > 0 else (3 if v[1] < 0 else 4)))
    if k == "Rotation3D": return (v[0] + v[1] + v[2]) % 6
    if k == "SpaceTimeOrder": return int(v[0] <= v[1] <= v[2])
    if k == "AllOf": return int(all(v))
    if k == "AnyOf": return int(any(v))
    if k == "NoneOf": return int(not any(v))
    if k == "ExactlyK": return int(sum(bool(x) for x in v[1:]) == v[0])
    if k == "SyllogismMood": return int(v[0] + v[1] == v[2])
    if k == "AnalogyRelation": return int(v[0] - v[1] == v[2] - v[3])
    if k == "SortingKey": return int(v[0] <= v[1])
    if k == "PermutationCheck": return int(sum(v[:3]) == 6)
    if k == "ConstraintCount": return sum(x >= v[0] for x in v)
    if k == "CompositeProof": return int(v[0] < v[1] and v[2] != 0)
    raise KeyError(k)

def xorshift64(state: int) -> int:
    state &= (1 << 64) - 1
    state ^= (state << 13) & ((1 << 64) - 1)
    state ^= state >> 7
    state ^= (state << 17) & ((1 << 64) - 1)
    return state & ((1 << 64) - 1)

def generated(seed: int, difficulty: str = "standard") -> Challenge:
    try:
        from .difficulty_profiles import profile
    except ImportError:  # script execution
        from difficulty_profiles import profile
    settings = profile(difficulty)
    state = (seed + 1) & ((1 << 64) - 1)
    if state == 0: state = 0x9e3779b97f4a7c15
    def below(n: int) -> int:
        nonlocal state
        state = xorshift64(state)
        return state % n
    kind = KINDS[below(90)]
    # Four values are the minimum required by the catalog; harder profiles add
    # retention load without changing the answer vocabulary.
    value_count = max(4, min(16, settings.premise_count))
    values = tuple(below(19) + 1 for _ in range(value_count))
    if kind == "LinearEquation": values = ((values[0] - 1) % 5 + 1, *values[1:])
    answer = solve(Challenge(kind, values, 0, 0))
    wrong = bool(below(2))
    claim = answer + 1 if wrong else answer
    return Challenge(kind, values, claim, answer,
                     nesting_depth=settings.boolean_depth,
                     distractor_count=min(settings.distractor_count,
                                          max(0, len(values) - 4)),
                     interference_permille=settings.interference_permille)
