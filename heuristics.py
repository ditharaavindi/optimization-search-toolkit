# ============================================================
# Heuristics for A* Search (Admissible and Finite)
# ============================================================

from typing import Tuple
from math import sqrt

Coord = Tuple[int, int]

def heuristic_manhattan(u: Coord, goal: Coord) -> float:
    """
    Return |ur - gr| + |uc - gc| (admissible for 4-neighborhood grids).
    """
    return abs(u[0] - goal[0]) + abs(u[1] - goal[1])


def heuristic_straight_line(u: Coord, goal: Coord) -> float:
    """
    Return Euclidean (straight-line) distance to goal (admissible and finite).
    """
    return sqrt((u[0] - goal[0])**2 + (u[1] - goal[1])**2)


def heuristic_custom(u: Coord, goal: Coord) -> float:
    """
    Custom admissible heuristic.
    I combine Manhattan and Euclidean heuristics in a convex combination.
    Since both are admissible and non-negative, any weighted average
    (with weights summing to 1) is also admissible.

    This heuristic slightly smooths Manhattan’s step-like contours,
    often improving node expansion efficiency on obstacle-rich grids.
    """
    manhattan = abs(u[0] - goal[0]) + abs(u[1] - goal[1])
    euclidean = sqrt((u[0] - goal[0])**2 + (u[1] - goal[1])**2)
    # Weighted average (still ≤ Manhattan, thus admissible)
    return 0.6 * manhattan + 0.4 * euclidean
