# heuristics.py
# ============================================================
# TASK
#   Implement three admissible (non-overestimating) heuristics.
#
# SIGNATURES (do not change):
#   heuristic_manhattan(u, goal)      -> float   (5%)
#   heuristic_straight_line(u, goal)  -> float   (5%)
#   heuristic_custom(u, goal)         -> float   (10%)
#
# PARAMETERS
#   u, goal: coordinates (r, c)
#
# RETURN
#   A non-negative number estimating the remaining cost from u to goal.
#
# RULES
# - Heuristics must be ADMISSIBLE for 4-neighbor grids with unit step cost,
#   i.e., h(u) <= true shortest path length from u to goal.
# - They must be finite (no NaN/inf) and >= 0.
# - We will probe many random states and compare h(u) against true distances.
#
# HINTS
# - Manhattan distance is admissible in 4-neighborhood: |dr| + |dc|.
# - Straight-line (Euclidean) distance is also admissible.
# - For the custom heuristic, keep it <= Manhattan to be safe,
#   OR design another admissible function and justify in your notes.
# ============================================================

from typing import Tuple
from math import hypot

Coord = Tuple[int, int]

def heuristic_manhattan(u: Coord, goal: Coord) -> float:
    """Return |ur - gr| + |uc - gc| (admissible for 4-neighborhood)."""
    (ur, uc), (gr, gc) = u, goal
    return float(abs(ur - gr) + abs(uc - gc))

def heuristic_straight_line(u: Coord, goal: Coord) -> float:
    """Return Euclidean (straight-line) distance to goal (admissible)."""
    (ur, uc), (gr, gc) = u, goal
    return float(hypot(ur - gr, uc - gc))

def heuristic_custom(u: Coord, goal: Coord) -> float:
    """
    Your own design. Must be admissible, non-negative, finite.
    Example idea (DON'T just copy this): 0.8 * Manhattan(u, goal)
    Explain your choice in the HTML summary notes.
    """
    # Convex combination of Manhattan and Euclidean. On a 4-connected
    # unit-cost grid Euclidean <= Manhattan, so any convex combo <= Manhattan
    # and therefore admissible. We choose weights that slightly favor Manhattan
    # while retaining Euclidean smoothing to reduce plateaus.
    man = heuristic_manhattan(u, goal)
    eu = heuristic_straight_line(u, goal)
    return float(0.6 * man + 0.4 * eu)
