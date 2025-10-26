# student_lp_dp.py
from __future__ import annotations
from typing import List, Tuple, Optional
from functools import lru_cache
import math

"""
===========================================================
Overall Pseudocode & Study Guide (LP + DP)
===========================================================

A) Linear Programming in 2 variables (vertex enumeration)
   Goal: maximize Z = c1*x + c2*y subject to a1*x + a2*y <= b (and x>=0, y>=0)

   1) Model the feasible region:
      - Collect all given constraints (<= type).
      - Add non-negativity constraints: x>=0, y>=0.

   2) Enumerate candidate vertices:
      - Intersect every pair of constraint boundary lines (treat each as equality).
      - Keep only well-defined intersections (ignore parallel lines).
      - (Optionally) include the origin explicitly.

   3) Feasibility test:
      - For each candidate (x,y), check all constraints (<= type) with a small numeric tolerance.

   4) Objective evaluation:
      - Evaluate Z at each feasible vertex.
      - Select the best according to Z; tie-break deterministically if needed.

B) 0/1 Knapsack (Dynamic Programming)
   Problem: given values[i], weights[i], capacity C, pick subset to maximize total value without
            exceeding C.

   1) Bottom-Up Table (iterative):
      - Define dp[i][cap] = best value using items from i..n-1 with remaining capacity 'cap'.
      - Fill the table in an order that ensures subproblems are ready (e.g., i from n-1→0).
      - Transition: choose between skipping item i or taking it (if it fits), then record the best.

   2) Top-Down with Memoization (recursive):
      - Define f(i, cap): best value using items from i..n-1 with capacity 'cap'.
      - Base cases: end of items or cap==0 -> return 0.
      - Transition: if item i doesn’t fit, skip; else max(skip, take).
      - Cache results to avoid recomputation.

Notes:
- Use a small tolerance EPS for LP comparisons with floats.
- Keep implementations simple, readable, and consistent with the above plan.
"""

# ---------- LP (12.5% of total grade) ----------
Constraint = Tuple[float, float, float]  # a1, a2, b  meaning  a1*x + a2*y <= b
EPS = 1e-9

def _intersect(c1: Constraint, c2: Constraint) -> Optional[Tuple[float, float]]:
    """
    Compute the intersection point of two *boundary lines* obtained from constraints.
    Each constraint (a1, a2, b) corresponds to a boundary line a1*x + a2*y = b.

    Detailed steps (do NOT paste final formulae; write the algebra yourself):
      1) Unpack both constraints into coefficients.
      2) Treat them as a 2x2 linear system in variables x and y.
      3) Compute the determinant of the 2x2 coefficient matrix.
         - If it's (near) zero, lines are parallel/ill-conditioned → return None.
      4) Otherwise, solve the system for (x, y) using your preferred method for 2x2 systems.
      5) Return (x, y) as floats.

    Return:
      (x, y) if a unique intersection exists and is well-conditioned; otherwise None.
    """
    # Unpack constraint coefficients
    a1, a2, b1 = c1  # a1*x + a2*y = b1
    c1_coeff, c2_coeff, b2 = c2  # c1*x + c2*y = b2
    
    # Compute determinant of the coefficient matrix
    # | a1  a2 |
    # | c1  c2 |
    det = a1 * c2_coeff - a2 * c1_coeff
    
    # Check if lines are parallel (near-zero determinant)
    if abs(det) < EPS:
        return None
    
    # Solve using Cramer's rule:
    # x = (b1*c2 - a2*b2) / det
    # y = (a1*b2 - b1*c1) / det
    x = (b1 * c2_coeff - a2 * b2) / det
    y = (a1 * b2 - b1 * c1_coeff) / det
    
    return (x, y)


def _is_feasible(pt: Tuple[float, float], constraints: List[Constraint]) -> bool:
    """
    Check whether point (x,y) satisfies ALL constraints a1*x + a2*y <= b (with tolerance).

    Detailed steps:
      1) For each constraint (a1, a2, b), compute the left-hand side at (x,y).
      2) Compare LHS to RHS with a small EPS slack to account for floating-point rounding.
      3) If any constraint is violated beyond tolerance, return False.
      4) If all pass, return True.
    """
    x, y = pt
    
    for a1, a2, b in constraints:
        # Compute left-hand side: a1*x + a2*y
        lhs = a1 * x + a2 * y
        
        # Check constraint: lhs <= b (with tolerance)
        if lhs > b + EPS:  # Violated beyond tolerance
            return False
    
    return True


def feasible_vertices(constraints: List[Constraint]) -> List[Tuple[float, float]]:
    """
    (6%) Enumerate and return all *feasible* vertices (x,y) of the polygonal feasible region.

    Detailed steps:
      1) Copy input constraints and append non-negativity:
         - Represent x>=0 and y>=0 as <=-type constraints suitable for your intersection logic.
           (Hint: you'll add two extra constraints to the list.)
      2) For every unordered pair of constraints, compute the intersection of their *boundary lines*.
         - Skip pairs that do not produce a unique intersection.
      3) Collect all intersection points plus the origin (as a simple additional candidate).
      4) Run the feasibility test on each candidate using _is_feasible.
      5) De-duplicate points robustly (e.g., rounding to fixed decimals or using a tolerance-based key).
      6) Return the list of unique feasible vertices.
    """
    # Copy input constraints and add non-negativity constraints
    all_constraints = constraints[:]
    all_constraints.append((-1.0, 0.0, 0.0))  # -x <= 0  (i.e., x >= 0)
    all_constraints.append((0.0, -1.0, 0.0))  # -y <= 0  (i.e., y >= 0)
    
    candidates = []
    
    # Add origin as a candidate
    candidates.append((0.0, 0.0))
    
    # Generate intersection points from all pairs of constraints
    n = len(all_constraints)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = _intersect(all_constraints[i], all_constraints[j])
            if intersection is not None:
                candidates.append(intersection)
    
    # Filter feasible candidates
    feasible = []
    for pt in candidates:
        if _is_feasible(pt, all_constraints):
            feasible.append(pt)
    
    # De-duplicate by rounding to avoid floating-point precision issues
    unique_points = []
    seen = set()
    
    for x, y in feasible:
        # Round to 10 decimal places for comparison
        rounded = (round(x, 10), round(y, 10))
        if rounded not in seen:
            seen.add(rounded)
            unique_points.append((x, y))
    
    return unique_points


def maximize_objective(vertices: List[Tuple[float, float]], c1: float, c2: float) -> Tuple[Tuple[float, float], float]:
    """
    (6.5%) Evaluate Z = c1*x + c2*y over feasible vertices and return (best_point, best_value).

    Detailed steps:
      1) Handle edge case: if vertices is empty, return a sensible default ((0.0, 0.0), 0.0).
      2) Initialize "best" with the first vertex and its objective value.
      3) Scan through remaining vertices:
         - Compute Z at each vertex.
         - If strictly better (beyond EPS), update best.
         - If tied within EPS, resolve deterministically (e.g., prefer larger x; if x ties, larger y).
      4) Return the best vertex and its value as a float.
    """
    if not vertices:
        return ((0.0, 0.0), 0.0)
    
    # Initialize with first vertex
    best_point = vertices[0]
    best_value = c1 * best_point[0] + c2 * best_point[1]
    
    # Scan through remaining vertices
    for vertex in vertices[1:]:
        x, y = vertex
        current_value = c1 * x + c2 * y
        
        # Check if strictly better
        if current_value > best_value + EPS:
            best_point = vertex
            best_value = current_value
        elif abs(current_value - best_value) <= EPS:
            # Tie-breaking: prefer larger x, then larger y
            if (x > best_point[0] + EPS) or \
               (abs(x - best_point[0]) <= EPS and y > best_point[1] + EPS):
                best_point = vertex
                best_value = current_value
    
    return (best_point, best_value)


# ---------- DP (12.5% of total grade) ----------
def knapsack_bottom_up(values: List[int], weights: List[int], capacity: int) -> int:
    """
    (6.5%) Bottom-up 0/1 knapsack. Return the optimal value (int).

    Table design (choose one and stick to it):
      Option A (common): dp[i][cap] = best value using items i..n-1 with remaining capacity 'cap'.
        - Dimensions: (n+1) x (capacity+1), initialized to 0.
        - Fill order: i from n-1 down to 0; cap from 0 to capacity.
        - Transition:
            skip = dp[i+1][cap]
            take = values[i] + dp[i+1][cap - weights[i]]  (only if it fits)
            dp[i][cap] = max(skip, take)

      Option B: dp[i][cap] = best value using first i items (0..i-1).
        - Dimensions: (n+1) x (capacity+1).
        - Fill order: i from 1 to n; cap from 0 to capacity.
        - Transition mirrors Option A but with shifted indices.

    Detailed steps:
      1) Validate input lengths and capacity.
      2) Allocate and initialize the 2D table to zeros.
      3) Implement your chosen formulation consistently, filling the table.
      4) Return the appropriate cell as the answer (depends on formulation).
    """
    # Validate inputs
    n = len(values)
    if n != len(weights) or capacity < 0:
        return 0
    
    # Use Option A: dp[i][cap] = best value using items i..n-1
    # Dimensions: (n+1) x (capacity+1)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill table from i = n-1 down to 0
    for i in range(n - 1, -1, -1):
        for cap in range(capacity + 1):
            # Option 1: Skip current item
            skip = dp[i + 1][cap]
            
            # Option 2: Take current item (if it fits)
            take = 0
            if weights[i] <= cap:
                take = values[i] + dp[i + 1][cap - weights[i]]
            
            # Take the maximum of skip and take
            dp[i][cap] = max(skip, take)
    
    # Answer is in dp[0][capacity]
    return dp[0][capacity]


def knapsack_top_down(values: List[int], weights: List[int], capacity: int) -> int:
    """
    (6%) Top-down (memoized) 0/1 knapsack. Return optimal value (int).

    Recurrence (typical):
      f(i, cap) = 0                                     if i==n or cap==0
      f(i, cap) = f(i+1, cap)                           if weights[i] > cap
      f(i, cap) = max(
                      f(i+1, cap),                      # skip item i
                      values[i] + f(i+1, cap - w[i])    # take item i
                   )                                    otherwise

    Detailed steps:
      1) Define an inner function f(i, cap) and decorate with @lru_cache(None).
      2) Implement the base cases (past last item or capacity empty).
      3) Implement the recurrence using the rule above.
      4) Return f(0, capacity).
    """
    n = len(values)
    if n != len(weights) or capacity < 0:
        return 0

    @lru_cache(maxsize=None)
    def f(i: int, cap: int) -> int:
        # Base cases
        if i == n or cap == 0:
            return 0
        
        # If current item doesn't fit, skip it
        if weights[i] > cap:
            return f(i + 1, cap)
        
        # Choose max of skipping or taking current item
        skip = f(i + 1, cap)
        take = values[i] + f(i + 1, cap - weights[i])
        
        return max(skip, take)

    return f(0, capacity)


# ------------- Optional local smoke test -------------
if __name__ == "__main__":
    # Minimal checks that won't reveal answers; just ensures your functions run.
    cons = [
        (1.0, 1.0, 6.0),
        (1.0, 0.0, 4.0),
        (0.0, 1.0, 5.0),
        (2.0, 1.0, 8.0),
    ]
    try:
        V = feasible_vertices(cons)
        print(f"[LP] #vertices found: {len(V)}")
        if V:
            bp, bv = maximize_objective(V, 3.0, 5.0)
            print(f"[LP] best vertex (masked): {bp}, value={bv:.2f}")
    except NotImplementedError:
        print("[LP] TODOs not yet implemented")

    vals = [6,5,18,15,10]
    wts  = [2,2,6,5,4]
    cap  = 10
    try:
        print("[DP] bottom-up (masked run):", knapsack_bottom_up(vals, wts, cap))
    except NotImplementedError:
        print("[DP] bottom-up TODO not implemented")
    try:
        print("[DP] top-down  (masked run):", knapsack_top_down(vals, wts, cap))
    except NotImplementedError:
        print("[DP] top-down  TODO not implemented")
