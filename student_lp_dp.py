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
    # Unpack constraints: a1*x + a2*y = b1, c1*x + c2*y = b2
    a1, a2, b1 = c1
    c1_coef, c2_coef, b2 = c2
    
    # Coefficient matrix determinant: | a1  a2 |
    #                                | c1  c2 |
    det = a1 * c2_coef - a2 * c1_coef
    
    # Check for near-zero determinant (parallel lines)
    if abs(det) < EPS:
        return None
    
    # Solve using Cramer's rule:
    # x = (b1*c2 - b2*a2) / det
    # y = (a1*b2 - c1*b1) / det
    x = (b1 * c2_coef - b2 * a2) / det
    y = (a1 * b2 - c1_coef * b1) / det
    
    return (float(x), float(y))


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
    for (a1, a2, b) in constraints:
        lhs = a1 * x + a2 * y
        if lhs > b + EPS:  # constraint violated beyond tolerance
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
    # Add non-negativity constraints: x >= 0 becomes -x <= 0, y >= 0 becomes -y <= 0
    all_constraints = constraints + [(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)]
    
    candidates = []
    
    # Generate intersection points from all pairs of constraints
    n = len(all_constraints)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _intersect(all_constraints[i], all_constraints[j])
            if pt is not None:
                candidates.append(pt)
    
    # Add origin as an additional candidate
    candidates.append((0.0, 0.0))
    
    # Filter feasible points
    feasible = []
    for pt in candidates:
        if _is_feasible(pt, all_constraints):
            feasible.append(pt)
    
    # De-duplicate using tolerance-based rounding
    unique = []
    for pt in feasible:
        # Round to avoid floating point precision issues
        rounded = (round(pt[0], 6), round(pt[1], 6))
        if rounded not in [(round(u[0], 6), round(u[1], 6)) for u in unique]:
            unique.append(pt)
    
    return unique


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
        return (0.0, 0.0), 0.0
    
    best_pt = vertices[0]
    best_val = c1 * best_pt[0] + c2 * best_pt[1]
    
    for pt in vertices[1:]:
        val = c1 * pt[0] + c2 * pt[1]
        
        # If strictly better, update
        if val > best_val + EPS:
            best_pt = pt
            best_val = val
        # If tied within tolerance, use deterministic tie-breaking
        elif abs(val - best_val) <= EPS:
            # Prefer larger x, if x ties then prefer larger y
            if pt[0] > best_pt[0] + EPS or (abs(pt[0] - best_pt[0]) <= EPS and pt[1] > best_pt[1] + EPS):
                best_pt = pt
                best_val = val
    
    return best_pt, float(best_val)


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
    n = len(values)
    if n != len(weights) or capacity < 0:
        return 0
    
    # Using Option A: dp[i][cap] = best value using items i..n-1 with capacity 'cap'
    # Dimensions: (n+1) x (capacity+1)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill from bottom-right to top-left
    for i in range(n - 1, -1, -1):
        for cap in range(capacity + 1):
            # Option 1: skip item i
            skip = dp[i + 1][cap]
            
            # Option 2: take item i (if it fits)
            take = 0
            if weights[i] <= cap:
                take = values[i] + dp[i + 1][cap - weights[i]]
            
            dp[i][cap] = max(skip, take)
    
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
        
        # Otherwise, choose max of skip vs take
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
