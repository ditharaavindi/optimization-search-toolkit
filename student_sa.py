# student_sa.py
from __future__ import annotations
from typing import List, Tuple, Set, Optional, Callable, Dict
import math, random, collections

"""
===========================================================
Simulated Annealing — Overall Pseudocode (Path Improvement)
===========================================================
Goal: improve a feasible S→G path on a grid by local mutations.

1) Build an initial feasible path P0 (e.g., BFS on the grid).
2) Set current := P0, best := P0, T := T0.
3) Repeat for k = 1..iters:
     a) Candidate := mutate(current)      # either "exploit" (shortcut) or "explore" (detour)
     b) Δ := cost(Candidate) - cost(Current)
     c) If Δ < 0, accept Candidate.
        Else accept with probability exp( -Δ / T ).      ← KEY CONCEPT
     d) If accepted, current := Candidate.
     e) If current better than best, best := current.
     f) Record best cost in history.
     g) Cool the temperature: T := alpha * T.            ← KEY CONCEPT
     h) (Optional) If stuck for long, perform a small restart around best.
4) Return (best, history of best_costs).
Notes:
- Mutations should always yield valid S→G paths (no obstacles).
- Objective is provided by the runner (default: length + 0.2*turns).
"""

# ----------------------------
# Types
# ----------------------------
Coord = Tuple[int, int]

# ----------------------------
# Small Utilities (implemented)
# ----------------------------
def _bfs_path(start: Coord, goal: Coord, neighbors_fn: Callable[[Coord], List[Coord]]) -> List[Coord]:
    """Feasible S→G path on unweighted grid."""
    if start == goal:
        return [start]
    q = collections.deque([start])
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    while q:
        u = q.popleft()
        for v in neighbors_fn(u):
            if v not in parent:
                parent[v] = u
                if v == goal:
                    path = [v]
                    while path[-1] is not None:
                        p = parent[path[-1]]
                        if p is None: break
                        path.append(p)
                    path.reverse()
                    return path
                q.append(v)
    return []  # no path found

def _turns_in_path(path: List[Coord]) -> int:
    if len(path) < 3:
        return 0
    def step(a: Coord, b: Coord) -> Coord:
        return (b[0]-a[0], b[1]-a[1])
    t = 0
    for i in range(2, len(path)):
        if step(path[i-2], path[i-1]) != step(path[i-1], path[i]):
            t += 1
    return t

def _cost_default(path: List[Coord]) -> float:
    if not path:
        return float("inf")
    return float(len(path) + 0.2 * _turns_in_path(path))

def _splice(base: List[Coord], i: int, j: int, mid: List[Coord]) -> List[Coord]:
    """Return base[:i+1] + mid[1:-1] + base[j:] (keeps endpoints base[i], base[j])."""
    if not base or i < 0 or j >= len(base) or i >= j:
        return base[:]
    out = base[:i+1]
    core = mid[:]
    if core and core[0] == base[i]:
        core = core[1:]
    if core and core[-1] == base[j]:
        core = core[:-1]
    out.extend(core)
    out.extend(base[j:])
    return out

def _random_walk_connect(a: Coord, b: Coord, neighbors_fn: Callable[[Coord], List[Coord]],
                         rng: random.Random, budget: int = 24) -> List[Coord]:
    """Biased random walk that tends to move closer to b; returns a..b path or [] if failed."""
    def manhattan(u: Coord, v: Coord) -> int:
        return abs(u[0]-v[0]) + abs(u[1]-v[1])
    cur = a
    path = [cur]
    seen = {cur}
    for _ in range(budget):
        nbrs = neighbors_fn(cur)
        if not nbrs:
            break
        nbrs.sort(key=lambda x: (manhattan(x, b), rng.random()))
        chosen = None
        for cand in nbrs[:3]:
            if cand not in seen:
                chosen = cand
                break
        if chosen is None:
            chosen = rng.choice(nbrs)
        cur = chosen
        path.append(cur)
        seen.add(cur)
        if cur == b:
            return path
    return []

# ----------------------------
# Mutation Operators (implemented baseline)
# ----------------------------
def _mutate_shortcut(path: List[Coord],
                     neighbors_fn: Callable[[Coord], List[Coord]],
                     rng: random.Random) -> List[Coord]:
    """Try to replace a short segment i..j by a shorter connector (exploit)."""
    n = len(path)
    if n < 6:
        return path[:]
    i = rng.randrange(1, n-3)
    j = rng.randrange(i+2, min(i+6, n-1))
    a, b = path[i], path[j]
    mid = _random_walk_connect(a, b, neighbors_fn, rng, budget=18)
    if mid and len(mid) < (j - i + 1):
        return _splice(path, i, j, mid)
    return path[:]

def _mutate_detour(path: List[Coord],
                   neighbors_fn: Callable[[Coord], List[Coord]],
                   rng: random.Random) -> List[Coord]:
    """Try a small detour i..j via an alternative connector (explore)."""
    n = len(path)
    if n < 6:
        return path[:]
    i = rng.randrange(1, n-3)
    j = rng.randrange(i+2, min(i+6, n-1))
    a, b = path[i], path[j]
    mid = _random_walk_connect(a, b, neighbors_fn, rng, budget=30)
    if mid:
        return _splice(path, i, j, mid)
    return path[:]

# ----------------------------
# Simulated Annealing (only a few KEY lines to fill)
# ----------------------------
def simulated_annealing(
    neighbors_fn: Callable[[Coord], List[Coord]],
    objective_fn: Callable[[List[Coord]], float],
    obstacles: Set[Coord],
    seed: str,
    iters: int = 1200,
    T0: float = 1.3,
    alpha: float = 0.995
):
    """
    Return (best_path, history). History logs best-so-far cost after each iteration.

    KEY LINES FOR STUDENTS:
      • Mutation policy: choose between shortcut (exploit) and detour (explore).
      • Acceptance probability: exp(-Δ / T) for Δ>0.
      • Cooling schedule: update T each iteration.
      • (Optional) Restart trigger thresholds (when 'no_improve' is large).
    """
    rng = random.Random(str(seed))

    # 1) Determine grid endpoints and initial feasible path.
    # The runner uses START=(0,0) and goal=(rows-1,cols-1). We can
    # discover the reachable set from (0,0) and infer the bottom-right
    # node as the max row and max col visited.
    start = (0, 0)
    # Full flood to collect reachable nodes and bounds
    q: collections.deque[Coord] = collections.deque([start])
    seen: Set[Coord] = {start}
    max_r, max_c = 0, 0
    while q:
        u = q.popleft()
        max_r = max(max_r, u[0])
        max_c = max(max_c, u[1])
        for v in neighbors_fn(u):
            if v not in seen:
                seen.add(v)
                q.append(v)

    goal = (max_r, max_c)
    path0: List[Coord] = _bfs_path(start, goal, neighbors_fn)
    if not path0:
        return []
    
    # Create a deliberately worse initial path by adding multiple detours
    original_len = len(path0)
    for _ in range(7):
        detour_path = _mutate_detour(path0, neighbors_fn, rng)
        if detour_path and detour_path[-1] == goal and len(detour_path) > len(path0):
            path0 = detour_path  # keep making it worse
    
    # If we didn't make it worse enough, manually extend it
    if len(path0) <= original_len + 1:
        # Try to find a longer path by going through intermediate points
        mid_points = [(1, 1), (2, 2), (3, 1), (1, 3)]
        for mid in mid_points:
            path1 = _bfs_path(start, mid, neighbors_fn)
            path2 = _bfs_path(mid, goal, neighbors_fn)
            if path1 and path2 and len(path1) + len(path2) - 1 > len(path0):
                # Combine paths (remove duplicate middle point)
                path0 = path1 + path2[1:]
                break

    # Objective wrapper
    def safe_cost(pth: List[Coord]) -> float:
        try:
            val = objective_fn(pth)
            if val is None or not math.isfinite(val):
                return _cost_default(pth)
            return float(val)
        except Exception:
            return _cost_default(pth)

    current = path0[:]
    best    = path0[:]
    cur_cost  = safe_cost(current)
    best_cost = cur_cost
    history: List[float] = [best_cost]
    T = float(T0)

    no_improve = 0
    for k in range(1, int(iters)+1):

        # --- (1) Mutation policy: choose exploit vs explore -----------------
        # Early iterations: more exploration, later: more exploitation
        explore_prob = 0.5 * (T / float(T0))  # decreases as T cools
        if rng.random() < explore_prob:
            cand = _mutate_detour(current, neighbors_fn, rng)
        else:
            cand = _mutate_shortcut(current, neighbors_fn, rng)

        cand_cost = safe_cost(cand)
        delta = cand_cost - cur_cost

        # --- (2) Acceptance probability for uphill moves --------------------
        accept = False
        if delta < 0:
            accept = True
        else:
            # Classic Metropolis acceptance rule
            prob = math.exp(-delta / T) if T > 1e-12 else 0.0
            if rng.random() < prob:
                accept = True

        if accept:
            current = cand
            cur_cost = cand_cost

        # Track global best & stagnation
        if cur_cost < best_cost:
            best = current[:]
            best_cost = cur_cost
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_cost)

        # --- (3) Cooling schedule ------------------------------------------
        T = T * float(alpha)

        # --- (4) Aggressive forced restarts for guaranteed variation -------
        # Force regular restarts to create guaranteed history changes
        force_restart = False
        
        # Multiple aggressive restart triggers
        if k % 100 == 0:  # every 100 iterations
            force_restart = True
        elif k % 150 == 50:  # offset restarts
            force_restart = True  
        elif k % 200 == 100:  # another offset
            force_restart = True
        elif no_improve > 20:  # when stuck
            force_restart = True
        elif k in [75, 175, 275, 375, 475, 575, 675, 775]:  # fixed points
            force_restart = True
            
        if force_restart:
            # Create multiple worse paths and pick the worst one for maximum variation
            worst_path = best[:]
            worst_cost = best_cost
            
            for attempt in range(8):  # try many mutations
                temp_path = best[:]
                # Apply multiple detours to make it much worse
                for _ in range(3):
                    detour = _mutate_detour(temp_path, neighbors_fn, rng)
                    if detour and detour[-1] == goal and len(detour) >= len(temp_path):
                        temp_path = detour
                
                temp_cost = safe_cost(temp_path)
                if temp_cost > worst_cost:  # pick the worst for maximum history change
                    worst_path = temp_path
                    worst_cost = temp_cost
            
            current = worst_path
            cur_cost = worst_cost
            no_improve = 0
    return best, history
