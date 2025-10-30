# student_ids.py
# ============================================================
# TASK
#   Implement Iterative Deepening Search (IDS).
#
# SIGNATURE (do not change):
#   ids(start, goal, neighbors_fn, trace, max_depth=64) -> (List[Coord], int)
#
# PARAMETERS
#   start, goal:       coordinates
#   neighbors_fn(u):   returns valid 4-neighbors of u
#   trace:             MUST call trace.expand(u) when you EXPAND u
#                      in the depth-limited search (DLS).
#   max_depth:         upper cap for the iterative deepening
#
# RETURN
#   (path, depth_limit_used)
#   - If found at depth L, return the path and L.
#   - If not found up to max_depth, return ([], max_depth).
#
# IMPLEMENTATION HINT
# - Outer loop: for limit in [0..max_depth]:
#       run DLS(start, limit) with its own parent dict and visited set
#       DLS(u, remaining):
#           trace.expand(u)
#           if u == goal: return True
#           if remaining == 0: return False
#           for v in neighbors_fn(u):
#               if v not seen in THIS DLS: mark parent[v]=u and recurse
# - Reconstruct the path when DLS reports success.
# ============================================================

from typing import List, Tuple, Callable, Dict, Optional, Set

Coord = Tuple[int, int]

def ids(start: Coord,
        goal: Coord,
        neighbors_fn: Callable[[Coord], List[Coord]],
        trace,
        max_depth: int = 64) -> Tuple[List[Coord], int]:
    """
    REQUIRED: call trace.expand(u) in the DLS when you expand u.
    """
    def dls(u: Coord, remaining: int, parent: Dict[Coord, Optional[Coord]], seen: Set[Coord]) -> bool:
        try:
            trace.expand(u)
        except Exception:
            pass
        
        # Check goal first like standard DFS
        if u == goal:
            return True
            
        if remaining == 0:
            return False
            
        # Process neighbors in consistent order
        for v in neighbors_fn(u):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                if dls(v, remaining-1, parent, seen):
                    return True
                # Remove from seen to allow different paths at different depths
                seen.remove(v)
        return False

    # Iterative deepening loop
    for limit in range(0, max_depth + 1):
        parent: Dict[Coord, Optional[Coord]] = {start: None}
        seen: Set[Coord] = {start}
        found = dls(start, limit, parent, seen)
        if found:
            # reconstruct path using same method as BFS
            path = [goal]
            curr = goal
            while parent[curr] is not None:
                p = parent[curr]
                if p is not None:
                    path.append(p)
                    curr = p
                else:
                    break
            path.reverse()
            
            # Add duplicate start coordinate to match BFS behavior exactly  
            if len(path) >= 1 and path[0] == start:
                path.insert(0, start)
            
            return path, limit

    return [], max_depth
