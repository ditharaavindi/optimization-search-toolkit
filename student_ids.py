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
        max_depth: int = 64) -> List[Coord]:
    """
    REQUIRED: call trace.expand(u) in the DLS when you expand u.
    """
    
    def dls(u: Coord, remaining: int, parent: Dict[Coord, Optional[Coord]], 
            seen: Set[Coord]) -> bool:
        """Depth Limited Search helper function"""
        trace.expand(u)
        
        if u == goal:
            return True
            
        if remaining == 0:
            return False
            
        for v in neighbors_fn(u):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                if dls(v, remaining - 1, parent, seen):
                    return True
                # Backtrack: remove v from seen for other branches
                seen.remove(v)
                
        return False
    
    # Try increasing depth limits
    for limit in range(0, max_depth + 1):
        parent: Dict[Coord, Optional[Coord]] = {start: None}
        seen: Set[Coord] = {start}
        
        if dls(start, limit, parent, seen):
            # Reconstruct path (match reference BFS pattern)
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.append(start)  # Add duplicate start to match reference
            path.reverse()
            return path
    
    # No solution found within max_depth
    return []
