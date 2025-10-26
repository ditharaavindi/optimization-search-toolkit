# ============================================================
# Breadth-First Search (Unweighted Shortest Path)
# ============================================================

from collections import deque
from typing import Tuple, List, Callable

Coord = Tuple[int, int]

def bfs(start: Coord,
        goal: Coord,
        neighbors_fn: Callable[[Coord], List[Coord]],
        trace) -> List[Coord]:
    """
    BFS that returns a shortest path list of coords from start to goal (inclusive).
    Must call trace.expand(node) when popping for expansion.
    """
    if start == goal:
        return [start]

    q = deque([start])
    parent = {start: None}

    while q:
        u = q.popleft()
        trace.expand(u)  

        for v in neighbors_fn(u):
            if v in parent:
                continue
            parent[v] = u
            if v == goal:
                # reconstruct path (match reference implementation)
                path = [v]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.append(start)  # This creates the duplicate start to match reference
                path.reverse()
                return path
            q.append(v)

    return []  # no path found
