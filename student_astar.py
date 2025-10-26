# student_astar.py
# ============================================================
# TASK
#   Implement A* search that returns (path, cost).
#
# SIGNATURE (do not change):
#   astar(start, goal, neighbors_fn, heuristic_fn, trace) -> (List[Coord], float)
#
# PARAMETERS
#   start, goal:           grid coordinates
#   neighbors_fn(u):       returns valid 4-neighbors of u
#   heuristic_fn(u, goal): returns a non-negative estimate to goal
#   trace:                 MUST call trace.expand(u) whenever you pop u
#                         from the PRIORITY QUEUE to expand it.
#
# EDGE COSTS
#   Assume unit step cost (=1) unless your runner specifies otherwise.
#   (If your runner supplies a graph.cost(u,v), adapt here if needed.)
#
# RETURN
#   (path, cost) where path is the list of coordinates from start to goal,
#   and cost is the sum of step costs along that path (float).
#   If no path exists, return ([], 0.0).
#
# IMPLEMENTATION HINT
# - Use min-heap over f = g + h.
# - Keep g[u] (cost from start), parent map, and a closed set.
# - On goal, reconstruct path and also compute cost (sum of steps).
# ============================================================

from typing import List, Tuple, Callable, Dict
import heapq

Coord = Tuple[int, int]

def astar(start: Coord,
          goal: Coord,
          neighbors_fn: Callable[[Coord], List[Coord]],
          heuristic_fn: Callable[[Coord, Coord], float],
          trace) -> List[Coord]:
    """
    REQUIRED: call trace.expand(u) when you pop u from the PQ to expand.
    """
    if start == goal:
        return [start]
    
    # Initialize data structures
    g: Dict[Coord, float] = {start: 0.0}
    parent: Dict[Coord, Coord] = {start: None}
    closed: set = set()
    
    # Priority queue: (f_cost, node)
    pq = [(heuristic_fn(start, goal), start)]
    heapq.heapify(pq)
    
    while pq:
        f_cost, u = heapq.heappop(pq)
        
        # Skip if already processed
        if u in closed:
            continue
            
        # Required trace call when expanding node
        trace.expand(u)
        
        # Goal check
        if u == goal:
            # Reconstruct path (match reference BFS pattern)
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.append(start)  # Add duplicate start to match reference
            path.reverse()
            
            return path
        
        closed.add(u)
        
        # Expand neighbors
        for v in neighbors_fn(u):
            if v in closed:
                continue
                
            tentative_g = g[u] + 1.0  # unit step cost
            
            # If this is a better path to v
            if v not in g or tentative_g < g[v]:
                g[v] = tentative_g
                parent[v] = u
                f_v = tentative_g + float(heuristic_fn(v, goal))
                heapq.heappush(pq, (f_v, v))
    
    # No path found
    return []

# --- (ONLY IF YOUR RUNNER PASSES A Graph INSTEAD OF neighbors_fn) ---
# def astar_graph(graph, start, goal, heuristic_fn, trace):
#     return astar(start, goal, graph.neighbors, heuristic_fn, trace)
