# Optimization & Search Algorithms

This project implements a comprehensive suite of artificial intelligence search and optimization algorithms for solving pathfinding, constraint optimization, and combinatorial problems. The implementation includes classic search algorithms with performance analysis and interactive visualization.

## Project Overview

This SE3062 assignment demonstrates the implementation and comparative analysis of fundamental AI algorithms:

- **Uninformed Search**: Breadth-First Search (BFS), Iterative Deepening Search (IDS)
- **Informed Search**: A\* with multiple heuristic functions
- **Metaheuristic Optimization**: Simulated Annealing with temperature scheduling
- **Mathematical Programming**: Linear Programming using vertex enumeration
- **Dynamic Programming**: 0/1 Knapsack problem with dual implementation approaches
- **Heuristic Design**: Manhattan, Euclidean, and custom admissible heuristics

## 📁 Project Structure

```
SE3062_Assignment/
├── README.md                   # Project documentation
├── runner_Msc.py              # Main execution controller (read-only)
├── common.py                  # Shared utilities and trace functionality
├── problem.json               # Generated grid problem instance
├── results.json               # Complete algorithm performance data
├── index.html                 # Interactive results dashboard
├── heuristics.py              # Admissible heuristic function implementations
├── student_bfs.py             # BFS implementation with optimal pathfinding
├── student_astar.py           # A* search with heuristic guidance
├── student_ids.py             # Iterative deepening with depth-limited search
├── student_sa.py              # Simulated annealing path optimization
├── student_lp_dp.py           # Linear & dynamic programming solutions
├── results/                   # Detailed algorithm output directory
│   ├── search_metrics.json    # Node expansion and timing data
│   ├── knapsack_result.json   # DP algorithm detailed results
│   ├── lp_result.json         # Linear programming solution steps
│   └── heuristics_stats.json  # Heuristic admissibility validation
└── __pycache__/              # Python bytecode cache
```

## System Requirements

- **Python 3.8+** (tested with Python 3.13)
- Standard Python libraries (no external dependencies)
  - `json`, `math`, `random`, `collections`
  - `typing`, `heapq`, `itertools`
  - Built-in `http.server` for visualization

## Quick Start Guide

### Basic Execution

Execute the complete algorithm suite:

```bash
python3 runner_Msc.py
```

### Custom Problem Generation

Generate specific problem instances:

```bash
python3 runner_Msc.py --student_id "SE3062_2025" --seed "CUSTOM" --rows 8 --cols 8 --density 0.25
```

### Configuration Parameters

- `--student_id`: Student identifier (default: "TEST")
- `--seed`: Reproducibility seed (default: derived from student_id)
- `--rows`: Grid height dimension (default: 6)
- `--cols`: Grid width dimension (default: 6)
- `--density`: Obstacle coverage ratio 0.0-1.0 (default: 0.22)

### Example Executions

```bash
# Large sparse grid
python3 runner_Msc.py --rows 12 --cols 12 --density 0.15

# Dense challenging grid
python3 runner_Msc.py --rows 8 --cols 8 --density 0.4

# Reproducible benchmark
python3 runner_Msc.py --seed "BENCHMARK_2025" --student_id "EVAL"
```

## Results & Visualization

### Generated Output Files

After execution, the system produces:

1. **`problem.json`** - Problem instance specification:

   - Grid topology with obstacle positions
   - Start (0,0) and goal (rows-1,cols-1) coordinates
   - Random seed for reproducibility

2. **`results.json`** - Comprehensive algorithm evaluation:

   - Path solutions and optimality verification
   - Performance metrics (expansions, path length, execution time)
   - Scoring breakdown and validation status
   - Mathematical programming solutions

3. **`index.html`** - Interactive performance dashboard:
   - Visual grid representation with algorithm paths
   - Comparative performance analysis charts
   - Algorithm efficiency metrics

### Visualization Access

1. **Launch HTTP server** (recommended for full functionality):

   ```bash
   python3 -m http.server 8080
   ```

   Navigate to: `http://localhost:8080/index.html`

2. **Direct file access** (limited features):

   ```bash
   open index.html  # macOS
   # or double-click index.html in file manager
   ```

3. **Results inspection**:
   ```bash
   python3 -c "import json; print(json.dumps(json.load(open('results.json')), indent=2))"
   ```

## Algorithm Implementation Details

### 1. Search Algorithms

- **BFS** (`student_bfs.py`):

  - Complete breadth-first exploration
  - Guaranteed optimal path discovery
  - Node expansion tracking via Trace class

- **A\*** (`student_astar.py`):

  - Priority queue with f(n) = g(n) + h(n)
  - Admissible heuristic integration
  - Closed set optimization for efficiency

- **IDS** (`student_ids.py`):
  - Iterative depth-limited search
  - Memory-efficient depth-first exploration
  - Completeness with optimal solution guarantee

### 2. Heuristic Functions (`heuristics.py`)

- **Manhattan Distance**: Grid-based L1 metric `|Δx| + |Δy|`
- **Euclidean Distance**: Straight-line L2 distance `√(Δx² + Δy²)`
- **Custom Heuristic**: Weighted combination for improved search guidance

### 3. Optimization Algorithm (`student_sa.py`)

- **Simulated Annealing**:
  - Probabilistic acceptance via Metropolis criterion
  - Exponential temperature cooling schedule
  - Path mutation operators (shortcut/detour)
  - Convergence tracking and restart mechanisms

### 4. Mathematical Programming (`student_lp_dp.py`)

- **Linear Programming**:

  - Constraint intersection and vertex enumeration
  - Feasible region boundary identification
  - Objective function optimization

- **Dynamic Programming**:
  - 0/1 Knapsack with capacity constraints
  - Bottom-up tabulation approach
  - Top-down memoization implementation

## 🎮 Usage Workflow

1. **Execute algorithms**:

   ```bash
   python3 runner_Msc.py --student_id "YOUR_ID"
   ```

2. **Verify completion**:

   ```
   Output: Wrote results.json and problem.json
   ```

3. **Launch visualization**:

   ```bash
   python3 -m http.server 8080
   ```

4. **Analyze performance**:
   - Open `http://localhost:8080/index.html`
   - Compare algorithm efficiency and optimality
   - Review scoring breakdown

## Performance Evaluation

### Scoring Criteria

The system evaluates algorithms across multiple dimensions:

- **Correctness**: Valid path from start to goal
- **Optimality**: Minimal path length achievement
- **Efficiency**: Node expansion minimization
- **Improvement**: Optimization gain over baseline

### Point Distribution

- **BFS (10 points)**: Optimal pathfinding completeness
- **Heuristics (20 points)**: Admissibility validation (Manhattan: 5, Euclidean: 5, Custom: 10)
- **A\* (15 points)**: Informed optimal search
- **IDS (15 points)**: Memory-efficient optimal search
- **Simulated Annealing (15 points)**: Path optimization and annealing behavior
- **Linear Programming (12.5 points)**: Constraint optimization accuracy
- **Dynamic Programming (12.5 points)**: Knapsack solution correctness

### Performance Metrics

- **Path Optimality**: Length comparison against known optimal
- **Search Efficiency**: Node expansions relative to problem size
- **Heuristic Quality**: Admissibility and search guidance effectiveness
- **Optimization Convergence**: SA improvement trajectory and plateau detection

## Troubleshooting Guide

### Common Issues

1. **Import/Module Errors**:

   ```bash
   # Verify file presence
   ls -la student_*.py heuristics.py common.py
   ```

2. **Execution Permissions**:

   ```bash
   chmod +x runner_Msc.py
   python3 runner_Msc.py  # Alternative execution
   ```

3. **Empty or Invalid Results**:

   - Check Python syntax in implementation files
   - Ensure all required functions are implemented
   - Verify return types match expected signatures

4. **Visualization Loading Issues**:
   - Use HTTP server instead of direct file access
   - Check browser console for JavaScript errors
   - Ensure `results.json` is properly formatted

### Debugging Strategies

Add diagnostic output to algorithm implementations:

```python
# Example debugging in student_*.py files
print(f"Algorithm: Current state = {current_state}")
print(f"Search: Expanding node {node} with f-value {f_value}")
```

## Testing Scenarios

### Algorithm Stress Testing

```bash
# Minimal obstacles (easy pathfinding)
python3 runner_Msc.py --density 0.05 --rows 4 --cols 4

# High obstacle density (challenging search)
python3 runner_Msc.py --density 0.45 --rows 10 --cols 10

# Large grid performance test
python3 runner_Msc.py --rows 15 --cols 15 --density 0.3
```

### Reproducibility Validation

```bash
# Fixed seed for consistent results
python3 runner_Msc.py --seed "VALIDATION_SEED"
```

## Academic Context

This implementation covers classical AI algorithms from:

- **Uninformed Search**: BFS completeness and optimality guarantees
- **Informed Search**: A\* admissibility and heuristic design principles
- **Local Search**: Simulated annealing convergence and parameter tuning
- **Constraint Optimization**: Linear programming duality and vertex methods
- **Dynamic Programming**: Optimal substructure and memoization techniques

## Learning Outcomes

Upon completion, this project demonstrates:

- **Search Algorithm Design**: Understanding of completeness, optimality, and complexity
- **Heuristic Development**: Admissibility constraints and search guidance
- **Optimization Techniques**: Metaheuristic parameter tuning and convergence analysis
- **Mathematical Programming**: Constraint handling and solution methodologies
- **Performance Analysis**: Comparative evaluation and efficiency measurement

## Development Guidelines

When extending or modifying implementations:

1. Preserve function signatures for compatibility with runner
2. Maintain Trace class usage for expansion counting
3. Ensure heuristic admissibility properties
4. Include proper error handling and edge cases
5. Test with multiple problem instances and seeds

## References

Implementation based on established algorithms from:

- Russell, S. & Norvig, P.: "Artificial Intelligence: A Modern Approach" (4th Ed.)
- Cormen, T. et al.: "Introduction to Algorithms" (3rd Ed.)
- Kirkpatrick, S. et al.: "Optimization by Simulated Annealing" (1983)
- Academic literature on admissible heuristics and search optimization

---
