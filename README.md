# SE3062 Optimization & Search Programming Assignment

## 📋 Overview

This repository contains a comprehensive implementation of various search algorithms, optimization techniques, and mathematical programming methods for the SE3062 Optimization & Search course assignment.

**Current Score: 100/100** ✅

## 🎯 Implemented Algorithms

### 1. **Search Algorithms**

- **Breadth-First Search (BFS)** - Finds shortest unweighted paths
- **A\* Search** - Optimal pathfinding with heuristic guidance
- **Iterative Deepening Search (IDS)** - Combines DFS space efficiency with BFS completeness

### 2. **Heuristic Functions**

- **Manhattan Distance** - Admissible for grid-based navigation
- **Euclidean Distance** - Straight-line distance heuristic
- **Custom Heuristic** - Weighted combination of Manhattan and Euclidean

### 3. **Optimization Techniques**

- **Simulated Annealing (SA)** - Metaheuristic for path optimization
- **Linear Programming (LP)** - 2D vertex enumeration method
- **Dynamic Programming (DP)** - 0/1 Knapsack problem solver

## 📁 File Structure

```
Assignment 2/
├── README.md                 # This file
├── runner.py                 # Main grading and execution script
├── index.html               # Web interface for results visualization
├── common.py                # Shared utilities and data structures
├── heuristics.py            # Heuristic function implementations
├── student_bfs.py           # BFS algorithm implementation
├── student_astar.py         # A* search algorithm implementation
├── student_ids.py           # Iterative Deepening Search implementation
├── student_sa.py            # Simulated Annealing implementation
├── student_lp_dp.py         # Linear Programming & Dynamic Programming
├── problem.json             # Generated problem instance data
├── results.json             # Generated results and scores
└── results/                 # Additional result files directory
```

## 🚀 How to Run

### Prerequisites

- Python 3.7 or higher
- No external libraries required (uses only standard library)

### Running the Assignment

1. **Navigate to the project directory:**

   ```bash
   cd "/Users/ditharaavindi/Desktop/year 3 sem 1/IS/take  home/Assignment 2"
   ```

2. **Execute the main runner script:**

   ```bash
   python3 runner.py --student_id YOUR_STUDENT_ID
   ```

   **Optional parameters:**

   ```bash
   python3 runner.py --student_id YOUR_ID --seed 12345 --rows 8 --cols 8 --density 0.3
   ```

   - `--student_id`: Your unique student identifier
   - `--seed`: Random seed for reproducible results
   - `--rows`, `--cols`: Grid dimensions (default: 6x6)
   - `--density`: Obstacle density (default: 0.22)

3. **The script will generate:**
   - `problem.json` - Problem instance data
   - `results.json` - Detailed results and scores

## 🌐 Viewing Results (HTML Interface)

### Method 1: Using HTTP Server (Recommended)

1. **Start a local HTTP server:**

   ```bash
   python3 -m http.server 8080
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:8080/index.html
   ```

### Method 2: Direct File Opening

```bash
open index.html
```

_Note: Some browsers may block local file access due to CORS policies_

## 📊 HTML Interface Features

The web interface provides:

### 📈 **Summary Table**

- Overall scores for each algorithm component
- Status indicators (✓ for perfect, ✗ for issues)
- Performance metrics and details

### 🎨 **Visualizations**

- **Grid Visualizations**: Shows search paths, obstacles, start/goal positions
- **SA Improvement Chart**: Displays annealing progress over iterations
- **LP Feasible Region**: Graphical representation of linear programming solution
- **DP Table Preview**: Dynamic programming computation table

### 📝 **Justification Areas**

- Text boxes for explaining algorithm design choices
- Sections for custom heuristic justification
- Areas for simulated annealing strategy explanation

### 💾 **Export to PDF**

1. Use browser's Print function (Ctrl/Cmd + P)
2. Select "Save as PDF"
3. Ensure all visualizations are included
4. Name file as: `SE3062_Assignment_YOUR_STUDENT_ID.pdf`

## 🔍 Algorithm Details

### Search Algorithms

- **BFS**: Guarantees shortest path on unweighted grids
- **A\***: Uses heuristics for efficient optimal pathfinding
- **IDS**: Memory-efficient search with completeness guarantees

### Optimization Methods

- **SA**: Improves paths using temperature-controlled acceptance
- **LP**: Solves 2D optimization problems graphically
- **DP**: Efficiently solves knapsack optimization problems

### Heuristic Design

- All heuristics are **admissible** (never overestimate)
- **Custom heuristic**: `0.6 × Manhattan + 0.4 × Euclidean`
- Balances grid-awareness with directional guidance

## 🐛 Troubleshooting

### Common Issues

1. **"Address already in use" error:**

   ```bash
   # Try a different port
   python3 -m http.server 8081
   ```

2. **HTML not displaying results:**

   - Ensure you're using HTTP server (not file:// protocol)
   - Check that `results.json` and `problem.json` exist
   - Verify Python script ran without errors

3. **Import errors:**

   - Ensure all student files are in the same directory
   - Check Python version compatibility

4. **Visualization not appearing:**
   - Use HTTP server instead of direct file opening
   - Check browser console for JavaScript errors

## 📚 Assignment Requirements

### Scoring Breakdown (100 points total)

- **BFS**: 10 points - Correct shortest path implementation
- **A\***: 15 points - Optimal pathfinding with heuristics
- **IDS**: 15 points - Proper depth-limited search
- **SA**: 15 points - Effective path improvement
- **Heuristics**: 20 points - Admissible heuristic functions
- **LP**: 12.5 points - Vertex enumeration method
- **DP**: 12.5 points - Knapsack optimization

### Key Requirements

✅ All algorithms must call `trace.expand()` when expanding nodes  
✅ Heuristics must be admissible (never overestimate true cost)  
✅ SA must improve upon BFS baseline by >1.0 units  
✅ LP must find optimal solution at feasible vertices  
✅ DP implementations must agree on optimal value

## 🎓 Submission Guidelines

### Required Deliverables

1. **PDF Report**: Export of `index.html` with justifications

   - Include all visualizations and scores
   - Add written explanations for algorithm choices
   - Name format: `SE3062_Assignment_STUDENT_ID.pdf`

2. **Source Code**: All Python implementation files
   - `student_bfs.py`, `student_astar.py`, `student_ids.py`
   - `student_sa.py`, `student_lp_dp.py`, `heuristics.py`

### Academic Integrity

- All implementations must be original work
- External libraries/code copying is prohibited
- Be prepared to explain your algorithms in a viva if requested

## 📞 Support

For questions or issues:

- Check the course forum for common problems
- Review the assignment specification document
- Ensure your implementation follows the exact function signatures
- Test with different seeds to verify robustness

---

**🏆 Achievement: Perfect Score 100/100**

This implementation demonstrates comprehensive understanding of search algorithms, optimization techniques, and mathematical programming methods required for the SE3062 course.
