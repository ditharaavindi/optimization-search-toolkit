# Optimization & Search — Run & Usage

Updated run instructions for the repository. This file documents how to run the included runner and view results locally after cloning the git repository.

## Quick summary

- The primary entrypoint for grading / running is `runner.py`.
- Running `runner.py` (from the repo root) will create `problem.json` and `results.json` which the `index.html` visualization consumes.
- The project uses only Python's standard library (no requirements file by default). A virtual environment is recommended.

## 1) Clone and open the project

Clone the repository (if you haven't already) and change into the project folder:

```bash
git clone <your-repo-url>
cd optimization-search-toolkit
```

Replace `<your-repo-url>` with the Git remote you use (GitHub, local path, etc.).

## 2) Create a Python virtualenv (recommended)

macOS (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If the project provides a `requirements.txt` later, install it with:

```bash
pip install -r requirements.txt
```

## 3) Run the main runner

Run from the repository root (where `runner.py` and the `student_*.py` modules live). Example:

```bash
python3 runner.py --student_id YOUR_STUDENT_ID
```

Example with overrides:

```bash
python3 runner.py --student_id TEST123 --seed 42 --rows 8 --cols 8 --density 0.25
```

Flags:

- `--student_id` (string): student id used as seed default and saved in output files (required-ish; default is `TEST`)
- `--seed` (string): optional seed override (if omitted the `student_id` is used as seed)
- `--rows`, `--cols` (int): grid dimensions (default 6x6)
- `--density` (float): obstacle density (default 0.22)

What the runner produces:

- `problem.json` — generated problem (grid size, obstacles, seed)
- `results.json` — grading output including per-algorithm results, scores, and hidden checks

Important: run `runner.py` from the repository root so the script can `import student_bfs`, `student_astar`, etc. (these are imported as top-level modules).

## 4) View the visualization (index.html)

Recommended: serve the repo root with a simple HTTP server and open `index.html` in a browser:

```bash
python3 -m http.server 8080
# then open http://localhost:8080/index.html in your browser
```

You can also use `open index.html` on macOS, but many browsers restrict certain file access when using the `file://` protocol. Serving via `http://` avoids that.

## 5) Running or testing individual modules (for development)

- To quickly test a student's BFS implementation interactively you can run a small REPL or a short script that imports `student_bfs` and calls its `bfs` function with a neighbors function. Example (python REPL):

```py
from importlib import import_module
S = import_module('student_bfs')
from runner import neighbors_4, START
obs = set()
n4 = neighbors_4(6, 6, obs)
trace = type('T', (), {'expanded': [], 'expand': lambda self, n: self.expanded.append(n)})()
path = S.bfs(START, (5,5), n4, trace)
print(path)
```

But note: the official grader uses the signatures shown in `runner.py` (e.g., `student.astar(start, goal, neighbors_fn, heur_fn, trace)`). When developing, keep those signatures.

## Troubleshooting & tips

- If you see `ModuleNotFoundError` when running `runner.py`, make sure your working directory is the repository root and the `student_*.py` files are present there.
- If results do not appear in the HTML, confirm `problem.json` and `results.json` exist and are up-to-date (re-run `runner.py`).
- If the HTTP server reports "Address already in use", pick another port (e.g., `8081`).

## Notes for maintainers

- `runner.py` uses only Python standard library modules (argparse, json, random, importlib, typing, etc.). If you add third-party packages, add a `requirements.txt` so users can `pip install -r requirements.txt`.
- Keep `student_*` files in repository root so the dynamic imports (`importlib.import_module`) work without package changes.

---
