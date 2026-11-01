# Stage 4: A* pathfinding

This fixed-seed run uses a 0.5 m occupancy grid and four-connected A* search.
Tom and Jerry move at 3.0 m/s and 1.1 m/s. A surprise wall appears at frame 24
and blocks the original upper route, forcing Tom to replan through the lower
route.

The full per-frame CSV, PNG sequence, and GIF remain in the Brev output
directory. The reviewed JSON summary is kept here for reproducibility.

A* is a deterministic graph-search algorithm, not machine learning. It uses a
map and a heuristic; it does not train or learn a policy.
