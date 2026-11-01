# Stage 3: Direct pursuit

This directory contains reviewed, reproducible results from the first
controller baseline. The full per-frame CSV, PNG sequence, and GIF remain in
the Brev output directory because generated media is large and is not needed
to reproduce the controller.

The result is intentionally a failure: Tom sees Jerry's relative position but
not the obstacles, collides with the route-splitting wall, and does not catch
Jerry. Stage 4 will compare this same fixed-seed scenario with A* pathfinding.
