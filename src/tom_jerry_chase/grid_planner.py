"""Simulator-independent occupancy grid and A* path planner.

The grid is a simplified top-down map of the room.  A cell is either free or
blocked; A* searches free cells for the lowest-cost route from Tom to Jerry.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math


Cell = tuple[int, int]
Point = tuple[float, float]
Rectangle = tuple[float, float, float, float]


@dataclass(frozen=True)
class GridSpec:
    """Describe a regularly spaced grid in world coordinates."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution: float

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError("grid maximums must not be smaller than minimums")

    @property
    def width(self) -> int:
        return int(round((self.max_x - self.min_x) / self.resolution)) + 1

    @property
    def height(self) -> int:
        return int(round((self.max_y - self.min_y) / self.resolution)) + 1

    def contains(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def world_to_cell(self, point: Point) -> Cell:
        """Return the closest in-bounds grid cell to a world position."""

        col = round((point[0] - self.min_x) / self.resolution)
        row = round((point[1] - self.min_y) / self.resolution)
        return (
            min(max(int(col), 0), self.width - 1),
            min(max(int(row), 0), self.height - 1),
        )

    def cell_to_world(self, cell: Cell) -> Point:
        if not self.contains(cell):
            raise ValueError(f"cell is outside grid: {cell}")
        return (
            self.min_x + cell[0] * self.resolution,
            self.min_y + cell[1] * self.resolution,
        )


def blocked_cells_from_rectangles(
    grid: GridSpec,
    rectangles: tuple[Rectangle, ...] | list[Rectangle],
    inflation: float = 0.0,
) -> set[Cell]:
    """Rasterize obstacle rectangles after expanding them by Tom's radius."""

    if inflation < 0.0:
        raise ValueError("inflation must be non-negative")

    blocked: set[Cell] = set()
    for col in range(grid.width):
        for row in range(grid.height):
            cell = (col, row)
            x, y = grid.cell_to_world(cell)
            if any(
                min_x - inflation <= x <= max_x + inflation
                and min_y - inflation <= y <= max_y + inflation
                for min_x, max_x, min_y, max_y in rectangles
            ):
                blocked.add(cell)
    return blocked


def _heuristic(first: Cell, second: Cell) -> int:
    """Manhattan distance is exact in an empty four-neighbor grid."""

    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _neighbors(grid: GridSpec, cell: Cell, blocked: set[Cell]) -> list[Cell]:
    candidates = (
        (cell[0] + 1, cell[1]),
        (cell[0] - 1, cell[1]),
        (cell[0], cell[1] + 1),
        (cell[0], cell[1] - 1),
    )
    return [candidate for candidate in candidates if grid.contains(candidate) and candidate not in blocked]


def astar_path(grid: GridSpec, start: Cell, goal: Cell, blocked: set[Cell]) -> list[Cell]:
    """Find a shortest free-cell path, including start and goal.

    Returns an empty list when no route exists.  The deterministic counter
    makes equal-cost choices reproducible across Python versions.
    """

    if not grid.contains(start) or not grid.contains(goal):
        raise ValueError("start and goal must be inside the grid")
    if start in blocked or goal in blocked:
        return []
    if start == goal:
        return [start]

    counter = itertools.count()
    frontier: list[tuple[int, int, Cell]] = [(0, next(counter), start)]
    came_from: dict[Cell, Cell] = {}
    cost_so_far: dict[Cell, int] = {start: 0}

    while frontier:
        _priority, _order, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in _neighbors(grid, current, blocked):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + _heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, next(counter), neighbor))

    return []


def path_length_m(path: list[Cell], resolution: float) -> float:
    """Measure a four-connected grid path in meters."""

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if len(path) < 2:
        return 0.0
    return math.fsum(
        math.dist(path[index], path[index + 1]) * resolution
        for index in range(len(path) - 1)
    )
