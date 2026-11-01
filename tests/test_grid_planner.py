from __future__ import annotations

import unittest

from src.tom_jerry_chase.grid_planner import (
    GridSpec,
    astar_path,
    blocked_cells_from_rectangles,
    path_length_m,
)


class GridSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GridSpec(-2.0, 2.0, -2.0, 2.0, 1.0)

    def test_world_cell_round_trip(self) -> None:
        cell = self.grid.world_to_cell((1.1, -0.9))
        self.assertEqual(cell, (3, 1))
        self.assertEqual(self.grid.cell_to_world(cell), (1.0, -1.0))

    def test_rectangle_is_inflated_for_robot_clearance(self) -> None:
        blocked = blocked_cells_from_rectangles(
            self.grid,
            [(-0.1, 0.1, -0.1, 0.1)],
            inflation=1.0,
        )
        self.assertTrue({(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)} <= blocked)


class AStarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GridSpec(0.0, 4.0, 0.0, 4.0, 1.0)

    def test_path_goes_around_wall(self) -> None:
        blocked = {(2, 0), (2, 1), (2, 2), (2, 3)}
        path = astar_path(self.grid, (0, 2), (4, 2), blocked)
        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1], (4, 2))
        self.assertTrue(all(cell not in blocked for cell in path))
        self.assertGreater(path_length_m(path, self.grid.resolution), 4.0)

    def test_no_path_when_wall_seals_grid(self) -> None:
        blocked = {(2, row) for row in range(self.grid.height)}
        self.assertEqual(astar_path(self.grid, (0, 2), (4, 2), blocked), [])

    def test_new_blockage_changes_replanned_route(self) -> None:
        first = astar_path(self.grid, (0, 2), (4, 2), set())
        new_blocked = {(2, 2)}
        second = astar_path(self.grid, (0, 2), (4, 2), new_blocked)
        self.assertIn((2, 2), first)
        self.assertNotIn((2, 2), second)


if __name__ == "__main__":
    unittest.main()
