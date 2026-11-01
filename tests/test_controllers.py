from __future__ import annotations

import math
import unittest

from src.tom_jerry_chase.controllers import direct_pursuit_velocity, waypoint_velocity


class DirectPursuitTests(unittest.TestCase):
    def test_points_at_target_with_requested_speed(self) -> None:
        vx, vy = direct_pursuit_velocity((0.0, 0.0), (3.0, 4.0), speed=2.0)
        self.assertAlmostEqual(vx, 1.2)
        self.assertAlmostEqual(vy, 1.6)
        self.assertAlmostEqual(math.hypot(vx, vy), 2.0)

    def test_zero_distance_means_zero_action(self) -> None:
        self.assertEqual(direct_pursuit_velocity((1.0, 1.0), (1.0, 1.0), 2.0), (0.0, 0.0))

    def test_negative_speed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            direct_pursuit_velocity((0.0, 0.0), (1.0, 0.0), -1.0)


class WaypointTests(unittest.TestCase):
    def test_reached_waypoint_advances_to_next_target(self) -> None:
        velocity, index = waypoint_velocity(
            (0.0, 0.0), [(0.0, 0.0), (0.0, 2.0)], 0, speed=1.0
        )
        self.assertEqual(index, 1)
        self.assertEqual(velocity, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
