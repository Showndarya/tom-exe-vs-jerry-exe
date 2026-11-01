"""Small, simulator-independent controllers used by the chase baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence


Vec2 = tuple[float, float]


def direct_pursuit_velocity(tom_xy: Vec2, jerry_xy: Vec2, speed: float) -> Vec2:
    """Point Tom directly at Jerry with a constant planar speed.

    This controller deliberately knows nothing about walls.  It is the Stage 3
    baseline we will later compare with A* and reinforcement learning.
    """

    if speed < 0.0:
        raise ValueError("speed must be non-negative")

    dx = jerry_xy[0] - tom_xy[0]
    dy = jerry_xy[1] - tom_xy[1]
    distance = math.hypot(dx, dy)
    if distance == 0.0 or speed == 0.0:
        return (0.0, 0.0)
    return (speed * dx / distance, speed * dy / distance)


def waypoint_velocity(
    position_xy: Vec2,
    waypoints: Sequence[Vec2],
    waypoint_index: int,
    speed: float,
    tolerance: float = 0.2,
) -> tuple[Vec2, int]:
    """Drive toward one waypoint and advance cyclically when it is reached."""

    if not waypoints:
        raise ValueError("at least one waypoint is required")
    if speed < 0.0:
        raise ValueError("speed must be non-negative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    index = waypoint_index % len(waypoints)
    target = waypoints[index]
    if math.dist(position_xy, target) <= tolerance:
        index = (index + 1) % len(waypoints)
        target = waypoints[index]

    return direct_pursuit_velocity(position_xy, target, speed), index
