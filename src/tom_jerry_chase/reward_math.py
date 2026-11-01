"""Simulator-independent reward constants and scalar checks for Stage 5."""

from __future__ import annotations


PROGRESS_WEIGHT = 4.0
DISTANCE_WEIGHT = 0.02
ACTION_WEIGHT = 0.002
STEP_PENALTY = 0.01
CATCH_BONUS = 12.0
BOUNDARY_PENALTY = 3.0


def chase_reward_scalar(
    previous_distance: float,
    distance: float,
    action_squared_norm: float,
    *,
    caught: bool = False,
    out_of_bounds: bool = False,
) -> float:
    """Return the same shaped reward used by the vectorized Isaac Lab task.

    This scalar version lets us unit-test the learning signal on the Mac without
    importing Isaac Lab, CUDA, or Torch.
    """

    progress = previous_distance - distance
    return (
        PROGRESS_WEIGHT * progress
        - DISTANCE_WEIGHT * distance
        - ACTION_WEIGHT * action_squared_norm
        - STEP_PENALTY
        + CATCH_BONUS * float(caught)
        - BOUNDARY_PENALTY * float(out_of_bounds)
    )

