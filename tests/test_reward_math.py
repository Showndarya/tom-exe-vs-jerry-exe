from __future__ import annotations

import unittest

from src.tom_jerry_chase.reward_math import chase_reward_scalar


class ChaseRewardTests(unittest.TestCase):
    def test_moving_toward_jerry_is_better_than_moving_away(self) -> None:
        toward = chase_reward_scalar(2.0, 1.8, 0.5)
        away = chase_reward_scalar(2.0, 2.2, 0.5)
        self.assertGreater(toward, away)

    def test_catch_bonus_dominates_a_single_step(self) -> None:
        normal = chase_reward_scalar(0.55, 0.50, 1.0)
        caught = chase_reward_scalar(0.55, 0.45, 1.0, caught=True)
        self.assertGreater(caught, normal + 10.0)

    def test_leaving_arena_is_penalized(self) -> None:
        inside = chase_reward_scalar(2.0, 2.0, 0.0)
        outside = chase_reward_scalar(2.0, 2.0, 0.0, out_of_bounds=True)
        self.assertLess(outside, inside - 2.0)


if __name__ == "__main__":
    unittest.main()

