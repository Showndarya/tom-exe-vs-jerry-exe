"""Direct Isaac Lab environment for the first PPO chase curriculum step."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv

from ..reward_math import (
    ACTION_WEIGHT,
    BOUNDARY_PENALTY,
    CATCH_BONUS,
    DISTANCE_WEIGHT,
    PROGRESS_WEIGHT,
    STEP_PENALTY,
)

if TYPE_CHECKING:
    from .tom_jerry_env_cfg import TomJerryEasyEnvCfg


class TomJerryEasyEnv(DirectRLEnv):
    """Train a blue rigid sphere to reach an orange stationary target."""

    cfg: TomJerryEasyEnvCfg

    def __init__(self, cfg: TomJerryEasyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.tom = self.scene["tom"]
        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._caught = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _tom_position(self) -> torch.Tensor:
        return self.tom.data.root_pos_w.torch[:, :2]

    def _tom_velocity(self) -> torch.Tensor:
        return self.tom.data.root_lin_vel_w.torch[:, :2]

    def _relative_target(self) -> torch.Tensor:
        # Jerry is placed at each cloned environment's local origin.
        return self.scene.env_origins[:, :2] - self._tom_position()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        self._forces.zero_()
        self._forces[:, 0, :2] = self.cfg.force_scale * self.actions
        self.tom.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            is_global=True,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        relative_target = self._relative_target() / self.cfg.arena_half_extent
        velocity = self._tom_velocity() / self.cfg.max_speed
        return {"policy": torch.cat((relative_target, velocity), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        distance = torch.linalg.vector_norm(self._relative_target(), dim=-1)
        progress = self._previous_distance - distance
        action_squared_norm = torch.sum(torch.square(self.actions), dim=-1)
        caught = distance <= self.cfg.catch_radius
        out_of_bounds = torch.any(
            torch.abs(self._tom_position() - self.scene.env_origins[:, :2])
            > self.cfg.arena_half_extent,
            dim=-1,
        )

        reward = (
            PROGRESS_WEIGHT * progress
            - DISTANCE_WEIGHT * distance
            - ACTION_WEIGHT * action_squared_norm
            - STEP_PENALTY
            + CATCH_BONUS * caught.float()
            - BOUNDARY_PENALTY * out_of_bounds.float()
        )
        self._previous_distance = distance
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.vector_norm(self._relative_target(), dim=-1)
        self._caught = distance <= self.cfg.catch_radius
        self._out_of_bounds = torch.any(
            torch.abs(self._tom_position() - self.scene.env_origins[:, :2])
            > self.cfg.arena_half_extent,
            dim=-1,
        )
        time_out = self.episode_length_buf >= self.max_episode_length
        return self._caught | self._out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.tom._ALL_INDICES

        success = self._caught[env_ids].float()
        failure = self._out_of_bounds[env_ids].float()
        episode_seconds = self.episode_length_buf[env_ids].float() * self.step_dt
        mean_catch_time = torch.sum(episode_seconds * success) / torch.clamp(
            torch.sum(success), min=1.0
        )
        self.extras.setdefault("log", {})["Metrics/catch_rate"] = success.mean()
        self.extras.setdefault("log", {})["Metrics/out_of_bounds_rate"] = failure.mean()
        self.extras.setdefault("log", {})["Metrics/mean_catch_time_s"] = mean_catch_time

        super()._reset_idx(env_ids)

        count = len(env_ids)
        angle = 2.0 * math.pi * torch.rand(count, device=self.device)
        radius = self.cfg.reset_radius_min + (
            self.cfg.reset_radius_max - self.cfg.reset_radius_min
        ) * torch.rand(count, device=self.device)

        root_pose = self.tom.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 0] += radius * torch.cos(angle)
        root_pose[:, 1] += radius * torch.sin(angle)
        root_velocity = self.tom.data.default_root_vel.torch[env_ids].clone()

        self.tom.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.tom.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self.tom.permanent_wrench_composer.set_forces_and_torques(
            forces=torch.zeros((count, 1, 3), device=self.device),
            env_ids=env_ids,
            is_global=True,
        )

        self._previous_distance[env_ids] = radius
        self._caught[env_ids] = False
        self._out_of_bounds[env_ids] = False

