"""Full-room chase curriculum with static walls and a scripted sweeper."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from ..reward_math import CATCH_BONUS
from .tom_jerry_moving_env import TomJerryMovingEnv

if TYPE_CHECKING:
    from .tom_jerry_env_cfg import TomJerryRoomEnvCfg


CHARACTER_DETAILS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/TomJerryCharacterDetails",
    markers={
        "tom_ear": sim_utils.ConeCfg(
            radius=0.15,
            height=0.30,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.30, 0.82), roughness=0.58
            ),
        ),
        "tom_muzzle": sim_utils.SphereCfg(
            radius=0.14,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.78, 0.88, 1.0), roughness=0.72
            ),
        ),
        "jerry_ear": sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.16, 0.03), roughness=0.7
            ),
        ),
        "jerry_tail": sim_utils.CapsuleCfg(
            radius=0.035,
            height=0.48,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.94, 0.40, 0.16), roughness=0.72
            ),
        ),
    },
)


class TomJerryRoomEnv(TomJerryMovingEnv):
    """Chase Jerry through a fixed map while reacting to a moving obstacle."""

    cfg: TomJerryRoomEnvCfg

    def __init__(self, cfg: TomJerryRoomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.sweeper = self.scene["sweeper"]
        self._sweeper_phase = torch.zeros(self.num_envs, device=self.device)
        self._sweeper_position_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._sweeper_velocity_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._collision_events = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._in_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._control_step_count = 0
        self._difficulty = 0.0 if self.cfg.curriculum_enabled else 1.0
        # Debug markers are visual-only.  They give the simple rigid bodies a
        # recognizable cat face and mouse silhouette without adding colliders,
        # masses, observations, or actions to the learned control problem.
        self._character_markers = None
        if self.sim.visualizers:
            self._character_markers = VisualizationMarkers(CHARACTER_DETAILS_CFG)
        self._spawn_points = torch.tensor(
            [
                (-2.8, 0.0),
                (-2.6, 2.0),
                (-2.2, 3.2),
                (0.2, 3.2),
                (2.7, 2.2),
                (3.2, 0.4),
                (2.2, -2.7),
                (0.2, -2.8),
            ],
            device=self.device,
        )
        self._visualize_character_details()

    @staticmethod
    def _heading_and_perpendicular(velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        fallback = torch.zeros_like(velocity)
        fallback[:, 0] = 1.0
        heading = torch.where(speed > 0.05, velocity / torch.clamp(speed, min=1.0e-6), fallback)
        perpendicular = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
        return heading, perpendicular

    def _visualize_character_details(self) -> None:
        if self._character_markers is None:
            return

        tom_position = self.tom.data.root_pos_w.torch[:, :3]
        jerry_position = self.jerry.data.root_pos_w.torch[:, :3]
        tom_heading, tom_perpendicular = self._heading_and_perpendicular(
            self._tom_velocity()
        )
        jerry_heading, jerry_perpendicular = self._heading_and_perpendicular(
            self._jerry_velocity_w
        )

        def offset_position(
            origin: torch.Tensor,
            forward: torch.Tensor,
            sideways: torch.Tensor,
            z_offset: float,
        ) -> torch.Tensor:
            result = origin.clone()
            result[:, :2] += forward + sideways
            result[:, 2] += z_offset
            return result

        tom_ear_left = offset_position(
            tom_position, 0.10 * tom_heading, 0.15 * tom_perpendicular, 0.37
        )
        tom_ear_right = offset_position(
            tom_position, 0.10 * tom_heading, -0.15 * tom_perpendicular, 0.37
        )
        tom_muzzle = offset_position(
            tom_position, 0.33 * tom_heading, torch.zeros_like(tom_heading), 0.02
        )
        jerry_ear_left = offset_position(
            jerry_position, -0.03 * jerry_heading, 0.12 * jerry_perpendicular, 0.18
        )
        jerry_ear_right = offset_position(
            jerry_position, -0.03 * jerry_heading, -0.12 * jerry_perpendicular, 0.18
        )
        jerry_tail = offset_position(
            jerry_position, -0.34 * jerry_heading, torch.zeros_like(jerry_heading), -0.02
        )

        translations = torch.cat(
            (
                tom_ear_left,
                tom_ear_right,
                tom_muzzle,
                jerry_ear_left,
                jerry_ear_right,
                jerry_tail,
            ),
            dim=0,
        )
        marker_indices = torch.cat(
            (
                torch.zeros(2 * self.num_envs, dtype=torch.int32, device=self.device),
                torch.ones(self.num_envs, dtype=torch.int32, device=self.device),
                torch.full((2 * self.num_envs,), 2, dtype=torch.int32, device=self.device),
                torch.full((self.num_envs,), 3, dtype=torch.int32, device=self.device),
            )
        )
        orientations = torch.zeros((6 * self.num_envs, 4), device=self.device)
        orientations[:, 0] = 1.0
        # Capsules point along local Z. Rotate each tail 90 degrees so its long
        # axis follows the opposite of Jerry's current direction of travel.
        tail_orientation = orientations[-self.num_envs :]
        tail_orientation[:, 0] = math.sqrt(0.5)
        tail_orientation[:, 1] = -math.sqrt(0.5) * jerry_heading[:, 1]
        tail_orientation[:, 2] = math.sqrt(0.5) * jerry_heading[:, 0]
        tail_orientation[:, 3] = 0.0

        self._character_markers.visualize(
            translations=translations,
            orientations=orientations,
            marker_indices=marker_indices,
        )

    def _update_jerry_trajectory(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self._all_env_ids

        half = self.cfg.jerry_route_half_extent
        segment_length = 2.0 * half
        perimeter = 4.0 * segment_length
        time_s = self.episode_length_buf[env_ids].float() * self.step_dt
        route_speed = self.cfg.jerry_route_speed * self._difficulty
        route_distance = torch.remainder(
            route_speed * time_s + self._jerry_phase[env_ids],
            perimeter,
        )
        segment = torch.floor(route_distance / segment_length).long()
        offset = torch.remainder(route_distance, segment_length)

        local_x = torch.where(
            segment == 0,
            -half + offset,
            torch.where(segment == 1, half, torch.where(segment == 2, half - offset, -half)),
        )
        local_y = torch.where(
            segment == 0,
            torch.full_like(offset, -half),
            torch.where(
                segment == 1,
                -half + offset,
                torch.where(segment == 2, torch.full_like(offset, half), half - offset),
            ),
        )
        velocity_x = torch.where(
            segment == 0,
            torch.full_like(offset, route_speed),
            torch.where(
                segment == 2,
                torch.full_like(offset, -route_speed),
                torch.zeros_like(offset),
            ),
        )
        velocity_y = torch.where(
            segment == 1,
            torch.full_like(offset, route_speed),
            torch.where(
                segment == 3,
                torch.full_like(offset, -route_speed),
                torch.zeros_like(offset),
            ),
        )

        self._jerry_position_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + local_x
        self._jerry_position_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + local_y
        self._jerry_velocity_w[env_ids, 0] = velocity_x
        self._jerry_velocity_w[env_ids, 1] = velocity_y

        root_pose = self.jerry.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 0] += local_x
        root_pose[:, 1] += local_y
        root_velocity = self.jerry.data.default_root_vel.torch[env_ids].clone()
        root_velocity[:, 0] = velocity_x
        root_velocity[:, 1] = velocity_y
        self.jerry.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.jerry.write_root_velocity_to_sim_index(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )

    def _update_sweeper_trajectory(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ):
        if env_ids is None:
            env_ids = self._all_env_ids

        time_s = self.episode_length_buf[env_ids].float() * self.step_dt
        theta = self.cfg.sweeper_angular_speed * time_s + self._sweeper_phase[env_ids]
        sweeper_amplitude = self.cfg.sweeper_amplitude * self._difficulty
        local_x = self.cfg.sweeper_center_x + sweeper_amplitude * torch.sin(theta)
        velocity_x = (
            sweeper_amplitude
            * self.cfg.sweeper_angular_speed
            * torch.cos(theta)
        )
        self._sweeper_position_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + local_x
        self._sweeper_position_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + self.cfg.sweeper_y
        self._sweeper_velocity_w[env_ids, 0] = velocity_x
        self._sweeper_velocity_w[env_ids, 1] = 0.0

        root_pose = self.sweeper.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 0] += local_x - self.cfg.sweeper_center_x
        root_velocity = self.sweeper.data.default_root_vel.torch[env_ids].clone()
        root_velocity[:, 0] = velocity_x
        self.sweeper.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.sweeper.write_root_velocity_to_sim_index(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        if self.cfg.curriculum_enabled:
            curriculum_fraction = (
                self._control_step_count - self.cfg.curriculum_warmup_steps
            ) / self.cfg.curriculum_ramp_steps
            self._difficulty = max(0.0, min(1.0, curriculum_fraction))
        self._control_step_count += 1
        self._update_jerry_trajectory()
        self._update_sweeper_trajectory()
        self._visualize_character_details()

    @staticmethod
    def _circle_hits_rect(
        positions: torch.Tensor,
        rect_min: torch.Tensor,
        rect_max: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        closest = torch.maximum(rect_min, torch.minimum(positions, rect_max))
        return torch.sum(torch.square(positions - closest), dim=-1) <= radius * radius

    def _collision_mask(self) -> torch.Tensor:
        local_position = self._tom_position() - self.scene.env_origins[:, :2]
        radius = 0.38
        tall = self._circle_hits_rect(
            local_position,
            torch.tensor((-1.225, -0.9), device=self.device),
            torch.tensor((-0.775, 2.9), device=self.device),
            radius,
        )
        low = self._circle_hits_rect(
            local_position,
            torch.tensor((-0.6, -1.225), device=self.device),
            torch.tensor((3.0, -0.775), device=self.device),
            radius,
        )
        sweeper_local = self._sweeper_position_w - self.scene.env_origins[:, :2]
        sweeper = self._circle_hits_rect(
            local_position,
            sweeper_local - torch.tensor((0.4, 0.65), device=self.device),
            sweeper_local + torch.tensor((0.4, 0.65), device=self.device),
            radius,
        )
        wall = torch.any(
            torch.abs(local_position) >= self.cfg.arena_half_extent - radius,
            dim=-1,
        )
        return tall | low | sweeper | wall

    def _get_observations(self) -> dict[str, torch.Tensor]:
        relative_target = self._relative_target() / self.cfg.arena_half_extent
        tom_velocity = self._tom_velocity() / self.cfg.max_speed
        jerry_velocity = self._jerry_velocity_w / self.cfg.jerry_max_speed
        local_position = (
            self._tom_position() - self.scene.env_origins[:, :2]
        ) / self.cfg.arena_half_extent
        relative_sweeper = (
            self._sweeper_position_w - self._tom_position()
        ) / self.cfg.arena_half_extent
        sweeper_velocity = self._sweeper_velocity_w / self.cfg.sweeper_max_speed
        difficulty = torch.full(
            (self.num_envs, 1),
            self._difficulty,
            device=self.device,
        )
        return {
            "policy": torch.cat(
                (
                    relative_target,
                    tom_velocity,
                    jerry_velocity,
                    local_position,
                    relative_sweeper,
                    sweeper_velocity,
                    difficulty,
                ),
                dim=-1,
            )
        }

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        self.extras.setdefault("log", {})["Metrics/difficulty"] = torch.tensor(
            self._difficulty,
            device=self.device,
        )
        caught = torch.linalg.vector_norm(self._relative_target(), dim=-1) <= self.cfg.catch_radius
        collision = self._collision_mask()
        new_collision = collision & ~self._in_collision
        self._collision_events += new_collision.long()
        self._in_collision = collision
        return (
            reward
            + (self.cfg.room_catch_bonus - CATCH_BONUS) * caught.float()
            - self.cfg.collision_penalty * collision.float()
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._all_env_ids

        previous_collision_events = self._collision_events[env_ids].float()
        super()._reset_idx(env_ids)
        self.extras.setdefault("log", {})[
            "Metrics/mean_collision_events"
        ] = previous_collision_events.mean()

        count = len(env_ids)
        route_perimeter = 8.0 * self.cfg.jerry_route_half_extent
        self._jerry_phase[env_ids] = route_perimeter * torch.rand(
            count, device=self.device
        )
        self._sweeper_phase[env_ids] = 2.0 * math.pi * torch.rand(
            count, device=self.device
        )
        self._update_jerry_trajectory(env_ids)
        self._update_sweeper_trajectory(env_ids)

        spawn_index = torch.randint(
            0,
            len(self._spawn_points),
            (count,),
            device=self.device,
        )
        spawn_local = self._spawn_points[spawn_index].clone()
        spawn_world = spawn_local + self.scene.env_origins[env_ids, :2]
        too_close = torch.linalg.vector_norm(
            spawn_world - self._jerry_position_w[env_ids], dim=-1
        ) < 1.5
        opposite_index = torch.remainder(spawn_index + 4, len(self._spawn_points))
        spawn_local[too_close] = self._spawn_points[opposite_index[too_close]]

        root_pose = self.tom.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, :2] = spawn_local + self.scene.env_origins[env_ids, :2]
        root_velocity = self.tom.data.default_root_vel.torch[env_ids].clone()
        self.tom.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.tom.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self._jerry_position_w[env_ids] - root_pose[:, :2], dim=-1
        )
        self._collision_events[env_ids] = 0
        self._in_collision[env_ids] = False
