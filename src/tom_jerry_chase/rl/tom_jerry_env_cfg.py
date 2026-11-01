"""Configuration for the first, deliberately easy Tom-versus-Jerry RL task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class TomJerryPhysicsCfg(PresetCfg):
    """Physics choices exposed through Isaac Lab's ``presets=`` selector."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=4,
            nconmax=2,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class TomJerrySceneCfg(InteractiveSceneCfg):
    """A cloned training scene containing one dynamic Tom and one target Jerry."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.92, 0.84, 0.58)),
    )

    tom: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tom",
        spawn=sim_utils.SphereCfg(
            radius=0.38,
            # Isaac Lab 3.0 composes advanced rigid-body properties through
            # schema fragments.  The backend-neutral base config deliberately
            # contains only enable/kinematic flags, so gravity and damping use
            # Newton's defaults for this first task.
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.38, 0.95), metallic=0.05, roughness=0.55
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.42)),
    )

    jerry = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Jerry",
        spawn=sim_utils.SphereCfg(
            radius=0.26,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.30, 0.03), metallic=0.0, roughness=0.65
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.86, 0.92, 1.0)),
    )


@configclass
class TomJerryEasyEnvCfg(DirectRLEnvCfg):
    """Continuous-control chase with a stationary target and no obstacles."""

    decimation = 2
    episode_length_s = 4.0
    action_space = 2
    observation_space = 4
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=TomJerryPhysicsCfg(),
    )
    scene: TomJerrySceneCfg = TomJerrySceneCfg(
        num_envs=256,
        env_spacing=7.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    force_scale = 8.0
    max_speed = 3.5
    arena_half_extent = 3.4
    catch_radius = 0.48
    reset_radius_min = 1.4
    reset_radius_max = 2.7

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, -8.0, 6.5))


@configclass
class TomJerryMovingSceneCfg(TomJerrySceneCfg):
    """Training scene whose Jerry is a script-controlled rigid object."""

    jerry: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Jerry",
        spawn=sim_utils.SphereCfg(
            radius=0.26,
            # Jerry is intentionally non-colliding.  The task detects a catch
            # from distance, while the environment prescribes Jerry's pose and
            # velocity every control step.  This stays backend-neutral in the
            # Isaac Lab 3 tensor API and prevents Tom from physically punting
            # the target across the room.
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.25),
            # Newton uses this disabled collider to derive a valid sphere
            # inertia tensor without allowing Tom/Jerry contact forces.
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.30, 0.03), metallic=0.0, roughness=0.65
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30)),
    )


@configclass
class TomJerryMovingEnvCfg(TomJerryEasyEnvCfg):
    """Second curriculum step: pursue a scripted figure-eight Jerry."""

    episode_length_s = 5.0
    observation_space = 6
    scene: TomJerryMovingSceneCfg = TomJerryMovingSceneCfg(
        num_envs=256,
        env_spacing=7.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    jerry_x_radius = 1.15
    jerry_y_radius = 0.85
    jerry_angular_speed = 1.05
    jerry_max_speed = 1.25
    catch_radius = 0.52
    reset_radius_min = 1.6
    reset_radius_max = 2.9


@configclass
class TomJerryRoomPhysicsCfg(TomJerryPhysicsCfg):
    """Newton capacity for walls, interior barriers, and a moving sweeper."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=32,
            nconmax=8,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class TomJerryRoomSceneCfg(TomJerryMovingSceneCfg):
    """A colorful enclosed room with two routes and a moving blocker."""

    north_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room/NorthWall",
        spawn=sim_utils.CuboidCfg(
            size=(8.4, 0.3, 1.2),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.72, 0.92), roughness=0.7
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 4.05, 0.60)),
    )
    south_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room/SouthWall",
        spawn=sim_utils.CuboidCfg(
            size=(8.4, 0.3, 1.2),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.72, 0.92), roughness=0.7
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -4.05, 0.60)),
    )
    east_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room/EastWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 7.8, 1.2),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.72, 0.92), roughness=0.7
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.05, 0.0, 0.60)),
    )
    west_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room/WestWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 7.8, 1.2),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.72, 0.92), roughness=0.7
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.05, 0.0, 0.60)),
    )
    tall_barrier = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room/TallBarrier",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 3.8, 1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.66, 0.30, 0.84), roughness=0.65
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 1.0, 0.50)),
    )
    low_barrier = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room/LowBarrier",
        spawn=sim_utils.CuboidCfg(
            size=(3.6, 0.45, 1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.36, 0.58), roughness=0.65
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, -1.0, 0.50)),
    )

    # Collision-disabled child shapes inherit each rigid body's transform.
    # They make the characters recognizable without changing masses,
    # colliders, observations, actions, or the PPO network.
    tom_ear_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tom/EarLeft",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.30,
            axis="Z",
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.30, 0.82), roughness=0.58
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.10, 0.15, 0.37)),
    )
    tom_ear_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tom/EarRight",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.30,
            axis="Z",
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.30, 0.82), roughness=0.58
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.10, -0.15, 0.37)),
    )
    tom_muzzle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tom/Muzzle",
        spawn=sim_utils.SphereCfg(
            radius=0.14,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.78, 0.88, 1.0), roughness=0.72
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.34, 0.0, 0.02)),
    )
    jerry_ear_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Jerry/EarLeft",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.16, 0.03), roughness=0.70
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.03, 0.12, 0.18)),
    )
    jerry_ear_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Jerry/EarRight",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.16, 0.03), roughness=0.70
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.03, -0.12, 0.18)),
    )
    jerry_tail = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Jerry/Tail",
        spawn=sim_utils.CapsuleCfg(
            radius=0.035,
            height=0.48,
            axis="X",
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.94, 0.40, 0.16), roughness=0.72
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.36, 0.0, -0.02)),
    )

    sweeper: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Room/Sweeper",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.3, 0.8),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=30.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.76, 0.04), metallic=0.05, roughness=0.5
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5, -2.0, 0.40)),
    )


@configclass
class TomJerryRoomEnvCfg(TomJerryMovingEnvCfg):
    """Third curriculum step: a fixed full room plus a dynamic obstacle."""

    episode_length_s = 10.0
    observation_space = 13
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics=TomJerryRoomPhysicsCfg(),
    )
    scene: TomJerryRoomSceneCfg = TomJerryRoomSceneCfg(
        num_envs=256,
        env_spacing=10.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    force_scale = 9.0
    max_speed = 4.0
    arena_half_extent = 3.85
    catch_radius = 0.56
    # Keep Jerry outside both interior barriers.  The old 2.8 m square cut
    # through the right end of the low barrier and the top of the tall one.
    # Jerry is script-driven, so a collider cannot correct a bad waypoint;
    # the trajectory itself must respect obstacle clearance.
    jerry_route_half_extent = 3.45
    jerry_route_speed = 1.25
    jerry_max_speed = 1.25
    sweeper_center_x = -1.5
    sweeper_amplitude = 0.9
    sweeper_y = -2.0
    sweeper_angular_speed = 1.0
    sweeper_max_speed = 0.9
    collision_penalty = 1.0
    room_catch_bonus = 32.0
    curriculum_enabled = True
    curriculum_warmup_steps = 7200
    curriculum_ramp_steps = 4800

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(11.0, -11.0, 8.5))


@configclass
class TomJerryRoomEvalEnvCfg(TomJerryRoomEnvCfg):
    """Full-difficulty room used for evaluation and live playback."""

    curriculum_enabled = False

