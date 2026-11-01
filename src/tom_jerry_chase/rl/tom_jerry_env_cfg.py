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

