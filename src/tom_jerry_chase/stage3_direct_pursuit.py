"""Stage 3: Tom blindly drives toward a scripted Jerry.

The controller observes only the two planar positions and commands Tom's
planar linear velocity directly.  It has no obstacle observation or map, so a
wall on the straight-line route should block it.  This is deliberately a
controller baseline, not machine learning.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


STAGE_PATH = Path("/workspace/output/tom_jerry_stage2.usda")
FRAME_DIR = Path("/workspace/output/stage3_direct_pursuit/frames")
METRICS_PATH = Path("/workspace/output/stage3_direct_pursuit/direct_pursuit_metrics.csv")
SUMMARY_PATH = Path("/workspace/output/stage3_direct_pursuit/direct_pursuit_summary.json")

RESOLUTION = (960, 540)
RENDER_HZ = 15
DURATION_SECONDS = 8
FRAME_COUNT = RENDER_HZ * DURATION_SECONDS
TOM_SPEED = 1.8
JERRY_SPEED = 0.70
CATCH_DISTANCE = 0.75
SEED = 7

JERRY_WAYPOINTS = (
    (3.6, 2.4),
    (4.7, 2.4),
    (4.7, 0.2),
    (3.2, 0.2),
    (3.2, 3.1),
)


def _world_xy(prim) -> tuple[float, float]:
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    return (float(translation[0]), float(translation[1]))


def _configure_planar_body(prim) -> None:
    from pxr import PhysxSchema

    # 4 locks Z translation; 7 locks rotation around X, Y, and Z.  Tom and
    # Jerry therefore behave like top-down mobile robots while contacts remain
    # fully physical in X/Y.
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateLockedPosAxisAttr(4)
    api.CreateLockedRotAxisAttr(7)
    api.CreateLinearDampingAttr(0.0)
    api.CreateAngularDampingAttr(10.0)


def main() -> None:
    simulation_app = SimulationApp(
        launch_config={
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        }
    )
    print("STAGE3 step=app_ready", flush=True)

    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings import _physx
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdPhysics

    from tom_jerry_chase.controllers import direct_pursuit_velocity, waypoint_velocity

    if not STAGE_PATH.is_file():
        raise FileNotFoundError(f"Stage does not exist: {STAGE_PATH}")

    context = omni.usd.get_context()
    if not context.open_stage(str(STAGE_PATH)):
        raise RuntimeError(f"Isaac Sim could not open stage: {STAGE_PATH}")
    for _ in range(20):
        simulation_app.update()

    stage = context.get_stage()
    tom_prim = stage.GetPrimAtPath("/World/Tom")
    jerry_prim = stage.GetPrimAtPath("/World/Jerry")
    if not tom_prim.IsValid() or not jerry_prim.IsValid():
        raise RuntimeError("Stage must contain /World/Tom and /World/Jerry")

    _configure_planar_body(tom_prim)
    _configure_planar_body(jerry_prim)
    tom_body = UsdPhysics.RigidBodyAPI(tom_prim)
    jerry_body = UsdPhysics.RigidBodyAPI(jerry_prim)
    tom_velocity = tom_body.CreateVelocityAttr()
    jerry_velocity = jerry_body.CreateVelocityAttr()
    tom_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0))
    jerry_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0))

    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(tom_prim)
    # Isaac Sim 6.2's generated Python schema exposes a shorter method name
    # than the current Omni Physics documentation.  Authoring the canonical
    # USD attribute keeps this script compatible with both bindings.
    threshold_factory = getattr(contact_api, "CreateThresholdAttr", None)
    if callable(threshold_factory):
        threshold_factory(0.0)
    else:
        tom_prim.CreateAttribute(
            "physxContactReport:threshold", Sdf.ValueTypeNames.Float
        ).Set(0.0)
    collision_count = 0
    active_obstacle_contacts: set[str] = set()

    def on_contact_report(contact_headers, _contact_data) -> None:
        nonlocal collision_count
        for header in contact_headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            pair = (collider0, collider1)
            if not any("/World/Tom/" in path for path in pair):
                continue
            obstacle = next(
                (
                    path
                    for path in pair
                    if path.startswith("/World/Obstacles/") or path.startswith("/World/Walls/")
                ),
                None,
            )
            if obstacle is None:
                continue
            if header.type == _physx.ContactEventType.CONTACT_FOUND:
                if obstacle not in active_obstacle_contacts:
                    active_obstacle_contacts.add(obstacle)
                    collision_count += 1
            elif header.type == _physx.ContactEventType.CONTACT_LOST:
                active_obstacle_contacts.discard(obstacle)

    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact_report
    )

    camera = rep.functional.create.camera(
        position=(0.0, -16.0, 18.5),
        look_at=(0.0, 0.0, 0.2),
        focal_length=21.0,
        parent="/World",
        name="Stage3Camera",
    )
    render_product = rep.create.render_product(camera, RESOLUTION, name="Stage3Render")
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(FRAME_DIR), rgb=True)
    writer.attach([render_product])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    rows: list[dict[str, float | int | bool]] = []
    waypoint_index = 0
    caught = False
    stuck_frames = 0
    previous_tom_xy = _world_xy(tom_prim)
    initial_distance = math.dist(previous_tom_xy, _world_xy(jerry_prim))

    print(
        f"STAGE3 step=simulation_start frames={FRAME_COUNT} seed={SEED}",
        flush=True,
    )
    for frame in range(FRAME_COUNT):
        tom_xy = _world_xy(tom_prim)
        jerry_xy = _world_xy(jerry_prim)
        observation = (jerry_xy[0] - tom_xy[0], jerry_xy[1] - tom_xy[1])
        action = direct_pursuit_velocity(tom_xy, jerry_xy, TOM_SPEED)
        jerry_action, waypoint_index = waypoint_velocity(
            jerry_xy,
            JERRY_WAYPOINTS,
            waypoint_index,
            JERRY_SPEED,
        )

        tom_velocity.Set(Gf.Vec3f(action[0], action[1], 0.0))
        jerry_velocity.Set(Gf.Vec3f(jerry_action[0], jerry_action[1], 0.0))
        rep.orchestrator.step(delta_time=1.0 / RENDER_HZ, pause_timeline=False)

        updated_tom_xy = _world_xy(tom_prim)
        updated_jerry_xy = _world_xy(jerry_prim)
        distance = math.dist(updated_tom_xy, updated_jerry_xy)
        displacement = math.dist(previous_tom_xy, updated_tom_xy)
        if math.hypot(*action) > 0.5 and displacement < 0.005:
            stuck_frames += 1
        previous_tom_xy = updated_tom_xy
        caught = distance <= CATCH_DISTANCE

        rows.append(
            {
                "frame": frame,
                "time_s": round((frame + 1) / RENDER_HZ, 4),
                "tom_x": round(updated_tom_xy[0], 5),
                "tom_y": round(updated_tom_xy[1], 5),
                "jerry_x": round(updated_jerry_xy[0], 5),
                "jerry_y": round(updated_jerry_xy[1], 5),
                "observation_dx": round(observation[0], 5),
                "observation_dy": round(observation[1], 5),
                "action_vx": round(action[0], 5),
                "action_vy": round(action[1], 5),
                "distance": round(distance, 5),
                "collision_count": collision_count,
                "caught": caught,
            }
        )
        if caught:
            print(f"STAGE3 step=caught frame={frame}", flush=True)
            break

    rep.orchestrator.wait_until_complete()
    timeline.stop()
    writer.detach()
    render_product.destroy()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    final_distance = float(rows[-1]["distance"])
    summary = {
        "controller": "direct_pursuit",
        "seed": SEED,
        "frames": len(rows),
        "simulated_seconds": round(len(rows) / RENDER_HZ, 4),
        "caught": caught,
        "collision_count": collision_count,
        "stuck_frames": stuck_frames,
        "initial_distance": round(initial_distance, 5),
        "final_distance": round(final_distance, 5),
        "metrics_csv": str(METRICS_PATH),
        "frames_dir": str(FRAME_DIR),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Keep the subscription alive until all physics steps are complete.
    contact_subscription = None
    print(f"STAGE3_OK {json.dumps(summary, sort_keys=True)}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
