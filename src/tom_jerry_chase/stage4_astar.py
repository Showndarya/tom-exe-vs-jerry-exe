"""Stage 4: Tom chases Jerry with a replanning A* path.

A* searches a top-down occupancy grid built from the known obstacle shapes.
Jerry still follows scripted waypoints.  A yellow surprise wall appears during
the run, invalidating Tom's original path and demonstrating replanning.  This
is classical graph search, not machine learning.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


STAGE_PATH = Path("/workspace/output/tom_jerry_stage2.usda")
OUTPUT_DIR = Path("/workspace/output/stage4_astar")
FRAME_DIR = OUTPUT_DIR / "frames"
METRICS_PATH = OUTPUT_DIR / "astar_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "astar_summary.json"

RESOLUTION = (960, 540)
RENDER_HZ = 15
DURATION_SECONDS = 12
FRAME_COUNT = RENDER_HZ * DURATION_SECONDS
TOM_SPEED = 3.0
JERRY_SPEED = 1.1
CATCH_DISTANCE = 0.75
SEED = 7

GRID_RESOLUTION = 0.5
ROBOT_CLEARANCE = 0.58
REPLAN_INTERVAL_FRAMES = 5
WAYPOINT_TOLERANCE = 0.28
SURPRISE_FRAME = 24

# Rectangles use (min_x, max_x, min_y, max_y).  They are expanded by Tom's
# radius plus a small safety margin when converted into blocked grid cells.
BASE_OBSTACLES = (
    (-1.4, -0.4, -1.3, 2.1),
    (0.65, 2.95, -1.85, -0.95),
)
SURPRISE_OBSTACLE = (-1.5, 1.0, 2.65, 3.35)

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

    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateLockedPosAxisAttr(4)
    api.CreateLockedRotAxisAttr(7)
    api.CreateLinearDampingAttr(0.0)
    api.CreateAngularDampingAttr(10.0)


def _make_surprise_wall(stage):
    """Create a collidable yellow wall outside the room, ready to pop in."""

    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, "/World/Obstacles/SurpriseWall")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.72, 0.05)])
    xformable = UsdGeom.Xformable(cube.GetPrim())
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(8.0, 8.0, 0.75))
    xformable.AddScaleOp().Set(Gf.Vec3f(2.5, 0.7, 1.5))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return translate_op


def main() -> None:
    simulation_app = SimulationApp(
        launch_config={
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        }
    )
    print("STAGE4 step=app_ready", flush=True)

    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings import _physx
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdPhysics

    from tom_jerry_chase.controllers import direct_pursuit_velocity, waypoint_velocity
    from tom_jerry_chase.grid_planner import (
        GridSpec,
        astar_path,
        blocked_cells_from_rectangles,
        path_length_m,
    )

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
    surprise_translate = _make_surprise_wall(stage)

    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(tom_prim)
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
                    if path.startswith("/World/Obstacles/")
                    or path.startswith("/World/Walls/")
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
        name="Stage4Camera",
    )
    render_product = rep.create.render_product(camera, RESOLUTION, name="Stage4Render")
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(FRAME_DIR), rgb=True)
    writer.attach([render_product])

    grid = GridSpec(-5.0, 5.0, -3.5, 3.5, GRID_RESOLUTION)
    obstacle_rectangles = list(BASE_OBSTACLES)
    blocked = blocked_cells_from_rectangles(grid, obstacle_rectangles, ROBOT_CLEARANCE)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    rows: list[dict[str, float | int | bool | str]] = []
    planned_cells: list[tuple[int, int]] = []
    path_waypoints: list[tuple[float, float]] = []
    path_index = 0
    planned_goal = None
    waypoint_index = 0
    caught = False
    replans = 0
    environment_changes = 0
    total_planned_distance = 0.0
    initial_distance = math.dist(_world_xy(tom_prim), _world_xy(jerry_prim))

    print(
        f"STAGE4 step=simulation_start frames={FRAME_COUNT} seed={SEED} "
        f"tom_speed={TOM_SPEED} jerry_speed={JERRY_SPEED}",
        flush=True,
    )

    for frame in range(FRAME_COUNT):
        environment_changed = False
        if frame == SURPRISE_FRAME:
            surprise_translate.Set(Gf.Vec3d(-0.25, 3.0, 0.75))
            obstacle_rectangles.append(SURPRISE_OBSTACLE)
            blocked = blocked_cells_from_rectangles(
                grid, obstacle_rectangles, ROBOT_CLEARANCE
            )
            environment_changed = True
            environment_changes += 1
            print(f"STAGE4 step=surprise_wall frame={frame}", flush=True)

        tom_xy = _world_xy(tom_prim)
        jerry_xy = _world_xy(jerry_prim)
        start_cell = grid.world_to_cell(tom_xy)
        goal_cell = grid.world_to_cell(jerry_xy)
        remaining_path_blocked = any(cell in blocked for cell in planned_cells[path_index:])
        goal_moved = goal_cell != planned_goal
        periodic_goal_check = frame % REPLAN_INTERVAL_FRAMES == 0 and goal_moved
        needs_plan = not planned_cells or path_index >= len(path_waypoints)

        replan_reason = ""
        if environment_changed or remaining_path_blocked:
            replan_reason = "blocked_path"
        elif periodic_goal_check:
            replan_reason = "jerry_moved"
        elif needs_plan:
            replan_reason = "path_complete"

        if replan_reason:
            planning_blocked = set(blocked)
            planning_blocked.discard(start_cell)
            planning_blocked.discard(goal_cell)
            planned_cells = astar_path(grid, start_cell, goal_cell, planning_blocked)
            planned_goal = goal_cell
            path_index = 0
            path_waypoints = [grid.cell_to_world(cell) for cell in planned_cells[1:]]
            replans += 1
            total_planned_distance += path_length_m(planned_cells, GRID_RESOLUTION)
            print(
                f"STAGE4 step=replan frame={frame} reason={replan_reason} "
                f"cells={len(planned_cells)}",
                flush=True,
            )

        while (
            path_index < len(path_waypoints)
            and math.dist(tom_xy, path_waypoints[path_index]) <= WAYPOINT_TOLERANCE
        ):
            path_index += 1

        if path_index < len(path_waypoints):
            target = path_waypoints[path_index]
            action = direct_pursuit_velocity(tom_xy, target, TOM_SPEED)
        else:
            target = tom_xy
            action = (0.0, 0.0)

        jerry_action, waypoint_index = waypoint_velocity(
            jerry_xy,
            JERRY_WAYPOINTS,
            waypoint_index,
            JERRY_SPEED,
            tolerance=0.25,
        )
        tom_velocity.Set(Gf.Vec3f(action[0], action[1], 0.0))
        jerry_velocity.Set(Gf.Vec3f(jerry_action[0], jerry_action[1], 0.0))
        rep.orchestrator.step(delta_time=1.0 / RENDER_HZ, pause_timeline=False)

        updated_tom_xy = _world_xy(tom_prim)
        updated_jerry_xy = _world_xy(jerry_prim)
        distance = math.dist(updated_tom_xy, updated_jerry_xy)
        caught = distance <= CATCH_DISTANCE
        rows.append(
            {
                "frame": frame,
                "time_s": round((frame + 1) / RENDER_HZ, 4),
                "tom_x": round(updated_tom_xy[0], 5),
                "tom_y": round(updated_tom_xy[1], 5),
                "jerry_x": round(updated_jerry_xy[0], 5),
                "jerry_y": round(updated_jerry_xy[1], 5),
                "target_x": round(target[0], 5),
                "target_y": round(target[1], 5),
                "action_vx": round(action[0], 5),
                "action_vy": round(action[1], 5),
                "distance": round(distance, 5),
                "path_cells": len(planned_cells),
                "replans": replans,
                "replan_reason": replan_reason,
                "environment_changed": environment_changed,
                "collision_count": collision_count,
                "caught": caught,
            }
        )
        if caught:
            print(f"STAGE4 step=caught frame={frame}", flush=True)
            break

    rep.orchestrator.wait_until_complete()
    timeline.stop()
    writer.detach()
    render_product.destroy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    summary = {
        "controller": "astar",
        "seed": SEED,
        "tom_speed": TOM_SPEED,
        "jerry_speed": JERRY_SPEED,
        "frames": len(rows),
        "simulated_seconds": round(len(rows) / RENDER_HZ, 4),
        "caught": caught,
        "catch_time_s": round(len(rows) / RENDER_HZ, 4) if caught else None,
        "collision_count": collision_count,
        "replans": replans,
        "environment_changes": environment_changes,
        "initial_distance": round(initial_distance, 5),
        "final_distance": round(float(rows[-1]["distance"]), 5),
        "total_planned_distance": round(total_planned_distance, 5),
        "metrics_csv": str(METRICS_PATH),
        "frames_dir": str(FRAME_DIR),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    contact_subscription = None
    print(f"STAGE4_OK {json.dumps(summary, sort_keys=True)}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
