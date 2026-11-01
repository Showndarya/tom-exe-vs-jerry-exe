"""Stage 2: build the first Tom.exe vs Jerry.exe room in Isaac Sim 6.2.

The scene intentionally uses only OpenUSD geometric primitives.  There are no
downloaded character models, textures, or hidden assets, so every visible item
is easy to explain and reproduce.
"""

from __future__ import annotations

from pathlib import Path

import omni.usd
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics


ROOM_WIDTH = 12.0
ROOM_DEPTH = 9.0
WALL_HEIGHT = 2.5
WALL_THICKNESS = 0.25
PHYSICS_HZ = 60


def set_color(prim: Usd.Prim, rgb: tuple[float, float, float]) -> None:
    """Apply a simple viewport color without external material files."""

    imageable = UsdGeom.Gprim(prim)
    imageable.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def set_transform(
    prim: Usd.Prim,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Give a primitive a position, rotation in degrees, and scale."""

    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation_xyz != (0.0, 0.0, 0.0):
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz))
    if scale != (1.0, 1.0, 1.0):
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def add_box(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = True,
) -> Usd.Prim:
    """Create a cube scaled to the requested world-space dimensions."""

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    set_transform(prim, position, size)
    set_color(prim, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def add_sphere(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float],
) -> Usd.Prim:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    prim = sphere.GetPrim()
    set_transform(prim, position)
    set_color(prim, color)
    return prim


def make_rigid_body(root: Usd.Prim, mass_kg: float) -> None:
    """Turn an Xform into a dynamic rigid body with a specified mass."""

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass_kg)


def build_tom(stage: Usd.Stage) -> None:
    """Create a blue capsule cat with visual ears and collider body."""

    root = UsdGeom.Xform.Define(stage, "/World/Tom").GetPrim()
    set_transform(root, (-4.0, -2.0, 0.75))
    make_rigid_body(root, mass_kg=3.0)

    body = UsdGeom.Capsule.Define(stage, "/World/Tom/Body")
    body.CreateAxisAttr("Z")
    body.CreateRadiusAttr(0.46)
    body.CreateHeightAttr(0.75)
    set_color(body.GetPrim(), (0.12, 0.42, 0.95))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())

    # Ears are deliberately visual-only so Tom has one simple, stable collider.
    for side, x in (("Left", -0.26), ("Right", 0.26)):
        ear = UsdGeom.Cone.Define(stage, f"/World/Tom/Ear{side}")
        ear.CreateAxisAttr("Z")
        ear.CreateRadiusAttr(0.20)
        ear.CreateHeightAttr(0.42)
        set_transform(ear.GetPrim(), (x, 0.0, 0.73))
        set_color(ear.GetPrim(), (0.18, 0.52, 1.0))

    nose = add_sphere(stage, "/World/Tom/Nose", (0.0, -0.44, 0.16), 0.10, (1.0, 0.55, 0.72))
    # Child coordinates are local to Tom, so this appears on his face.
    nose.SetMetadata("comment", "Visual-only pink nose")


def build_jerry(stage: Usd.Stage) -> None:
    """Create a small orange mouse with ears, nose, and a simple tail."""

    root = UsdGeom.Xform.Define(stage, "/World/Jerry").GetPrim()
    set_transform(root, (3.6, 2.4, 0.38))
    make_rigid_body(root, mass_kg=0.6)

    body = UsdGeom.Sphere.Define(stage, "/World/Jerry/Body")
    body.CreateRadiusAttr(0.34)
    set_color(body.GetPrim(), (1.0, 0.38, 0.06))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())

    for side, x in (("Left", -0.23), ("Right", 0.23)):
        ear = add_sphere(stage, f"/World/Jerry/Ear{side}", (x, 0.0, 0.29), 0.18, (1.0, 0.58, 0.18))
        ear.SetMetadata("comment", "Visual-only mouse ear")

    add_sphere(stage, "/World/Jerry/Nose", (0.0, -0.32, 0.02), 0.065, (0.25, 0.05, 0.04))

    tail = UsdGeom.Cylinder.Define(stage, "/World/Jerry/Tail")
    tail.CreateAxisAttr("X")
    tail.CreateRadiusAttr(0.045)
    tail.CreateHeightAttr(0.85)
    set_transform(tail.GetPrim(), (0.0, 0.55, -0.05), rotation_xyz=(0.0, 0.0, 22.0))
    set_color(tail.GetPrim(), (1.0, 0.55, 0.28))


def build_scene(output_path: Path) -> Usd.Stage:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(PHYSICS_HZ)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr(PHYSICS_HZ)

    add_box(stage, "/World/Floor", (0.0, 0.0, -0.10), (ROOM_WIDTH, ROOM_DEPTH, 0.20), (0.94, 0.86, 0.58))

    wall_color = (0.35, 0.78, 0.82)
    add_box(stage, "/World/Walls/North", (0.0, ROOM_DEPTH / 2, WALL_HEIGHT / 2), (ROOM_WIDTH, WALL_THICKNESS, WALL_HEIGHT), wall_color)
    add_box(stage, "/World/Walls/South", (0.0, -ROOM_DEPTH / 2, WALL_HEIGHT / 2), (ROOM_WIDTH, WALL_THICKNESS, WALL_HEIGHT), wall_color)
    add_box(stage, "/World/Walls/East", (ROOM_WIDTH / 2, 0.0, WALL_HEIGHT / 2), (WALL_THICKNESS, ROOM_DEPTH, WALL_HEIGHT), wall_color)
    add_box(stage, "/World/Walls/West", (-ROOM_WIDTH / 2, 0.0, WALL_HEIGHT / 2), (WALL_THICKNESS, ROOM_DEPTH, WALL_HEIGHT), wall_color)

    # Two offset obstacles leave multiple viable routes between Tom and Jerry.
    add_box(stage, "/World/Obstacles/BlueBlock", (-0.9, 0.4, 0.75), (1.0, 3.4, 1.5), (0.25, 0.35, 0.86))
    add_box(stage, "/World/Obstacles/PinkBlock", (1.8, -1.4, 0.65), (2.3, 0.9, 1.3), (0.92, 0.28, 0.54))

    build_tom(stage)
    build_jerry(stage)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(1800.0)
    sun.CreateAngleAttr(1.5)
    set_transform(sun.GetPrim(), (0.0, 0.0, 8.0), rotation_xyz=(48.0, -28.0, 25.0))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    set_transform(camera.GetPrim(), (10.5, -12.0, 10.0), rotation_xyz=(58.0, 0.0, 40.0))
    camera.CreateFocalLengthAttr(28.0)

    stage.GetRootLayer().Export(str(output_path))
    return stage


def main() -> None:
    output_path = Path("/workspace/output/tom_jerry_stage2.usda")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = build_scene(output_path)

    required_prims = ("/World/Floor", "/World/Tom/Body", "/World/Jerry/Body")
    missing = [path for path in required_prims if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Missing required scene prims: {missing}")

    prim_count = sum(1 for _ in stage.Traverse())
    print(
        f"STAGE2_OK path={output_path} prims={prim_count} physics_hz={PHYSICS_HZ}",
        flush=True,
    )


if __name__ == "__main__":
    main()
