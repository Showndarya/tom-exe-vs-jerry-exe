"""Render the Stage 2 USD scene to PNG without an interactive stream.

This is the reliable visual-verification path for cloud machines where the
WebRTC livestream extension cannot start.  The scene is still rendered by
Isaac Sim's RTX renderer on the NVIDIA GPU.
"""

from __future__ import annotations

from pathlib import Path

from isaacsim import SimulationApp


STAGE_PATH = Path("/workspace/output/tom_jerry_stage2.usda")
OUTPUT_DIR = Path("/workspace/output/stage2_render")
RESOLUTION = (1280, 720)


def main() -> None:
    simulation_app = SimulationApp(
        launch_config={
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        }
    )
    print("STAGE2_RENDER step=app_ready", flush=True)

    import omni.replicator.core as rep
    import omni.usd
    print("STAGE2_RENDER step=imports_ready", flush=True)

    if not STAGE_PATH.is_file():
        raise FileNotFoundError(f"Stage does not exist: {STAGE_PATH}")

    context = omni.usd.get_context()
    if not context.open_stage(str(STAGE_PATH)):
        raise RuntimeError(f"Isaac Sim could not open stage: {STAGE_PATH}")
    print("STAGE2_RENDER step=stage_open", flush=True)

    # Let USD composition, materials, lighting, and RTX resources settle.
    for _ in range(20):
        simulation_app.update()
    print("STAGE2_RENDER step=stage_settled", flush=True)

    camera = rep.functional.create.camera(
        position=(0.0, -16.0, 18.0),
        look_at=(0.0, 0.0, 0.25),
        focal_length=24.0,
        parent="/World",
        name="PortfolioCamera",
    )
    render_product = rep.create.render_product(camera, RESOLUTION, name="Stage2Render")
    print("STAGE2_RENDER step=render_product_ready", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(OUTPUT_DIR), rgb=True)
    writer.attach([render_product])
    print("STAGE2_RENDER step=writer_attached", flush=True)

    # Capture a few identical frames so the RTX renderer has time to converge.
    for _ in range(3):
        # Keep the timeline active while Replicator waits for the render event.
        # Pausing here can deadlock a headless writer before its first frame.
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
        print("STAGE2_RENDER step=frame_captured", flush=True)
    rep.orchestrator.wait_until_complete()

    writer.detach()
    render_product.destroy()
    simulation_app.update()

    images = sorted(OUTPUT_DIR.rglob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG was written under {OUTPUT_DIR}")
    print(f"STAGE2_RENDER_OK image={images[-1]} frames={len(images)}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
