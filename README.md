# Tom.exe vs Jerry.exe — The Great AI Chase

A playful robotics project built with NVIDIA Isaac Sim 6.2 and Isaac Lab 3.0.
Original geometric characters chase through a tiny obstacle-filled room while
we compare direct pursuit, A* path planning, and a PPO policy honestly.

## Current milestone

- [x] NVIDIA Brev L40S provisioned and secured
- [x] Isaac Sim 6.2 compatibility check passed
- [x] Full headless Isaac Sim runtime reached `app ready`
- [x] Stage 2 primitive room, Tom, and Jerry
- [x] Stage 2 RTX PNG render verified on the L40S
- [x] Authenticated Brev viewer endpoint configured
- [x] Authenticated live Newton/Viser browser visualization verified
- [x] Stage 3 direct-pursuit baseline and animated RTX replay
- [x] Stage 4 A* planner with moving-target and blocked-route replanning
- [x] Stage 5 easy Isaac Lab PPO environment, training, playback, and evaluation
- [ ] Stage 6 reproducible comparison
- [ ] Stage 7 portfolio documentation and demo

## Design principles

- Use only original geometric characters and simple materials.
- Keep path planning distinct from reinforcement learning.
- Record fixed seeds and real metrics; never invent convergence.
- Keep cloud endpoints private and avoid idle GPU cost.
- Build the simplest working version before adding complexity.

## Repository layout

```text
src/tom_jerry_chase/   Isaac Sim and Isaac Lab source code
infra/                 Docker Compose overrides for cloud streaming
docs/                  Setup and architecture notes
experiments/           Reproducible configurations and CSV results
outputs/               User-facing deliverables
```

## Verified Stage 2 result

`src/tom_jerry_chase/scene_stage2.py` builds a 12 m × 9 m room from OpenUSD
primitives. It includes four collidable walls, two route-splitting obstacles,
a blue capsule cat, an orange sphere mouse, rigid bodies, gravity, lighting,
and a camera.

The scene was executed inside the official Isaac Sim 6.2 container on the Brev
L40S. Its self-check reported:

```text
STAGE2_OK path=/workspace/output/tom_jerry_stage2.usda prims=37 physics_hz=60
```

The generated USDA was then checked for `/World/PhysicsScene`, `/World/Tom`,
and `/World/Jerry`. Detailed setup and repeatable run commands will continue to
be added as each milestone is verified on the Brev GPU rather than written
speculatively.

### Verified headless RTX render

Interactive streaming and rendering are separate pieces. The livestream
extension can fail while Isaac Sim's RTX renderer and the simulation remain
healthy. `src/tom_jerry_chase/render_stage2.py` opens the saved USD stage,
creates a Replicator camera and RGB writer, and saves three PNG frames using
the L40S GPU.

The following command was executed successfully on the Brev VM. It reuses the
workspace volume from the Compose-created Isaac Sim container, which may be
stopped while the one-off renderer runs:

```bash
docker run --gpus all --rm --name tomjerry-stage2-render \
  --shm-size=2g \
  --volumes-from tomjerry-isaac-sim-1 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  --entrypoint /isaac-sim/python.sh \
  nvcr.io/nvidia/isaac-sim:6.2.0 \
  /workspace/src/tom_jerry_chase/render_stage2.py
```

The verification marker and final image path were:

```text
STAGE2_RENDER_OK image=/workspace/output/stage2_render/rgb_0002.png frames=3
```

The render script follows NVIDIA's
[Replicator writer workflow](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html).

## Secure Brev streaming

The official Isaac Sim Docker Compose viewer uses three ports:

| Port | Protocol | Purpose |
| --- | --- | --- |
| 8210 | TCP | Web viewer |
| 49100 | TCP | WebRTC signaling |
| 47998 | UDP | WebRTC media |

The web page is exposed through a Brev-authenticated HTTPS endpoint. The raw
WebRTC ports are restricted on the VM itself to the current client IP. Do not
leave unauthenticated streaming ports open without that host firewall.

Create one authenticated route for the viewer. The viewer proxies the
`/sign_in` WebSocket path to Isaac Sim inside Docker, following NVIDIA's HTTPS
deployment pattern:

```bash
brev ports open tom-exe-jerry-exe 8210 --protocol http --json
```

Set both `WEB_VIEWER_ALLOWED_HOST` and `ISAACSIM_SIGNAL_HOST` to the same
viewer hostname returned by Brev, without the `https://` prefix. The custom
viewer keeps the page and signaling channel on one HTTPS/WSS origin while
sending media to the VM's IP-restricted UDP port. Port `49100` does not need a
separate Brev route:

```bash
WEB_VIEWER_ALLOWED_HOST="<authenticated-viewer-host>" \
ISAACSIM_SIGNAL_HOST="<authenticated-viewer-host>" \
ISAACSIM_SIGNAL_PORT=443 \
ISAACSIM_FORCE_WSS=true \
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0 \
ISAACSIM_HOST="<brev-public-ip>" \
ISAAC_SIM_DATA=/home/ubuntu/docker/isaac-sim \
docker compose -p tomjerry \
  -f /home/ubuntu/isaacsim-streaming/tools/docker/docker-compose.yml \
  -f /home/ubuntu/tom-exe-vs-jerry-exe/infra/docker-compose.stream.override.yml \
  up --build -d
```

The tested Brev VM reached the viewer login but did not start Isaac Sim's
primary stream server. Isaac Sim logged `NVST_R_INTERNAL_ERROR` and
`Failed to start the primary stream server`, so port `49100` never listened
and the browser correctly remained at **Waiting for stream**. This is a
server-side livestream/driver compatibility issue, not an authentication
failure. The scene itself still runs and renders on the GPU; use the verified
headless render path below while the VM image is on its current R565 driver.

The temporary `iptables` rules used during development allow only
`<current-client-ip>/32` to the three streaming ports, followed by a drop rule
for every other source. These rules are deliberately not persisted across a
VM reboot; verify and reapply them whenever the instance or client network
changes.

The Brev L40S used for this project costs `$1.79/hour` in the selected region.
Stop the VM whenever development pauses so credits are not consumed while idle.

## Verified Stage 3 result: Dumb Tom

`src/tom_jerry_chase/stage3_direct_pursuit.py` turns the Stage 2 characters
into moving rigid bodies. Jerry follows a fixed waypoint loop. On every frame,
Tom observes only the relative position `(jerry_x - tom_x, jerry_y - tom_y)`
and commands a constant-speed velocity pointing directly at Jerry.

This is a hand-written controller, not A* and not machine learning. Tom has no
map or obstacle observation, so it cannot reason about going around a wall.
PhysX still resolves the collision, which means Tom pushes into the obstacle
instead of passing through it.

Run the experiment in the official Isaac Sim 6.2 container:

```bash
docker run --gpus all --rm --name tomjerry-stage3 \
  --shm-size=2g \
  --volumes-from tomjerry-isaac-sim-1 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/src \
  --entrypoint /isaac-sim/python.sh \
  nvcr.io/nvidia/isaac-sim:6.2.0 \
  /workspace/src/tom_jerry_chase/stage3_direct_pursuit.py
```

The fixed-seed Brev run produced these actual results:

| Metric | Result |
| --- | ---: |
| Random seed | 7 |
| Simulated time | 8.0 s |
| Rendered frames | 120 |
| Initial Tom–Jerry distance | 8.7818 m |
| Final Tom–Jerry distance | 6.60316 m |
| Obstacle/wall collisions | 1 |
| Stuck frames | 14 |
| Caught Jerry | No |

The raw per-frame CSV, JSON summary, PNG frames, and animated GIF were written
to `/workspace/output/stage3_direct_pursuit/` on the Brev instance. The reviewed
summary is preserved in `experiments/stage3/direct_pursuit_summary.json`.

### Collision detection

Tom uses PhysX contact reports. A contact callback receives `CONTACT_FOUND`
and `CONTACT_LOST` events, identifies pairs involving Tom and a wall or
obstacle, and counts each newly formed contact once. This follows NVIDIA's
[PhysX contact report workflow](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html).

Two compatibility issues found during the cloud run are intentionally recorded:

1. Isaac Sim 6.2's installed Python schema uses a shorter contact-threshold
   factory name than newer documentation. The script tries that binding and
   falls back to authoring the canonical USD attribute.
2. The container runs as UID 1234 while the Jupyter workspace is owned by the
   `ubuntu` user. Generated experiment data therefore goes into the existing
   container-writable `/workspace/output` directory; only reviewed summaries
   are copied into Git.

## Live rendering status

The Stage 3 GIF is a genuine RTX-rendered animation of the physics run, but it
is a replay rather than an interactive live viewport. It can be viewed in
Jupyter immediately after an experiment finishes.

The current Brev VM uses host networking, but its public address is provided
through NAT and is not assigned to a VM network interface. Isaac Sim's WebRTC
server must bind its media address to a real interface, so it reports
`NVST_R_INTERNAL_ERROR` and never opens the stream port when given that public
address. This explains why the authenticated viewer remains at **Waiting for
stream** even though RTX rendering and physics work correctly. NVIDIA's
[Isaac Sim Docker guide](https://github.com/isaac-sim/IsaacSim/blob/develop/tools/docker/README.md)
documents the host-network and real-interface requirement.

A true interactive viewport remains possible with a directly routable VM
interface or a supported secure UDP/VPN route. That change may involve a new
paid VM, so it will only be attempted after explicit approval. Until then, the
rendered GIF gives a truthful, low-risk visual check of every experiment.

## Verified Stage 4 result: A* Tom

`src/tom_jerry_chase/grid_planner.py` converts the room into a 0.5 m occupancy
grid. Free cells are graph nodes, horizontal and vertical moves are graph
edges, and obstacle cells are removed from the graph. Obstacles are expanded
by 0.58 m so the planned route leaves space for Tom's physical collider.

A* chooses the next cell with the smallest sum of:

- the route cost already traveled, `g(n)`; and
- the Manhattan-distance estimate to Jerry, `h(n)`.

`src/tom_jerry_chase/stage4_astar.py` follows the resulting cell centers as
waypoints. It replans when Jerry enters another grid cell or when an obstacle
blocks the remaining route. At frame 24, a yellow surprise wall appears across
the initially chosen upper corridor, so Tom discards that path and searches
for a lower route.

This is classical search, not machine learning: no policy is trained, no
reward is optimized, and A* does not improve from experience.

Stage 4 also increases Tom from 1.8 to 3.0 m/s and Jerry from 0.7 to 1.1 m/s.
Before the final Stage 6 comparison, every controller will be rerun with the
same speeds so the comparison remains fair.

The verified Brev run reported:

| Metric | Result |
| --- | ---: |
| Random seed | 7 |
| Simulated time / catch time | 11.4 s |
| Rendered frames | 171 |
| Replans | 14 |
| Environment changes | 1 |
| Obstacle/wall collisions | 0 |
| Initial Tom–Jerry distance | 8.7818 m |
| Final Tom–Jerry distance | 0.79115 m |
| Caught Jerry | Yes |

Two run-time corrections are preserved in Git history:

1. The first catch threshold was 0.75 m, but Tom's 0.46 m radius plus Jerry's
   0.34 m radius means the colliders touch at about 0.80 m. The corrected
   0.82 m threshold includes a 2 cm numerical tolerance without requiring the
   bodies to overlap.
2. A shorter successful run left tail PNGs from an earlier timeout run. Stage
   4 now removes only its own old generated PNGs before rendering, preventing
   stale frames from contaminating the animation.

The raw CSV, JSON summary, 171 PNG frames, and 86-frame GIF are in
`/workspace/output/stage4_astar/` on Brev. The reviewed summary is stored in
`experiments/stage4/astar_summary.json`.

## Verified Stage 5 result: Learning Tom, three-step curriculum

The first RL curriculum deliberately removes walls and keeps Jerry stationary.
This is the minimum task that answers one question cleanly: can PPO learn a
continuous 2D pursuit controller from observations and reward, rather than
being handed a direction or path?

`TomJerry-Chase-Easy-Direct` is an Isaac Lab `DirectRLEnv` with 64 parallel
rooms during training. Each blue Tom is a dynamic rigid sphere and each orange
Jerry is a stationary target. The policy receives four numbers:

- Jerry's relative x and y position; and
- Tom's x and y velocity.

The policy outputs two continuous actions. They become world-frame x/y forces
through Isaac Lab's permanent wrench composer. The reward favors progress,
adds a 12-point catch bonus, and penalizes distance, excessive action, elapsed
steps, and leaving the arena. A* is not used anywhere in this task.

The official Isaac Lab 3.0 kitless image required CUDA 13, while the Brev host
driver exposed CUDA 12.7 compatibility. `infra/Dockerfile.isaaclab-cuda-compat`
installs NVIDIA's `cuda-compat-13-0` package in a derived image; the host was
not modified or rebooted. The resulting container verified `CUDA_AVAILABLE
True` and identified the GPU as an NVIDIA L40S.

An 80-iteration PPO run using RSL-RL 5.5.1 and NVIDIA Newton MJWarp 1.6.1
completed in 29.1 seconds. The final training batch reported a 1.0 catch rate,
but that number alone is not treated as proof. A separate fixed-seed evaluation
loaded `model_79.pt` and produced:

| Metric | Result |
| --- | ---: |
| Evaluation seed | 20261101 |
| Parallel environments | 64 |
| Completed episodes | 513 |
| Catches | 511 |
| Catch success rate | 99.61% |
| Out-of-bounds failures | 2 |
| Mean successful catch time | 0.7282 s |

This easy result established the learning pipeline, so the curriculum was then
extended twice instead of treating the 99.61% score as the finished project.

### Moving Jerry

`TomJerry-Chase-Moving-Direct` makes Jerry follow a scripted figure-eight. The
policy receives Jerry's velocity as well as the original relative-position and
Tom-velocity observations. A 160-iteration PPO run processed 245,760
environment steps in 48.35 seconds. Independent evaluation produced:

| Metric | Result |
| --- | ---: |
| Evaluation seed | 20261101 |
| Completed episodes | 512 |
| Catches | 455 |
| Catch success rate | 88.87% |
| Out-of-bounds failures | 46 |
| Timeouts | 11 |
| Mean successful catch time | 1.0660 s |

### Full room and dynamic obstacle

`TomJerry-Chase-Room-Direct` adds four enclosing walls, two interior barriers
that create alternate corridors, a yellow scripted sweeper, and a square Jerry
route. Its 13 observations describe the target, Tom, the sweeper, local room
position, and curriculum difficulty. The two actions are still continuous
world-frame x/y forces: there is no A* path hidden inside the policy.

Two shorter training attempts honestly failed with zero catches. The final
curriculum held Jerry and the sweeper still for 300 iterations, ramped their
motion over 200 iterations, and trained at full difficulty for another 400.
That 900-iteration run processed 1,382,400 environment steps in 245.81 seconds.
A fresh collision-instrumented evaluation of `model_899.pt` produced:

| Metric | Result |
| --- | ---: |
| Evaluation seed | 20261101 |
| Completed episodes | 512 |
| Catches | 332 |
| Catch success rate | 64.84% |
| Out-of-bounds failures | 0 |
| Timeouts | 180 |
| Collision-entry events | 729 |
| Mean collision-entry events / episode | 1.4238 |
| Mean successful catch time | 5.4653 s |

The seed, checkpoint, and dependency versions are recorded for reproduction.
GPU contact solving can still introduce small run-to-run numerical variation,
so this is a reproducible experiment specification rather than a promise of
bit-for-bit CUDA results.

This room is a fixed layout with a changing target and one dynamic obstacle.
It demonstrates partial learning in that layout; it does **not** yet prove
generalization to previously unseen or randomly generated rooms. Random-room
training and a same-layout direct/A*/PPO comparison remain Stage 6 work.

Live playback uses Isaac Lab's Viser visualizer through a Brev-authenticated
HTTPS route. This lightweight Newton view works despite the VM's RTX WebRTC
NAT limitation. It currently shows one policy-controlled full room for a clear
demo and provides pause, reset, and live-plot controls; it is not the
photorealistic Isaac Sim RTX viewport.

Reviewed results are preserved in `experiments/stage5/`. Large checkpoints and
TensorBoard event files remain under the ignored `work/isaaclab_logs/` folder
on Brev.
