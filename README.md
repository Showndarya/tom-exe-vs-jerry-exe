# Tom.exe vs Jerry.exe — The Great AI Chase

![Isaac Sim 6.2](https://img.shields.io/badge/Isaac%20Sim-6.2-76B900?logo=nvidia&logoColor=white)
![Isaac Lab 3.0](https://img.shields.io/badge/Isaac%20Lab-3.0-76B900?logo=nvidia&logoColor=white)
![Newton MJWarp 1.6.1](https://img.shields.io/badge/Newton%20MJWarp-1.6.1-5C3EE8)
![PPO with RSL--RL](https://img.shields.io/badge/PPO-RSL--RL%205.5.1-FF6B35)
![Tests](https://img.shields.io/badge/tests-12%20passing-2EA44F)

**A playful robotics project that asks one serious question: how should an
agent chase a moving target when walls keep getting in the way?**

Tom is a blue physics-driven robot. Jerry is an orange scripted target. They
race through small rooms built entirely from original geometric primitives—no
downloaded character models, rigging, or copyrighted assets. The project
implements three progressively more capable approaches:

1. direct pursuit, which runs straight at Jerry;
2. A* pathfinding, which searches around known obstacles; and
3. PPO reinforcement learning, which learns continuous forces from reward.

The simulation runs on an NVIDIA L40S in Brev. An Apple Silicon Mac acts as
the development machine while Isaac Sim, Isaac Lab, Newton physics, training,
evaluation, and rendering stay on the cloud GPU.

## Results at a glance

These are measured results from fixed-seed runs, not hand-authored demo
outcomes.

| Stage | Method | Challenge | Verified result |
| --- | --- | --- | --- |
| 3 | Direct pursuit | Moving Jerry with an obstructed straight line | Failed to catch; 1 collision and 14 stuck frames |
| 4 | A* | Known map plus a surprise wall | Caught in 11.4 s; 14 replans; 0 collisions |
| 5A | PPO | Stationary Jerry, open arena | 511 / 513 catches — **99.61%** |
| 5B | PPO | Figure-eight Jerry | 455 / 512 catches — **88.87%** |
| 5C | PPO | Fixed full room plus moving sweeper | 332 / 512 catches — **64.84%** |

The Stage 3 and Stage 4 demonstrations use different controller speeds and
termination conditions, so the table is a milestone summary—not a fair
head-to-head benchmark. A same-layout, same-speed comparison remains Stage 6
work.

### PPO curriculum

| Curriculum | Training iterations | Environment steps | Independent evaluation | Mean catch time |
| --- | ---: | ---: | ---: | ---: |
| Easy | 80 | 122,880 | 99.61% | 0.7282 s |
| Moving Jerry | 160 | 245,760 | 88.87% | 1.0660 s |
| Full room | 900 | 1,382,400 | 64.84% | 5.4653 s |

The first two full-room attempts produced **zero catches**. They remain in the
experiment record. The final curriculum used a 300-iteration warm-up, a
200-iteration difficulty ramp, and 400 iterations at full difficulty. The
64.84% result shows partial learning in one fixed room; it does **not** prove
generalization to unseen layouts.

Full-precision outputs live in
[`experiments/stage5/`](experiments/stage5/), including the seed, checkpoint
iteration, failures, collision events, and dependency versions.

## How the system fits together

```text
 scripted Jerry ── pose + velocity ─┐
                                    │
 room + sweeper ── geometry ────────┼──> observation vector
                                    │           │
 dynamic Tom ──── velocity ─────────┘           v
                                         controller / policy
                                       direct | A* | PPO
                                                │
                                                v
                                      x/y velocity or force
                                                │
                                                v
                                    Isaac Sim / Newton physics
                                                │
                              collisions, distance, catches, resets
                                                │
                              PPO only: reward + next observation
```

Pathfinding and reinforcement learning are intentionally kept separate. A*
receives an occupancy grid and searches it. PPO receives numbers describing
the current state and optimizes a policy from experience. The PPO policy does
not call A* internally.

## The three Toms

### Dumb Tom — direct pursuit

Tom observes only Jerry's relative position and points straight at it. This is
a useful baseline because its failure is easy to explain: Tom has no map and
cannot reason about going around a wall. PhysX stops the body at the obstacle,
but the controller keeps asking it to move forward.

Verified seed-7 result: final distance 6.6032 m, one collision, 14 stuck
frames, and no catch. See
[`experiments/stage3/direct_pursuit_summary.json`](experiments/stage3/direct_pursuit_summary.json).

### Planning Tom — A*

The room becomes a 0.5 m occupancy grid. Free cells are graph nodes; cardinal
moves are edges. Obstacles are inflated by 0.58 m so the route leaves room for
Tom's collider. A* minimizes:

```text
f(n) = g(n) + h(n)
```

`g(n)` is the route cost already traveled and `h(n)` is Manhattan distance to
Jerry. When a yellow surprise wall appears, Tom discards the blocked route and
searches again.

Verified seed-7 result: catch in 11.4 s, 14 replans, and zero collisions. See
[`experiments/stage4/astar_summary.json`](experiments/stage4/astar_summary.json).

### Learning Tom — PPO

The Isaac Lab tasks use continuous control. The policy emits two values in
`[-1, 1]`; they become x/y forces on Tom. The hardest task exposes 13
observations covering Jerry, Tom, the moving sweeper, room position, and
curriculum difficulty.

The reward combines progress toward Jerry, a catch bonus, distance cost,
action cost, collision penalty, time cost, and arena-boundary penalty. Episode
resets randomize the starting configuration while fixed evaluation seeds make
comparisons repeatable.

## What works today

- [x] Brev L40S setup with authenticated browser access
- [x] Isaac Sim 6.2 scene generation and headless RTX frames
- [x] Direct-pursuit baseline with contact-event metrics
- [x] A* planning with obstacle inflation and replanning
- [x] Isaac Lab 3.0 custom `DirectRLEnv` tasks
- [x] NVIDIA Newton MJWarp physics on the cloud GPU
- [x] Easy, moving-target, and full-room PPO curricula
- [x] Fixed-seed evaluation with catch, timeout, OOB, time, and collision metrics
- [x] Lightweight live Viser playback for the Newton room
- [x] Twelve local unit tests for controller, planner, and reward logic
- [ ] Same-layout direct/A*/PPO comparison
- [ ] Randomized-room training and unseen-layout evaluation
- [ ] Final narrated demo video

## Repository map

```text
.
├── src/tom_jerry_chase/
│   ├── scene_stage2.py              # primitive Isaac Sim room
│   ├── render_stage2.py             # headless RTX capture
│   ├── stage3_direct_pursuit.py      # baseline controller
│   ├── grid_planner.py              # occupancy grid + A*
│   ├── stage4_astar.py               # replanning simulation
│   └── rl/
│       ├── tom_jerry_env_cfg.py      # Isaac Lab scene/task configs
│       ├── tom_jerry_*_env.py        # easy, moving, and room tasks
│       ├── evaluate_ppo.py           # fixed-seed evaluation
│       └── agents/                   # RSL-RL PPO configuration
├── experiments/                      # reviewed JSON/CSV evidence
├── infra/                            # cloud/container compatibility files
├── tests/                            # fast CPU-only unit tests
└── pyproject.toml                    # Isaac Lab task registration
```

Large checkpoints, TensorBoard logs, frame sequences, and local cloud scratch
files are intentionally ignored.

## Reproduce it

### 1. Run the fast tests locally

The planner, controller, and reward tests do not require Isaac Sim:

```bash
git clone https://github.com/Showndarya/tom-exe-vs-jerry-exe.git
cd tom-exe-vs-jerry-exe
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Expected result: `Ran 12 tests ... OK`.

### 2. Prepare the Isaac Lab container on an NVIDIA GPU

Isaac Sim and Isaac Lab do not run natively on Apple Silicon. Use a Linux host
with an NVIDIA GPU. This project used one L40S with 48 GB VRAM.

The official Isaac Lab 3.0 kitless image uses CUDA 13. The tested Brev host had
an R565 driver, so the derived image installs NVIDIA's forward-compatibility
package without modifying or rebooting the host:

```bash
docker build \
  --file infra/Dockerfile.isaaclab-cuda-compat \
  --tag tomjerry-isaaclab:3.0.0-r565-compat \
  .
```

Inside the container, install this package so Isaac Lab discovers its task
entry point:

```bash
/opt/isaaclab-venv/bin/python -m pip install --no-deps -e /workspace/project
```

### 3. Train the easy PPO task

```bash
isaaclab train \
  --rl_library rsl_rl \
  --task TomJerry-Chase-Easy-Direct \
  --num_envs 64 \
  --max_iterations 80 \
  presets=newton_mjwarp
```

### 4. Evaluate a saved policy

```bash
/opt/isaaclab-venv/bin/python \
  /workspace/project/src/tom_jerry_chase/rl/evaluate_ppo.py \
  --curriculum room \
  --num-envs 64 \
  --episodes 512 \
  --seed 20261101 \
  --output /workspace/project/work/stage5_room_eval.json
```

Use `--curriculum easy` or `--curriculum moving` for the earlier curriculum
steps. Checkpoint selection follows the installed Isaac Lab/RSL-RL launcher;
the exact verified checkpoints are recorded in the experiment JSON files.

### 5. Watch one policy live

```bash
isaaclab play \
  --rl_library rsl_rl \
  --task TomJerry-Chase-Room-Eval-Direct \
  --num_envs 1 \
  --checkpoint /path/to/model_899.pt \
  --viz viser \
  presets=newton_mjwarp
```

The Viser page provides pause, reset, plots, and camera controls. If the canvas
opens blank, choose **Configuration & diagnostics → Reset View**.

## Cloud and rendering notes

<details>
<summary><strong>Why the project uses Viser for live playback</strong></summary>

Isaac Sim's RTX renderer worked correctly on the tested L40S and produced PNG
frames. Its WebRTC server could not bind the Brev VM's NAT-provided public
address to a real network interface, so the primary stream reported
`NVST_R_INTERNAL_ERROR`. That is a livestream transport limitation, not a
simulation failure.

Isaac Lab's lightweight Viser visualizer works over a Brev-authenticated HTTPS
route and is sufficient for live Newton playback. It is not a replacement for
the photorealistic RTX viewport.

</details>

<details>
<summary><strong>Security and cost controls</strong></summary>

- Browser endpoints use Brev authentication.
- Raw streaming ports are never intended to remain open to the internet.
- Development ports must be restricted to the current client IP when used.
- The tested L40S compute rate was $1.79/hour.
- Stopping compute preserves the environment but continues storage billing.
- Stop the instance whenever active GPU work pauses.

</details>

## Experiment discipline

- Planning is never described as learning.
- Training success is verified with separate fixed-seed evaluation.
- Failed curricula remain in the CSV instead of disappearing from the story.
- Exact seeds, checkpoint iterations, dependency versions, and episode counts
  are stored beside the results.
- GPU physics may introduce small numerical variation; reproducible settings do
  not imply bit-for-bit identical CUDA execution.

## Roadmap

Stage 6 will put direct pursuit, A*, and PPO into the same room with identical
speeds, starts, Jerry routes, obstacle changes, and seeds. The comparison will
report catch rate, collision count, and completion time.

Stage 7 will add randomized or procedurally selected room layouts, an unseen
layout test, and a short narrated demo. The current full-room policy is a
fixed-layout result and should be presented that way in interviews.

## Upstream references

This project builds on NVIDIA's public APIs and documentation:

- [Isaac Sim 6.2 documentation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/)
- [Isaac Lab 3.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [Isaac Lab reinforcement-learning guide](https://isaac-sim.github.io/IsaacLab/main/source/concepts/reinforcement_learning.html)
- [Isaac Sim Replicator writer workflow](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html)
- [PhysX contact-report workflow](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html)
- [NVIDIA CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)

Built as a one-day learning sprint, with the debugging history left intact so
the repository shows how the system evolved—not just the final happy path.
