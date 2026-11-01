# Stage 5 — PPO curriculum

## What was tested

Tom learns continuous planar forces over three progressively harder tasks:

1. stationary Jerry in an open arena;
2. a scripted figure-eight Jerry; and
3. a fixed full room with a perimeter-running Jerry and moving sweeper.

## Why this task is intentionally easy

Reward debugging and simulator integration are much easier when the first task
has an obvious solution. If PPO cannot learn this version, adding walls or a
moving target would hide the cause. A successful easy run establishes the
training, checkpoint, playback, and evaluation pipeline. It does not establish
obstacle-navigation performance, which is why the harder tasks are evaluated
separately.

## Reproducible commands

The project must be installed in the Isaac Lab environment so its
`isaaclab.tasks` entry point is discovered:

```bash
/opt/isaaclab-venv/bin/python -m pip install --no-deps -e /workspace/project

isaaclab train \
  --rl_library rsl_rl \
  --task TomJerry-Chase-Easy-Direct \
  --num_envs 64 \
  --max_iterations 80 \
  presets=newton_mjwarp
```

Run the saved policy through the browser-based Viser visualizer:

```bash
isaaclab play \
  --rl_library rsl_rl \
  --task TomJerry-Chase-Easy-Direct \
  --num_envs 9 \
  --checkpoint latest \
  --viz viser \
  presets=newton_mjwarp
```

Run independent fixed-seed evaluation:

```bash
/opt/isaaclab-venv/bin/python \
  /workspace/project/src/tom_jerry_chase/rl/evaluate_ppo.py \
  --num-envs 64 \
  --episodes 512 \
  --seed 20261101 \
  --output /workspace/project/work/stage5_ppo_eval.json
```

Choose a harder evaluator with `--curriculum moving` or `--curriculum room`.
The room command is:

```bash
/opt/isaaclab-venv/bin/python \
  /workspace/project/src/tom_jerry_chase/rl/evaluate_ppo.py \
  --curriculum room \
  --num-envs 64 \
  --episodes 512 \
  --seed 20261101 \
  --output /workspace/project/work/stage5_room_eval.json
```

The legacy-compatible trainer checkpoints do not contain Isaac Lab's newer run
manifest, so live playback must use the explicit checkpoint path rather than
the `latest` alias:

```bash
isaaclab play \
  --rl_library rsl_rl \
  --task TomJerry-Chase-Room-Eval-Direct \
  --num_envs 1 \
  --checkpoint /workspace/isaaclab/logs/rsl_rl/tom_jerry_room/2026-11-01_22-20-20/model_899.pt \
  --viz viser \
  presets=newton_mjwarp
```

## Actual result

The easy task reached 99.61% success. Moving Jerry reached 88.87%. The full
room reached 64.84%, with zero out-of-bounds failures, 180 timeouts, and 729
collision-entry events across 512 episodes. The first 300-iteration room run
and a shorter 500-iteration curriculum both produced zero catches; they were
not hidden or relabeled as successes. The final 900-iteration curriculum is
the result reported here.

See the JSON files for full precision and `curriculum_results.csv` for the
cross-curriculum summary. The fixed full-room result is not evidence of
generalization to unseen room layouts.

## Upstream references

- [Isaac Lab 3.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [Isaac Lab reinforcement-learning guide](https://isaac-sim.github.io/IsaacLab/main/source/concepts/reinforcement_learning.html)
- [NVIDIA CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
