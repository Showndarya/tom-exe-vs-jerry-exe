# Stage 5 — Easy PPO chase

## What was tested

Tom learns continuous planar forces from four observations: Jerry's relative
position and Tom's planar velocity. Jerry is stationary and there are no
obstacles in this first curriculum step.

## Why this task is intentionally easy

Reward debugging and simulator integration are much easier when the first task
has an obvious solution. If PPO cannot learn this version, adding walls or a
moving target would hide the cause. A successful easy run establishes the
training, checkpoint, playback, and evaluation pipeline; it does not establish
obstacle-navigation performance.

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

## Actual result

The evaluator completed 513 episodes because several vectorized environments
terminated together on the final simulator step. It recorded 511 catches, two
out-of-bounds failures, a 99.61% success rate, and a 0.7282-second mean catch
time. See `easy_ppo_evaluation.json` for full precision and
`easy_ppo_results.csv` for the training/evaluation summary.

## Upstream references

- [Isaac Lab 3.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [Isaac Lab reinforcement-learning guide](https://isaac-sim.github.io/IsaacLab/main/source/concepts/reinforcement_learning.html)
- [NVIDIA CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
