"""Fixed-seed evaluation for the trained Stage 5 RSL-RL PPO policy."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the easy Tom chase policy.")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--log-root", default="/workspace/isaaclab/logs/rsl_rl/tom_jerry_easy")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    return parser.parse_args()


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(selector: str, log_root: Path) -> Path:
    if selector != "latest":
        path = Path(selector)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    candidates = list(log_root.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints found under {log_root}")
    return max(candidates, key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime))


def scalar(value) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def main() -> None:
    args = parse_args()

    import gymnasium as gym
    import torch

    import tom_jerry_chase.rl  # noqa: F401 - registers the Gym task
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner

    from tom_jerry_chase.rl.agents.rsl_rl_ppo_cfg import TomJerryEasyPPORunnerCfg
    from tom_jerry_chase.rl.tom_jerry_env_cfg import TomJerryEasyEnvCfg, TomJerryPhysicsCfg

    checkpoint = resolve_checkpoint(args.checkpoint, Path(args.log_root))
    env_cfg = TomJerryEasyEnvCfg()
    # The unified Isaac Lab CLI resolves PresetCfg selectors before launch.
    # This standalone evaluator must choose the concrete backend itself.
    env_cfg.sim.physics = TomJerryPhysicsCfg().newton_mjwarp
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = TomJerryEasyPPORunnerCfg()
    agent_cfg.seed = args.seed

    with launch_simulation(env_cfg, args):
        env = gym.make("TomJerry-Chase-Easy-Direct", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(wrapped, agent_cfg)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
        observations = wrapped.get_observations()

        completed = 0
        successes = 0.0
        out_of_bounds = 0.0
        catch_time_sum = 0.0
        vector_steps = 0
        max_vector_steps = (
            math.ceil(args.episodes / args.num_envs) * wrapped.unwrapped.max_episode_length * 2
        )

        with torch.inference_mode():
            while completed < args.episodes and vector_steps < max_vector_steps:
                observations, _, dones, extras = wrapped.step(policy(observations))
                policy.reset(dones)
                vector_steps += 1
                done_count = int(dones.sum().item())
                if done_count == 0:
                    continue

                log = extras.get("log", {})
                catch_rate = scalar(log.get("Metrics/catch_rate", 0.0))
                out_rate = scalar(log.get("Metrics/out_of_bounds_rate", 0.0))
                mean_catch_time = scalar(log.get("Metrics/mean_catch_time_s", 0.0))
                step_successes = catch_rate * done_count

                completed += done_count
                successes += step_successes
                out_of_bounds += out_rate * done_count
                catch_time_sum += mean_catch_time * step_successes

        wrapped.close()

    successes_int = int(round(successes))
    out_of_bounds_int = int(round(out_of_bounds))
    # Vectorized environments can finish several episodes on the final step,
    # so report the exact number observed rather than silently discarding them.
    evaluated = completed
    result = {
        "task": "TomJerry-Chase-Easy-Direct",
        "algorithm": "PPO via RSL-RL 5.5.1",
        "physics": "NVIDIA Newton MJWarp 1.6.1",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration(checkpoint),
        "seed": args.seed,
        "num_parallel_envs": args.num_envs,
        "episodes_requested": args.episodes,
        "episodes_completed": evaluated,
        "catch_count": successes_int,
        "catch_success_rate": successes_int / evaluated if evaluated else 0.0,
        "out_of_bounds_count": out_of_bounds_int,
        "mean_catch_time_s": catch_time_sum / successes if successes else None,
        "vector_steps": vector_steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "PPO_EVAL_OK "
        f"episodes={evaluated} catches={successes_int} "
        f"success_rate={result['catch_success_rate']:.4f} "
        f"out_of_bounds={out_of_bounds_int} "
        f"mean_catch_time_s={result['mean_catch_time_s']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

