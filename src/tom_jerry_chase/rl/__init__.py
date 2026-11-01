"""Gymnasium registration for the Tom-versus-Jerry Isaac Lab tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="TomJerry-Chase-Easy-Direct",
    entry_point=f"{__name__}.tom_jerry_env:TomJerryEasyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tom_jerry_env_cfg:TomJerryEasyEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:TomJerryEasyPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="TomJerry-Chase-Moving-Direct",
    entry_point=f"{__name__}.tom_jerry_moving_env:TomJerryMovingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tom_jerry_env_cfg:TomJerryMovingEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:TomJerryMovingPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

