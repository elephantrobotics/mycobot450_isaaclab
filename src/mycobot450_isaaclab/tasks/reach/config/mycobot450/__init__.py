"""450 + F100 end-effector pose tracking."""

import gymnasium as gym

for suffix, config in (("", "Mycobot450ReachEnvCfg"), ("-Play", "Mycobot450ReachEnvCfg_PLAY")):
    gym.register(
        id=f"Mycobot450Isaaclab-Reach{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mycobot450_reach_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
