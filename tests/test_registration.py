# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import mycobot450_isaaclab.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "Mycobot450Isaaclab-Reach": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "mycobot450_isaaclab.tasks.reach.config.mycobot450.mycobot450_reach_env_cfg:Mycobot450ReachEnvCfg"
            ),
            "default_agent": "rsl_rl",
        },
        "Mycobot450Isaaclab-Reach-Play": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "mycobot450_isaaclab.tasks.reach.config.mycobot450.mycobot450_reach_env_cfg:Mycobot450ReachEnvCfg_PLAY"
            ),
            "default_agent": "rsl_rl",
        },
        "Mycobot450Isaaclab-Lift-Cube": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "mycobot450_isaaclab.tasks.lift.config.mycobot450.joint_pos_env_cfg:Mycobot450CubeLiftEnvCfg"
            ),
            "default_agent": "rsl_rl",
        },
        "Mycobot450Isaaclab-Lift-Cube-Play": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "mycobot450_isaaclab.tasks.lift.config.mycobot450.joint_pos_env_cfg:Mycobot450CubeLiftEnvCfg_PLAY"
            ),
            "default_agent": "rsl_rl",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]


def test_registered_configs_can_be_loaded():
    """Catch upstream API changes hidden by string-only Gym registrations."""
    from importlib import import_module

    for task_id in (
        "Mycobot450Isaaclab-Reach",
        "Mycobot450Isaaclab-Reach-Play",
        "Mycobot450Isaaclab-Lift-Cube",
        "Mycobot450Isaaclab-Lift-Cube-Play",
    ):
        spec = gym.spec(task_id)
        for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"):
            module, name = spec.kwargs[key].split(":")
            cfg = getattr(import_module(module), name)()
            if key == "env_cfg_entry_point":
                assert cfg.scene.num_envs > 0
                assert cfg.decimation > 0
            else:
                assert cfg.num_steps_per_env > 0
                assert cfg.algorithm.class_name == "PPO"


def test_balance_removed():
    assert "Mycobot450Isaaclab-Balance-Cartpole" not in gym.registry
