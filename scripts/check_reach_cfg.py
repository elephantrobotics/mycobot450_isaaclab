# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity-check Reach env cfg without launching the simulator."""

import math

from mycobot450_isaaclab.tasks.manager_based.mycobot_pro450_reach.reach_env_cfg import (
    MycobotPro450ReachEnvCfg,
    MycobotPro450ReachEnvCfg_PLAY,
)


def main() -> None:
    cfg = MycobotPro450ReachEnvCfg()
    assert cfg.actions.gripper_action is None
    assert cfg.commands.ee_pose.body_name == "Link6"
    assert cfg.commands.ee_pose.ranges.pitch == (math.pi, math.pi)
    assert cfg.commands.ee_pose.ranges.roll == (0.0, 0.0)
    assert cfg.commands.ee_pose.ranges.yaw == (-math.pi / 4.0, math.pi / 4.0)
    assert cfg.commands.ee_pose.ranges.pos_x == (0.18, 0.32)
    assert cfg.commands.ee_pose.ranges.pos_y == (-0.12, 0.12)
    assert cfg.commands.ee_pose.ranges.pos_z == (0.18, 0.32)
    assert cfg.rewards.end_effector_position_tracking.params["asset_cfg"].body_names == ["Link6"]
    assert cfg.rewards.end_effector_position_tracking.weight == -0.4
    assert cfg.rewards.end_effector_position_tracking_fine_grained.params["std"] == 0.05
    assert cfg.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names == ["Link6"]
    assert cfg.rewards.end_effector_orientation_tracking.weight == -0.2
    assert cfg.curriculum.action_rate.params["num_steps"] == 20000
    assert cfg.curriculum.joint_vel.params["num_steps"] == 20000
    assert "joint[1-6]" in cfg.actions.arm_action.joint_names
    play = MycobotPro450ReachEnvCfg_PLAY()
    assert play.scene.num_envs == 50
    assert play.actions.gripper_action is None
    print("reach cfg check passed")


if __name__ == "__main__":
    main()
