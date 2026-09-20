# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MyCobot Pro450 reach: track a sampled Link6 pose, tool +Z pointing down.

Mirrors official Isaac-Reach-Franka-v0. The F100 gripper is in the USD but is not
an action; it stays at the default opening.
"""

import math

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from mycobot450_isaaclab.assets.robots.mycobot_pro450 import (
    MYCOBOT_PRO450_ARM_JOINT_NAMES,
    MYCOBOT_PRO450_CFG,
)

_EE_BODY = "Link6"


@configclass
class MycobotPro450ReachEnvCfg(ReachEnvCfg):
    """Reach sampled Link6 poses with joint-position actions on the 6 arm joints."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.scene.robot = MYCOBOT_PRO450_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=MYCOBOT_PRO450_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = None

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [_EE_BODY]
        self.rewards.end_effector_position_tracking.weight = -0.4
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [_EE_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.05
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [_EE_BODY]
        self.rewards.end_effector_orientation_tracking.weight = -0.2
        # Official curriculum tightens smoothness at 4500 steps and can freeze the last centimeters.
        self.curriculum.action_rate.params["num_steps"] = 20000
        self.curriculum.joint_vel.params["num_steps"] = 20000

        self.commands.ee_pose.body_name = _EE_BODY
        # Link6 +Z is the tool axis. pitch=pi points it at world -Z (same as Franka Reach).
        # Mid-workspace with F100 clearance: z>=0.18 keeps pads above the table; x>=0.18
        # avoids folding against the base. Yaw ±pi/4 is a small in-plane twist.
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.commands.ee_pose.ranges.roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.yaw = (-math.pi / 4.0, math.pi / 4.0)
        self.commands.ee_pose.ranges.pos_x = (0.18, 0.32)
        self.commands.ee_pose.ranges.pos_y = (-0.12, 0.12)
        self.commands.ee_pose.ranges.pos_z = (0.18, 0.32)
        self.commands.ee_pose.debug_vis = True

        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=MYCOBOT_PRO450_ARM_JOINT_NAMES
        )

        self.viewer.eye = (1.2, 1.2, 0.9)
        self.viewer.lookat = (0.22, 0.0, 0.15)


@configclass
class MycobotPro450ReachEnvCfg_PLAY(MycobotPro450ReachEnvCfg):
    """Fewer environments for visualization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
