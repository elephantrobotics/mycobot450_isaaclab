# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ActionsCfg, LiftEnvCfg, RewardsCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from mycobot450_isaaclab.assets.robots.mycobot_pro450 import (
    MYCOBOT_PRO450_ARM_JOINT_NAMES,
    MYCOBOT_PRO450_BASE_PRIM_PATH,
    MYCOBOT_PRO450_CFG,
    MYCOBOT_PRO450_EE_BODY_NAME,
    MYCOBOT_PRO450_EE_PRIM_PATH,
    MYCOBOT_PRO450_GRIPPER_CLOSE_POS,
    MYCOBOT_PRO450_GRIPPER_JOINT_NAMES,
    MYCOBOT_PRO450_GRIPPER_OPEN_POS,
    MYCOBOT_PRO450_TCP_OFFSET_POS,
)

from .mdp import rewards as mycobot_rewards

# Cube must sit in the F100 jaw box (TCP frame) to count as held.
_HOLD_DIST = 0.05
_THROW_DIST = 0.06
_JAW_TERM = {
    "jaw_depth": 0.04,
    "jaw_width": 0.03,
    "jaw_height": 0.03,
}
_GRIPPER_TERM = {
    "gripper_cfg": SceneEntityCfg("robot", joint_names=MYCOBOT_PRO450_GRIPPER_JOINT_NAMES),
    "gripper_open_pos": MYCOBOT_PRO450_GRIPPER_OPEN_POS,
    "gripper_close_pos": MYCOBOT_PRO450_GRIPPER_CLOSE_POS,
}
_WRIST_TERM = {
    "wrist_cfg": SceneEntityCfg("robot", body_names=["Link6"]),
}


@configclass
class MycobotRewardsCfg(RewardsCfg):
    """Grasp then carry: open approach, pinch, ungated lift, held goal tracking."""

    reaching_object = RewTerm(
        func=mycobot_rewards.ObjectEeDistanceCarry,
        params={"std": 0.1, "closed_ratio": 0.30, **_WRIST_TERM, **_GRIPPER_TERM, **_JAW_TERM},
        weight=1.0,
    )
    aligning_down = RewTerm(
        func=mycobot_rewards.ee_z_down,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Link6"])},
        weight=1.0,
    )
    ee_raised = RewTerm(
        func=mycobot_rewards.EeRaisedAfterGrasp,
        params={"minimal_height": 0.12, "closed_ratio": 0.30, **_GRIPPER_TERM, **_JAW_TERM},
        weight=4.0,
    )
    gripper_closed_away = RewTerm(
        func=mycobot_rewards.gripper_closed_away,
        params={"max_ee_object_dist": 0.06, "closed_ratio": 0.30, **_GRIPPER_TERM, **_JAW_TERM},
        weight=-3.0,
    )
    ee_goal_after_grasp = RewTerm(
        func=mycobot_rewards.EeGoalAfterGrasp,
        params={"std": 0.12, "command_name": "object_pose", "closed_ratio": 0.30, **_GRIPPER_TERM, **_JAW_TERM},
        weight=12.0,
    )
    holding_object = RewTerm(
        func=mycobot_rewards.object_pinch,
        params={
            "std": 0.08,
            "table_z": 0.055,
            "lift_bonus": 4.0,
            "max_bonus_height": 0.20,
            **_GRIPPER_TERM,
            **_JAW_TERM,
            **_WRIST_TERM,
        },
        weight=2.0,
    )
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.10}, weight=15.0)
    lifting_held = RewTerm(
        func=mycobot_rewards.object_is_lifted_and_held,
        params={"minimal_height": 0.10, "max_ee_object_dist": _HOLD_DIST, **_GRIPPER_TERM, **_JAW_TERM},
        weight=4.0,
    )
    object_tips_on_cube = RewTerm(
        func=mycobot_rewards.object_tips_on_cube,
        params={"max_ee_object_dist": _HOLD_DIST, **_JAW_TERM},
        weight=-3.0,
    )
    object_goal_tracking = RewTerm(
        func=mycobot_rewards.object_goal_distance_held,
        params={
            "std": 0.12,
            "minimal_height": 0.10,
            "command_name": "object_pose",
            "success_threshold": 0.05,
            "max_ee_object_dist": _HOLD_DIST,
            **_GRIPPER_TERM,
            **_JAW_TERM,
        },
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mycobot_rewards.object_goal_distance_held,
        params={
            "std": 0.05,
            "minimal_height": 0.10,
            "command_name": "object_pose",
            "max_ee_object_dist": _HOLD_DIST,
            **_GRIPPER_TERM,
            **_JAW_TERM,
        },
        weight=5.0,
    )
    object_goal_tracking_l1 = RewTerm(
        func=mycobot_rewards.object_goal_distance_held,
        params={
            "std": 1.0,
            "kernel": "neg_l1",
            "minimal_height": 0.10,
            "command_name": "object_pose",
            "max_ee_object_dist": _HOLD_DIST,
            **_GRIPPER_TERM,
            **_JAW_TERM,
        },
        weight=0.0,
    )
    object_throwing = RewTerm(
        func=mycobot_rewards.object_throwing,
        params={
            "airborne_height": 0.12,
            "max_ee_object_dist": _THROW_DIST,
            "vel_threshold": 0.6,
            "max_object_height": 0.35,
            **_GRIPPER_TERM,
            **_JAW_TERM,
        },
        weight=-4.0,
    )
    object_pressed_against_arm = RewTerm(
        func=mycobot_rewards.object_pressed_against_arm,
        params={
            "minimal_height": 0.10,
            "max_arm_object_dist": 0.055,
            "max_ee_object_dist": _HOLD_DIST,
            "arm_cfg": SceneEntityCfg("robot", body_names=["Link.*", "base_link"]),
            **_GRIPPER_TERM,
            **_JAW_TERM,
        },
        weight=-8.0,
    )
    # Official -0.1 curriculum on unclipped L2 made 450 joint spin return -1e4 and kill PPO.
    action_rate = RewTerm(func=mycobot_rewards.action_rate_l2_clipped, params={"max_l2": 5.0}, weight=-1.0e-2)
    joint_vel = RewTerm(
        func=mycobot_rewards.joint_vel_l2_clipped,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_l2": 25.0},
        weight=-1.0e-2,
    )


@configclass
class MycobotActionsCfg(ActionsCfg):
    """Arm joints stay relative; F100 uses a continuous opening angle, not binary open/close."""

    gripper_action: mdp.JointPositionToLimitsActionCfg = MISSING


@configclass
class MycobotPro450CubeLiftEnvCfg(LiftEnvCfg):
    """MyCobot Pro450 cube lift, aligned with Isaac-Lift-Cube-Franka-v0."""

    actions: MycobotActionsCfg = MycobotActionsCfg()
    rewards: MycobotRewardsCfg = MycobotRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.scene.robot = MYCOBOT_PRO450_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Convex-decomposed F100 fingers generate far more contact patches than convexHull.
        # Default gpu_max_rigid_patch_count is 5 * 2**15 (163840) and overflowed at ~1.7e5.
        self.sim.physics.gpu_max_rigid_patch_count = 2**20
        self.sim.physics.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physics.gpu_total_aggregate_pairs_capacity = 2**21

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=MYCOBOT_PRO450_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )
        # Continuous F100 angle: action in [-1, 1] maps to joint limits (open -1.012, close 0).
        # Binary open/close is what Franka-Lift uses; it chatters on an angular gripper.
        self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=MYCOBOT_PRO450_GRIPPER_JOINT_NAMES,
            scale=1.0,
            rescale_to_limits=True,
        )

        self.commands.object_pose.body_name = MYCOBOT_PRO450_EE_BODY_NAME
        self.commands.object_pose.debug_vis = True
        # Keep the command off the table cube. Overlapping XY + z=0.10 let a parked grasp farm goal.
        self.commands.object_pose.ranges.pos_x = (0.12, 0.26)
        self.commands.object_pose.ranges.pos_y = (-0.10, 0.10)
        self.commands.object_pose.ranges.pos_z = (0.16, 0.24)

        self.scene.table.init_state.pos = [0.22, 0.0, 0.0]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, 0.0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=2.0,
                    dynamic_friction=1.5,
                    restitution=0.0,
                    friction_combine_mode="max",
                ),
            ),
        )
        self.events.reset_object_position.params["pose_range"]["x"] = (-0.03, 0.03)
        self.events.reset_object_position.params["pose_range"]["y"] = (-0.05, 0.05)

        # Keep the official smoothness curriculum, but delay it and cap the jump.
        # At 1e5 env steps / weight -0.1 the 450's joint_vel_l2 exploded (~-2e4) and collapsed PPO.
        self.curriculum.action_rate.params["num_steps"] = 1_000_000
        self.curriculum.action_rate.params["weight"] = -1.0e-2
        self.curriculum.joint_vel.params["num_steps"] = 1_000_000
        self.curriculum.joint_vel.params["weight"] = -1.0e-2

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=MYCOBOT_PRO450_BASE_PRIM_PATH,
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=MYCOBOT_PRO450_EE_PRIM_PATH,
                    name="end_effector",
                    offset=OffsetCfg(pos=MYCOBOT_PRO450_TCP_OFFSET_POS),
                ),
            ],
        )

        self.viewer.eye = (1.2, 1.2, 0.9)
        self.viewer.lookat = (0.25, 0.0, 0.15)


@configclass
class MycobotPro450CubeLiftEnvCfg_PLAY(MycobotPro450CubeLiftEnvCfg):
    """Fewer environments for visualization and evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True
