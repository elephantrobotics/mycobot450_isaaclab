"""Franka-style pose reaching with the six-axis 450 and a fixed-open F100.

The tracked frame is f100_base_link (the gripper mount), not the fingertip TCP.
All policy quaternions use Isaac Lab 3's XYZW convention.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg
from isaaclab_tasks.utils import PresetCfg

from mycobot450_isaaclab.assets.lighting import configure_arm_lighting
from mycobot450_isaaclab.assets.robots.mycobot_pro450 import (
    MYCOBOT_PRO450_CFG,
    MYCOBOT_PRO450_EE_BODY_NAME,
)

from ...mdp import (
    ReachablePoseCommandCfg,
    SmoothJointVelocityActionCfg,
    filtered_target,
    pose_error,
    reset_joints_from_reached_states,
    settled_success,
    target_velocity,
)

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
# Absolute target clips in radians, inside the authored USD limits.
JOINT_TARGET_LIMITS = dict(
    zip(ARM_JOINTS, [(-2.8, 2.8), (-2.15, 2.15), (-2.65, 2.65), (-2.8, 2.8), (-2.8, 2.8), (-2.85, 2.85)], strict=True)
)


@configclass
class ReachPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.01)
    default: PhysxCfg = isaacsim_physx


@configclass
class ReachEventsCfg:
    # Reset targets as well as state: joints without an action (F100) otherwise
    # retain the actuator collection's zero target and close after the first step.
    reset_scene = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_robot_joints = EventTermCfg(
        func=reset_joints_from_reached_states,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "reached_state_probability": 0.0,
        },
    )


@configclass
class Mycobot450ReachEnvCfg(ReachEnvCfg):
    """Reach and hold one pose per episode, without resetting on arrival."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2048
        self.scene.env_spacing = 1.5
        # Moderate ambient and front fill keep the white arm readable, while
        # the stronger oblique key light preserves curved surfaces and edges.
        configure_arm_lighting(self.scene)
        # Match FrankaCableSceneCfg's table and robot-to-table placement.
        # The table top and robot base are at z=0; the ground is at z=-1.05.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
            spawn=sim_utils.CuboidCfg(
                size=(1.3, 0.9, 1.05),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)),
                physics_material=RigidBodyMaterialBaseCfg(static_friction=0.01, dynamic_friction=0.01),
            ),
        )
        self.scene.robot = MYCOBOT_PRO450_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["arm"].stiffness = 400.0
        self.scene.robot.actuators["arm"].damping = {"joint[1-2]": 60.0, "joint3": 40.0, "joint[4-6]": 20.0}
        self.scene.robot.actuators["arm"].joint_velocity_limit = 1.5
        self.scene.robot.actuators["gripper"].joint_effort_limit = 8.0
        self.sim.physics = ReachPhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.3, 1.3, 0.9), lookat=(0.2, 0.0, 0.25))
        self.actions.arm_action = SmoothJointVelocityActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            preserve_order=True,
            scale=0.8,
            use_default_offset=False,
            target_limits=JOINT_TARGET_LIMITS,
        )
        self.actions.gripper_action = None
        arm = SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)
        for name in ("joint_pos", "joint_vel"):
            getattr(self.observations.policy, name).params = {"asset_cfg": arm}
        # Explicit pose feedback follows the local Franka hold preset.
        self.observations.policy.pose_error = ObservationTermCfg(func=pose_error)
        self.observations.policy.filtered_target = ObservationTermCfg(func=filtered_target)
        self.observations.policy.target_velocity = ObservationTermCfg(func=target_velocity)
        self.events = ReachEventsCfg()
        self.commands.ee_pose = ReachablePoseCommandCfg(
            asset_name="robot",
            body_name=MYCOBOT_PRO450_EE_BODY_NAME,
            ranges=self.commands.ee_pose.ranges,
            resampling_time_range=(1.0e9, 1.0e9),
        )
        command = self.commands.ee_pose
        command.body_name = MYCOBOT_PRO450_EE_BODY_NAME
        command.make_quat_unique = True
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.goal_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        command.current_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        command.position_success_threshold = 0.02
        command.orientation_success_threshold = 0.15
        command.ranges.pos_x = (0.12, 0.35)
        command.ranges.pos_y = (-0.20, 0.20)
        command.ranges.pos_z = (0.20, 0.46)
        # Orientations come from FK of legal joints; Euler ranges are unused.
        command.ranges.pitch = (0.0, 0.0)
        ee = SceneEntityCfg("robot", body_names=[MYCOBOT_PRO450_EE_BODY_NAME])
        self.rewards.end_effector_position_tracking.params["asset_cfg"] = ee
        self.rewards.end_effector_position_tracking.weight = -2.0
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"] = ee
        # Mixed wrist orientations need enough reward to improve rotation even
        # when doing so briefly worsens the already-small position error.
        self.rewards.end_effector_orientation_tracking.weight = -1.5
        self.rewards.position_fine = RewardTermCfg(
            func=mdp.position_command_error_tanh,
            weight=2.0,
            params={"asset_cfg": ee, "command_name": "ee_pose", "std": 0.1},
        )
        self.rewards.success = RewardTermCfg(
            func=settled_success,
            weight=0.5,
        )
        self.rewards.joint_vel.params["asset_cfg"] = arm
        self.rewards.action_magnitude.weight = -0.02
        # Keep regularization fixed: an abrupt curriculum jump destabilizes holding.
        self.rewards.action_rate.weight = -0.02
        self.rewards.joint_vel.weight = -0.1
        self.rewards.joint_acceleration = RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": arm},
        )
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None
        self.terminations.success = None


@configclass
class Mycobot450ReachEnvCfg_PLAY(Mycobot450ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.sim.default_visualizer_cfg.eye = (0.9, 0.9, 0.75)
        self.sim.default_visualizer_cfg.lookat = (0.18, 0.0, 0.3)
        # Avoid frequent reset teleports being mistaken for policy motion in playback.
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
        # Cross-family wrist rotations need time to settle under the same speed cap.
        self.commands.ee_pose.resampling_time_range = (10.0, 10.0)
