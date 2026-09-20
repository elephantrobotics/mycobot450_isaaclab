# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for F100 angular-gripper lift: hold the cube, carry it, penalize throws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def _as_torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _ee_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = _as_torch(obj.data.root_pos_w)
    ee_w = _as_torch(ee_frame.data.target_pos_w)[..., 0, :]
    return torch.linalg.norm(cube_pos_w - ee_w, dim=1)


def _ee_object_distance_xy(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Horizontal TCP-to-cube distance. Vertical lift does not increase this."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = _as_torch(obj.data.root_pos_w)
    ee_w = _as_torch(ee_frame.data.target_pos_w)[..., 0, :]
    return torch.linalg.norm(cube_pos_w[:, :2] - ee_w[:, :2], dim=1)


def _tcp_pos_w(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return _as_torch(ee_frame.data.target_pos_w)[..., 0, :]


def _cube_pos_in_tcp_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Cube position in the F100 TCP frame. TCP +X is the finger approach axis."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = _as_torch(obj.data.root_pos_w)
    ee_w = _as_torch(ee_frame.data.target_pos_w)[..., 0, :]
    quat_w = getattr(ee_frame.data, "target_quat_w", None)
    if quat_w is None:
        return cube_pos_w - ee_w
    ee_quat_w = _as_torch(quat_w)[..., 0, :]
    pos_tcp, _ = subtract_frame_transforms(ee_w, ee_quat_w, cube_pos_w)
    return pos_tcp


def _in_jaws(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw_depth: float,
    jaw_width: float,
    jaw_height: float,
) -> torch.Tensor:
    """True when the cube center sits between the F100 pads, not beside the palm or arm."""
    tcp = _cube_pos_in_tcp_frame(env, object_cfg, ee_frame_cfg)
    return (tcp[:, 0].abs() <= jaw_depth) & (tcp[:, 1].abs() <= jaw_width) & (tcp[:, 2].abs() <= jaw_height)


def _gripper_closed(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg | None,
    gripper_open_pos: float,
    gripper_close_pos: float,
    closed_ratio: float,
) -> torch.Tensor:
    """True when commanded/measured gripper joints are closer to close than ``closed_ratio``."""
    if gripper_cfg is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = _as_torch(robot.data.joint_pos)[:, gripper_cfg.joint_ids]
    span = gripper_close_pos - gripper_open_pos
    ratio = (joint_pos - gripper_open_pos) / (span + 1e-6)
    return torch.all(ratio > closed_ratio, dim=1)


def _gripper_close_ratio(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg | None,
    gripper_open_pos: float,
    gripper_close_pos: float,
) -> torch.Tensor:
    if gripper_cfg is None:
        return torch.ones(env.num_envs, device=env.device)
    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = _as_torch(robot.data.joint_pos)[:, gripper_cfg.joint_ids]
    span = gripper_close_pos - gripper_open_pos
    ratio = (joint_pos - gripper_open_pos) / (span + 1e-6)
    return torch.clamp(ratio, 0.0, 1.0).mean(dim=1)


def _is_held(
    env: ManagerBasedRLEnv,
    max_ee_object_dist: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg | None,
    gripper_open_pos: float,
    gripper_close_pos: float,
    closed_ratio: float,
    require_gripper: bool = True,
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
) -> torch.Tensor:
    """True pinch: cube in the F100 jaw box, near TCP, and the gripper closing."""
    near_ee = _ee_object_distance(env, object_cfg, ee_frame_cfg) < max_ee_object_dist
    in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height)
    held = near_ee & in_jaws
    if not require_gripper:
        return held
    closed = _gripper_closed(env, gripper_cfg, gripper_open_pos, gripper_close_pos, closed_ratio)
    return held & closed


def _ee_z_down(env: ManagerBasedRLEnv, wrist_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """How much Link6 +Z points at world -Z. Sideways poses score 0."""
    if wrist_cfg is None:
        return torch.ones(env.num_envs, device=env.device)
    body_ids = getattr(wrist_cfg, "body_ids", None)
    robot: Articulation = env.scene[wrist_cfg.name]
    body_quat = getattr(robot.data, "body_quat_w", None)
    if body_ids is None or body_quat is None:
        return torch.ones(env.num_envs, device=env.device)
    quat = _as_torch(body_quat)[:, int(body_ids[0]), :]
    z_local = torch.zeros(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    z_local[:, 2] = 1.0
    z_axis = quat_apply(quat, z_local)
    return (-z_axis[:, 2]).clamp(min=0.0)


def ee_z_down(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Link6"]),
) -> torch.Tensor:
    """Reward J6/Link6 +Z pointing down at the table."""
    return _ee_z_down(env, asset_cfg)


def object_ee_distance_aligned(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Link6"]),
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
) -> torch.Tensor:
    """Reach the cube in 3D so the TCP must descend into the jaw, not hover above it."""
    near = 1.0 - torch.tanh(_ee_object_distance(env, object_cfg, ee_frame_cfg) / std)
    align = _ee_z_down(env, wrist_cfg)
    in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height).float()
    return near * (0.2 + 0.8 * align) * (0.5 + 0.5 * in_jaws)


class ObjectEeDistanceCarry(ManagerTermBase):
    """Approach the cube; after a pinch, stop pulling the TCP back down onto the table."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._committed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self._committed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Link6"]),
        gripper_cfg: SceneEntityCfg | None = None,
        gripper_open_pos: float = -1.012,
        gripper_close_pos: float = 0.0,
        closed_ratio: float = 0.30,
        jaw_depth: float = 0.04,
        jaw_width: float = 0.03,
        jaw_height: float = 0.03,
    ) -> torch.Tensor:
        in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height)
        closed = _gripper_closed(env, gripper_cfg, gripper_open_pos, gripper_close_pos, closed_ratio)
        self._committed |= in_jaws & closed
        approach = object_ee_distance_aligned(
            env, std, object_cfg, ee_frame_cfg, wrist_cfg, jaw_depth, jaw_width, jaw_height
        )
        align = _ee_z_down(env, wrist_cfg)
        after_grasp = 0.2 + 0.8 * align
        return torch.where(self._committed, after_grasp, approach)


class EeGoalAfterGrasp(ManagerTermBase):
    """Pull the TCP to the sampled pose after a pinch. Cube-gated goal never fires if the cube stays down."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._committed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        log = self._env.extras.setdefault("log", {})
        log["Metrics/grasp_commit_fraction"] = self._committed[env_ids].float().mean().item()
        self._committed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gripper_cfg: SceneEntityCfg | None = None,
        gripper_open_pos: float = -1.012,
        gripper_close_pos: float = 0.0,
        closed_ratio: float = 0.30,
        jaw_depth: float = 0.04,
        jaw_width: float = 0.03,
        jaw_height: float = 0.03,
    ) -> torch.Tensor:
        in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height)
        closed = _gripper_closed(env, gripper_cfg, gripper_open_pos, gripper_close_pos, closed_ratio)
        self._committed |= in_jaws & closed
        robot: RigidObject = env.scene[robot_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            _as_torch(robot.data.root_pos_w), _as_torch(robot.data.root_quat_w), command[:, :3]
        )
        ee_w = _tcp_pos_w(env, ee_frame_cfg)
        distance = torch.linalg.norm(des_pos_w - ee_w, dim=1)
        return self._committed.float() * (1.0 - torch.tanh(distance / std))


class EeRaisedAfterGrasp(ManagerTermBase):
    """Raise the TCP only after a real pinch. Hover-and-close above the cube does not count."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._committed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self._committed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        gripper_cfg: SceneEntityCfg | None = None,
        gripper_open_pos: float = -1.012,
        gripper_close_pos: float = 0.0,
        closed_ratio: float = 0.30,
        jaw_depth: float = 0.04,
        jaw_width: float = 0.03,
        jaw_height: float = 0.03,
    ) -> torch.Tensor:
        in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height)
        closed = _gripper_closed(env, gripper_cfg, gripper_open_pos, gripper_close_pos, closed_ratio)
        self._committed |= in_jaws & closed
        ee_z = _tcp_pos_w(env, ee_frame_cfg)[:, 2]
        return (self._committed & (ee_z > minimal_height)).float()


def gripper_closed_away(
    env: ManagerBasedRLEnv,
    max_ee_object_dist: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg | None = None,
    gripper_open_pos: float = -1.012,
    gripper_close_pos: float = 0.0,
    closed_ratio: float = 0.30,
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
) -> torch.Tensor:
    """Penalize closing in free space. That farms a parked closed pose."""
    closed = _gripper_closed(env, gripper_cfg, gripper_open_pos, gripper_close_pos, closed_ratio)
    near = _ee_object_distance(env, object_cfg, ee_frame_cfg) < max_ee_object_dist
    in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height)
    return (closed & ~near & ~in_jaws).float()


def joint_vel_l2_clipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_l2: float = 25.0,
) -> torch.Tensor:
    """L2 joint-velocity penalty, clipped so a spinning 450 cannot explode PPO."""
    robot: Articulation = env.scene[asset_cfg.name]
    vel = _as_torch(robot.data.joint_vel)[:, asset_cfg.joint_ids]
    return torch.clamp(torch.sum(torch.square(vel), dim=1), max=max_l2)


def action_rate_l2_clipped(env: ManagerBasedRLEnv, max_l2: float = 5.0) -> torch.Tensor:
    """L2 action-rate penalty, clipped against command chatter."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.clamp(torch.sum(torch.square(delta), dim=1), max=max_l2)


def object_pinch(
    env: ManagerBasedRLEnv,
    std: float = 0.08,
    table_z: float = 0.055,
    lift_bonus: float = 8.0,
    max_bonus_height: float = 0.14,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg | None = None,
    gripper_open_pos: float = -1.012,
    gripper_close_pos: float = 0.0,
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
    wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Link6"]),
) -> torch.Tensor:
    """Dense approach in 3D; lift bonus is on TCP height after the cube is in the jaws."""
    near = 1.0 - torch.tanh(_ee_object_distance(env, object_cfg, ee_frame_cfg) / std)
    closed = _gripper_close_ratio(env, gripper_cfg, gripper_open_pos, gripper_close_pos)
    in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height).float()
    ee_z = _tcp_pos_w(env, ee_frame_cfg)[:, 2]
    span = max(max_bonus_height - table_z, 1e-6)
    extra = torch.clamp((ee_z - table_z) / span, 0.0, 1.0) * in_jaws * closed
    align = _ee_z_down(env, wrist_cfg)
    close_in_jaws = closed * in_jaws
    return near * align * (1.0 + 1.5 * in_jaws + 3.0 * close_in_jaws) * (1.0 + lift_bonus * extra)


def object_is_lifted_and_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_ee_object_dist: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg | None = None,
    gripper_open_pos: float = -1.012,
    gripper_close_pos: float = 0.0,
    closed_ratio: float = 0.45,
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
) -> torch.Tensor:
    """Reward lift only while the cube is pinched between the F100 pads."""
    obj: RigidObject = env.scene[object_cfg.name]
    lifted = _as_torch(obj.data.root_pos_w)[:, 2] > minimal_height
    held = _is_held(
        env,
        max_ee_object_dist,
        object_cfg,
        ee_frame_cfg,
        gripper_cfg,
        gripper_open_pos,
        gripper_close_pos,
        closed_ratio,
        jaw_depth=jaw_depth,
        jaw_width=jaw_width,
        jaw_height=jaw_height,
    )
    return torch.where(lifted & held, 1.0, 0.0)


def object_throwing(
    env: ManagerBasedRLEnv,
    max_ee_object_dist: float,
    airborne_height: float = 0.10,
    vel_threshold: float = 0.6,
    max_object_height: float = 0.35,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg | None = None,
    gripper_open_pos: float = -1.012,
    gripper_close_pos: float = 0.0,
    closed_ratio: float = 0.45,
    minimal_height: float | None = None,
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
) -> torch.Tensor:
    """Penalize a cube that leaves the pinch above the table, scaled by speed.

    ``airborne_height`` must sit above the resting cube on the table (~0.055 m) so a
    still cube is not treated as a throw. ``minimal_height`` is accepted as an alias.
    """
    del minimal_height
    obj: RigidObject = env.scene[object_cfg.name]
    pos_w = _as_torch(obj.data.root_pos_w)
    lin_vel_w = _as_torch(obj.data.root_lin_vel_w)
    speed = torch.linalg.norm(lin_vel_w, dim=1)
    airborne = pos_w[:, 2] > airborne_height
    too_high = pos_w[:, 2] > max_object_height
    held = _is_held(
        env,
        max_ee_object_dist,
        object_cfg,
        ee_frame_cfg,
        gripper_cfg,
        gripper_open_pos,
        gripper_close_pos,
        closed_ratio,
        jaw_depth=jaw_depth,
        jaw_width=jaw_width,
        jaw_height=jaw_height,
    )
    lost = airborne & ~held
    speed_scale = 1.0 + torch.clamp(speed / vel_threshold, max=4.0)
    return lost.float() * speed_scale + too_high.float() * 2.0


def object_tips_on_cube(
    env: ManagerBasedRLEnv,
    max_ee_object_dist: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
) -> torch.Tensor:
    """Penalize parking with the fingertips on the cube instead of the cube in the jaw gap."""
    near = _ee_object_distance(env, object_cfg, ee_frame_cfg) < max_ee_object_dist
    in_jaws = _in_jaws(env, object_cfg, ee_frame_cfg, jaw_depth, jaw_width, jaw_height)
    return (near & ~in_jaws).float()


def object_pressed_against_arm(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_arm_object_dist: float = 0.055,
    max_ee_object_dist: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg | None = None,
    gripper_open_pos: float = -1.012,
    gripper_close_pos: float = 0.0,
    closed_ratio: float = 0.45,
    jaw_depth: float = 0.04,
    jaw_width: float = 0.03,
    jaw_height: float = 0.03,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Link.*", "base_link"]),
) -> torch.Tensor:
    """Penalize lifting the cube by squeezing it against the arm links instead of the F100 pads."""
    obj: RigidObject = env.scene[object_cfg.name]
    cube = _as_torch(obj.data.root_pos_w)
    lifted = cube[:, 2] > minimal_height
    held = _is_held(
        env,
        max_ee_object_dist,
        object_cfg,
        ee_frame_cfg,
        gripper_cfg,
        gripper_open_pos,
        gripper_close_pos,
        closed_ratio,
        jaw_depth=jaw_depth,
        jaw_width=jaw_width,
        jaw_height=jaw_height,
    )
    body_ids = getattr(arm_cfg, "body_ids", None)
    robot: Articulation = env.scene[arm_cfg.name]
    body_pos_attr = getattr(robot.data, "body_pos_w", None)
    if body_ids is None or body_pos_attr is None:
        return (lifted & ~held).float()
    body_pos = _as_torch(body_pos_attr)[:, body_ids, :3]
    dist = torch.linalg.norm(cube.unsqueeze(1) - body_pos, dim=2).min(dim=1).values
    near_arm = dist < max_arm_object_dist
    return (lifted & near_arm & ~held).float()


class object_goal_distance_held(ManagerTermBase):
    """Goal tracking that only pays when the cube is lifted and still pinched.

    Success (optional ``success_threshold``) also requires the hold, so a thrown
    cube flying through the target does not count.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        max_ee_object_dist: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        gripper_cfg: SceneEntityCfg | None = None,
        gripper_open_pos: float = -1.012,
        gripper_close_pos: float = 0.0,
        closed_ratio: float = 0.45,
        success_threshold: float | None = None,
        kernel: str = "tanh",
        jaw_depth: float = 0.04,
        jaw_width: float = 0.03,
        jaw_height: float = 0.03,
    ) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            _as_torch(robot.data.root_pos_w), _as_torch(robot.data.root_quat_w), command[:, :3]
        )
        object_pos_w = _as_torch(obj.data.root_pos_w)
        delta = des_pos_w - object_pos_w
        distance = torch.linalg.norm(delta, dim=1)
        lifted = object_pos_w[:, 2] > minimal_height
        held = _is_held(
            env,
            max_ee_object_dist,
            object_cfg,
            ee_frame_cfg,
            gripper_cfg,
            gripper_open_pos,
            gripper_close_pos,
            closed_ratio,
            jaw_depth=jaw_depth,
            jaw_width=jaw_width,
            jaw_height=jaw_height,
        )
        carried = lifted & held
        if success_threshold is not None:
            self._succeeded |= carried & (distance < success_threshold)
            # object_pose/position_error tracks f100_base_link, not the cube (TCP offset ~0.12 m).
            log = env.extras.setdefault("log", {})
            log["Metrics/cube_goal_distance"] = distance.mean().item()
            log["Metrics/held_fraction"] = carried.float().mean().item()
            if bool(carried.any()):
                held_delta = delta[carried]
                log["Metrics/held_cube_goal_distance"] = distance[carried].mean().item()
                log["Metrics/held_cube_goal_xy"] = torch.linalg.norm(held_delta[:, :2], dim=1).mean().item()
                log["Metrics/held_cube_goal_z"] = held_delta[:, 2].abs().mean().item()
        if kernel == "neg_l1":
            shaped = -distance
        elif kernel == "tanh":
            shaped = 1.0 - torch.tanh(distance / std)
        else:
            raise ValueError(f"Unknown goal kernel '{kernel}'")
        return carried.float() * shaped
