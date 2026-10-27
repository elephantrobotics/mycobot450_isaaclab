"""Regression checks for reachable goals and the deployable reach control interface."""

import math
from types import SimpleNamespace

import torch

from isaaclab.utils.math import quat_apply

from mycobot450_isaaclab.assets.robots.mycobot_pro450 import MYCOBOT_PRO450_CFG
from mycobot450_isaaclab.tasks.reach.config.mycobot450.agents.rsl_rl_ppo_cfg import ReachPPORunnerCfg
from mycobot450_isaaclab.tasks.reach.config.mycobot450.mycobot450_reach_env_cfg import (
    ARM_JOINTS,
    JOINT_TARGET_LIMITS,
    Mycobot450ReachEnvCfg,
    Mycobot450ReachEnvCfg_PLAY,
)
from mycobot450_isaaclab.tasks.reach.mdp import (
    ArmKinematics,
    ReachPoseSampler,
    SmoothJointVelocityAction,
    SmoothJointVelocityActionCfg,
)


def test_reach_interface_and_shared_asset_isolation():
    original_stiffness = MYCOBOT_PRO450_CFG.actuators["arm"].stiffness
    original_damping = MYCOBOT_PRO450_CFG.actuators["arm"].damping
    cfg = Mycobot450ReachEnvCfg()
    action = cfg.actions.arm_action
    assert action.joint_names == [f"joint{i}" for i in range(1, 7)]
    assert action.preserve_order and not action.use_default_offset
    assert action.scale == 0.8 and action.target_limits == JOINT_TARGET_LIMITS
    assert cfg.actions.gripper_action is None
    assert cfg.events.reset_scene.params["reset_joint_targets"]
    assert list(cfg.events.to_dict())[:2] == ["reset_scene", "reset_robot_joints"]
    assert cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names == ARM_JOINTS
    assert cfg.observations.policy.joint_vel.params["asset_cfg"].preserve_order
    assert cfg.commands.ee_pose.body_name == "f100_base_link"
    assert cfg.commands.ee_pose.make_quat_unique
    assert cfg.sim.dt * cfg.decimation == 1 / 30
    assert ReachPPORunnerCfg().clip_actions == 1.0
    assert MYCOBOT_PRO450_CFG.actuators["arm"].stiffness == original_stiffness
    assert MYCOBOT_PRO450_CFG.actuators["arm"].damping == original_damping
    assert cfg.scene.robot is not MYCOBOT_PRO450_CFG
    for joint, (low, high) in JOINT_TARGET_LIMITS.items():
        assert low < cfg.scene.robot.init_state.joint_pos[joint] < high
    for bounds, authored_degrees in zip(JOINT_TARGET_LIMITS.values(), (162, 125, 154, 162, 162, 165), strict=True):
        assert max(abs(v) for v in bounds) < math.radians(authored_degrees)


def test_play_matches_training_contract():
    train, play = Mycobot450ReachEnvCfg(), Mycobot450ReachEnvCfg_PLAY()
    assert train.observations.policy.enable_corruption
    assert not play.observations.policy.enable_corruption
    assert train.actions.to_dict() == play.actions.to_dict()
    assert train.commands.ee_pose.ranges.to_dict() == play.commands.ee_pose.ranges.to_dict()
    assert train.scene.robot.init_state.to_dict() == play.scene.robot.init_state.to_dict()
    assert train.terminations.success is None and play.terminations.success is None
    assert train.commands.ee_pose.resampling_time_range[0] > train.episode_length_s
    assert play.commands.ee_pose.resampling_time_range == (10.0, 10.0)


def _control():
    action = SmoothJointVelocityAction.__new__(SmoothJointVelocityAction)
    action.cfg = SmoothJointVelocityActionCfg()
    action._env = SimpleNamespace(physics_dt=1 / 120, num_envs=2)
    state = torch.tensor([[0.2] * 6, [-0.3] * 6])
    action._asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=state)),
        set_joint_position_target_index=lambda **kwargs: None,
    )
    action._joint_ids = list(range(6))
    action._scale, action._offset = 0.8, 0.0
    action._raw_actions = torch.zeros(2, 6)
    action.target = state.clone()
    action.velocity = torch.zeros_like(state)
    action.limits = torch.tensor([[-2.0, 2.0]] * 6)
    action.pending_reset = torch.zeros(2, dtype=torch.bool)
    return action


def test_filter_partial_reset_uses_post_event_joint_state():
    action = _control()
    action.velocity[:] = 0.6
    action.reset(torch.tensor([0]))
    action._asset.data.joint_pos.torch[0] = 0.4
    action.process_actions(torch.zeros(2, 6))
    torch.testing.assert_close(action.target[0], torch.full((6,), 0.4))
    torch.testing.assert_close(action.target[1], torch.full((6,), -0.3))
    torch.testing.assert_close(action.velocity[0], torch.zeros(6))
    torch.testing.assert_close(action.velocity[1], torch.full((6,), 0.6))
    assert not action.pending_reset.any()


def test_velocity_position_limits_and_anti_windup():
    action = _control()
    action.process_actions(torch.full((2, 6), 10.0))
    for _ in range(600):
        previous = action.target.clone()
        action.apply_actions()
        assert (action.target - previous).abs().max() <= 0.8 / 120 + 1e-7
        assert action.velocity.abs().max() <= 0.8
        assert action.target.abs().max() <= 2.0
    torch.testing.assert_close(action.target, torch.full((2, 6), 2.0))
    torch.testing.assert_close(action.velocity, torch.zeros(2, 6))
    # Reversing must immediately permit motion away from the clipped limit.
    action.process_actions(-torch.ones(2, 6))
    action.apply_actions()
    assert (action.target < 2.0).all()


def test_alternating_policy_targets_do_not_create_position_jumps():
    action = _control()
    history = []
    for step in range(800):
        if step % 4 == 0:
            action.process_actions(torch.full((2, 6), 1.0 if (step // 4) % 2 else -1.0))
        action.apply_actions()
        if step >= 700:
            history.append(action.target[0, 0].clone())
    trace = torch.stack(history)
    assert trace.max() - trace.min() < 0.003


def test_fk_workspace_and_joint_limits():
    cfg = Mycobot450ReachEnvCfg()
    bounds = torch.tensor(cfg.commands.ee_pose.joint_ranges)
    for i, joint in enumerate(ARM_JOINTS):
        assert JOINT_TARGET_LIMITS[joint][0] <= bounds[i, 0] < bounds[i, 1] <= JOINT_TARGET_LIMITS[joint][1]
    generator = torch.Generator().manual_seed(123)
    joints = bounds[:, 0] + torch.rand(20000, 6, generator=generator) * (bounds[:, 1] - bounds[:, 0])
    poses = ArmKinematics()(joints)
    ranges = cfg.commands.ee_pose.ranges
    low = torch.tensor([ranges.pos_x[0], ranges.pos_y[0], ranges.pos_z[0]])
    high = torch.tensor([ranges.pos_x[1], ranges.pos_y[1], ranges.pos_z[1]])
    valid = ((poses[:, :3] >= low) & (poses[:, :3] <= high)).all(-1)
    assert valid.float().mean() > 0.15
    assert ((poses[valid, :3].max(0).values - poses[valid, :3].min(0).values) > torch.tensor([0.20, 0.35, 0.23])).all()
    torch.testing.assert_close(poses[:, 3:].norm(dim=-1), torch.ones(len(poses)), atol=1e-5, rtol=0)


def test_diverse_goals_preserve_legacy_and_cover_downward_directions():
    cfg = Mycobot450ReachEnvCfg().commands.ee_pose
    cfg.reservoir_size = 64
    sampler = ReachPoseSampler(cfg)
    torch.testing.assert_close(sampler.poses, ReachPoseSampler(cfg).poses)
    cfg.sampling_seed += 1
    assert not torch.equal(sampler.poses, ReachPoseSampler(cfg).poses)
    q, poses = sampler.joints, sampler.poses
    limits = torch.tensor(list(JOINT_TARGET_LIMITS.values()))
    assert ((q >= limits[:, 0]) & (q <= limits[:, 1])).all()
    old = torch.tensor(cfg.joint_ranges)
    assert ((q[0] >= old[:, 0]) & (q[0] <= old[:, 1])).all()
    lo, hi = torch.tensor([0.12, -0.20, 0.20]), torch.tensor([0.35, 0.20, 0.46])
    assert ((poses[..., :3] >= lo) & (poses[..., :3] <= hi)).all()
    axes = quat_apply(poses[..., 3:].reshape(-1, 4), torch.tensor([1.0, 0.0, 0.0]).expand(256, -1)).reshape(4, 64, 3)
    for i, (direction, angle) in enumerate(
        zip(([0.0, 0.0, -1.0], [0.5, 0.5, -(2**-0.5)], [0.5, -0.5, -(2**-0.5)]), (8, 12, 12), strict=True), start=1
    ):
        assert ((axes[i] * torch.tensor(direction)).sum(-1) >= math.cos(math.radians(angle)) - 1e-6).all()
    assert (axes[2, :, 1] > 0).all() and (axes[3, :, 1] < 0).all()
