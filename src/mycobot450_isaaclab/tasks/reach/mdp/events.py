"""Optional reset-state coverage for reaching from previously attained poses."""

import math

import torch

from isaaclab.envs.mdp.events import reset_joints_by_offset
from isaaclab.managers import SceneEntityCfg


def reset_joints_from_reached_states(
    env,
    env_ids: torch.Tensor | slice,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    reached_state_probability: float = 0.0,
):
    """Mix home resets with measured, low-speed states from other live environments.

    Reset environments are excluded from the source pool: reset_scene has already
    moved them home. Source poses must currently meet the existing success and
    joint-speed thresholds, and lie inside the controller's position limits.
    The next goal is still sampled independently by CommandManager.reset().
    This assumes identical static scenes and checks the local robot base frames.
    Full joint geometry is copied, while the gripper's open targets are preserved.

    Zero probability follows the original reset exactly. An independent RNG for
    the optional state choice preserves the global target/action sampling stream.
    This only diversifies initial joint positions; it does not replay a trajectory
    or preserve the controller history of an uninterrupted target switch.
    """
    if not math.isfinite(reached_state_probability) or not 0.0 <= reached_state_probability <= 1.0:
        raise ValueError("reached_state_probability must be finite and inside [0, 1]")
    kwargs = dict(position_range=position_range, velocity_range=velocity_range, asset_cfg=asset_cfg)
    if reached_state_probability == 0.0:
        reset_joints_by_offset(env, env_ids, **kwargs)
        return

    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if len(ids) == 0:
        return
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term("ee_pose")
    limits = env.action_manager.get_term("arm_action").limits
    all_positions = asset.data.joint_pos.torch
    positions = all_positions[:, asset_cfg.joint_ids]
    velocities = asset.data.joint_vel.torch
    if positions.shape[1] != len(limits):
        raise ValueError("Reset joint count must match the arm controller")

    # Compute current errors without updating the command's sticky episode-success flag.
    position_error, orientation_error = command._compute_error()
    eligible = command._compute_success(position_error, orientation_error)
    eligible &= velocities.abs().max(-1).values < 0.1
    eligible &= ((positions >= limits[:, 0]) & (positions <= limits[:, 1])).all(-1)
    eligible[ids] = False
    source_ids = eligible.nonzero().flatten()
    source_positions = all_positions[source_ids].clone()
    local_root_position = asset.data.root_pos_w.torch - env.scene.env_origins
    root_orientation = asset.data.root_quat_w.torch

    # Keep the original home-reset draws, including for rows we later replace.
    reset_joints_by_offset(env, env_ids, **kwargs)
    for name in ("start_from_reached_state", "reached_reset_pool_size", "reached_reset_frame_rejection"):
        if name not in command.metrics:
            command.metrics[name] = torch.zeros(env.num_envs, device=env.device)
    command.metrics["start_from_reached_state"][ids] = 0.0
    command.metrics["reached_reset_pool_size"][ids] = len(source_positions)
    command.metrics["reached_reset_frame_rejection"][ids] = 0.0
    if not len(source_positions):
        return

    if not hasattr(env, "_reach_reset_generator"):
        env._reach_reset_generator = torch.Generator(device=env.device).manual_seed(int(env.cfg.seed or 0) + 1_000_003)
    generator = env._reach_reset_generator
    selected = ids[torch.rand(len(ids), generator=generator, device=env.device) < reached_state_probability]
    if not len(selected):
        return
    indices = torch.randint(len(source_positions), (len(selected),), generator=generator, device=env.device)
    donors = source_ids[indices]
    same_frame = (local_root_position[selected] - local_root_position[donors]).abs().max(-1).values < 1.0e-5
    same_frame &= (root_orientation[selected] * root_orientation[donors]).sum(-1).abs() > 1.0 - 1.0e-6
    command.metrics["reached_reset_frame_rejection"][selected] = (~same_frame).float()
    selected, indices = selected[same_frame], indices[same_frame]
    if not len(selected):
        return
    initial_position = source_positions[indices]
    joint_ids = asset_cfg.joint_ids
    # Copy the complete measured geometry, including the passive gripper joints.
    asset.write_joint_position_to_sim_index(position=initial_position, env_ids=selected, joint_ids=slice(None))
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(initial_position), env_ids=selected, joint_ids=slice(None)
    )
    # Avoid a home-directed actuator target before the action term consumes its pending reset.
    asset.set_joint_position_target_index(target=initial_position[:, joint_ids], env_ids=selected, joint_ids=joint_ids)
    command.metrics["start_from_reached_state"][selected] = 1.0
