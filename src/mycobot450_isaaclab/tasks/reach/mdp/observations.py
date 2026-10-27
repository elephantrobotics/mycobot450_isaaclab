"""Policy observations for pose feedback and controller memory."""

import torch

from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms


def filtered_target(env):
    """Expose integrated position targets so actuator/filter memory is observable."""
    action = env.action_manager.get_term("arm_action")
    current = action._asset.data.joint_pos.torch[:, action._joint_ids]
    home = action._asset.data.default_joint_pos.torch[:, action._joint_ids]
    return torch.where(action.pending_reset[:, None], current, action.target) - home


def target_velocity(env):
    action = env.action_manager.get_term("arm_action")
    return torch.where(action.pending_reset[:, None], 0.0, action.velocity)


def pose_error(env, command_name="ee_pose"):
    command = env.command_manager.get_term(command_name)
    data = command.robot.data
    position, orientation = subtract_frame_transforms(
        data.root_pos_w.torch,
        data.root_quat_w.torch,
        data.body_pos_w.torch[:, command.body_idx],
        data.body_quat_w.torch[:, command.body_idx],
    )
    delta_position, delta_rotation = compute_pose_error(
        position, orientation, command.command[:, :3], command.command[:, 3:], rot_error_type="axis_angle"
    )
    return torch.cat((delta_position, delta_rotation), dim=-1)
