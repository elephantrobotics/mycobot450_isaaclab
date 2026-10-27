"""Reach rewards requiring both pose accuracy and settled joint motion."""


def settled_success(env):
    command = env.command_manager.get_term("ee_pose")
    action = env.action_manager.get_term("arm_action")
    velocity = action._asset.data.joint_vel.torch[:, action._joint_ids]
    return command.compute_success() & (velocity.abs().max(-1).values < 0.1)
