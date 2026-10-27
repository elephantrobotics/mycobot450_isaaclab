"""Smoothed joint-velocity control integrated into position targets."""

import math

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class SmoothJointVelocityAction(JointPositionAction):
    """Bounded velocity actions integrated into position-servo targets at physics frequency."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.time_constant <= 0:
            raise ValueError("Target velocity filter time_constant must be positive")
        self.target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self.velocity = torch.zeros_like(self.target)
        self.limits = torch.tensor([cfg.target_limits[name] for name in self._joint_names], device=self.device)
        self.pending_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions):
        # Events can change joint state after the action manager's reset callback.
        self.target[:] = torch.where(
            self.pending_reset[:, None], self._asset.data.joint_pos.torch[:, self._joint_ids], self.target
        )
        self.velocity[self.pending_reset] = 0.0
        self.pending_reset[:] = False
        super().process_actions(actions.clamp(-1.0, 1.0))

    def apply_actions(self):
        dt = self._env.physics_dt
        alpha = 1.0 - math.exp(-dt / self.cfg.time_constant)
        self.velocity.lerp_(self.processed_actions, alpha)
        proposed = self.target + dt * self.velocity
        self.target[:] = proposed.clamp(self.limits[:, 0], self.limits[:, 1])
        # Anti-windup at position limits; the policy observes both controller states.
        self.velocity[:] = torch.where(proposed == self.target, self.velocity, 0.0)
        self._asset.set_joint_position_target_index(target=self.target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.pending_reset[slice(None) if env_ids is None else env_ids] = True
