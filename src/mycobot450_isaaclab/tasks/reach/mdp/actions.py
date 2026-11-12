"""Smoothed joint-velocity control integrated into position targets."""

import math

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class SmoothJointVelocityAction(JointPositionAction):
    """Physics-step integration with optional policy-period endpoint holds."""

    def __init__(self, cfg, env):
        update_decimation = cfg.target_update_decimation
        integration_substeps = cfg.integration_substeps
        valid_update = type(update_decimation) is int and update_decimation > 0
        if not valid_update:
            raise ValueError("target_update_decimation must be a positive integer")
        if integration_substeps is None:
            valid_update = valid_update and update_decimation in (1, env.cfg.decimation)
        else:
            valid_update = valid_update and env.cfg.decimation % update_decimation == 0
            if type(integration_substeps) is not int or not 1 <= integration_substeps <= update_decimation:
                raise ValueError("integration_substeps must be an integer within the target hold interval")
        if not valid_update:
            raise ValueError("target_update_decimation must be 1 or the policy decimation")
        super().__init__(cfg, env)
        if cfg.time_constant <= 0:
            raise ValueError("Target velocity filter time_constant must be positive")
        if cfg.max_tracking_error is not None and (
            not math.isfinite(cfg.max_tracking_error) or cfg.max_tracking_error <= 0
        ):
            raise ValueError("max_tracking_error must be finite and positive, or None")
        self.target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self.velocity = torch.zeros_like(self.target)
        self.limits = torch.tensor([cfg.target_limits[name] for name in self._joint_names], device=self.device)
        self.pending_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_update_phase = 0

    def process_actions(self, actions):
        # Events can change joint state after the action manager's reset callback.
        self.target[:] = torch.where(
            self.pending_reset[:, None], self._asset.data.joint_pos.torch[:, self._joint_ids], self.target
        )
        self.velocity[self.pending_reset] = 0.0
        self.pending_reset[:] = False
        super().process_actions(actions.clamp(-1.0, 1.0))
        # A new policy action starts a complete update/hold interval. Partial
        # resets change only the marked controller rows, not the other rows.
        self._target_update_phase = 0

    def apply_actions(self):
        update_decimation = getattr(self.cfg, "target_update_decimation", 1)
        phase = getattr(self, "_target_update_phase", 0)
        if phase == 0:
            if self.cfg.max_tracking_error is not None:
                self.integration_feedback = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
            # Advance the unchanged filter/integrator before the first physical
            # step. The endpoint is held for the remainder of this interval;
            # this changes the submitted trajectory and must be evaluated.
            substeps = getattr(self.cfg, "integration_substeps", None)
            for _ in range(update_decimation if substeps is None else substeps):
                self._integrate_substep()
        self._target_update_phase = (phase + 1) % update_decimation
        self._asset.set_joint_position_target_index(target=self.target, joint_ids=self._joint_ids)

    def _integrate_substep(self):
        dt = self._env.physics_dt
        alpha = 1.0 - math.exp(-dt / self.cfg.time_constant)
        self.velocity.lerp_(self.processed_actions, alpha)
        if self.cfg.max_tracking_error is not None:
            # Restrict only new integration, never snap an existing target to measured joints.
            error = self.target - self.integration_feedback
            limit = self.cfg.max_tracking_error
            feedback_dt = dt * getattr(self.cfg, "target_update_decimation", 1)
            forward = ((limit - error) / feedback_dt).clamp_min(0.0)
            backward = ((limit + error) / feedback_dt).clamp_min(0.0)
            self.velocity[:] = torch.maximum(torch.minimum(self.velocity, forward), -backward)
        proposed = self.target + dt * self.velocity
        self.target[:] = proposed.clamp(self.limits[:, 0], self.limits[:, 1])
        # Anti-windup at position limits; the policy observes both controller states.
        self.velocity[:] = torch.where(proposed == self.target, self.velocity, 0.0)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.pending_reset[slice(None) if env_ids is None else env_ids] = True
