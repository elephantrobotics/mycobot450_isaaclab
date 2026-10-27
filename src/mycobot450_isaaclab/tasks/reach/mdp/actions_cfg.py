"""Configuration for the Reach velocity-action filter."""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from .actions import SmoothJointVelocityAction


@configclass
class SmoothJointVelocityActionCfg(JointPositionActionCfg):
    class_type: type = SmoothJointVelocityAction
    use_default_offset: bool = False
    scale: float = 0.8
    time_constant: float = 0.12
    target_limits: dict | None = None
