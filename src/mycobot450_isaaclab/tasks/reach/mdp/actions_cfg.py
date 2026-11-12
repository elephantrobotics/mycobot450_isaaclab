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
    max_tracking_error: float | None = None
    """Optional integration guard in radians; None preserves the trained control contract."""
    target_update_decimation: int = 1
    """1 submits every physics step; policy decimation submits a four-substep endpoint and holds it."""
    integration_substeps: int | None = None
    """Optional nominal filter steps per held endpoint; None advances once per held physics step."""
