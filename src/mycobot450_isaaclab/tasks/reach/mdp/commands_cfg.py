"""Configuration for the four reachable target-pose families."""

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass

from .commands import ReachablePoseCommand


@configclass
class ReachablePoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = ReachablePoseCommand
    joint_ranges: tuple = ((-1.0, 1.0), (-1.2, 0.8), (-2.6, -0.5), (-0.5, 0.5), (0.9, 2.2), (-0.5, 0.5))
    diverse_joint_ranges: tuple = ((-1.2, 1.2), (-1.2, 0.8), (-2.6, -0.5), (-1.6, 1.6), (-2.2, 2.2), (-1.5, 1.5))
    family_weights: tuple = (0.25, 0.25, 0.25, 0.25)
    reservoir_size: int = 4096
    sampling_seed: int = 42
