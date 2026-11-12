"""Configuration for the four reachable target-pose families."""

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass

from mycobot450_isaaclab.utils.reach_goals import (
    DEFAULT_DIVERSE_JOINT_RANGES,
    DEFAULT_FAMILY_WEIGHTS,
    DEFAULT_JOINT_RANGES,
    DEFAULT_RESERVOIR_SIZE,
    DEFAULT_SAMPLING_SEED,
)

from .commands import ReachablePoseCommand


@configclass
class ReachablePoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = ReachablePoseCommand
    joint_ranges: tuple = DEFAULT_JOINT_RANGES
    diverse_joint_ranges: tuple = DEFAULT_DIVERSE_JOINT_RANGES
    family_weights: tuple = DEFAULT_FAMILY_WEIGHTS
    reservoir_size: int = DEFAULT_RESERVOIR_SIZE
    sampling_seed: int = DEFAULT_SAMPLING_SEED
    legacy_online_sampling: bool = False
