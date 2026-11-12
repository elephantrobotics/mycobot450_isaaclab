"""Forward kinematics and reachable end-effector pose sampling."""

import xml.etree.ElementTree as ET

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_unique

from mycobot450_isaaclab.assets import MYCOBOT450_ISAACLAB_ASSETS_DIR
from mycobot450_isaaclab.utils.reach_goals import GOAL_FAMILIES as GOAL_FAMILIES
from mycobot450_isaaclab.utils.reach_goals import PoseReservoir


class ArmKinematics:
    """Six-joint URDF FK to the F100 mount, in the robot base frame (XYZW)."""

    def __init__(self, device="cpu"):
        path = MYCOBOT450_ISAACLAB_ASSETS_DIR / "robots/sim_450_f100/config/450_f100.urdf"
        root = ET.parse(path).getroot()
        self.origins = []
        for name in [f"joint{i}" for i in range(1, 7)] + ["AssemblerFixedJoint"]:
            joint = root.find(f"joint[@name='{name}']")
            if name != "AssemblerFixedJoint" and joint.find("axis").get("xyz") != "0 0 1":
                raise ValueError(f"Unsupported rotation axis for {name}")
            origin = joint.find("origin")
            pos = torch.tensor([float(v) for v in origin.get("xyz").split()], device=device)
            rpy = torch.tensor([float(v) for v in origin.get("rpy").split()], device=device)
            self.origins.append((pos, quat_from_euler_xyz(*rpy)))

    def __call__(self, joints):
        position = torch.zeros_like(joints[:, :3])
        orientation = torch.zeros_like(joints[:, :4])
        orientation[:, 3] = 1.0
        for i, (offset, rotation) in enumerate(self.origins):
            position, orientation = combine_frame_transforms(
                position, orientation, offset.expand_as(position), rotation.expand_as(orientation)
            )
            if i < 6:
                zero = torch.zeros_like(joints[:, i])
                rotation = quat_from_euler_xyz(zero, zero, joints[:, i])
                position, orientation = combine_frame_transforms(
                    position, orientation, torch.zeros_like(position), rotation
                )
        return torch.cat((position, quat_unique(orientation)), dim=-1)


class ReachPoseSampler(PoseReservoir):
    """Seeded, in-memory FK reservoirs; no goal database or IK solution enters observations.

    +X of the gripper is its approach axis. Forward/left are +X/+Y of the robot
    base, independent of the viewer. The downward cone is 8 degrees; diagonal
    cones are 12 degrees around directions tilted 45 degrees from vertical.
    """

    def __init__(self, cfg, device="cpu"):
        ranges = cfg.ranges
        super().__init__(
            ArmKinematics(device),
            (ranges.pos_x, ranges.pos_y, ranges.pos_z),
            joint_ranges=cfg.joint_ranges,
            diverse_joint_ranges=cfg.diverse_joint_ranges,
            reservoir_size=cfg.reservoir_size,
            sampling_seed=cfg.sampling_seed,
            device=device,
        )


class OnlineLegacyPoseSampler:
    """Fresh FK samples for basic reaching, without a finite training goal bank."""

    def __init__(self, cfg, device="cpu"):
        if tuple(cfg.family_weights) != (1.0, 0.0, 0.0, 0.0):
            raise ValueError("Online legacy sampling requires legacy-only family weights")
        self.fk = ArmKinematics(device)
        self.device = device
        self.bounds = torch.tensor(cfg.joint_ranges, device=device)
        ranges = cfg.ranges
        self.low = torch.tensor([ranges.pos_x[0], ranges.pos_y[0], ranges.pos_z[0]], device=device)
        self.high = torch.tensor([ranges.pos_x[1], ranges.pos_y[1], ranges.pos_z[1]], device=device)

    def sample(self, families):
        joints = torch.empty(len(families), 6, device=self.device)
        poses = torch.empty(len(families), 7, device=self.device)
        remaining = torch.arange(len(families), device=self.device)
        for _ in range(128):
            if len(remaining) == 0:
                return poses, joints
            q = self.bounds[:, 0] + torch.rand(len(remaining), 6, device=self.device) * (
                self.bounds[:, 1] - self.bounds[:, 0]
            )
            candidate = self.fk(q)
            valid = ((candidate[:, :3] >= self.low) & (candidate[:, :3] <= self.high)).all(-1)
            poses[remaining[valid]], joints[remaining[valid]] = candidate[valid], q[valid]
            remaining = remaining[~valid]
        if len(remaining):
            raise RuntimeError("Cannot sample reachable legacy poses inside the workspace")
        return poses, joints


class ReachablePoseCommand(UniformPoseCommand):
    """Sample joint configurations then reject poses outside the requested XYZ box.

    Position/orientation are correlated, not uniformly distributed SE(3). The joint
    witness guarantees kinematic reachability, not collision-free transition paths.
    Witness joints are never supplied to the policy or used to control the arm.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sampler_type = OnlineLegacyPoseSampler if cfg.legacy_online_sampling else ReachPoseSampler
        self.sampler = sampler_type(cfg, self.device)
        self.goal_joints = torch.zeros(self.num_envs, 6, device=self.device)
        self.goal_family = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.family_weights = torch.tensor(cfg.family_weights, device=self.device)
        if (self.family_weights < 0).any() or self.family_weights.sum() <= 0:
            raise ValueError("Goal family weights must be nonnegative with positive sum")

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(ids) == 0:
            return
        families = (
            torch.zeros(len(ids), dtype=torch.long, device=self.device)
            if self.cfg.legacy_online_sampling
            else torch.multinomial(self.family_weights, len(ids), replacement=True)
        )
        self.pose_command_b[ids], self.goal_joints[ids] = self.sampler.sample(families)
        self.goal_family[ids] = families
