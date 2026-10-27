"""Forward kinematics and reachable end-effector pose sampling."""

import math
import xml.etree.ElementTree as ET

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_euler_xyz, quat_unique

from mycobot450_isaaclab.assets import MYCOBOT450_ISAACLAB_ASSETS_DIR


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


GOAL_FAMILIES = ("legacy", "down", "forward_left_down", "forward_right_down")


class ReachPoseSampler:
    """Seeded, in-memory FK reservoirs; no goal database or IK solution enters observations.

    +X of the gripper is its approach axis. Forward/left are +X/+Y of the robot
    base, independent of the viewer. The downward cone is 8 degrees; diagonal
    cones are 12 degrees around directions tilted 45 degrees from vertical.
    """

    def __init__(self, cfg, device="cpu"):
        if cfg.reservoir_size < 1:
            raise ValueError("Goal reservoir_size must be positive")
        self.fk = ArmKinematics(device)
        self.generator = torch.Generator(device=device).manual_seed(cfg.sampling_seed)
        self.device = device
        banks = []
        ranges = cfg.ranges
        low = torch.tensor([ranges.pos_x[0], ranges.pos_y[0], ranges.pos_z[0]], device=device)
        high = torch.tensor([ranges.pos_x[1], ranges.pos_y[1], ranges.pos_z[1]], device=device)
        directions = torch.tensor([[0, 0, -1], [0.5, 0.5, -(2**-0.5)], [0.5, -0.5, -(2**-0.5)]], device=device)
        thresholds = torch.tensor([math.cos(math.radians(a)) for a in (8, 12, 12)], device=device)
        axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        # Build all three new families together. Reservoirs are regenerated with
        # a different seed for evaluation, so validation goals are unseen.
        for bounds, families in ((cfg.joint_ranges, [0]), (cfg.diverse_joint_ranges, [1, 2, 3])):
            bounds = torch.tensor(bounds, device=device)
            collected = {family: [] for family in families}
            counts = {family: 0 for family in families}
            for _ in range(256):
                q = bounds[:, 0] + torch.rand(65536, 6, generator=self.generator, device=device) * (
                    bounds[:, 1] - bounds[:, 0]
                )
                poses = self.fk(q)
                valid = ((poses[:, :3] >= low) & (poses[:, :3] <= high)).all(-1)
                approach = quat_apply(poses[:, 3:], axis.expand(len(q), -1))
                for family in families:
                    mask = (
                        valid
                        if family == 0
                        else valid & ((approach * directions[family - 1]).sum(-1) >= thresholds[family - 1])
                    )
                    accepted = q[mask][: max(0, cfg.reservoir_size - counts[family])]
                    collected[family].append(accepted)
                    counts[family] += len(accepted)
                if all(count >= cfg.reservoir_size for count in counts.values()):
                    break
            else:
                raise RuntimeError(f"Cannot fill reachable pose reservoirs: {counts}")
            banks.extend(torch.cat(collected[family]) for family in families)
        self.joints = torch.stack(banks)
        self.poses = self.fk(self.joints.flatten(0, 1)).reshape(4, cfg.reservoir_size, 7)

    def sample(self, families):
        indices = torch.randint(self.joints.shape[1], families.shape, device=self.device)
        return self.poses[families, indices], self.joints[families, indices]


class ReachablePoseCommand(UniformPoseCommand):
    """Sample joint configurations then reject poses outside the requested XYZ box.

    Position/orientation are correlated, not uniformly distributed SE(3). The joint
    witness guarantees kinematic reachability, not collision-free transition paths.
    Witness joints are never supplied to the policy or used to control the arm.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sampler = ReachPoseSampler(cfg, self.device)
        self.goal_joints = torch.zeros(self.num_envs, 6, device=self.device)
        self.goal_family = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.family_weights = torch.tensor(cfg.family_weights, device=self.device)
        if (self.family_weights < 0).any() or self.family_weights.sum() <= 0:
            raise ValueError("Goal family weights must be nonnegative with positive sum")

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(ids) == 0:
            return
        families = torch.multinomial(self.family_weights, len(ids), replacement=True)
        self.pose_command_b[ids], self.goal_joints[ids] = self.sampler.sample(families)
        self.goal_family[ids] = families
