"""Reproducible Reach goals; kinematic witnesses never command a robot.

Plans can use a local joint-space walk or the default four-family training
sampler. Reachability does not establish collision-free transitions.
"""

import copy
import json
import math
import secrets
from pathlib import Path

import torch

if __package__:
    from .reach_policy import JOINT_NAMES, ArmKinematics, pose_error, quat_mul
else:
    from reach_policy import JOINT_NAMES, ArmKinematics, pose_error, quat_mul

GOAL_FAMILIES = ("legacy", "down", "forward_left_down", "forward_right_down")
DEFAULT_JOINT_RANGES = ((-1.0, 1.0), (-1.2, 0.8), (-2.6, -0.5), (-0.5, 0.5), (0.9, 2.2), (-0.5, 0.5))
DEFAULT_DIVERSE_JOINT_RANGES = ((-1.2, 1.2), (-1.2, 0.8), (-2.6, -0.5), (-1.6, 1.6), (-2.2, 2.2), (-1.5, 1.5))
DEFAULT_FAMILY_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
DEFAULT_RESERVOIR_SIZE = 4096
DEFAULT_SAMPLING_SEED = 42


class PoseReservoir:
    """Shared training/deployment FK reservoirs, with the caller's batch FK.

    The gripper's +X approach axis defines the three direction cones. Building
    banks uses a private generator; sample() uses training's global RNG by
    default, while exported plans can pass their own generator.
    """

    def __init__(
        self,
        fk,
        workspace,
        *,
        joint_ranges=DEFAULT_JOINT_RANGES,
        diverse_joint_ranges=DEFAULT_DIVERSE_JOINT_RANGES,
        reservoir_size=DEFAULT_RESERVOIR_SIZE,
        sampling_seed=DEFAULT_SAMPLING_SEED,
        device="cpu",
    ):
        if type(reservoir_size) is not int or reservoir_size < 1:
            raise ValueError("Goal reservoir_size must be a positive integer")
        _seed(sampling_seed)
        self.fk, self.device = fk, device
        self.generator = torch.Generator(device=device).manual_seed(sampling_seed)
        workspace = torch.as_tensor(workspace, dtype=torch.float32, device=device)
        low, high = workspace[:, 0], workspace[:, 1]
        directions = torch.tensor([[0, 0, -1], [0.5, 0.5, -(2**-0.5)], [0.5, -0.5, -(2**-0.5)]], device=device)
        thresholds = torch.tensor([math.cos(math.radians(a)) for a in (8, 12, 12)], device=device)
        axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        banks = []
        for bounds, families in ((joint_ranges, [0]), (diverse_joint_ranges, [1, 2, 3])):
            bounds = torch.tensor(bounds, device=device)
            collected = {family: [] for family in families}
            counts = {family: 0 for family in families}
            for _ in range(256):
                q = bounds[:, 0] + torch.rand(65536, 6, generator=self.generator, device=device) * (
                    bounds[:, 1] - bounds[:, 0]
                )
                poses = self.fk(q)
                valid = ((poses[:, :3] >= low) & (poses[:, :3] <= high)).all(-1)
                rotation = poses[:, 3:]
                uv = torch.linalg.cross(rotation[:, :3], axis.expand(len(q), -1))
                approach = axis + 2 * (rotation[:, 3:] * uv + torch.linalg.cross(rotation[:, :3], uv))
                for family in families:
                    mask = (
                        valid
                        if family == 0
                        else valid & ((approach * directions[family - 1]).sum(-1) >= thresholds[family - 1])
                    )
                    accepted = q[mask][: max(0, reservoir_size - counts[family])]
                    collected[family].append(accepted)
                    counts[family] += len(accepted)
                if all(count >= reservoir_size for count in counts.values()):
                    break
            else:
                raise RuntimeError(f"Cannot fill reachable pose reservoirs: {counts}")
            banks.extend(torch.cat(collected[family]) for family in families)
        self.joints = torch.stack(banks)
        self.poses = self.fk(self.joints.flatten(0, 1)).reshape(4, reservoir_size, 7)

    def sample(self, families, *, generator=None):
        indices = torch.randint(self.joints.shape[1], families.shape, generator=generator, device=self.device)
        return self.poses[families, indices], self.joints[families, indices]


class BatchKinematics:
    """CPU batch FK from the same exported chain as the scalar controller."""

    def __init__(self, chain):
        self.origins = ArmKinematics(chain).origins

    def __call__(self, joints):
        position = torch.zeros_like(joints[:, :3])
        rotation = torch.zeros_like(joints[:, :4])
        rotation[:, 3] = 1.0
        for index, (offset, orientation) in enumerate(self.origins):
            uv = torch.linalg.cross(rotation[:, :3], offset.expand_as(position))
            position += offset + 2 * (rotation[:, 3:] * uv + torch.linalg.cross(rotation[:, :3], uv))
            rotation = quat_mul(rotation, orientation.expand_as(rotation))
            if index < 6:
                half = joints[:, index] / 2
                rotation = quat_mul(rotation, torch.stack((half * 0, half * 0, half.sin(), half.cos()), dim=-1))
        rotation /= rotation.norm(dim=-1, keepdim=True)
        return torch.cat((position, torch.where(rotation[:, 3:] < 0, -rotation, rotation)), dim=-1)


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a finite{' positive' if positive else ''} number")
    return float(value)


def _vector(values, length, name):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    result = torch.tensor([_number(value, name) for value in values], dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} is not representable as finite float32")
    return result


def _bounds(values, size, training, home, name):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"{name} must contain {size} lower/upper pairs")
    result = torch.stack([_vector(row, 2, name) for row in values])
    if (result[:, 0] >= result[:, 1]).any() or (home < result[:, 0]).any() or (home > result[:, 1]).any():
        raise ValueError(f"{name} must be ordered and contain home")
    if training is not None and ((result[:, 0] < training[:, 0]).any() or (result[:, 1] > training[:, 1]).any()):
        raise ValueError(f"{name} must remain inside the training bounds")
    return result


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("policy_sha256 must be a lowercase SHA256 hex digest")
    return value


class GoalPlan:
    """Validate base-frame goals and a separate measured-motion envelope.

    Version 1 retains its single workspace for both purposes. Version 2 declares
    both explicitly; only the goal workspace is bounded by the training domain.
    """

    def __init__(self, data, controller, policy_sha256):
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] not in (1, 2):
            raise ValueError("Expected a version-1 or version-2 goal plan")
        if data["version"] == 1:
            if "goal_workspace_m" in data or "motion_workspace_m" in data:
                raise ValueError("Version-1 plans use workspace_m only; use version 2 for separate goal/motion ranges")
            goal_workspace_key = "workspace_m"
        else:
            if "workspace_m" in data:
                raise ValueError("Version-2 plans require goal_workspace_m and motion_workspace_m without workspace_m")
            goal_workspace_key = "goal_workspace_m"
        if _hash(data.get("policy_sha256")) != _hash(policy_sha256):
            raise ValueError("Goal plan policy_sha256 does not match the loaded policy")
        if data.get("joint_order") != JOINT_NAMES:
            raise ValueError("Goal plan joint_order must be joint1..joint6")
        self.interval_s = _number(data.get("interval_s"), "interval_s", positive=True)
        home_pose = controller.fk(controller.home)
        self.joint_bounds = _bounds(
            data.get("joint_bounds_rad"), 6, controller.limits, controller.home, "joint_bounds_rad"
        )
        self.goal_workspace = _bounds(
            data.get(goal_workspace_key), 3, controller.workspace, home_pose[:3], goal_workspace_key
        )
        self.motion_workspace = (
            self.goal_workspace.clone()
            if data["version"] == 1
            else _bounds(data.get("motion_workspace_m"), 3, None, home_pose[:3], "motion_workspace_m")
        )
        self.workspace = self.goal_workspace  # Compatibility alias: never use it for measured-motion guards.
        sampling = data.get("sampling")
        if (
            not isinstance(sampling, dict)
            or not isinstance(sampling.get("method"), str)
            or not sampling["method"].strip()
        ):
            raise ValueError("Goal plan sampling must describe its method and seed")
        _seed(sampling.get("seed"))
        entries = data.get("goals")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Goal plan must contain at least one goal")
        goals, witnesses = [], []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Goal {index} must contain a pose and kinematic witness")
            pose = _vector(entry.get("pose_xyz_xyzw"), 7, f"goal {index} pose_xyz_xyzw")
            witness = _vector(entry.get("witness_joint_rad"), 6, f"goal {index} witness_joint_rad")
            if not math.isclose(pose[3:].norm().item(), 1.0, rel_tol=0, abs_tol=1e-5):
                raise ValueError(f"Goal {index} quaternion must have unit length")
            if ((pose[:3] < self.goal_workspace[:, 0]) | (pose[:3] > self.goal_workspace[:, 1])).any():
                raise ValueError(f"Goal {index} pose lies outside the plan goal workspace")
            if ((pose[:3] < self.motion_workspace[:, 0]) | (pose[:3] > self.motion_workspace[:, 1])).any():
                raise ValueError(f"Goal {index} pose lies outside motion_workspace_m")
            if ((witness < self.joint_bounds[:, 0]) | (witness > self.joint_bounds[:, 1])).any():
                raise ValueError(f"Goal {index} witness lies outside the plan joint bounds")
            expected = controller.fk(witness)
            orientation_difference = min(
                (pose[3:] - expected[3:]).abs().max().item(), (pose[3:] + expected[3:]).abs().max().item()
            )
            if (pose[:3] - expected[:3]).abs().max() > 1e-5 or orientation_difference > 1e-5:
                raise ValueError(f"Goal {index} pose does not match its witness FK within 1e-5")
            goals.append(controller.goal(pose))
            witnesses.append(witness)
        self.goals = torch.stack(goals)
        self.witnesses = torch.stack(witnesses)
        self.data = copy.deepcopy(data)

    @classmethod
    def from_file(cls, path, controller, policy_sha256):
        return cls.from_data(json.loads(Path(path).read_text()), controller, policy_sha256)

    @classmethod
    def from_data(cls, data, controller, policy_sha256):
        return cls(data, controller, policy_sha256)

    def validate_goal(self, index):
        """Return a goal copy; witness joints are intentionally a separate field."""
        if type(index) is not int or not 0 <= index < len(self.goals):
            raise IndexError("Goal index is outside the plan")
        return self.goals[index].clone()


class DefaultGoalSampler:
    """Draw indefinitely from the training reservoirs without resetting the controller.

    Build banks before connecting hardware; each live switch only draws one
    family/index and checks FK. Keep constant memory, with the seed for replay.
    """

    def __init__(self, plan, controller, kinematic_chain, *, seed=None, reservoir_size=DEFAULT_RESERVOIR_SIZE):
        self.seed = _seed(secrets.randbits(63) if seed is None else seed)
        if not torch.equal(plan.goal_workspace, controller.workspace):
            raise ValueError("Random goals require the complete training goal workspace")
        if (plan.motion_workspace[:, 0] > controller.workspace[:, 0]).any() or (
            plan.motion_workspace[:, 1] < controller.workspace[:, 1]
        ).any():
            raise ValueError("Motion bounds must contain the complete random goal workspace")
        for ranges in (DEFAULT_JOINT_RANGES, DEFAULT_DIVERSE_JOINT_RANGES):
            bounds = torch.tensor(ranges)
            if ((bounds[:, 0] < plan.joint_bounds[:, 0]) | (bounds[:, 1] > plan.joint_bounds[:, 1])).any():
                raise ValueError("Default sampling joint ranges exceed the plan joint bounds")
        self.controller = controller
        self.reservoir = PoseReservoir(
            BatchKinematics(kinematic_chain), controller.workspace, reservoir_size=reservoir_size
        )
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.weights = torch.tensor(DEFAULT_FAMILY_WEIGHTS)
        home_pose = controller.fk(controller.home)
        if not torch.allclose(self.reservoir.fk(controller.home[None])[0], home_pose, atol=1e-5, rtol=0):
            raise ValueError("Sampling kinematic chain does not match the deployment controller")
        self.count = 0
        self.family_counts = dict.fromkeys(GOAL_FAMILIES, 0)
        self.initial_goal = self.sample()  # Homing already verifies arrival; no extra home dwell is needed.

    def sample(self):
        family = torch.multinomial(self.weights, 1, replacement=True, generator=self.generator)
        poses, witnesses = self.reservoir.sample(family, generator=self.generator)
        pose = self.controller.fk(witnesses[0])
        if not torch.allclose(poses[0], pose, atol=1e-5, rtol=0):
            raise ValueError("Sampling kinematic chain does not match the deployment controller")
        pose = self.controller.goal(pose)
        name = GOAL_FAMILIES[family.item()]
        self.last_goal = {
            "family": name,
            "pose_xyz_xyzw": pose.tolist(),
            "witness_joint_rad": witnesses[0].tolist(),
        }
        self.count += 1
        self.family_counts[name] += 1
        return pose

    def report(self):
        return {
            "method": "runtime_default_four_family_independent",
            "seed": self.seed,
            "reservoir_seed": DEFAULT_SAMPLING_SEED,
            "reservoir_size_per_family": self.reservoir.joints.shape[1],
            "family_weights": list(DEFAULT_FAMILY_WEIGHTS),
            "sample_count": self.count,
            "initial_home_hold_s": 0.0,
            "family_counts": dict(self.family_counts),
            "last_goal": copy.deepcopy(self.last_goal),
            "note": "Plan supplies interval/bounds only. Fixed goals are ignored; no transition filtering.",
        }


def sample_plan(
    controller,
    policy_sha256,
    *,
    seed=450,
    count=6,
    interval_s=10,
    joint_radius=0.8,
    joint_step=0.15,
    max_position_step=0.04,
    max_angle_step=0.20,
    min_position_step=0.015,
):
    """Sample N outward goals, then reverse them to home: 2*N+1 total poses.

    Uses at most 10,000 proposals in total. All sampling and FK validation happen
    before control; the runtime supplies only the selected Cartesian pose to policy.
    """
    _hash(policy_sha256)
    _seed(seed)
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    interval_s = _number(interval_s, "interval_s", positive=True)
    joint_radius = _number(joint_radius, "joint_radius", positive=True)
    joint_step = _number(joint_step, "joint_step", positive=True)
    max_position_step = _number(max_position_step, "max_position_step", positive=True)
    max_angle_step = _number(max_angle_step, "max_angle_step", positive=True)
    min_position_step = _number(min_position_step, "min_position_step", positive=True)
    if min_position_step > max_position_step:
        raise ValueError("min_position_step must not exceed max_position_step")
    bounds = torch.stack(
        (
            torch.maximum(controller.home - joint_radius, controller.limits[:, 0]),
            torch.minimum(controller.home + joint_radius, controller.limits[:, 1]),
        ),
        dim=1,
    )
    home = controller.home.clone()
    pose = controller.fk(home)
    # Validate the envelope and home before spending time on candidate sampling.
    _bounds(bounds.tolist(), 6, controller.limits, home, "joint_bounds_rad")
    _bounds(controller.workspace.tolist(), 3, controller.workspace, pose[:3], "workspace_m")
    outward = [{"pose_xyz_xyzw": pose.tolist(), "witness_joint_rad": home.tolist()}]
    previous = home
    generator = torch.Generator(device="cpu").manual_seed(seed)
    attempts = 0
    while len(outward) <= count and attempts < 10000:
        attempts += 1
        candidate = previous + (2 * torch.rand(6, generator=generator) - 1) * joint_step
        if ((candidate < bounds[:, 0]) | (candidate > bounds[:, 1])).any():
            continue
        candidate_pose = controller.fk(candidate)
        outside_workspace = (candidate_pose[:3] < controller.workspace[:, 0]) | (
            candidate_pose[:3] > controller.workspace[:, 1]
        )
        if outside_workspace.any():
            continue
        difference = pose_error(pose, candidate_pose)
        distance, angle = difference[:3].norm().item(), difference[3:].norm().item()
        if not min_position_step <= distance <= max_position_step or angle > max_angle_step:
            continue
        outward.append({"pose_xyz_xyzw": candidate_pose.tolist(), "witness_joint_rad": candidate.tolist()})
        previous, pose = candidate, candidate_pose
    if len(outward) != count + 1:
        raise RuntimeError(f"Cannot sample {count} goals within 10000 attempts; accepted {len(outward) - 1}")
    data = {
        "version": 2,
        "policy_sha256": policy_sha256,
        "joint_order": JOINT_NAMES.copy(),
        "interval_s": interval_s,
        # The policy may use a different joint solution than a goal's witness.
        # Its execution envelope is the training limit, also checked against the
        # calibrated hardware limits at runtime; sampling has its own smaller box.
        "joint_bounds_rad": controller.limits.tolist(),
        "goal_workspace_m": controller.workspace.tolist(),
        "motion_workspace_m": controller.workspace.tolist(),
        "goals": outward + copy.deepcopy(outward[-2::-1]),
        "sampling": {
            "method": "local_joint_random_walk_with_reverse_return",
            "seed": seed,
            "outward_count": count,
            "joint_radius_rad": joint_radius,
            "joint_bounds_rad": bounds.tolist(),
            "joint_step_rad": joint_step,
            "min_position_step_m": min_position_step,
            "max_position_step_m": max_position_step,
            "max_angle_step_rad": max_angle_step,
            "attempts": attempts,
            "note": "Kinematic witnesses validate reachability; only Cartesian goals enter policy observations.",
        },
    }
    GoalPlan.from_data(data, controller, policy_sha256)
    return data


def sample_default_plan(
    controller,
    policy_sha256,
    kinematic_chain,
    *,
    seed=2026,
    count=24,
    interval_s=10.0,
    reservoir_size=DEFAULT_RESERVOIR_SIZE,
    sampling_seed=DEFAULT_SAMPLING_SEED,
):
    """Home, N independent default-family targets, then home; no step filtering.

    This uses the training sampler's four reservoirs and equal family weights.
    A finite random draw need not contain equal family counts. Home endpoints
    are preparation/return goals and are not counted as random family draws.
    """
    _hash(policy_sha256)
    _seed(seed)
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    interval_s = _number(interval_s, "interval_s", positive=True)
    home_pose = controller.fk(controller.home)
    _bounds(controller.workspace.tolist(), 3, controller.workspace, home_pose[:3], "workspace_m")
    for ranges in (DEFAULT_JOINT_RANGES, DEFAULT_DIVERSE_JOINT_RANGES):
        bounds = torch.tensor(ranges)
        if ((bounds[:, 0] < controller.limits[:, 0]) | (bounds[:, 1] > controller.limits[:, 1])).any():
            raise ValueError("Default sampling joint ranges exceed the deployment contract's limits")
    sampler = PoseReservoir(
        BatchKinematics(kinematic_chain),
        controller.workspace,
        reservoir_size=reservoir_size,
        sampling_seed=sampling_seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    families = torch.multinomial(torch.tensor(DEFAULT_FAMILY_WEIGHTS), count, replacement=True, generator=generator)
    sampled_poses, witnesses = sampler.sample(families, generator=generator)
    home = {"pose_xyz_xyzw": home_pose.tolist(), "witness_joint_rad": controller.home.tolist(), "family": "home"}
    goals = [home]
    for family, sampled_pose, witness in zip(families.tolist(), sampled_poses, witnesses, strict=True):
        pose = controller.fk(witness)
        if not torch.allclose(sampled_pose, pose, atol=1e-5, rtol=0):
            raise ValueError("Sampling kinematic chain does not match the deployment controller")
        goals.append(
            {
                "pose_xyz_xyzw": pose.tolist(),
                "witness_joint_rad": witness.tolist(),
                "family": GOAL_FAMILIES[family],
            }
        )
    goals.append(copy.deepcopy(home))
    data = {
        "version": 2,
        "policy_sha256": policy_sha256,
        "joint_order": JOINT_NAMES.copy(),
        "interval_s": interval_s,
        "joint_bounds_rad": controller.limits.tolist(),
        "goal_workspace_m": controller.workspace.tolist(),
        "motion_workspace_m": controller.workspace.tolist(),
        "goals": goals,
        "sampling": {
            "method": "default_four_family_independent",
            "seed": seed,
            "count": count,
            "family_names": list(GOAL_FAMILIES),
            "family_weights": list(DEFAULT_FAMILY_WEIGHTS),
            "family_counts": {name: int((families == i).sum()) for i, name in enumerate(GOAL_FAMILIES)},
            "joint_ranges_rad": [list(pair) for pair in DEFAULT_JOINT_RANGES],
            "diverse_joint_ranges_rad": [list(pair) for pair in DEFAULT_DIVERSE_JOINT_RANGES],
            "reservoir_size_per_family": reservoir_size,
            "reservoir_seed": sampling_seed,
            "adjacent_position_limit_m": None,
            "adjacent_orientation_limit_rad": None,
            "home_endpoints": True,
            "note": (
                "Shared default Reach training reservoirs; independent family/index draws without adjacent-goal "
                "filtering or success-based selection. Witness joints only establish endpoint reachability."
            ),
        },
    }
    GoalPlan.from_data(data, controller, policy_sha256)
    return data
