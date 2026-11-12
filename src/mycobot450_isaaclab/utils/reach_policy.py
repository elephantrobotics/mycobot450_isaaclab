"""CPU Reach inference and controller state, independent of Isaac Lab and robot SDKs.

Quaternions are XYZW. The tracked pose is base_link -> f100_base_link.
Keep numerical parity with tasks/reach/mdp/{observations,actions,commands}.py.
"""

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
OBSERVATION_TERMS = {
    "joint_pos": "isaaclab.envs.mdp.observations:joint_pos_rel",
    "joint_vel": "isaaclab.envs.mdp.observations:joint_vel_rel",
    "pose_command": "isaaclab.envs.mdp.observations:generated_commands",
    "actions": "isaaclab.envs.mdp.observations:last_action",
    "pose_error": "mycobot450_isaaclab.tasks.reach.mdp.observations:pose_error",
    "filtered_target": "mycobot450_isaaclab.tasks.reach.mdp.observations:filtered_target",
    "target_velocity": "mycobot450_isaaclab.tasks.reach.mdp.observations:target_velocity",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, size, name):
    result = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if result.shape != (size,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def quat_mul(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.linalg.cross(a[..., :3], b[..., :3])
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_rpy(rpy):
    r, p, y = (rpy * 0.5).unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )


def pose_error(current, goal):
    inverse = current[3:].clone()
    inverse[:3] *= -1
    inverse /= current[3:].square().sum()
    error = quat_mul(goal[3:], inverse)
    error = torch.where(error[3] < 0, -error, error)
    half_angle = torch.atan2(error[:3].norm(), error[3])
    angle = 2 * half_angle
    divisor = torch.where(angle.abs() > 1e-6, half_angle.sin() / angle, 0.5 - angle.square() / 48)
    return torch.cat((goal[:3] - current[:3], error[:3] / divisor))


def read_kinematic_chain(path):
    """Extract the same seven transforms used by training; meshes are not needed."""
    root = ET.parse(path).getroot()
    chain, parent = [], "base_link"
    for name in JOINT_NAMES + ["AssemblerFixedJoint"]:
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.find("parent").get("link") != parent:
            raise ValueError(f"Invalid URDF chain at {name}")
        if name in JOINT_NAMES and joint.find("axis").get("xyz") != "0 0 1":
            raise ValueError(f"Unsupported axis for {name}")
        origin = joint.find("origin")
        chain.append({key: [float(v) for v in origin.get(key).split()] for key in ("xyz", "rpy")})
        parent = joint.find("child").get("link")
    if parent != "f100_base_link":
        raise ValueError("URDF chain must end at f100_base_link")
    return chain


class ArmKinematics:
    def __init__(self, chain):
        if len(chain) != 7:
            raise ValueError("Expected six arm joints and the fixed F100 mounting transform")
        self.origins = [(vector(x["xyz"], 3, "origin"), quat_rpy(vector(x["rpy"], 3, "rpy"))) for x in chain]

    def __call__(self, joints):
        position, rotation = torch.zeros(3), torch.tensor([0.0, 0.0, 0.0, 1.0])
        for index, (offset, orientation) in enumerate(self.origins):
            uv = torch.linalg.cross(rotation[:3], offset)
            position += offset + 2 * (rotation[3] * uv + torch.linalg.cross(rotation[:3], uv))
            rotation = quat_mul(rotation, orientation)
            if index < 6:
                half = joints[index] / 2
                rotation = quat_mul(rotation, torch.stack((half * 0, half * 0, half.sin(), half.cos())))
        rotation /= rotation.norm()
        return torch.cat((position, torch.where(rotation[3] < 0, -rotation, rotation)))


class ReachController:
    """One robot, float32 observations and the training controller's fixed-step state."""

    def __init__(self, contract):
        if contract["version"] != 1 or contract["joint_order"] != JOINT_NAMES:
            raise ValueError("Unsupported deployment contract version or joint order")
        if contract["action_type"] != "filtered_joint_velocity_integrated_position_v1":
            raise ValueError("Unsupported action type")
        self.home = vector(contract["home_rad"], 6, "home")
        self.limits = torch.tensor(contract["joint_limits_rad"], dtype=torch.float32)
        if (
            self.limits.shape != (6, 2)
            or not torch.isfinite(self.limits).all()
            or not (self.limits[:, 0] < self.limits[:, 1]).all()
        ):
            raise ValueError("Invalid joint limits")
        self.dt = float(contract["control_dt_s"])
        self.policy_dt = float(contract["policy_dt_s"])
        self.tau = float(contract["filter_time_constant_s"])
        self.scale = float(contract["velocity_scale_rad_s"])
        if not all(math.isfinite(x) and x > 0 for x in (self.dt, self.policy_dt, self.tau, self.scale)):
            raise ValueError("Invalid timing, action scale or filter constant")
        self.decimation = round(self.policy_dt / self.dt)
        if self.decimation < 1 or not math.isclose(self.decimation * self.dt, self.policy_dt):
            raise ValueError("Policy period must be an integer number of control periods")
        if contract.get("max_tracking_error_rad") is not None:
            raise ValueError("Tracking-guard policies are not supported by this deployment version")
        self.fk = ArmKinematics(contract["kinematic_chain"])
        # The bundle stores the training goal domain; runtime motion bounds live in GoalPlan.
        self.workspace = torch.tensor(contract["workspace_m"], dtype=torch.float32)
        if (
            self.workspace.shape != (3, 2)
            or not torch.isfinite(self.workspace).all()
            or not (self.workspace[:, 0] < self.workspace[:, 1]).all()
        ):
            raise ValueError("Invalid goal workspace")
        self.reset(self.home)

    def reset(self, measured):
        self.target = vector(measured, 6, "measured joints").clone()
        self.velocity = torch.zeros(6)
        self.previous_action = torch.zeros(6)
        self.lead_limited_substeps = 0

    def goal(self, pose):
        result = vector(pose, 7, "goal XYZ/XYZW").clone()
        norm = result[3:].norm().item()
        if not math.isclose(norm, 1, abs_tol=1e-3):
            raise ValueError("Goal quaternion must be unit length (XYZW)")
        result[3:] /= norm
        if result[6] < 0:
            result[3:] *= -1
        if ((result[:3] < self.workspace[:, 0]) | (result[:3] > self.workspace[:, 1])).any():
            raise ValueError("Goal is outside the training workspace")
        return result

    def observe(self, measured, velocity, goal):
        q, dq = vector(measured, 6, "joints"), vector(velocity, 6, "velocity")
        return torch.cat(
            (
                q - self.home,
                dq,
                goal,
                self.previous_action,
                pose_error(self.fk(q), goal),
                self.target - self.home,
                self.velocity,
            )
        ).unsqueeze(0)

    def process(self, action):
        self.previous_action = vector(action, 6, "policy action").clamp(-1, 1)

    def step(self, measured=None, *, target_lead_limit_rad=None, action_gain=1.0):
        if type(action_gain) not in (int, float) or not math.isfinite(action_gain) or not 0 < action_gain <= 1:
            raise ValueError("action_gain must be finite and in (0, 1]")
        if target_lead_limit_rad is not None:
            if (
                type(target_lead_limit_rad) not in (int, float)
                or not math.isfinite(target_lead_limit_rad)
                or target_lead_limit_rad <= 0
            ):
                raise ValueError("target_lead_limit_rad must be finite and positive")
            measured = vector(measured, 6, "measured joints")
        self.velocity.lerp_(self.scale * action_gain * self.previous_action, 1 - math.exp(-self.dt / self.tau))
        if target_lead_limit_rad is not None:
            # Match SmoothJointVelocityAction: restrict new integration, never snap an existing target.
            error = self.target - measured
            # Feedback is held for a policy period; substep-sized corrections amplify filter-state errors.
            forward = ((target_lead_limit_rad - error) / self.policy_dt).clamp_min(0.0)
            backward = ((target_lead_limit_rad + error) / self.policy_dt).clamp_min(0.0)
            limited = torch.maximum(torch.minimum(self.velocity, forward), -backward)
            self.lead_limited_substeps += int((limited != self.velocity).any())
            self.velocity = limited
        proposed = self.target + self.dt * self.velocity
        self.target = proposed.clamp(self.limits[:, 0], self.limits[:, 1])
        self.velocity = torch.where(proposed == self.target, self.velocity, 0.0)
        return self.target.clone()


class ReachPolicy:
    def __init__(self, bundle):
        self.directory = Path(bundle)
        self.contract = json.loads((self.directory / "contract.json").read_text())
        model = self.directory / "policy.pt"
        if sha256(model) != self.contract["policy_sha256"]:
            raise ValueError("policy.pt does not match contract.json")
        self.controller = ReachController(self.contract)
        # Keep the original float32 FK operations, without per-tick Python dispatch.
        self.controller.fk = torch.jit.trace(self.controller.fk.__call__, self.controller.home, check_trace=True)
        self.model = torch.jit.load(str(model), map_location="cpu").eval()
        # Tracing does not execute every optimized path. Warm the actual observation
        # and policy before opening hardware, without advancing controller state.
        goal = self.controller.fk(self.controller.home)
        for _ in range(3):
            self.infer(self.controller.observe(self.controller.home, torch.zeros(6), goal))

    @torch.inference_mode()
    def infer(self, observation):
        if observation.shape != (1, 43) or not torch.isfinite(observation).all():
            raise ValueError("Expected a finite [1, 43] observation")
        action = self.model(observation)
        if not isinstance(action, torch.Tensor) or action.shape != (1, 6) or not torch.isfinite(action).all():
            raise ValueError("Policy must return a finite [1, 6] action")
        return action[0].clone()
