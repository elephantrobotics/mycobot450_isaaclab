"""Play an exported Reach TorchScript policy in Isaac Sim.

Follows Isaac Lab's scripts/tutorials/03_envs/policy_inference_in_usd.py.
The Reach task supplies the scene, observations, goals and action controller.
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch

from mycobot450_isaaclab.utils.deploy import HardwareConfig, command_schedule
from mycobot450_isaaclab.utils.reach_goals import DefaultGoalSampler, GoalPlan
from mycobot450_isaaclab.utils.reach_motion_guard import MotionGuard
from mycobot450_isaaclab.utils.reach_policy import JOINT_NAMES, ReachController, sha256

TASK = "Mycobot450Isaaclab-Reach-Play"


def load_policy(path: Path):
    """Load only a TorchScript export, before starting the simulator."""
    try:
        return torch.jit.load(str(path), map_location="cpu").eval()
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot load {path} as TorchScript. Use the exported policy.pt, "
            "not a training checkpoint such as model_2999.pt."
        ) from exc


def load_goal_plan(path: Path, policy_path: Path):
    """A simulation plan must refer to the same self-contained deployment bundle."""
    contract_path = policy_path.with_name("contract.json")
    if not contract_path.is_file():
        raise ValueError("--goal-plan requires --policy inside a deployment bundle containing contract.json")
    contract = json.loads(contract_path.read_text())
    policy_hash = sha256(policy_path)
    if contract.get("policy_sha256") != policy_hash:
        raise ValueError("policy.pt does not match the deployment contract")
    controller = ReachController(contract)
    plan = GoalPlan.from_file(path, controller, policy_hash)
    plan.training_limits = controller.limits.clone()
    plan.contract = contract
    return plan


def load_motion_guard(path, plan, *, time_scale, velocity_source, steps):
    """Validate the explicit development hardware profile without opening a robot."""
    if plan is None or time_scale != 1.0 or velocity_source != "finite_difference":
        raise ValueError("--hardware-config requires --goal-plan, --time-scale 1 and finite_difference velocity")
    if type(steps) is not int or not 0 < steps <= 18000:
        raise ValueError("--hardware-config requires 1..18000 steps (at most 600 seconds at 30 Hz)")
    controller = ReachController(plan.contract)
    data = json.loads(Path(path).read_text())
    if "motion_guard" not in data:
        raise ValueError("Simulation --hardware-config requires a staged motion_guard profile")
    hardware = HardwareConfig(data, controller)
    hardware.require_continuous_ready(30, time_scale, response_trial=True, duration=steps / 30)
    command_schedule(controller, 30, time_scale)
    return MotionGuard(data)


def configure_plan_timing(cfg, plan, *, time_scale=1.0, target_update_decimation=None):
    """Keep PhysX at its original rate while matching the deployment command clock."""
    if type(time_scale) not in (int, float) or time_scale not in (1.0, 0.5, 0.25):
        raise ValueError("time_scale must be 1, 0.5 or 0.25")
    if time_scale == 1.0:
        if target_update_decimation is not None:
            if target_update_decimation not in (1, cfg.decimation):
                raise ValueError("--target-update-decimation must be 1 or the policy decimation")
            cfg.actions.arm_action.target_update_decimation = target_update_decimation
        return
    if plan is None:
        raise ValueError("--time-scale below 1 requires --goal-plan and its deployment contract")
    dt, policy_dt = plan.contract["control_dt_s"], plan.contract["policy_dt_s"]
    if not math.isclose(cfg.sim.dt, dt) or not math.isclose(cfg.decimation * cfg.sim.dt, policy_dt):
        raise ValueError("Simulation timing must match the deployment contract before time scaling")
    hold_steps = round(1 / (15 * dt))
    nominal_steps = round(hold_steps * time_scale)
    policy_steps = round(policy_dt / (time_scale * dt))
    if (
        nominal_steps < 1
        or not math.isclose(hold_steps * dt, 1 / 15)
        or not math.isclose(nominal_steps, hold_steps * time_scale)
        or not math.isclose(policy_steps * dt * time_scale, policy_dt)
        or policy_steps % hold_steps != 0
    ):
        raise ValueError("Scaled playback requires whole nominal substeps and 15 Hz command holds")
    if target_update_decimation is not None and target_update_decimation != hold_steps:
        raise ValueError(f"Scaled playback requires --target-update-decimation {hold_steps}")
    cfg.decimation = policy_steps
    cfg.actions.arm_action.target_update_decimation = hold_steps
    cfg.actions.arm_action.integration_substeps = nominal_steps


def infer_action(policy, observation: torch.Tensor) -> torch.Tensor:
    """Preserve the current Reach actor's observation and action contract."""
    if observation.ndim != 2 or observation.shape[1] != 43:
        raise ValueError(f"Reach requires observations shaped (num_envs, 43), got {tuple(observation.shape)}")
    if not torch.isfinite(observation).all():
        raise ValueError("Reach observations contain NaN or infinity")
    # The export already contains the trained observation normalizer.
    action = policy(observation)
    expected_shape = (observation.shape[0], 6)
    if not isinstance(action, torch.Tensor) or action.shape != expected_shape:
        raise ValueError(f"Reach policy must return one action tensor shaped {expected_shape}")
    if not torch.isfinite(action).all():
        raise ValueError("Reach policy returned NaN or infinity")
    # Match RSL-RL's clip_actions=1.0 before ActionManager stores last_action.
    return action.clamp(-1.0, 1.0)


def validate_velocity_source(source, *, plan, target_hold_steps, policy_steps):
    if source not in ("native", "finite_difference"):
        raise ValueError("velocity_source must be native or finite_difference")
    if source == "finite_difference":
        if plan is None:
            raise ValueError("--velocity-source finite_difference requires --goal-plan")
        if (
            type(target_hold_steps) is not int
            or type(policy_steps) is not int
            or policy_steps < 1
            or target_hold_steps != policy_steps
        ):
            raise ValueError("Finite-difference velocity requires exactly one held command per policy step")


class PolicyVelocityObservation:
    """Sample actual joints at policy calls, retaining difference history across goals."""

    def __init__(self, env, plan, *, source, time_scale):
        action = env.action_manager.get_term("arm_action")
        validate_velocity_source(
            source,
            plan=plan,
            target_hold_steps=getattr(getattr(action, "cfg", None), "target_update_decimation", None),
            policy_steps=getattr(getattr(env, "cfg", None), "decimation", None),
        )
        self.robot = env.command_manager.get_term("ee_pose").robot
        ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if names != JOINT_NAMES:
            raise ValueError("Velocity observations require measured joints in deployment order")
        self.ids = ids.torch if hasattr(ids, "torch") else ids
        self.source, self.time_scale, self.dt = source, time_scale, env.step_dt
        self.previous_q = None
        self.count = 0
        self.maxima = torch.zeros(3, device=env.device)

    def prepare(self, observation):
        if observation.shape != (1, 43):
            raise ValueError("Velocity observations require a single 43-dimensional observation")
        try:
            q = self.robot.data.joint_pos.torch[:, self.ids].clone()
            native = self.robot.data.joint_vel.torch[:, self.ids].clone()
        except (AttributeError, TypeError) as exc:
            raise ValueError("Velocity observations require native measured joint positions and velocities") from exc
        if q.shape != (1, 6) or native.shape != (1, 6) or not torch.isfinite(torch.cat((q, native))).all():
            raise ValueError("Velocity observations require six finite measured positions and velocities")
        difference = torch.zeros_like(q) if self.previous_q is None else (q - self.previous_q) / self.dt
        self.previous_q = q
        self.maxima = torch.maximum(
            self.maxima,
            torch.stack((native.abs().max(), difference.abs().max(), (native - difference).abs().max())),
        )
        self.count += 1
        if self.source == "finite_difference":
            observation = observation.clone()
            observation[:, 6:12] = difference / self.time_scale
        elif self.time_scale != 1.0:
            observation = observation.clone()
            observation[:, 6:12] /= self.time_scale
        return observation

    def result(self):
        native, difference, discrepancy = self.maxima.tolist()
        return {
            "samples": self.count,
            "sample_hz": 1 / self.dt,
            "max_native_speed_rad_s": native,
            "max_finite_difference_speed_rad_s": difference,
            "max_native_vs_finite_difference_rad_s": discrepancy,
            "first_finite_difference_sample": "zero; no previous position sample at stationary startup",
            "note": (
                "Physical-time velocities sampled before each policy inference. These peaks differ from "
                "max_joint_speed_rad_s, which samples native velocity at every physics step. Position "
                "differences use unquantized simulation joints and fixed env.step_dt, not SDK jitter."
            ),
        }


@torch.inference_mode()
def play_policy(env, policy, *, steps: int = 0, real_time: bool = True) -> int:
    """Run inference and let the environment handle commands and episode resets."""
    obs, _ = env.reset()
    step_count = 0
    while env.sim.is_running() and (steps == 0 or step_count < steps):
        start = time.perf_counter()
        action = infer_action(policy, obs["policy"])
        obs, _, _, _, _ = env.step(action)
        step_count += 1
        if real_time:
            remaining = env.step_dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)
    return step_count


class WorkspaceSamples:
    """Measured Cartesian bounds at one explicit sampling rate, in the robot root frame."""

    def __init__(self, bounds, *, period, phase):
        self.bounds, self.period, self.phase = bounds, period, phase
        self.samples = self.violations = 0
        self.minimum = torch.full((3,), math.inf, device=bounds.device)
        self.maximum = -self.minimum.clone()
        self.lower_counts = torch.zeros(3, dtype=torch.int64, device=bounds.device)
        self.upper_counts = torch.zeros_like(self.lower_counts)
        self.lower_excess = torch.zeros(3, device=bounds.device)
        self.upper_excess = torch.zeros_like(self.lower_excess)
        self.first = self.worst = None

    def sample(self, q, position, *, time_in_goal):
        if not torch.isfinite(torch.cat((q, position))).all():
            raise RuntimeError("Non-finite measured state in workspace diagnostics")
        lower = (self.bounds[:, 0] - position).clamp_min(0)
        upper = (position - self.bounds[:, 1]).clamp_min(0)
        self.minimum = torch.minimum(self.minimum, position)
        self.maximum = torch.maximum(self.maximum, position)
        self.lower_counts += (lower > 0).to(torch.int64)
        self.upper_counts += (upper > 0).to(torch.int64)
        self.lower_excess = torch.maximum(self.lower_excess, lower)
        self.upper_excess = torch.maximum(self.upper_excess, upper)
        self.samples += 1
        excess = float(torch.maximum(lower, upper).max())
        if excess > 0:
            self.violations += 1
            if self.first is None or self.worst is None or excess > self.worst["max_excess_m"]:
                record = {
                    "time_in_goal_s": time_in_goal,
                    "q_rad": q.tolist(),
                    "position_xyz_m": position.tolist(),
                    "workspace_m": self.bounds.tolist(),
                    "lower_excess_m": lower.tolist(),
                    "upper_excess_m": upper.tolist(),
                    "max_excess_m": excess,
                }
                if self.first is None:
                    self.first = record
                if self.worst is None or excess > self.worst["max_excess_m"]:
                    self.worst = record
        return excess > 0

    def result(self):
        return {
            "samples": self.samples,
            "sample_period_s": self.period,
            "sample_phase": self.phase,
            "position_min_m": self.minimum.tolist() if self.samples else None,
            "position_max_m": self.maximum.tolist() if self.samples else None,
            "lower_violation_counts": self.lower_counts.tolist(),
            "upper_violation_counts": self.upper_counts.tolist(),
            "max_lower_excess_m": self.lower_excess.tolist(),
            "max_upper_excess_m": self.upper_excess.tolist(),
            "violation_samples": self.violations,
            "sampled_violation_duration_s": self.violations * self.period,
            "first_violation": self.first,
            "max_violation": self.worst,
        }


def combine_workspace_results(rows, name, empty):
    """Combine goal diagnostics without losing the time and measured state of extrema."""
    if not rows:
        return empty
    result = {"bounds_m": rows[0][name]["bounds_m"], "axis_order": ["x", "y", "z"]}
    for rate in ("physics", "policy"):
        entries = [row[name][rate] for row in rows]
        sampled = [entry for entry in entries if entry["samples"]]
        combined = {
            "samples": sum(entry["samples"] for entry in entries),
            "sample_period_s": entries[0]["sample_period_s"],
            "sample_phase": entries[0]["sample_phase"],
            "violation_samples": sum(entry["violation_samples"] for entry in entries),
            "sampled_violation_duration_s": sum(entry["sampled_violation_duration_s"] for entry in entries),
        }
        for field, aggregate in (
            ("position_min_m", min),
            ("position_max_m", max),
            ("lower_violation_counts", sum),
            ("upper_violation_counts", sum),
            ("max_lower_excess_m", max),
            ("max_upper_excess_m", max),
        ):
            sources = sampled if field.startswith("position_") else entries
            combined[field] = (
                [aggregate(entry[field][axis] for entry in sources) for axis in range(3)] if sources else None
            )
        for field in ("first_violation", "max_violation"):
            records = [
                {
                    **row[name][rate][field],
                    "goal_index": row["goal_index"],
                    "time_sim_s": row["start_time_s"] + row[name][rate][field]["time_in_goal_s"],
                }
                for row in rows
                if row[name][rate][field] is not None
            ]
            combined[field] = (
                min(records, key=lambda item: item["time_sim_s"])
                if records and field == "first_violation"
                else max(records, key=lambda item: item["max_excess_m"])
                if records
                else None
            )
        result[rate] = combined
    return result


class PlanMetrics:
    """Accumulate native physics state, without computing a second robot FK."""

    def __init__(self, env, plan):
        self.command = env.command_manager.get_term("ee_pose")
        self.action = env.action_manager.get_term("arm_action")
        self.robot = self.command.robot
        ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if names != JOINT_NAMES:
            raise ValueError("Simulation joint order does not match the deployment plan")
        self.ids = ids.torch if hasattr(ids, "torch") else ids
        self.bounds = plan.joint_bounds.to(env.device)
        self.workspace = getattr(plan, "motion_workspace", plan.workspace).to(env.device)
        goal_workspace = getattr(plan, "goal_workspace", plan.workspace).to(env.device)
        self.workspaces = {
            name: {
                "bounds": bounds,
                "physics": WorkspaceSamples(bounds, period=env.physics_dt, phase="after_physics_step"),
                "policy": WorkspaceSamples(bounds, period=env.step_dt, phase="before_policy_inference"),
            }
            for name, bounds in (("motion_workspace", self.workspace), ("goal_workspace", goal_workspace))
        }
        self.controller_limits = self.action.limits.to(env.device)
        if hasattr(plan, "training_limits") and not torch.equal(
            self.controller_limits, plan.training_limits.to(env.device)
        ):
            raise ValueError("Simulation controller limits differ from the deployment contract")
        self.simulation_limits = self.robot.data.joint_pos_limits.torch[0, self.ids].clone()
        self.maxima = torch.zeros(4, device=env.device)
        self.violations = torch.zeros(7, dtype=torch.int64, device=env.device)
        self.measured_min = torch.full((6,), math.inf, device=env.device)
        self.measured_max = -self.measured_min.clone()
        self.target_min = self.measured_min.clone()
        self.target_max = self.measured_max.clone()
        self.measured_excess = torch.zeros(6, device=env.device)
        self.target_excess = torch.zeros(6, device=env.device)
        self.samples = 0
        self.final = None
        self.physics_dt = env.physics_dt
        self.policy_dt = env.step_dt
        self.policy_samples = 0

    def measured_pose(self):
        from isaaclab.utils.math import subtract_frame_transforms

        data = self.robot.data
        q = data.joint_pos.torch[0, self.ids]
        position, _ = subtract_frame_transforms(
            data.root_pos_w.torch,
            data.root_quat_w.torch,
            data.body_pos_w.torch[:, self.command.body_idx],
            data.body_quat_w.torch[:, self.command.body_idx],
        )
        return q, position[0]

    def sample_policy(self):
        q, position = self.measured_pose()
        for workspace in self.workspaces.values():
            workspace["policy"].sample(q, position, time_in_goal=self.policy_samples * self.policy_dt)
        self.policy_samples += 1

    def sample(self):
        q, position = self.measured_pose()
        velocity = self.robot.data.joint_vel.torch[0, self.ids]
        target = self.action.target[0]
        position_error, orientation_error = self.command._compute_error()
        values = torch.stack((position_error[0], orientation_error[0], velocity.abs().max(), (target - q).abs().max()))
        if not torch.isfinite(torch.cat((q, velocity, target, position, values))).all():
            raise RuntimeError("Non-finite simulation state during goal-plan validation")
        self.maxima = torch.maximum(self.maxima, values)
        self.measured_min = torch.minimum(self.measured_min, q)
        self.measured_max = torch.maximum(self.measured_max, q)
        self.target_min = torch.minimum(self.target_min, target)
        self.target_max = torch.maximum(self.target_max, target)
        self.measured_excess = torch.maximum(
            self.measured_excess, torch.maximum(self.bounds[:, 0] - q, q - self.bounds[:, 1])
        )
        self.target_excess = torch.maximum(
            self.target_excess, torch.maximum(self.bounds[:, 0] - target, target - self.bounds[:, 1])
        )
        for workspace in self.workspaces.values():
            workspace["physics"].sample(q, position, time_in_goal=(self.samples + 1) * self.physics_dt)
        outside_workspace = ((position < self.workspace[:, 0]) | (position > self.workspace[:, 1])).any()
        self.violations += torch.stack(
            (
                ((q < self.bounds[:, 0]) | (q > self.bounds[:, 1])).any(),
                ((target < self.bounds[:, 0]) | (target > self.bounds[:, 1])).any(),
                outside_workspace,
                ((q < self.controller_limits[:, 0]) | (q > self.controller_limits[:, 1])).any(),
                ((target < self.controller_limits[:, 0]) | (target > self.controller_limits[:, 1])).any(),
                ((q < self.simulation_limits[:, 0]) | (q > self.simulation_limits[:, 1])).any(),
                ((target < self.simulation_limits[:, 0]) | (target > self.simulation_limits[:, 1])).any(),
            )
        ).to(torch.int64)
        self.final = values
        self.samples += 1

    def result(self):
        maxima, counts = self.maxima.tolist(), self.violations.tolist()
        final = None if self.final is None else self.final.tolist()
        workspaces = {
            name: {
                "bounds_m": workspace["bounds"].tolist(),
                "axis_order": ["x", "y", "z"],
                "physics": workspace["physics"].result(),
                "policy": workspace["policy"].result(),
            }
            for name, workspace in self.workspaces.items()
        }
        return {
            "physics_samples": self.samples,
            "position_error_m": None if final is None else final[0],
            "orientation_error_rad": None if final is None else final[1],
            "final_max_joint_speed_rad_s": None if final is None else final[2],
            "max_position_error_m": maxima[0],
            "max_orientation_error_rad": maxima[1],
            "max_joint_speed_rad_s": maxima[2],
            "max_target_tracking_error_rad": maxima[3],
            "joint_envelope_violation_samples": counts[0],
            "target_envelope_violation_samples": counts[1],
            "workspace_violation_samples": counts[2],
            "first_workspace_violation": workspaces["motion_workspace"]["physics"]["first_violation"],
            "policy_workspace_violation_samples": workspaces["motion_workspace"]["policy"]["violation_samples"],
            **workspaces,
            "controller_joint_limit_violation_samples": counts[3],
            "controller_target_limit_violation_samples": counts[4],
            "simulation_joint_limit_violation_samples": counts[5],
            "simulation_target_limit_violation_samples": counts[6],
            "controller_joint_limits_rad": self.controller_limits.tolist(),
            "simulation_joint_limits_rad": self.simulation_limits.tolist(),
            "measured_joint_min_rad": self.measured_min.tolist() if self.samples else None,
            "measured_joint_max_rad": self.measured_max.tolist() if self.samples else None,
            "target_joint_min_rad": self.target_min.tolist() if self.samples else None,
            "target_joint_max_rad": self.target_max.tolist() if self.samples else None,
            "measured_joint_envelope_excess_rad": self.measured_excess.tolist(),
            "target_joint_envelope_excess_rad": self.target_excess.tolist(),
            "final_settled_success": bool(final and final[0] < 0.02 and final[1] < 0.15 and final[2] < 0.1),
        }


@torch.inference_mode()
def play_goal_plan(
    env,
    policy,
    plan,
    *,
    steps=0,
    real_time=True,
    report=None,
    time_scale=1.0,
    velocity_source="native",
    motion_guard=None,
    goal_sampler=None,
):
    """Cycle goals on the existing controller timeline; never reset between them."""
    if env.num_envs != 1:
        raise ValueError("Goal-plan simulation requires exactly one environment")
    if type(time_scale) not in (int, float) or time_scale not in (1.0, 0.5, 0.25):
        raise ValueError("time_scale must be 1, 0.5 or 0.25")
    real_interval = plan.interval_s / time_scale
    interval_steps = round(real_interval / env.step_dt)
    if interval_steps < 1 or not math.isclose(interval_steps * env.step_dt, real_interval, abs_tol=1e-9):
        raise ValueError("Goal interval must be an integer number of policy steps")
    velocity = PolicyVelocityObservation(env, plan, source=velocity_source, time_scale=time_scale)
    shadow = None
    integration_steps = None
    if motion_guard is not None:
        if time_scale != 1.0 or velocity_source != "finite_difference" or not math.isclose(env.step_dt, 1 / 30):
            raise ValueError("Simulation motion guards require finite_difference velocity at 30 Hz and time_scale=1")
        if type(steps) is not int or not 0 < steps <= 18000:
            raise ValueError("Simulation motion guards require 1..18000 steps")
        shadow = ReachController(plan.contract)
        period, integration_steps, policy_every = command_schedule(shadow, 30, time_scale)
        if not math.isclose(period, env.step_dt) or integration_steps != 4 or policy_every != 1:
            raise ValueError("Simulation motion guards require four original controller substeps per policy call")
        action_cfg = env.action_manager.get_term("arm_action").cfg
        if getattr(action_cfg, "max_tracking_error", None) != motion_guard.target_lead_limit_rad:
            raise ValueError("Simulation and deployment target lead limits must match")
    report = {} if report is None else report
    if motion_guard is not None:
        report.update(max_shadow_target_difference_rad=0.0, max_shadow_velocity_difference_rad_s=0.0)
    report.update(
        mode="goal_plan_simulation",
        policy_sha256=plan.data["policy_sha256"],
        goal_count=None if goal_sampler is not None else len(plan.goals),
        goal_interval_s=real_interval,
        nominal_goal_interval_s=plan.interval_s,
        time_scale=time_scale,
        velocity_source=velocity_source,
        policy_hz=1 / env.step_dt,
        physics_hz=1 / env.physics_dt,
        requested_steps=steps,
        steps=0,
        goal_switch_count=0,
        reset_count=0,
        reset_attempts=0,
        goal_results=[],
        error=None,
        note=(
            "Errors use simulator body state; kinematic witnesses are never applied to joints. "
            "Workspace durations are sample count times sample period, not exact continuous crossing times. "
            "Physics and policy-call diagnostics have independent sampling phases. No hardware is connected."
        ),
    )
    obs, _ = env.reset()
    command = env.command_manager.get_term("ee_pose")
    goals = plan.goals.to(env.device)
    current_goal = (goal_sampler.initial_goal if goal_sampler is not None else goals[0]).to(env.device)
    original_update, original_reset = env.scene.update, env._reset_idx
    metrics = PlanMetrics(env, plan)
    goal_index, segment_start, step_count = 0, 0, 0

    def select_goal(index):
        nonlocal current_goal
        current_goal = (
            (goal_sampler.initial_goal if index == 0 else goal_sampler.sample()).to(env.device)
            if goal_sampler is not None
            else goals[index]
        )
        command.pose_command_b[:] = current_goal
        command.time_left.fill_(1.0e9)
        env.obs_buf = env.observation_manager.compute()
        return env.obs_buf

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        metrics.sample()
        return result

    def reject_reset(*args, **kwargs):
        report["reset_attempts"] += 1
        raise RuntimeError("Reset requested during uninterrupted goal-plan validation")

    def finish_segment(reason):
        if metrics.samples or metrics.policy_samples:
            report["goal_results"].append(
                {
                    "goal_index": goal_index,
                    "goal_pose_xyz_xyzw": current_goal.tolist(),
                    "sampled_goal": dict(goal_sampler.last_goal) if goal_sampler is not None else None,
                    "start_time_s": segment_start * env.step_dt,
                    "end_time_s": step_count * env.step_dt,
                    "interval_completed": step_count - segment_start == interval_steps,
                    "end_reason": reason,
                    **metrics.result(),
                }
            )
            row = report["goal_results"][-1]
            position = "n/a" if row["position_error_m"] is None else f"{row['position_error_m'] * 1000:.2f}mm"
            orientation = (
                "n/a"
                if row["orientation_error_rad"] is None
                else f"{math.degrees(row['orientation_error_rad']):.2f}deg"
            )
            print(
                f"[GOAL {goal_index + 1}/{len(goals) if goal_sampler is None else 'random'}] "
                f"t={step_count * env.step_dt:.1f}s "
                f"position={position} orientation={orientation} "
                f"settled={row['final_settled_success']} "
                f"workspace_violations={row['workspace_violation_samples']}",
                flush=True,
            )

    ending = "window_closed"
    env.scene.update, env._reset_idx = update, reject_reset
    try:
        obs = select_goal(0)
        while env.sim.is_running() and (steps == 0 or step_count < steps):
            if step_count and step_count % interval_steps == 0:
                finish_segment("goal_switch")
                segment_start = step_count
                goal_index = goal_index + 1 if goal_sampler is not None else (goal_index + 1) % len(goals)
                metrics = PlanMetrics(env, plan)
                obs = select_goal(goal_index)
                report["goal_switch_count"] += 1
            started = time.perf_counter()
            observation = velocity.prepare(obs["policy"])
            metrics.sample_policy()  # Same measured state, before inference/commands; never advance physics here.
            if motion_guard is not None:
                state = observation[0].detach().cpu()
                q = velocity.previous_q[0].detach().cpu().clone()
                if step_count == 0:
                    shadow.reset(q)
                target_difference = (shadow.target - (state[31:37] + shadow.home)).abs().max().item()
                velocity_difference = (shadow.velocity - state[37:43]).abs().max().item()
                report["max_shadow_target_difference_rad"] = max(
                    report["max_shadow_target_difference_rad"], target_difference
                )
                report["max_shadow_velocity_difference_rad_s"] = max(
                    report["max_shadow_velocity_difference_rad_s"], velocity_difference
                )
                if not torch.allclose(shadow.target, state[31:37] + shadow.home, atol=5e-5, rtol=0):
                    raise RuntimeError("Simulation integrated target differs from the deployment shadow controller")
                if not torch.allclose(shadow.velocity, state[37:43], atol=5e-5, rtol=0):
                    raise RuntimeError(
                        "Simulation filtered velocity differs from the deployment shadow controller: "
                        f"{velocity_difference:.9g} rad/s"
                    )
                motion_guard.check_feedback(
                    q, state[6:12] * time_scale, shadow.target, env.step_dt, step_count * env.step_dt
                )
            action = infer_action(policy, observation)
            if motion_guard is not None:
                shadow.process(action[0].detach().cpu())
                for _ in range(integration_steps):
                    target = shadow.step(
                        q,
                        target_lead_limit_rad=motion_guard.target_lead_limit_rad,
                        action_gain=motion_guard.action_gain,
                    )
                motion_guard.check_target(q, target, step_count * env.step_dt)
            obs, _, terminated, truncated, _ = env.step(action)
            if motion_guard is not None and motion_guard.target_lead_limit_rad is not None:
                sampled = env.action_manager.get_term("arm_action").integration_feedback[0].detach().cpu()
                if not torch.equal(q, sampled):
                    report["integration_feedback_difference_rad"] = (q - sampled).tolist()
                    raise RuntimeError("Simulation and deployment integrated with different measured joints")
            step_count += 1
            if bool(terminated.any() or truncated.any()):
                report["reset_count"] += int((terminated | truncated).sum())
                raise RuntimeError("Unexpected episode boundary during goal-plan validation")
            if not torch.equal(command.pose_command_b[0], current_goal):
                raise RuntimeError("Goal changed outside the supplied plan")
            if real_time:
                time.sleep(max(0.0, env.step_dt - (time.perf_counter() - started)))
        if steps and step_count == steps:
            ending = "step_limit"
    except BaseException as exc:
        ending = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.scene.update, env._reset_idx = original_update, original_reset
        finish_segment(ending)
        rows = report["goal_results"]
        completed = [row for row in rows if row["interval_completed"]]
        report.update(
            steps=step_count,
            elapsed_simulation_s=step_count * env.step_dt,
            end_reason=ending,
            completed_goal_intervals=len(completed),
            all_completed_goals_settled=bool(completed) and all(row["final_settled_success"] for row in completed),
            velocity_observation_samples=velocity.result(),
            goal_sampling=goal_sampler.report() if goal_sampler is not None else None,
        )
        if motion_guard is not None:
            report["target_lead_limit_rad"] = motion_guard.target_lead_limit_rad
            report["action_gain"] = motion_guard.action_gain
            report["lead_limited_substeps"] = shadow.lead_limited_substeps
            report["motion_guard"] = motion_guard.report()
            report["motion_guard"]["simulation_note"] = (
                "Same development monitor as hardware, with an independently checked deployment controller. "
                "Initial finite-difference speed is zero at stationary startup. Simulator timestamps and angles "
                "are exact samples; this does not model SDK response lag or certify hardware operation."
            )
        for name in ("max_joint_speed_rad_s", "max_target_tracking_error_rad"):
            report[name] = max((row[name] for row in rows), default=0.0)
        for name in (
            "joint_envelope_violation_samples",
            "target_envelope_violation_samples",
            "workspace_violation_samples",
            "controller_joint_limit_violation_samples",
            "controller_target_limit_violation_samples",
            "simulation_joint_limit_violation_samples",
            "simulation_target_limit_violation_samples",
        ):
            report[name] = sum(row[name] for row in rows)
        for name in ("controller_joint_limits_rad", "simulation_joint_limits_rad"):
            report[name] = metrics.result()[name]
        for name in ("motion_workspace", "goal_workspace"):
            report[name] = combine_workspace_results(rows, name, metrics.result()[name])
        report["policy_workspace_violation_samples"] = report["motion_workspace"]["policy"]["violation_samples"]
        report["first_workspace_violation"] = report["motion_workspace"]["physics"]["first_violation"]
        report["max_workspace_violation"] = report["motion_workspace"]["physics"]["max_violation"]
        for name in (
            "measured_joint_min_rad",
            "measured_joint_max_rad",
            "target_joint_min_rad",
            "target_joint_max_rad",
            "measured_joint_envelope_excess_rad",
            "target_joint_envelope_excess_rad",
        ):
            aggregate = min if name.endswith("min_rad") else max
            sampled_rows = [row for row in rows if row[name] is not None]
            report[name] = [aggregate(row[name][j] for row in sampled_rows) for j in range(6)] if sampled_rows else None
        report["passed"] = (
            report["error"] is None
            and report["reset_count"] == report["reset_attempts"] == 0
            and report["all_completed_goals_settled"]
            and report["joint_envelope_violation_samples"] == 0
            and report["target_envelope_violation_samples"] == 0
            and report["workspace_violation_samples"] == 0
            and report["policy_workspace_violation_samples"] == 0
            and report["controller_joint_limit_violation_samples"] == 0
            and report["controller_target_limit_violation_samples"] == 0
            and report["simulation_joint_limit_violation_samples"] == 0
            and report["simulation_target_limit_violation_samples"] == 0
            and (steps == 0 or step_count == steps)
        )
    return step_count


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils import instantiate

    from isaaclab_tasks.utils import parse_env_cfg

    import mycobot450_isaaclab.tasks  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported TorchScript policy.pt to play.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of simulated arms (default: 1).")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many policy steps; 0 runs until closed.")
    parser.add_argument("--goal-plan", type=Path, help="Cycle a validated deployment goal plan without episode resets.")
    parser.add_argument(
        "--random-goals",
        action="store_true",
        help="Sample default four-family goals at runtime; use plan bounds/interval only",
    )
    parser.add_argument("--goal-seed", type=int, help="Replay the same stream as deploy.py; omitted chooses a new seed")
    parser.add_argument(
        "--hardware-config",
        type=Path,
        help="Apply a staged development motion-guard profile in simulation; never connects hardware.",
    )
    parser.add_argument(
        "--report", type=Path, help="New result JSON path; goal-plan playback defaults to timestamped logs."
    )
    parser.add_argument(
        "--velocity-source",
        choices=("native", "finite_difference"),
        default="native",
        help="Native PhysX velocity or consecutive measured positions; differences require goal-plan command holds.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        choices=(1.0, 0.5, 0.25),
        default=1.0,
        help="Nominal policy seconds per simulated second; below 1 uses 15 Hz held commands with --goal-plan.",
    )
    parser.add_argument(
        "--target-update-decimation", type=int, help="Override target holds: 1 for 120 Hz or 4 for 30 Hz endpoints."
    )
    parser.add_argument(
        "--real-time", action=argparse.BooleanOptionalAction, default=True, help="Pace playback to simulation time."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args, overrides = parser.parse_known_args()
    if args.num_envs < 1 or args.steps < 0:
        parser.error("--num_envs must be positive and --steps must be nonnegative")
    if args.goal_plan and args.num_envs != 1:
        parser.error("--goal-plan requires --num_envs 1")
    if args.report and not args.goal_plan:
        parser.error("--report requires --goal-plan")
    if args.random_goals and not args.goal_plan:
        parser.error("--random-goals requires --goal-plan")
    if args.goal_seed is not None and not args.random_goals:
        parser.error("--goal-seed requires --random-goals")
    if args.goal_plan and args.report is None:
        args.report = (
            Path(__file__).resolve().parents[1]
            / "outputs/reach/simulation"
            / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            / "report.json"
        )
    if args.report and args.report.exists():
        parser.error("Use a new --report path; simulation results are never overwritten")
    if args.report:
        print(f"[LOG] Simulation report: {args.report.resolve()}", flush=True)
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        parser.error(f"Policy file does not exist: {policy_path}")
    policy = load_policy(policy_path)
    plan = load_goal_plan(args.goal_plan, policy_path) if args.goal_plan else None
    goal_sampler = (
        DefaultGoalSampler(plan, ReachController(plan.contract), plan.contract["kinematic_chain"], seed=args.goal_seed)
        if args.random_goals
        else None
    )
    if goal_sampler is not None:
        print(f"[INFO] Runtime goal sampling: {json.dumps(goal_sampler.report())}", flush=True)
    try:
        motion_guard = (
            load_motion_guard(
                args.hardware_config,
                plan,
                time_scale=args.time_scale,
                velocity_source=args.velocity_source,
                steps=args.steps,
            )
            if args.hardware_config is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs, overrides=overrides)
    env_cfg.play_mode()
    env_cfg.terminations.time_out = None
    try:
        configure_plan_timing(
            env_cfg, plan, time_scale=args.time_scale, target_update_decimation=args.target_update_decimation
        )
        if motion_guard is not None:
            env_cfg.actions.arm_action.max_tracking_error = motion_guard.target_lead_limit_rad
            env_cfg.actions.arm_action.scale *= motion_guard.action_gain
        validate_velocity_source(
            args.velocity_source,
            plan=plan,
            target_hold_steps=env_cfg.actions.arm_action.target_update_decimation,
            policy_steps=env_cfg.decimation,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if plan is not None:
        env_cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
        # The one startup reset matches the deployment home; goal switches never reset.
        params = env_cfg.events.reset_robot_joints.params
        params.update(position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), reached_state_probability=0.0)
    if args.device == "cpu":
        env_cfg.sim.use_fabric = False

    with launch_simulation(env_cfg, args):
        env = instantiate(env_cfg)
        report = {}
        try:
            policy = policy.to(env.device)
            print(f"[INFO] TorchScript policy: {policy_path}", flush=True)
            print(f"[INFO] {TASK}: {env.num_envs} env(s), policy period {env.step_dt:.6f} s. Ctrl+C to exit.")
            if plan is None:
                count = play_policy(env, policy, steps=args.steps, real_time=args.real_time)
            else:
                report["goal_plan"] = str(args.goal_plan.resolve())
                report["target_update_decimation"] = env_cfg.actions.arm_action.target_update_decimation
                report["integration_substeps"] = env_cfg.actions.arm_action.integration_substeps
                report["command_hz"] = 1 / (env.physics_dt * env_cfg.actions.arm_action.target_update_decimation)
                count = play_goal_plan(
                    env,
                    policy,
                    plan,
                    steps=args.steps,
                    real_time=args.real_time,
                    report=report,
                    time_scale=args.time_scale,
                    velocity_source=args.velocity_source,
                    motion_guard=motion_guard,
                    goal_sampler=goal_sampler,
                )
            print(f"[INFO] Finished {count} policy steps.", flush=True)
        except KeyboardInterrupt:
            print("\n[INFO] Playback interrupted.", flush=True)
        finally:
            try:
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    with args.report.open("x") as file:
                        json.dump(report, file, indent=2, allow_nan=False)
                        file.write("\n")
                        file.flush()
            finally:
                env.close()
        if plan is not None:
            print(f"[RESULT] Goal-plan validation passed: {report.get('passed', False)}", flush=True)
            if not report.get("passed", False):
                parser.exit(1, "Goal-plan validation failed; inspect the result before hardware execution.\n")


if __name__ == "__main__":
    main()
