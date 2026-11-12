# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.11.0", "pymycobot==4.0.7", "numpy==2.3.1"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reach deployment: self-check, read-only timing probe, and continuous SDK control.

No ROS, Isaac Lab, simulator or camera is required. `run --home` first executes
the checked single-target home and starts the policy only after success. `run` is
read-only unless --execute is supplied AND hardware validation is recorded in the
configuration. A successful read-only probe does not certify command streaming.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import uuid
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch

if __package__:
    from .reach_goals import DefaultGoalSampler, GoalPlan, sample_default_plan, sample_plan
    from .reach_home import HomeMotionBounds, HomeTelemetry, run_direct_home
    from .reach_motion_guard import MotionGuard
    from .reach_policy import JOINT_NAMES, ReachPolicy, pose_error, vector
    from .reach_telemetry import TelemetryPublisher
    from .robot_interface import Pro450Interface, prepare_stream
else:
    from reach_goals import DefaultGoalSampler, GoalPlan, sample_default_plan, sample_plan
    from reach_home import HomeMotionBounds, HomeTelemetry, run_direct_home
    from reach_motion_guard import MotionGuard
    from reach_policy import JOINT_NAMES, ReachPolicy, pose_error, vector
    from reach_telemetry import TelemetryPublisher
    from robot_interface import Pro450Interface, prepare_stream


def hardware_template(controller):
    return {
        "joint_order": JOINT_NAMES,
        "sign": [1] * 6,
        "offset_rad": [0.0] * 6,
        "joint_limits_rad": controller.limits.tolist(),
        "calibration_verified": False,
        "streaming_and_stop_verified": False,
        "verified_command_hz": None,
        "verification_notes": "",
        "sdk_speed": 10,
        "gripper_id": 14,
        "start_tolerance_rad": 0.05,
        "max_tracking_error_rad": 0.15,
        "max_measured_velocity_rad_s": 1.5,
        "max_goal_distance_m": 0.05,
        "max_goal_angle_rad": 0.25,
        "home_motion_workspaces_m": [],
    }


class HardwareConfig:
    def __init__(self, data, controller):
        if data["joint_order"] != JOINT_NAMES:
            raise ValueError("Hardware joint order must be joint1..joint6")
        self.data = data
        self.sign = vector(data["sign"], 6, "joint signs")
        self.offset = vector(data["offset_rad"], 6, "joint offsets")
        if not (self.sign.abs() == 1).all():
            raise ValueError("Joint signs must be +1 or -1")
        self.limits = torch.tensor(data["joint_limits_rad"], dtype=torch.float32)
        if (
            self.limits.shape != (6, 2)
            or not torch.isfinite(self.limits).all()
            or not (self.limits[:, 0] < self.limits[:, 1]).all()
            or (self.limits[:, 0] < controller.limits[:, 0]).any()
            or (self.limits[:, 1] > controller.limits[:, 1]).any()
        ):
            raise ValueError("Hardware limits must be ordered and inside the training limits")
        for name in (
            "start_tolerance_rad",
            "max_tracking_error_rad",
            "max_measured_velocity_rad_s",
            "max_goal_distance_m",
            "max_goal_angle_rad",
        ):
            if not isinstance(data[name], (float, int)) or not math.isfinite(data[name]) or data[name] <= 0:
                raise ValueError(f"Invalid {name}")
        if type(data["sdk_speed"]) is not int or not 1 <= data["sdk_speed"] <= 100:
            raise ValueError("sdk_speed must be an integer in 1..100 (not rad/s)")
        if type(data["gripper_id"]) is not int or not 1 <= data["gripper_id"] <= 254:
            raise ValueError("Invalid gripper ID")
        if "motion_guard" in data:
            MotionGuard.validate(data)
        elif any(name in data for name in ("max_new_target_error_rad", "target_lead_limit_rad", "action_gain")):
            raise ValueError("Runtime target limits and action_gain require motion_guard configuration")

    def require_execution_ready(self, command_hz=120.0):
        if "motion_guard" in self.data:
            raise ValueError("Development motion guards require --continuous --response-trial")
        if self.data["calibration_verified"] is not True:
            raise ValueError("Verify joint signs, zeros and limits before enabling execution")
        if self.data["streaming_and_stop_verified"] is not True or not self.data["verification_notes"].strip():
            raise ValueError("Command streaming and stopping are not verified; use probe/run without --execute")
        verified = self.data.get("verified_command_hz")
        if type(verified) not in (int, float) or not math.isfinite(verified) or not math.isclose(verified, command_hz):
            raise ValueError(f"Hardware verification must explicitly cover {command_hz:g} Hz commands")

    def require_continuous_ready(self, command_hz, time_scale=1.0, *, response_trial=False, duration=None):
        """Explicit continuous exploration; it does not certify indefinite operation."""
        if self.data["calibration_verified"] is not True:
            raise ValueError("Verify the SDK-to-simulation joint mapping before continuous execution")
        if type(time_scale) not in (int, float) or time_scale not in (1.0, 0.5, 0.25):
            raise ValueError("time_scale must be 1, 0.5 or 0.25")
        if not math.isfinite(command_hz) or not (
            math.isclose(command_hz, 30.0) or (math.isclose(command_hz, 15.0) and time_scale in (0.5, 0.25))
        ):
            raise ValueError("Supervised continuous execution requires 30 Hz, or 15 Hz with time scaling")
        if type(response_trial) is not bool:
            raise ValueError("response_trial must be boolean")
        staged = "motion_guard" in self.data
        if staged:
            MotionGuard.validate(self.data)
            if not response_trial:
                raise ValueError("Development motion guards require --response-trial")
        if response_trial:
            if (
                type(duration) not in (int, float)
                or not math.isfinite(duration)
                or not 0 <= duration <= 600
                or not math.isclose(command_hz, 30.0)
                or time_scale != 1.0
            ):
                raise ValueError(
                    "Response trials require 30 Hz, time_scale=1 and duration in [0, 600] seconds (0 until Ctrl+C)"
                )
            if staged:
                cfg = self.data["motion_guard"]
                # The original-speed policy reaches 0.8 rad/s; permit bounded feedback overshoot.
                if (
                    self.data["sdk_speed"] > 30
                    or self.data["max_measured_velocity_rad_s"] > 1.0
                    or self.data["max_tracking_error_rad"] > 0.25
                    or self.data["max_new_target_error_rad"] > 0.28
                    or cfg["tracking_soft_rad"] > 0.15
                    or cfg["new_target_soft_rad"] > 0.18
                    or cfg["speed_soft_rad_s"] > 0.85
                    or not 0.1 <= cfg["tracking_grace_s"] <= 0.3
                    or not 0.1 <= cfg["speed_window_s"] <= 0.2
                ):
                    raise ValueError("Development motion guard configuration exceeds the bounded response-trial limits")
            elif (
                self.data["sdk_speed"] > 15
                or self.data["max_measured_velocity_rad_s"] > 0.5
                or self.data["max_tracking_error_rad"] > 0.15
            ):
                raise ValueError("Response trials require SDK speed <= 15, measured speed <= 0.5 and tracking <= 0.15")
        elif self.data["sdk_speed"] > 10:
            raise ValueError("Supervised continuous execution requires SDK speed <= 10")

    def to_sim(self, degrees):
        return self.sign * torch.deg2rad(vector(degrees, 6, "SDK joint feedback")) + self.offset

    def to_sdk(self, radians):
        self.check_limits(radians)
        return torch.rad2deg(self.sign * (radians - self.offset)).tolist()

    def check_limits(self, joints):
        if not torch.isfinite(joints).all() or ((joints < self.limits[:, 0]) | (joints > self.limits[:, 1])).any():
            raise ValueError("Joints/targets are outside calibrated limits; refusing to silently clip the policy")


def summarize_timings(values):
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "count": len(values),
        "p50_ms": tensor.quantile(0.5).item() * 1000,
        "p95_ms": tensor.quantile(0.95).item() * 1000,
        "max_ms": tensor.max().item() * 1000,
    }


def probe(robot, duration, rate, clock=time.monotonic, sleep=time.sleep):
    """Only get_angles requests, with absolute deadlines and no burst catch-up."""
    period = 1 / rate
    start = deadline = clock()
    latencies, intervals, previous, missed = [], [], None, 0
    last = None
    while clock() - start < duration:
        sleep(max(0, deadline - clock()))
        last, received, latency = robot.read()
        latencies.append(latency)
        if previous is not None:
            intervals.append(received - previous)
        previous = received
        deadline += period
        if clock() > deadline:
            missed += 1
            deadline = clock() + period
    return {
        "mode": "read_only_probe",
        "requested_hz": rate,
        "actual_hz": len(latencies) / (clock() - start),
        "response": summarize_timings(latencies),
        "sample_interval": summarize_timings(intervals) if intervals else None,
        "missed_deadlines": missed,
        "last_angles_deg": last,
        "command_streaming_verified": False,
        "note": "Host receipt timestamps only. get_angles timing does not measure send_angles or servo response.",
    }


def command_schedule(controller, command_hz=None, time_scale=1.0):
    """Schedule nominal controller substeps on the real command clock."""
    hz = 1 / controller.dt if command_hz is None else command_hz
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("Command frequency must be finite and positive")
    if type(time_scale) not in (int, float) or time_scale not in (1.0, 0.5, 0.25):
        raise ValueError("time_scale must be 1, 0.5 or 0.25")
    if time_scale != 1.0 and not any(math.isclose(hz, allowed) for allowed in (15.0, 30.0)):
        raise ValueError("Time scaling requires 15 or 30 Hz real commands")
    period = 1 / hz
    if not any(math.isclose(period, allowed) for allowed in (controller.dt, controller.policy_dt)) and not (
        math.isclose(hz, 15.0) and time_scale in (0.5, 0.25)
    ):
        raise ValueError("Commands require the original physics/policy rate, or 15 Hz with time scaling")
    policy_period = period * time_scale
    integration_steps = round(policy_period / controller.dt)
    policy_every = round(controller.policy_dt / policy_period)
    if (
        integration_steps < 1
        or policy_every < 1
        or not math.isclose(integration_steps * controller.dt, policy_period)
        or not math.isclose(policy_every * policy_period, controller.policy_dt)
    ):
        raise ValueError("Time scaling must preserve whole controller substeps and whole policy intervals")
    return period, integration_steps, policy_every


class PolicyTrialError(RuntimeError):
    """Preserve the control failure and independent stop outcome of a physical trial."""

    def __init__(self, report):
        self.report = report
        super().__init__(report["error"] or (report["stop"] or {}).get("error") or "Policy trial did not complete")


def run_control(
    policy,
    robot,
    hardware,
    goal_pose,
    duration,
    *,
    execute=False,
    continuous=False,
    goal_plan=None,
    goal_sampler=None,
    command_hz=None,
    time_scale=1.0,
    allow_goal_jumps=False,
    response_trial=False,
    trace=None,
    telemetry=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    controller = policy.controller
    motion_guard = MotionGuard(hardware.data) if "motion_guard" in hardware.data else None
    lead_limit = hardware.data.get("target_lead_limit_rad")
    action_gain = hardware.data.get("action_gain", 1.0)
    vr_mode = None
    if type(allow_goal_jumps) is not bool or (allow_goal_jumps and not continuous):
        raise ValueError("allow_goal_jumps requires continuous goal-plan execution")
    if type(response_trial) is not bool or (response_trial and (not continuous or not execute)):
        raise ValueError("response_trial requires continuous execution")
    if goal_sampler is not None and (not continuous or not allow_goal_jumps):
        raise ValueError("Random goals require continuous execution and allow_goal_jumps")
    period, integration_steps, policy_every = command_schedule(controller, command_hz, time_scale)
    if continuous:
        if goal_pose is not None or goal_plan is None:
            raise ValueError("Continuous mode requires a goal plan, without --goal")
        if (
            not math.isfinite(duration)
            or duration < 0
            or not any(math.isclose(1 / period, allowed) for allowed in (15.0, 30.0))
        ):
            raise ValueError("Continuous mode requires 15/30 Hz and a nonnegative duration (0 until interrupted)")
    elif goal_plan is not None:
        raise ValueError("A goal plan requires continuous mode")
    if execute:
        if continuous:
            hardware.require_continuous_ready(1 / period, time_scale, response_trial=response_trial, duration=duration)
        else:
            hardware.require_execution_ready(1 / period)
        robot.check_ready(hardware.data["gripper_id"])
        if continuous:
            vr_mode = robot.call("get_vr_mode")
            if type(vr_mode) is not int or vr_mode != 1:
                raise RuntimeError(
                    "Continuous Reach requires VR buffer mode 1; run reach_home --current-start --execute first"
                )
    # Get two distinct replies to initialize the finite-difference velocity estimate.
    degrees, previous_time, _ = robot.read()
    previous_q = hardware.to_sim(degrees)
    sleep(period)
    degrees, received, _ = robot.read()
    q = hardware.to_sim(degrees)
    hardware.check_limits(q)
    if received <= previous_time:
        raise ValueError("Non-increasing joint feedback timestamp")
    initial_velocity = (q - previous_q) / (received - previous_time)
    if execute and (
        (q - controller.home).abs().max() > hardware.data["start_tolerance_rad"] or initial_velocity.abs().max() > 0.05
    ):
        raise ValueError("Start stationary near the calibrated training home; this program does not home the robot")
    controller.reset(q)
    initial_q = q.clone()
    initial_pose = controller.fk(q)
    goal_index = goal_switches = 0
    goal = controller.goal(
        goal_sampler.initial_goal
        if goal_sampler is not None
        else (goal_plan.validate_goal(0) if continuous else (initial_pose if goal_pose is None else goal_pose))
    )
    if continuous and ((q < goal_plan.joint_bounds[:, 0]) | (q > goal_plan.joint_bounds[:, 1])).any():
        raise ValueError("Initial joints are outside the continuous goal plan's joint envelope")
    if (
        continuous
        and (
            (initial_pose[:3] < goal_plan.motion_workspace[:, 0])
            | (initial_pose[:3] > goal_plan.motion_workspace[:, 1])
        ).any()
    ):
        raise ValueError("Initial mount position is outside the continuous goal plan's motion workspace")
    if execute and not (goal_sampler is not None and allow_goal_jumps):
        error = pose_error(initial_pose, goal)
        if (
            error[:3].norm() > hardware.data["max_goal_distance_m"]
            or error[3:].norm() > hardware.data["max_goal_angle_rad"]
        ):
            raise ValueError("Initial goal exceeds the configured local commissioning range")
    previous_q, previous_time = q, received
    start = deadline = clock()
    end_time = math.inf if continuous and duration == 0 else start + duration
    goal_interval_real_s = goal_plan.interval_s / time_scale if continuous else None
    next_goal_time = start + goal_interval_real_s if continuous else math.inf
    tick, writes, missed = 0, 0, 0
    skipped = 0
    previous_overrun = False

    def history():
        return deque(maxlen=4096) if continuous else []

    work_times = history()
    feedback_times, policy_times, acknowledgement_times = history(), history(), history()
    pose_samples = history()
    combined_feedback = continuous and callable(getattr(robot, "read_state", None))
    attempted_writes = 0
    last_attempted_target = None
    max_steps = math.ceil(duration / period) if continuous and duration > 0 else None
    failure = None
    failure_sample = None
    interrupted = False
    stop_report = None
    motion_attempted = False
    run_id = uuid.uuid4().hex
    telemetry_seq = 0
    policy_sample = {}
    dq = initial_velocity
    mount_min = torch.full((3,), math.inf)
    mount_max = torch.full((3,), -math.inf)
    workspace_samples = goal_workspace_exits = motion_workspace_exits = 0

    def tracking_error(target, message):
        difference = (target - q).abs()
        joint = int(difference.argmax())
        return RuntimeError(
            f"{message}: joint{joint + 1}, measured={q[joint].item():.6f} rad, "
            f"target={target[joint].item():.6f} rad, error={difference[joint].item():.6f} rad, "
            f"limit={hardware.data['max_tracking_error_rad']:.6f} rad"
        )

    def publish(state, error=None):
        nonlocal telemetry_seq
        if telemetry is not None:
            telemetry.publish(
                {
                    "schema": 1,
                    "type": "reach_state",
                    "run_id": run_id,
                    "seq": telemetry_seq,
                    "monotonic_s": received,
                    "published_monotonic_s": clock(),
                    "state": state,
                    "q_rad": q.tolist(),
                    "dq_rad_s": dq.tolist(),
                    "target_rad": controller.target.tolist(),
                    "goal_xyz_xyzw": goal.tolist(),
                    "goal_index": goal_index,
                    "sent": execute and motion_attempted,
                    "policy_sha256": getattr(policy, "contract", {}).get("policy_sha256", "unknown"),
                    "error": error,
                    **policy_sample,
                }
            )
            telemetry_seq += 1

    try:
        while clock() < end_time and (max_steps is None or tick + skipped < max_steps):
            sleep(max(0, deadline - clock()))
            if continuous and clock() >= end_time:
                break
            if execute and clock() >= deadline + period:
                raise TimeoutError("Missed control deadline; refusing to send a late target")
            work_start = clock()
            degrees, received, latency = robot.read_state() if combined_feedback else robot.read()
            feedback_finished = clock()
            if continuous:
                feedback_times.append(feedback_finished - work_start)
            q = hardware.to_sim(degrees)
            hardware.check_limits(q)
            if continuous and ((q < goal_plan.joint_bounds[:, 0]) | (q > goal_plan.joint_bounds[:, 1])).any():
                raise RuntimeError("Measured joints left the continuous goal plan's joint envelope")
            sample_dt = received - previous_time
            if sample_dt <= 0 or sample_dt > 0.1:
                raise TimeoutError("Joint feedback gap is invalid or exceeds 100 ms")
            dq = (q - previous_q) / sample_dt
            previous_q, previous_time = q, received
            guard_warnings = []
            if execute:
                if motion_guard is not None:
                    guard_warnings.extend(motion_guard.check_feedback(q, dq, controller.target, sample_dt, received))
                elif dq.abs().max() > hardware.data["max_measured_velocity_rad_s"]:
                    joint = int(dq.abs().argmax())
                    raise RuntimeError(
                        "Measured joint velocity exceeds the commissioning limit: "
                        f"joint{joint + 1}={dq[joint].item():.6f} rad/s, "
                        f"limit={hardware.data['max_measured_velocity_rad_s']:.6f} rad/s, "
                        f"sample_dt={sample_dt:.6f} s, angles_deg={degrees!r}"
                    )
                if (
                    motion_guard is None
                    and (controller.target - q).abs().max() > hardware.data["max_tracking_error_rad"]
                ):
                    raise tracking_error(controller.target, "Joint tracking error exceeds the commissioning limit")
            if tick % policy_every == 0:
                # Include the same read-only fault-query cost in dry runs and execution.
                if not combined_feedback:
                    robot.check_fault()
                if continuous and clock() >= next_goal_time:
                    candidate_index = (
                        goal_index + 1 if goal_sampler is not None else (goal_index + 1) % len(goal_plan.goals)
                    )
                    candidate = controller.goal(
                        goal_sampler.sample() if goal_sampler is not None else goal_plan.validate_goal(candidate_index)
                    )
                    transition_error = pose_error(controller.fk(q), candidate)
                    if (
                        execute
                        and not allow_goal_jumps
                        and (
                            transition_error[:3].norm() > hardware.data["max_goal_distance_m"]
                            or transition_error[3:].norm() > hardware.data["max_goal_angle_rad"]
                        )
                    ):
                        raise RuntimeError("Next goal exceeds the transition range from measured pose")
                    goal, goal_index = candidate, candidate_index
                    goal_switches += 1
                    next_goal_time += goal_interval_real_s
                observation = controller.observe(q, dq / time_scale, goal)
                if telemetry is not None:
                    policy_observation = observation[0].tolist()
                action = policy.infer(observation)
                controller.process(action)
                if telemetry is not None:
                    # Preserve actual inference I/O; later packets retain the same sample identity.
                    policy_sample = {
                        "policy_observation": policy_observation,
                        "policy_action": action.tolist(),
                        "policy_step": policy_sample.get("policy_step", -1) + 1,
                        "policy_monotonic_s": received,
                    }
            # Substeps and filter state remain in nominal policy time. The real
            # command clock and every hardware guard retain their original units.
            for _ in range(integration_steps):
                target = (
                    controller.step(q, target_lead_limit_rad=lead_limit, action_gain=action_gain)
                    if lead_limit is not None or action_gain != 1
                    else controller.step()
                )
            if continuous:
                # Off-inference ticks still need the current measured pose. Never
                # reuse an observation cached before the latest joint feedback.
                error = observation[0, 25:31] if tick % policy_every == 0 else pose_error(controller.fk(q), goal)
                if ((target < goal_plan.joint_bounds[:, 0]) | (target > goal_plan.joint_bounds[:, 1])).any():
                    raise RuntimeError("Policy target left the continuous goal plan's joint envelope")
                measured_xyz = goal[:3] - error[:3]
                mount_min = torch.minimum(mount_min, measured_xyz)
                mount_max = torch.maximum(mount_max, measured_xyz)
                workspace_samples += 1
                goal_workspace_exits += int(
                    (
                        (measured_xyz < goal_plan.goal_workspace[:, 0])
                        | (measured_xyz > goal_plan.goal_workspace[:, 1])
                    ).any()
                )
                outside_motion = (
                    (measured_xyz < goal_plan.motion_workspace[:, 0])
                    | (measured_xyz > goal_plan.motion_workspace[:, 1])
                ).any()
                motion_workspace_exits += int(outside_motion)
                if outside_motion:
                    raise RuntimeError(
                        "Measured mount position left the continuous goal plan's motion workspace: "
                        f"xyz_m={measured_xyz.tolist()}, bounds_m={goal_plan.motion_workspace.tolist()}"
                    )
                if execute and motion_guard is not None:
                    guard_warnings.extend(motion_guard.check_target(q, target, received))
                elif (target - q).abs().max() > hardware.data["max_tracking_error_rad"]:
                    raise tracking_error(target, "New policy target exceeds the tracking-error limit")
                pose_samples.append(
                    {
                        "t_s": received - start,
                        "goal_index": goal_index,
                        "position_error_m": error[:3].norm().item(),
                        "orientation_error_rad": error[3:].norm().item(),
                        "max_joint_speed_rad_s": dq.abs().max().item(),
                        "max_target_error_rad": (target - q).abs().max().item(),
                        "measured_mount_xyz_m": (goal[:3] - error[:3]).tolist(),
                    }
                )
            if execute:
                angles = hardware.to_sdk(target)
                before_send = clock()
                if continuous:
                    policy_times.append(before_send - feedback_finished)
                if response_trial and before_send - received > period:
                    raise TimeoutError("Feedback is older than one control period before send")
                # A development trial may consume one extra slot, but never replay targets to catch up.
                cycle_deadline = deadline + period * (2 if response_trial else 1)
                send_deadline = min(cycle_deadline, end_time) if continuous else cycle_deadline
                remaining = send_deadline - clock()
                if remaining <= 0:
                    raise TimeoutError("No command budget left in this control period")
                motion_attempted = True  # Even an unacknowledged send might have moved the robot.
                attempted_writes += 1
                last_attempted_target = angles
                robot.send(angles, hardware.data["sdk_speed"], timeout=remaining)
                if continuous:
                    acknowledgement_times.append(clock() - before_send)
                writes += 1
            for warning in guard_warnings:
                print(json.dumps({"event": "motion_guard_warning", **warning}), file=sys.stderr, flush=True)
            work_times.append(clock() - work_start)
            publish("running" if execute else "read_only")
            if trace is not None:
                trace.write(
                    json.dumps(
                        {
                            "t_s": received - start,
                            "policy_time_s": (received - start) * time_scale,
                            "q_rad": q.tolist(),
                            "dq_rad_s": dq.tolist(),
                            "target_rad": target.tolist(),
                            "action": controller.previous_action.tolist(),
                            "lead_limited_substeps": controller.lead_limited_substeps,
                            "goal_xyz_xyzw": goal.tolist(),
                            "goal_index": goal_index,
                            "sampled_goal": goal_sampler.last_goal if goal_sampler is not None else None,
                            "read_latency_s": latency,
                            "sent": execute,
                            "motion_guard": motion_guard.snapshot() if motion_guard is not None else None,
                            "guard_warnings": guard_warnings,
                            **(
                                {
                                    "position_error_m": pose_samples[-1]["position_error_m"],
                                    "orientation_error_rad": pose_samples[-1]["orientation_error_rad"],
                                    "measured_mount_xyz_m": pose_samples[-1]["measured_mount_xyz_m"],
                                }
                                if continuous
                                else {}
                            ),
                            "timings_s": (
                                {
                                    "start_lateness": work_start - deadline,
                                    "feedback": feedback_times[-1],
                                    "policy_and_controller": policy_times[-1],
                                    "send_acknowledgement": acknowledgement_times[-1],
                                    "work": work_times[-1],
                                }
                                if continuous and execute
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
            tick += 1
            deadline += period
            if clock() > deadline:
                missed += 1
                if execute:
                    now = clock()
                    if not response_trial or now >= deadline + period or previous_overrun:
                        raise TimeoutError("Control period overrun; stopping without replaying queued targets")
                    skipped += 1
                    previous_overrun = True
                    deadline += period
                    print(
                        json.dumps({"event": "control_slot_skipped", "t_s": now - start}), file=sys.stderr, flush=True
                    )
                else:
                    deadline = clock() + period
            else:
                previous_overrun = False
    except BaseException as exc:
        if not continuous:
            raise
        if continuous and isinstance(exc, KeyboardInterrupt):
            interrupted = True
        else:
            failure = exc
            # Capture the failing cycle before stop feedback updates q/dq. A new
            # integrated target may have been refused and must not replace the
            # last attempted send or the last successful control sample.
            failure_sample = {
                "goal_index": goal_index,
                "t_s": clock() - start,
                "feedback_t_s": received - start,
                "q_rad": q.tolist(),
                "target_rad": controller.target.tolist(),
                "dq_rad_s": dq.tolist(),
                "previous_action": controller.previous_action.tolist(),
                "goal_xyz_xyzw": goal.tolist(),
                "measured_mount_xyz_m": controller.fk(q)[:3].tolist(),
                "motion_guard": motion_guard.snapshot() if motion_guard is not None else None,
            }
    finally:
        if motion_attempted:
            if continuous:
                try:
                    stop_report = robot.stop_and_observe(clock=clock, sleep=sleep)
                except BaseException as exc:
                    stop_report = {
                        "stop_acknowledged": False,
                        "post_stop_stationary": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                robot.stop()  # Keeps power; no release-servos/power-off on exit.
        if stop_report and stop_report.get("samples"):
            last = stop_report["samples"][-1]
            if "q_deg" in last:
                q = hardware.to_sim(last["q_deg"])
                dq = torch.zeros_like(q)
                received = stop_report["stop_request_monotonic_s"] + last["received_after_stop_s"]
        try:
            publish(
                "fault"
                if failure or (stop_report and stop_report.get("error"))
                else ("stopped" if execute else "read_only"),
                str(failure) if failure else ((stop_report or {}).get("error")),
            )
        except BaseException as exc:
            if failure is None:
                failure = exc
    result = {
        "run_id": run_id,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "mode": "continuous_policy" if continuous else ("execute" if execute else "read_only_inference"),
        "command_hz": 1 / period,
        "time_scale": time_scale,
        "allow_goal_jumps": allow_goal_jumps,
        "response_trial": response_trial,
        "vr_mode": vr_mode,
        "target_lead_limit_rad": lead_limit,
        "action_gain": action_gain,
        "effective_velocity_scale_rad_s": controller.scale * action_gain,
        "lead_limited_substeps": controller.lead_limited_substeps,
        "integration_hz": time_scale / controller.dt,
        "policy_hz": time_scale / controller.policy_dt,
        "nominal_integration_hz": 1 / controller.dt,
        "nominal_policy_hz": 1 / controller.policy_dt,
        "goal_interval_real_s": goal_interval_real_s,
        "target_schedule": (
            "time_scaled_substeps"
            if time_scale != 1.0
            else ("physics_step" if integration_steps == 1 else "policy_endpoint_hold")
        ),
        "ticks": tick,
        "writes": writes,
        "missed_deadlines": missed,
        "skipped_control_slots": skipped,
        "control_work": summarize_timings(work_times),
        "note": "Dry-run targets are virtual; they do not establish tracking success or streaming capability.",
    }
    if continuous:
        end = pose_samples[-1]["t_s"] if pose_samples else 0
        last_second = [row for row in pose_samples if row["t_s"] >= end - 1 and row["goal_index"] == goal_index]
        result.update(
            passed=(
                failure is None
                and (interrupted or max_steps is None or tick + skipped == max_steps)
                and (
                    not execute
                    or bool(
                        stop_report
                        and stop_report["stop_acknowledged"]
                        and stop_report["post_stop_stationary"]
                        and not stop_report.get("error")
                    )
                )
            ),
            error=(stop_report or {}).get("error") if failure is None else f"{type(failure).__name__}: {failure}",
            failure_sample=failure_sample,
            attempted_writes=attempted_writes,
            planned_writes=max_steps,
            interrupted=interrupted,
            goal_switches=goal_switches,
            goal_index=goal_index,
            goal_plan=goal_plan.data,
            goal_sampling=goal_sampler.report() if goal_sampler is not None else None,
            workspace_observations={
                "goal_workspace_m": goal_plan.goal_workspace.tolist(),
                "motion_workspace_m": goal_plan.motion_workspace.tolist(),
                "samples": workspace_samples,
                "position_min_m": mount_min.tolist() if workspace_samples else None,
                "position_max_m": mount_max.tolist() if workspace_samples else None,
                "outside_goal_workspace_samples": goal_workspace_exits,
                "outside_motion_workspace_samples": motion_workspace_exits,
                "note": "Measured mount positions at command feedback times; only motion bounds stop execution.",
            },
            hardware_config=hardware.data,
            motion_guard=motion_guard.report() if motion_guard is not None else None,
            timing_window_capacity=4096,
            last_attempted_target_deg=last_attempted_target,
            policy_sha256=getattr(policy, "contract", {}).get("policy_sha256"),
            initial_joint_position_rad=initial_q.tolist(),
            initial_pose_xyz_xyzw=initial_pose.tolist(),
            goal_xyz_xyzw=goal.tolist(),
            feedback_mode="combined_status_and_angles" if combined_feedback else "separate_status_and_angles",
            stage_timings={
                "feedback": summarize_timings(feedback_times),
                "policy_and_controller": summarize_timings(policy_times),
                "send_acknowledgement": summarize_timings(acknowledgement_times),
            },
            pose_samples=list(pose_samples),
            final_control_sample=pose_samples[-1] if pose_samples else None,
            final_second_tracking_success=(
                bool(last_second)
                and len(last_second) >= math.floor(min(end, 1) / period) - 1
                and all(
                    row["position_error_m"] < 0.02
                    and row["orientation_error_rad"] < 0.15
                    and row["max_joint_speed_rad_s"] < 0.1
                    for row in last_second
                )
            ),
            stop=stop_report,
            telemetry_dropped=getattr(telemetry, "dropped", 0),
            note=(
                "Supervised policy execution. Pose errors use measured joints and model FK, "
                "not external TCP measurement; timing/pose windows are bounded during continuous operation."
            ),
        )
        if not result["passed"]:
            raise PolicyTrialError(result) from failure
    return result


def run_startup_home(robot, policy, bounds, goal, *, sdk_speed=10, open_gripper=False, telemetry=None):
    """Reuse the checked home and its measured telemetry before policy execution."""
    display = HomeTelemetry(telemetry, policy.contract["policy_sha256"], goal, execute=True) if telemetry else None
    preparation = None
    report = {"mode": "direct_home", "passed": False, "error": None}
    try:
        preparation = prepare_stream(robot, low_latency=True)
        print(
            f"[HOME] Checking current joints and path before homing, SDK speed={sdk_speed}.",
            file=sys.stderr,
            flush=True,
        )
        report = run_direct_home(
            robot,
            policy.contract,
            bounds,
            execute=True,
            sdk_speed=sdk_speed,
            open_gripper=open_gripper,
            progress=display,
            progress_interval=0.0,
        )
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            passed=False, error=f"{type(exc).__name__}: {exc}", interrupted=isinstance(exc, KeyboardInterrupt)
        )
    report["stream_preparation"] = preparation
    if not report["passed"]:
        report["error"] = report.get("error") or "Home arrival and stop verification failed"
    if display is not None:
        display.finish(report)
    if not report["passed"]:
        raise PolicyTrialError({"mode": "home_then_policy", "passed": False, "error": report["error"], "home": report})
    print("[HOME] Settled and verified; starting policy.", file=sys.stderr, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("self-check", "init-config", "sample-goals", "run"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        if name in ("init-config", "sample-goals"):
            p.add_argument("--output", type=Path, required=True)
        if name == "sample-goals":
            p.add_argument(
                "--sampler",
                choices=("local", "default"),
                default="local",
                help="Local walk or the default simulation's independent four-family targets",
            )
            p.add_argument("--seed", type=int, default=450)
            p.add_argument(
                "--count", type=int, default=6, help="Number of new targets; default sampler adds home at both ends"
            )
            p.add_argument("--interval", type=float, default=10.0)
            p.add_argument(
                "--joint-radius", type=float, default=0.8, help="Sampling-witness radius about home, in radians"
            )
            p.add_argument("--joint-step", type=float, default=0.15)
            p.add_argument("--position-step", type=float, default=0.035)
            p.add_argument("--angle-step", type=float, default=0.15)
        if name == "run":
            p.add_argument("--hardware-config", type=Path, required=True)
            p.add_argument(
                "--home", action="store_true", help="Home once before continuous execution; abort on failure"
            )
            p.add_argument("--home-speed", type=int, choices=range(1, 11), default=10)
            p.add_argument(
                "--home-plan", type=Path, help="Override automatic home bounds from the goal plan and hardware config"
            )
            p.add_argument("--open-gripper", action="store_true", help="Allow opening an empty F100 during --home")
            p.add_argument("--goal", type=float, nargs=7, metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"))
            p.add_argument("--execute", action="store_true")
            p.add_argument(
                "--continuous", action="store_true", help="Supervised goal sequence, retaining controller state"
            )
            p.add_argument("--goal-plan", type=Path, help="Reviewed reachable target sequence from sample-goals")
            p.add_argument(
                "--random-goals",
                action="store_true",
                help="Sample default four-family goals at runtime; use plan bounds/interval only",
            )
            p.add_argument(
                "--goal-seed", type=int, help="Replay a random goal stream; omitted chooses and reports a new seed"
            )
            p.add_argument(
                "--allow-goal-jumps",
                action="store_true",
                help="Allow continuous goal position/orientation jumps; physical motion guards remain active",
            )
            p.add_argument(
                "--response-trial",
                action="store_true",
                help=(
                    "Supervised trial: 30 Hz, time-scale 1, SDK <=30 with staged guards, "
                    "duration <=600 s or 0 until Ctrl+C"
                ),
            )
            p.add_argument("--telemetry-port", type=int, help="Publish measured state to the local read-only viewer")
            p.add_argument(
                "--command-hz",
                type=float,
                help="Original physics rate by default; 15 Hz requires explicit --time-scale below 1",
            )
            p.add_argument(
                "--time-scale",
                type=float,
                choices=(1.0, 0.5, 0.25),
                default=1.0,
                help="Nominal policy seconds per real second; values below 1 require --command-hz 15 or 30",
            )
            p.add_argument(
                "--trace", type=Path, help="Explicit new JSONL path; omit both log paths for automatic timestamped logs"
            )
            p.add_argument(
                "--report", type=Path, help="Explicit new JSON result path; omit both log paths for automatic logs"
            )
    p = sub.add_parser("probe", help="Measure joint feedback only; never sends motion or changes robot configuration")
    p.add_argument("--rate", type=float, default=120)
    for name in ("probe", "run"):
        p = sub.choices[name]
        p.add_argument("--ip", required=True)
        p.add_argument("--port", type=int, default=4500)
        p.add_argument(
            "--duration", type=float, default=None, help="Seconds; continuous mode defaults to 0 (until Ctrl+C)"
        )
    args = parser.parse_args()
    if hasattr(args, "duration") and args.duration is None:
        args.duration = 0 if getattr(args, "continuous", False) else 5
    if getattr(args, "continuous", False) and (args.goal is not None or args.goal_plan is None):
        parser.error("--continuous requires --goal-plan and excludes --goal")
    if getattr(args, "goal_plan", None) is not None and not args.continuous:
        parser.error("--goal-plan requires --continuous")
    if getattr(args, "allow_goal_jumps", False) and not args.continuous:
        parser.error("--allow-goal-jumps requires --continuous")
    if getattr(args, "response_trial", False) and (not args.continuous or not args.execute):
        parser.error("--response-trial requires --continuous --execute")
    if getattr(args, "home", False) and (not args.continuous or not args.execute):
        parser.error("--home requires --continuous --execute --goal-plan")
    if getattr(args, "open_gripper", False) and not args.home:
        parser.error("--open-gripper requires --home")
    if getattr(args, "home_plan", None) is not None and not args.home:
        parser.error("--home-plan requires --home")
    if getattr(args, "random_goals", False) and (not args.continuous or not args.allow_goal_jumps):
        parser.error("--random-goals requires --continuous --goal-plan --allow-goal-jumps")
    if getattr(args, "goal_seed", None) is not None and not args.random_goals:
        parser.error("--goal-seed requires --random-goals")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    torch.set_num_threads(1)
    for name in ("duration", "rate"):
        value = getattr(args, name, 1)
        zero_allowed = name == "duration" and getattr(args, "continuous", False)
        if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
            parser.error(f"--{name} must be finite and positive")
    if args.command == "probe":
        with Pro450Interface(args.ip, args.port, timeout=0.5) as robot:
            print(json.dumps(probe(robot, args.duration, args.rate), indent=2))
        return
    policy = ReachPolicy(args.bundle)
    if args.command == "init-config":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as file:
            json.dump(hardware_template(policy.controller), file, indent=2)
            file.write("\n")
        print(f"Unverified hardware configuration: {args.output}")
    elif args.command == "sample-goals":
        sampling_options = {"seed": args.seed, "count": args.count, "interval_s": args.interval}
        if args.sampler == "default":
            plan = sample_default_plan(
                policy.controller,
                policy.contract["policy_sha256"],
                policy.contract["kinematic_chain"],
                **sampling_options,
            )
        else:
            plan = sample_plan(
                policy.controller,
                policy.contract["policy_sha256"],
                **sampling_options,
                joint_radius=args.joint_radius,
                joint_step=args.joint_step,
                max_position_step=args.position_step,
                max_angle_step=args.angle_step,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(plan, output, indent=2)
            output.write("\n")
        print(
            f"Goal plan: {args.output}; {len(plan['goals'])} goals, {args.interval:g} s per goal. No hardware accessed."
        )
    elif args.command == "self-check":
        controller = policy.controller
        goal = controller.fk(controller.home)
        observation = controller.observe(controller.home, torch.zeros(6), goal)
        print(
            json.dumps(
                {
                    "checkpoint": policy.contract["checkpoint"],
                    "policy_sha256": policy.contract["policy_sha256"],
                    "home_pose_xyz_xyzw": goal.tolist(),
                    "observation_shape": list(observation.shape),
                    "action": policy.infer(observation).tolist(),
                    "control_hz": 1 / controller.dt,
                    "policy_hz": 1 / controller.policy_dt,
                    "hardware_connected": False,
                },
                indent=2,
            )
        )
    else:
        hardware = HardwareConfig(json.loads(args.hardware_config.read_text()), policy.controller)
        goal_plan = (
            GoalPlan.from_file(args.goal_plan, policy.controller, policy.contract["policy_sha256"])
            if args.goal_plan
            else None
        )
        period, _, _ = command_schedule(policy.controller, args.command_hz, args.time_scale)
        goal_sampler = (
            DefaultGoalSampler(goal_plan, policy.controller, policy.contract["kinematic_chain"], seed=args.goal_seed)
            if args.random_goals
            else None
        )
        if goal_sampler is not None:
            print(json.dumps({"event": "goal_sampling", **goal_sampler.report()}), file=sys.stderr, flush=True)
        if args.execute:
            if args.continuous:
                hardware.require_continuous_ready(
                    1 / period, args.time_scale, response_trial=args.response_trial, duration=args.duration
                )
            else:
                hardware.require_execution_ready(1 / period)  # Fail before opening a socket.
        else:
            print("Read-only inference: no motor commands. Calibration may still be unverified.", file=sys.stderr)
        home_bounds = None
        if args.home:
            if (hardware.sign != 1).any() or (hardware.offset != 0).any() or hardware.data["gripper_id"] != 14:
                raise ValueError("Integrated home requires nominal SDK joint mapping and F100 ID 14")
            home_data = json.loads(args.home_plan.read_text()) if args.home_plan else goal_plan.data
            home_bounds = HomeMotionBounds(
                policy.contract,
                home_data,
                extra_workspaces=() if args.home_plan else hardware.data.get("home_motion_workspaces_m", []),
            )
            home_joints = torch.tensor(home_bounds.joints, dtype=hardware.limits.dtype)
            if (home_joints[:, 0] < hardware.limits[:, 0]).any() or (home_joints[:, 1] > hardware.limits[:, 1]).any():
                raise ValueError("Home goal plan joint bounds must stay inside calibrated hardware limits")
        if args.trace is None and args.report is None:
            run_dir = (
                Path(__file__).resolve().parents[3]
                / "outputs/reach/deploy"
                / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            )
            run_dir.mkdir(parents=True)
            args.trace, args.report = run_dir / "trace.jsonl", run_dir / "report.json"
            print(f"[LOG] Run directory: {run_dir}", file=sys.stderr, flush=True)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.trace.open("x") if args.trace else nullcontext(None) as trace,
            args.report.open("x") if args.report else nullcontext(None) as report_file,
            TelemetryPublisher(args.telemetry_port)
            if args.telemetry_port is not None
            else nullcontext(None) as telemetry,
            Pro450Interface(
                args.ip, args.port, writable=args.execute, gripper_range=(0, 100) if args.open_gripper else None
            ) as robot,
        ):
            failed = False
            home_report = None
            try:
                if args.home:
                    home_report = run_startup_home(
                        robot,
                        policy,
                        home_bounds,
                        policy.controller.fk(policy.controller.home).tolist(),
                        sdk_speed=args.home_speed,
                        open_gripper=args.open_gripper,
                        telemetry=telemetry,
                    )
                result = run_control(
                    policy,
                    robot,
                    hardware,
                    args.goal,
                    args.duration,
                    execute=args.execute,
                    continuous=args.continuous,
                    goal_plan=goal_plan,
                    goal_sampler=goal_sampler,
                    command_hz=args.command_hz,
                    time_scale=args.time_scale,
                    allow_goal_jumps=args.allow_goal_jumps,
                    response_trial=args.response_trial,
                    trace=trace,
                    telemetry=telemetry,
                )
            except PolicyTrialError as exc:
                result, failed = exc.report, True
            except (Exception, KeyboardInterrupt) as exc:
                if home_report is None:
                    raise
                result = {
                    "mode": "home_then_policy",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "interrupted": isinstance(exc, KeyboardInterrupt),
                }
                failed = True
            if home_report is not None:
                result["home"] = home_report
            if report_file:
                json.dump(result, report_file, indent=2)
                report_file.write("\n")
                report_file.flush()
        print(json.dumps(result, indent=2))
        if failed:
            parser.exit(
                1,
                f"Policy execution failed: {result['error']}\nNo automatic motion retry or return was attempted.\n",
            )


if __name__ == "__main__":
    main()
