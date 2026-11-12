# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pymycobot==4.0.7", "numpy==2.3.1"]
# ///
"""Preview or execute a checked move to the Reach training home.

--full-home explicitly selects the contract home. Legacy segmented paths are limited
to 300 waypoints / 150 degrees; --current-start checks a single-target home path.
Default operation only reads. --execute permits the reviewed path and stop(1);
--open-gripper separately permits one F100 opening to 100 before arm motion.
--current-start uses the shared Lift single-target home, default SDK speed 10.
With --execute it prepares fresh-target mode on a stationary arm first.
No policy inference, power changes, retries, or automatic return are used.
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
from pathlib import Path

import numpy as np

if __package__:
    from .reach_telemetry import TelemetryPublisher, validate_state
    from .robot_interface import Pro450Interface, _observe_stop, encode_sdk_angles, prepare_stream, validate_home_path
else:
    from reach_telemetry import TelemetryPublisher, validate_state
    from robot_interface import Pro450Interface, _observe_stop, encode_sdk_angles, prepare_stream, validate_home_path


def contract_home_degrees(contract):
    supported = contract.get("version") == 1 or contract.get("interface") == "lift_state_v1"
    if not supported or contract.get("joint_order") != [f"joint{i}" for i in range(1, 7)]:
        raise ValueError("Expected a six-joint Reach or Lift deployment contract")
    home = contract.get("home_rad")
    if (
        not isinstance(home, list)
        or len(home) != 6
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in home)
    ):
        raise ValueError("Contract must contain six finite home_rad values")
    return [math.degrees(v) for v in home]


def make_plan(contract, start, end_angles, *, sdk_speed=1, full_home=False):
    """Compute a nominal SDK-degree path; this does not establish physical calibration."""
    home = contract_home_degrees(contract)
    encode_sdk_angles(start)
    encode_sdk_angles(end_angles)
    if type(sdk_speed) is not int or not 1 <= sdk_speed <= 10:
        raise ValueError("This home path permits SDK speed 1..10")
    if full_home is not True:
        raise ValueError("Only full-home paths are supported; full_home must be the boolean True")
    initial = [round(v * 100) for v in start]
    target = [round(v * 100) for v in home]
    end = [round(v * 100) for v in end_angles]
    if end != target:
        raise ValueError("Full-home endpoint must equal the contract home at centidegree precision")
    distance = max(abs(a - b) for a, b in zip(initial, end, strict=True))
    if distance <= 10:
        raise ValueError("Reviewed endpoint is within 0.1 degrees of start; this motion is unnecessary")
    if distance > 15000:
        raise ValueError("Full-home path must stay within 150 degrees of its reviewed start")
    count = (distance + 49) // 50
    path = [
        [(a + round((b - a) * tick / count)) / 100 for a, b in zip(initial, end, strict=True)]
        for tick in range(count + 1)
    ]
    path = validate_home_path(path, full_home=full_home)
    return {
        "full_home": full_home,
        "reviewed_start_deg": list(path[0]),
        "training_home_deg": [v / 100 for v in target],
        "reviewed_end_deg": list(path[-1]),
        "waypoint_count": count,
        "waypoints_deg": [list(pose) for pose in path[1:]],
        "sdk_speed": sdk_speed,
        "measured_speed_limit_deg_s": 3 * sdk_speed,
        "minimum_waypoint_interval_s": 0.2,
        "waypoint_timeout_s": 3,
        "total_budget_s": 300,
        "gripper_command_if_approved": {"method": "set_pro_gripper_angle", "args": [100, 14]},
        "mapping": "Nominal SDK joint degrees equal simulation joint degrees; physical calibration is not certified.",
        "note": "Only this full home path is approved by --execute; no Reach policy is run.",
    }


def read_f100_opening(
    robot,
    *,
    attempts=1,
    timeout=1.0,
    retry_delay_s=0.1,
    clock=time.monotonic,
    sleep=time.sleep,
    diagnostics=None,
    context="F100 feedback",
):
    """Read a fresh opening, optionally retrying invalid returns before arm motion.

    SDK/transport exceptions and cancellation propagate immediately. No cached
    opening is substituted, and this helper never sends a gripper target.
    """
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("F100 startup reads permit one to three attempts")
    if not math.isfinite(retry_delay_s) or not 0.1 <= retry_delay_s <= 1.0:
        raise ValueError("F100 retry delay must be 0.1..1 seconds")
    rows = []
    for attempt in range(1, attempts + 1):
        started = clock()
        try:
            opening = robot.call("get_pro_gripper_angle", 14, timeout=timeout)
        except (Exception, KeyboardInterrupt) as exc:
            row = dict(context=context, attempt=attempt, error=f"{type(exc).__name__}: {exc}")
            row["latency_ms"] = (clock() - started) * 1000
            if diagnostics is not None:
                diagnostics.append(row)
            raise
        valid = type(opening) in (int, float) and math.isfinite(opening) and 0 <= opening <= 100
        row = dict(
            context=context,
            attempt=attempt,
            raw=repr(opening),
            latency_ms=(clock() - started) * 1000,
            valid=valid,
        )
        rows.append(row)
        if diagnostics is not None:
            diagnostics.append(row)
        if valid:
            return opening
        if attempt < attempts:
            sleep(retry_delay_s)
    raise ValueError(f"Invalid {context}: {rows}")


def read_current_start(
    robot,
    *,
    sleep=time.sleep,
    clock=time.monotonic,
    allow_closed_gripper=False,
    gripper_read_attempts=1,
    diagnostics=None,
):
    """Read a ready, stationary start without sending any robot command."""
    if allow_closed_gripper:
        for method, expected in (("is_power_on", 1), ("get_control_mode", 0), ("get_fresh_mode", 1)):
            if robot.call(method) != expected:
                raise RuntimeError(f"Current-start requires {method}={expected}")
        read_f100_opening(
            robot,
            attempts=gripper_read_attempts,
            clock=clock,
            sleep=sleep,
            diagnostics=diagnostics,
            context="current-start F100 feedback",
        )
        robot.check_fault()
    else:
        robot.check_ready(14)
    previous, previous_time, _ = robot.read()
    encode_sdk_angles(previous)
    sleep(0.05)
    robot.check_fault()
    current, received, _ = robot.read()
    encode_sdk_angles(current)
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (previous_time, received)):
        raise ValueError("Invalid current-start feedback timing")
    dt = received - previous_time
    if not 0 < dt <= 0.1:
        raise RuntimeError("Current-start feedback gap exceeded 100 ms or timestamp did not increase")
    if max(abs(a - b) for a, b in zip(current, previous, strict=True)) > 0.02 + 1e-9:
        raise RuntimeError("Current start is not stationary; no home path was prepared")
    return {"angles_deg": list(current), "received_monotonic_s": received, "stationary_interval_s": dt}


class HomeMotionBounds:
    """Check the approved joint/mount envelope; this is not whole-arm collision checking.

    FK uses the contract's fixed Rz(yaw) Ry(pitch) Rx(roll) origins and six local
    joint-Z rotations, matching the deployment chain without importing Torch.
    SDK and simulation joint coordinates must already have the nominal mapping.
    """

    @staticmethod
    def _array(values, shape, name):
        try:
            result = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name}") from exc
        if result.shape != shape or result.dtype.kind not in "fi" or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
        return result.astype(np.float64)

    @classmethod
    def _bounds(cls, values, rows, name):
        result = cls._array(values, (rows, 2), name)
        if not (result[:, 0] < result[:, 1]).all():
            raise ValueError(f"{name} must contain ordered lower/upper bounds")
        return result

    def __init__(self, contract, goal_plan, *, extra_workspaces=()):
        if (
            not isinstance(goal_plan, dict)
            or type(goal_plan.get("version")) is not int
            or goal_plan["version"] != 2
            or "workspace_m" in goal_plan
        ):
            raise ValueError("Home bounds require a version-2 goal plan with motion_workspace_m")
        policy_hash = contract.get("policy_sha256")
        if (
            not isinstance(policy_hash, str)
            or len(policy_hash) != 64
            or any(c not in "0123456789abcdef" for c in policy_hash)
            or goal_plan.get("policy_sha256") != policy_hash
        ):
            raise ValueError("Goal plan policy_sha256 must match the deployment contract")
        if goal_plan.get("joint_order") != contract.get("joint_order"):
            raise ValueError("Goal plan joint_order must match the deployment contract")
        home = contract_home_degrees(contract)
        training = self._bounds(contract.get("joint_limits_rad"), 6, "contract joint_limits_rad")
        self.joints = self._bounds(goal_plan.get("joint_bounds_rad"), 6, "joint_bounds_rad")
        self.workspace = self._bounds(goal_plan.get("motion_workspace_m"), 3, "motion_workspace_m")
        if not isinstance(extra_workspaces, (list, tuple)):
            raise ValueError("home_motion_workspaces_m must be a list of workspace boxes")
        self.workspaces = np.stack(
            [self.workspace] + [self._bounds(box, 3, "home_motion_workspaces_m") for box in extra_workspaces]
        )
        # GoalPlan serializes float32 limits. Compare in that same convention,
        # then retain the tighter float64 intersection for actual home checks.
        joint32, training32 = self.joints.astype(np.float32), training.astype(np.float32)
        if (joint32[:, 0] < training32[:, 0]).any() or (joint32[:, 1] > training32[:, 1]).any():
            raise ValueError("Goal plan joint bounds must stay inside the contract joint limits")
        self.joints[:, 0] = np.maximum(self.joints[:, 0], training[:, 0])
        self.joints[:, 1] = np.minimum(self.joints[:, 1], training[:, 1])
        chain = contract.get("kinematic_chain")
        if not isinstance(chain, list) or len(chain) != 7:
            raise ValueError("Contract must contain the six-joint and fixed-mount kinematic chain")
        self.origins = []
        for item in chain:
            if not isinstance(item, dict):
                raise ValueError("Invalid contract kinematic origin")
            xyz = self._array(item.get("xyz"), (3,), "origin xyz")
            roll, pitch, yaw = self._array(item.get("rpy"), (3,), "origin rpy")
            cr, sr, cp, sp, cy, sy = (
                math.cos(roll),
                math.sin(roll),
                math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw),
                math.sin(yaw),
            )
            origin = np.eye(4)
            origin[:3, :3] = [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
            origin[:3, 3] = xyz
            self.origins.append(origin)
        self.check(home)  # The approved envelope must contain the contract home.

    def mount_position(self, degrees):
        radians = np.deg2rad(self._array(degrees, (6,), "joint feedback"))
        transform = np.eye(4)
        for index, origin in enumerate(self.origins):
            transform = transform @ origin
            if index < 6:
                c, s = math.cos(radians[index]), math.sin(radians[index])
                transform = transform @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return transform[:3, 3]

    def check(self, degrees):
        radians = np.deg2rad(self._array(degrees, (6,), "joint feedback"))
        if ((radians < self.joints[:, 0]) | (radians > self.joints[:, 1])).any():
            raise ValueError("Home joints are outside the goal plan's joint bounds")
        position = self.mount_position(degrees)
        inside = ((position >= self.workspaces[:, :, 0]) & (position <= self.workspaces[:, :, 1])).all(axis=1)
        if not inside.any():
            raise ValueError(
                f"Home mount position is outside motion_workspace_m and approved home regions: {position.tolist()}"
            )
        return position.tolist()

    def validate_path(self, path):
        positions = np.array([self.check(q) for q in path])
        return {
            "motion_workspace_m": self.workspace.tolist(),
            "home_motion_workspaces_m": self.workspaces.tolist(),
            "joint_bounds_rad": self.joints.tolist(),
            "path_sample_count": len(path),
            "mount_path_min_m": positions.min(axis=0).tolist(),
            "mount_path_max_m": positions.max(axis=0).tolist(),
            "path_check": (
                "Joint limits and union of approved mount regions at every path sample and measured feedback; "
                "not whole-arm collision checking."
            ),
        }


class HomeError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["error"])


class HomeTelemetry:
    """Adapt existing measured home progress to the read-only localhost viewer."""

    def __init__(self, publisher, policy_sha256, goal, *, execute=False, clock=time.monotonic):
        self.publisher, self.execute, self.clock = publisher, execute, clock
        self.packet = validate_state(
            {
                "schema": 1,
                "type": "reach_state",
                "run_id": uuid.uuid4().hex,
                "seq": 0,
                "policy_sha256": policy_sha256,
                "monotonic_s": 0.0,
                "state": "running" if execute else "read_only",
                "sent": False,
                "error": None,
                "q_rad": [0.0] * 6,
                "dq_rad_s": [0.0] * 6,
                "target_rad": [0.0] * 6,
                "goal_index": 0,
                "goal_xyz_xyzw": list(goal),
            }
        )
        self.has_feedback = False

    def __call__(self, row):
        velocity = row["joint_velocity_deg_s"]
        if velocity is None:
            return  # No velocity estimate exists for the first measurement.
        self.packet.update(
            monotonic_s=row["received_monotonic_s"],
            q_rad=list(map(math.radians, row["q_deg"])),
            dq_rad_s=list(map(math.radians, velocity)),
            target_rad=list(map(math.radians, row["target_deg"])),
            sent=self.execute and row["waypoint"] > 0,
        )
        self.has_feedback = True
        self._publish()

    def _publish(self):
        self.packet["published_monotonic_s"] = self.clock()
        self.publisher.publish(dict(self.packet))
        self.packet["seq"] += 1

    def finish(self, report=None, *, error=None):
        report = {} if report is None else report
        # Stop samples are already collected by the original safety procedure.
        # Recover their original receipt times, never the time this status is sent.
        samples = [(row["received_monotonic_s"], row["q_deg"]) for row in report.get("samples", [])]
        for stop in report.get("stops", []):
            samples.extend(
                (stop["request_monotonic_s"] + row["received_after_stop_s"], row["q_deg"])
                for row in stop["samples"]
                if "received_after_stop_s" in row and "q_deg" in row
            )
        direct_stop = report.get("stop")
        if direct_stop:
            samples.extend(
                (direct_stop["stop_request_monotonic_s"] + row["received_after_stop_s"], row["q_deg"])
                for row in direct_stop["samples"]
                if "received_after_stop_s" in row and "q_deg" in row
            )
        valid = sorted(
            (stamp, q)
            for stamp, q in samples
            if type(stamp) in (int, float)
            and math.isfinite(stamp)
            and stamp >= 0
            and len(q) == 6
            and all(type(v) in (int, float) and math.isfinite(v) for v in q)
        )
        if len(valid) >= 2 and valid[-1][0] > valid[-2][0]:
            (previous_time, previous), (received, q) = valid[-2:]
            self.packet.update(
                monotonic_s=received,
                q_rad=list(map(math.radians, q)),
                dq_rad_s=[math.radians(a - b) / (received - previous_time) for a, b in zip(q, previous, strict=True)],
                target_rad=list(
                    map(
                        math.radians,
                        report.get(
                            "target_deg", report.get("last_attempted_target_deg", report.get("reviewed_start_deg"))
                        ),
                    )
                ),
            )
            self.has_feedback = True
        error = error or report.get("error")
        stopped = report.get("stops", []) and all(
            report["stops"][-1].get(key) is True for key in ("acknowledged", "stationary")
        )
        if "stop" in report:
            stop = report["stop"]
            stopped = report.get("attempted_writes") == 0 or bool(
                stop and stop.get("stop_acknowledged") and stop.get("post_stop_stationary") and not stop.get("error")
            )
        if self.execute and not error and not (report.get("passed") and stopped):
            error = "Home completion and stopping were not verified"
        self.packet.update(
            state="fault" if error else ("stopped" if self.execute else "read_only"),
            error=error,
            sent=self.execute and report.get("attempted_writes", 0) > 0,
        )
        if self.has_feedback:
            self._publish()
        report.update(run_id=self.packet["run_id"], telemetry_dropped=self.publisher.dropped)


def run_home(
    robot,
    plan,
    *,
    execute=False,
    allow_gripper_open=False,
    progress=None,
    motion_bounds=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    full_home = plan["full_home"]
    path = validate_home_path([plan["reviewed_start_deg"], *plan["waypoints_deg"]], full_home=full_home)
    if tuple(plan["training_home_deg"]) != path[-1]:
        raise ValueError("Full-home endpoint must equal the contract home")
    path_check = {} if motion_bounds is None else motion_bounds.validate_path(path)
    budget_s = 300
    sdk_speed = plan["sdk_speed"]
    if type(sdk_speed) is not int or not 1 <= sdk_speed <= 10:
        raise ValueError("This home path permits SDK speed 1..10")
    speed_limit = 3 * sdk_speed  # A monitoring ceiling, not an SDK-to-deg/s conversion.
    if execute and (
        getattr(robot, "home_path", None) != path
        or getattr(robot, "allow_gripper_open", False) != allow_gripper_open
        or getattr(robot, "home_speed", None) != sdk_speed
        or getattr(robot, "home_full", False) != full_home
    ):
        raise ValueError("Execution requires the exact reviewed home-path capability")
    state = {key: robot.call(key) for key in ("is_power_on", "get_control_mode", "get_fresh_mode")}
    state["f100_opening"] = robot.call("get_pro_gripper_angle", 14, timeout=1.0)
    opening = state["f100_opening"]
    if type(opening) not in (int, float) or not math.isfinite(opening) or not 0 <= opening <= 100:
        raise ValueError("Invalid F100 opening feedback; no opening or arm movement is permitted")
    limits = [(robot.call("get_joint_min_angle", j), robot.call("get_joint_max_angle", j)) for j in range(1, 7)]
    for j, (low, high) in enumerate(limits):
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in (low, high))
            or not low < high
            or min(p[j] for p in path) - 0.15 < low
            or max(p[j] for p in path) + 0.15 > high
        ):
            raise ValueError(f"J{j + 1} home envelope is outside measured joint limits")
    report = {
        **plan,
        **path_check,
        "mode": "full_home" if execute else "read_only_home_preview",
        "state": state,
        "gripper_open_required": not _open(state["f100_opening"]),
        "gripper_open_authorized": allow_gripper_open,
        "passed": False,
        "attempted_writes": 0,
        "writes": 0,
        "gripper_open_attempted": False,
        "stops": [],
        "samples": [],
        "settled_waypoints": [],
        "error": None,
        "full_home_verified": False,
        "reach_policy_verified": False,
    }
    lower = [round(min(p[j] for p in path) * 100) - 15 for j in range(6)]
    upper = [round(max(p[j] for p in path) * 100) + 15 for j in range(6)]
    last_q = last_received = None
    last_target = path[0]
    deadline = math.inf
    observation_started = clock()
    last_progress = -math.inf

    def sample(*, starting=False, preparing=False, until=math.inf):
        nonlocal last_q, last_received, last_progress
        limit = min(deadline, until)

        def bounded(function):
            remaining = limit - clock()
            if remaining <= 0:
                raise TimeoutError("Home path sample exceeded its time budget")
            original = robot.timeout
            robot.timeout = min(original, remaining)
            try:
                return function()
            finally:
                robot.timeout = original

        bounded(robot.check_fault)
        q, received, latency = bounded(robot.read)
        encode_sdk_angles(q)
        units = [round(v * 100) for v in q]
        row = {"q_deg": q, "received_monotonic_s": received, "latency_s": latency}
        report["samples"].append(row)
        previous_q, previous_time = last_q, last_received
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (received, latency)) or latency < 0:
            raise ValueError("Invalid feedback timing")
        last_q, last_received = list(q), received
        if clock() >= limit or received >= limit:
            raise TimeoutError("Home path sample exceeded its time budget")
        if any(not low <= v <= high for v, low, high in zip(units, lower, upper, strict=True)):
            raise RuntimeError("Joint feedback left the reviewed path envelope")
        if motion_bounds is not None:
            row["mount_xyz_m"] = motion_bounds.check(q)
        error = max(abs(v - round(t * 100)) for v, t in zip(units, last_target, strict=True))
        if error > (5 if starting else 75):
            raise RuntimeError("Reviewed start changed" if starting else "Tracking error exceeded 0.75 degrees")
        if previous_q is not None:
            dt = received - previous_time
            # No arm target exists during F100 preparation. Modbus may block;
            # retain absolute start checks and require a new stationary window
            # before sending any waypoint. Motion samples never waive this gap.
            if dt <= 0 or (dt > 0.1 and not preparing):
                raise TimeoutError("Joint feedback gap exceeded 100 ms")
            speeds = [(a - b) / dt for a, b in zip(q, previous_q, strict=True)]
            speed = max(map(abs, speeds))
            row["joint_velocity_deg_s"] = speeds
            row["max_speed_deg_s"] = speed
            if speed > speed_limit + 1e-9:
                raise RuntimeError(f"Measured joint speed exceeded {speed_limit:g} degrees/s")
        if progress is not None and received - last_progress >= 0.2:
            progress(
                {
                    "event": "joint_feedback",
                    "received_monotonic_s": received,
                    "elapsed_s": received - observation_started,
                    "waypoint": report["writes"],
                    "q_deg": list(q),
                    "target_deg": list(last_target),
                    "joint_velocity_deg_s": row.get("joint_velocity_deg_s"),
                    "max_tracking_error_deg": error / 100,
                }
            )
            last_progress = received
            if clock() >= limit or (not preparing and clock() - received > 0.1):
                raise TimeoutError("Live output delayed feedback beyond its time budget")
        return units, received

    def settle(target, until, *, starting=False):
        window = deque()
        while clock() < min(until, deadline):
            units, received = sample(starting=starting, until=until)
            window.append((received, units))
            while len(window) > 5 and received - window[1][0] >= 0.2:
                window.popleft()
            errors = [max(abs(v - round(t * 100)) for v, t in zip(row[1], target, strict=True)) for row in window]
            ranges = [max(row[1][j] for row in window) - min(row[1][j] for row in window) for j in range(6)]
            if len(window) >= 5 and received - window[0][0] >= 0.2 - 1e-9 and max(errors) <= 10 and max(ranges) <= 2:
                return
            sleep(0.02)
        raise TimeoutError("Waypoint did not settle within its time budget")

    def stop_and_observe():
        nonlocal last_q, last_received
        details = {
            "request_monotonic_s": clock(),
            "samples": [],
            "acknowledged": False,
            "stationary": False,
            "last_feedback_deg": None,
            "observed_stop_time_upper_bound_s": None,
            "pre_stop_feedback_deg": last_q,
            "pre_stop_received_monotonic_s": last_received,
            "ack_monotonic_s": None,
            "ack_latency_s": None,
        }
        report["stops"].append(details)
        bounds_failure = None
        try:
            robot.stop()
            details["ack_monotonic_s"] = clock()
            details["ack_latency_s"] = clock() - details["request_monotonic_s"]
            details["acknowledged"] = True
            _observe_stop(
                robot,
                last_q,
                last_q,
                last_received,
                details,
                clock=clock,
                sleep=sleep,
                monitor_all_joint_speeds=True,
                max_speed_deg_s=speed_limit,
                max_displacement_deg=0.15 if sdk_speed == 1 else 0.75,
            )
        except BaseException as exc:
            details["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if motion_bounds is not None:
                for row in details["samples"]:
                    if "q_deg" in row:
                        try:
                            row["mount_xyz_m"] = motion_bounds.check(row["q_deg"])
                        except ValueError as exc:
                            row["motion_bounds_error"] = str(exc)
                            bounds_failure = bounds_failure or exc
            for row in reversed(details["samples"]):
                offset = row.get("received_after_stop_s")
                if "q_deg" in row and type(offset) in (int, float) and math.isfinite(offset):
                    received = details["request_monotonic_s"] + offset
                    if last_received is None or received > last_received:
                        last_q, last_received = row["q_deg"], received
                        break
        if bounds_failure is not None:
            details["error"] = f"{type(bounds_failure).__name__}: {bounds_failure}"
            raise bounds_failure

    settle(path[0], clock() + 1, starting=True)
    report["measured_start_deg"] = last_q.copy()
    if not execute:
        return report
    if any(state[key] != value for key, value in (("is_power_on", 1), ("get_control_mode", 0), ("get_fresh_mode", 1))):
        raise ValueError("Home path requires existing power, position mode and refresh mode 1; no modes are changed")
    if report["gripper_open_required"] and not allow_gripper_open:
        raise ValueError("F100 must be open; explicit --open-gripper approval is required to open it")
    deadline = clock() + budget_s
    failure = None
    initial_stop_verified = False
    try:
        stop_and_observe()
        initial_stop_verified = True
        if not _open(state["f100_opening"]):
            report["gripper_open_attempted"] = True
            opening_deadline = min(deadline, clock() + 3)
            result = robot.call("set_pro_gripper_angle", 100, 14, timeout=min(1.0, opening_deadline - clock()))
            if result != 1:
                raise RuntimeError("F100 opening was not acknowledged")
            while True:
                if clock() >= opening_deadline:
                    raise TimeoutError("F100 did not open within three seconds")
                # Opening is bounded independently; arm motion still requires the readback below.
                value = robot.call("get_pro_gripper_angle", 14, timeout=min(1.0, opening_deadline - clock()))
                sample(starting=True, preparing=True, until=opening_deadline)
                if _open(value):
                    break
                sleep(0.02)
        else:
            value = robot.call("get_pro_gripper_angle", 14, timeout=min(1.0, deadline - clock()))
            sample(starting=True, preparing=True)
            if not _open(value):
                raise RuntimeError("F100 opening changed after preflight")
        settle(path[0], clock() + 1, starting=True)
        last_sent = None
        for index, target in enumerate(path[1:], 1):
            sample()
            if last_sent is not None and clock() - last_sent < 0.2:
                sleep(0.2 - (clock() - last_sent))
                sample()
            if clock() >= deadline:
                raise TimeoutError(f"Home path exceeded its {budget_s}-second budget")
            last_sent = clock()
            report["attempted_writes"] += 1
            report["last_attempted_target_deg"] = list(target)
            result = robot.call(
                "send_angles", encode_sdk_angles(target), sdk_speed, timeout=min(0.1, deadline - clock())
            )
            if result != 1:
                raise RuntimeError("Waypoint was not acknowledged")
            report["writes"] += 1
            last_target = target
            settle(target, last_sent + 3)
            report["settled_waypoints"].append(
                {"index": index, "target_deg": list(target), "measured_deg": last_q.copy()}
            )
    except BaseException as exc:
        failure = exc
    finally:
        if initial_stop_verified:
            try:
                stop_and_observe()
            except BaseException as exc:
                failure = failure or exc
    if clock() >= deadline:
        failure = failure or TimeoutError(f"Home path exceeded its {budget_s}-second budget")
    if motion_bounds is not None:
        try:
            motion_bounds.check(last_q)
        except ValueError as exc:
            failure = failure or exc
    final_error = max(abs(round(a * 100) - round(b * 100)) for a, b in zip(last_q, path[-1], strict=True))
    if failure is None and final_error > 10:
        failure = RuntimeError("Final stopped feedback is not within 0.1 degrees of the reviewed endpoint")
    report["final_angles_deg"] = last_q
    report["final_error_deg"] = final_error / 100
    report["passed"] = (
        failure is None
        and report["writes"] == len(path) - 1
        and len(report["settled_waypoints"]) == len(path) - 1
        and report["stops"][-1]["stationary"]
    )
    report["full_home_verified"] = report["passed"]
    if failure is not None:
        report["error"] = f"{type(failure).__name__}: {failure}"
        raise HomeError(report) from failure
    return report


def run_direct_home(
    robot,
    contract,
    bounds,
    *,
    execute=False,
    sdk_speed=10,
    open_gripper=False,
    timeout_s=60,
    progress=None,
    progress_interval=0.2,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """The path is checked offline; only its final joint target is sent."""
    if (
        type(sdk_speed) is not int
        or not 1 <= sdk_speed <= 10
        or not math.isfinite(timeout_s)
        or not 1 <= timeout_s <= 120
    ):
        raise ValueError("Use SDK speed 1..10 and timeout 1..120 seconds")
    home = [round(v, 2) for v in contract_home_degrees(contract)]
    f100_reads = []
    observed = read_current_start(
        robot,
        sleep=sleep,
        clock=clock,
        allow_closed_gripper=open_gripper,
        gripper_read_attempts=3,
        diagnostics=f100_reads,
    )
    start = observed["angles_deg"]
    already_home = max(abs(a - b) for a, b in zip(start, home, strict=True)) <= 0.1
    bounds.check(start)
    bounds.check(home)
    # These are offline path samples, not the legacy segmented motion commands.
    count = max(1, math.ceil(max(abs(a - b) for a, b in zip(start, home, strict=True)) / 0.5))
    path = np.linspace(start, home, count + 1).tolist()
    checked = bounds.validate_path(path)
    for joint in range(6):
        low = robot.call("get_joint_min_angle", joint + 1)
        high = robot.call("get_joint_max_angle", joint + 1)
        if (
            not all(type(v) in (int, float) and math.isfinite(v) for v in (low, high))
            or not low <= min(start[joint], home[joint]) <= max(start[joint], home[joint]) <= high
        ):
            raise ValueError(f"J{joint + 1} home path exceeds measured joint limits")
    report = dict(
        mode="direct_home" if execute else "read_only_home_preview",
        sdk_speed=sdk_speed,
        policy_sha256=contract["policy_sha256"],
        measured_start_deg=start,
        target_deg=home,
        writes=0,
        attempted_writes=0,
        gripper_open_attempted=False,
        f100_reads=f100_reads,
        passed=False,
        samples=[],
        stop=None,
        error=None,
        **checked,
    )
    if not execute:
        return report
    started, last_progress = clock(), -math.inf
    deadline, attempted = started + timeout_s, False
    last_q, last_stamp = start, observed["received_monotonic_s"]
    try:

        def read_opening():
            return read_f100_opening(
                robot,
                attempts=3,
                timeout=0.2,
                clock=clock,
                sleep=sleep,
                diagnostics=f100_reads,
                context="home F100 feedback",
            )

        opening = read_opening()
        if opening < 100:
            if not open_gripper:
                raise ValueError("Opening F100 requires --open-gripper")
            report["gripper_open_attempted"] = True
            if robot.call("set_pro_gripper_angle", 100, 14, timeout=0.2) != 1:
                raise RuntimeError("F100 opening was not acknowledged")
            until = min(deadline, clock() + 3)
            while opening < 100 and clock() < until:
                sleep(0.05)
                opening = read_opening()
            if opening < 100:
                raise TimeoutError("F100 did not reach the policy's initial opening of 100")
        report["initial_gripper_sdk"] = opening
        q, stamp, _ = robot.read_state(timeout=0.2)
        bounds.check(q)
        if max(abs(a - b) for a, b in zip(q, start, strict=True)) > 0.05:
            raise RuntimeError("Reviewed start changed before homing")
        last_q, last_stamp = q, stamp
        if not already_home:
            attempted = True
            report["attempted_writes"] = 1
            # Exactly one send_angles(home, speed); SDK plans the continuous move.
            robot.send(encode_sdk_angles(home), sdk_speed, timeout=0.2)
            report["writes"] = 1
        window = deque()
        while clock() < deadline:
            sleep(0.04)
            q, stamp, latency = robot.read_state(timeout=min(0.2, deadline - clock()))
            xyz = bounds.check(q)
            dt = stamp - last_stamp
            if not 0 < dt <= 0.2:
                raise TimeoutError("Home feedback gap exceeded 200ms")
            speed = max(abs(a - b) / dt for a, b in zip(q, last_q, strict=True))
            if speed > 3 * sdk_speed:
                raise RuntimeError("Measured home speed exceeded the monitoring bound")
            if any(not min(a, b) - 1 <= v <= max(a, b) + 1 for v, a, b in zip(q, start, home, strict=True)):
                raise RuntimeError("Measured joints left the reviewed home envelope")
            error = max(abs(a - b) for a, b in zip(q, home, strict=True))
            row = dict(
                time_s=clock() - started,
                received_monotonic_s=stamp,
                q_deg=q,
                target_deg=home,
                waypoint=int(attempted),
                joint_velocity_deg_s=[(a - b) / dt for a, b in zip(q, last_q, strict=True)],
                target_error_deg=error,
                max_speed_deg_s=speed,
                read_ms=latency * 1000,
                mount_xyz_m=xyz,
            )
            report["samples"].append(row)
            last_q, last_stamp = q, stamp
            if progress and clock() - last_progress >= progress_interval:
                progress(row)
                last_progress = clock()
            window.append((stamp, q, error))
            while len(window) > 5 and stamp - window[1][0] >= 0.3:
                window.popleft()
            if (
                len(window) >= 5
                and stamp - window[0][0] >= 0.3
                and max(v[2] for v in window) <= 0.1
                and all(max(v[1][j] for v in window) - min(v[1][j] for v in window) <= 0.02 + 1e-9 for j in range(6))
            ):
                report["passed"] = True
                break
        else:
            raise TimeoutError("Direct home did not settle before its deadline")
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["interrupted"] = isinstance(exc, KeyboardInterrupt)
    finally:
        if attempted:
            try:
                report["stop"] = robot.stop_and_observe(clock=clock, sleep=sleep)
                if (
                    not report["stop"].get("stop_acknowledged")
                    or not report["stop"].get("post_stop_stationary")
                    or report["stop"].get("error")
                ):
                    raise RuntimeError("Home stop was not verified")
                if report["stop"].get("samples"):
                    last_q = report["stop"]["samples"][-1]["q_deg"]
                    bounds.check(last_q)
            except Exception as exc:
                report["error"] = report["error"] or f"{type(exc).__name__}: {exc}"
        report["final_angles_deg"] = last_q
        report["final_error_deg"] = max(abs(a - b) for a, b in zip(last_q, home, strict=True))
        report["elapsed_s"] = clock() - started
        report["passed"] = report["passed"] and report["error"] is None and report["final_error_deg"] <= 0.1
    return report


def _open(value):
    return type(value) in (int, float) and math.isfinite(value) and 95 <= value <= 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    start_options = parser.add_mutually_exclusive_group(required=True)
    start_options.add_argument("--start-angles", type=float, nargs=6)
    start_options.add_argument(
        "--current-start", action="store_true", help="Read a stationary start; requires --full-home and --goal-plan"
    )
    parser.add_argument(
        "--end-angles",
        type=float,
        nargs=6,
        help="Fixed reviewed endpoint in SDK degrees; current-start defaults to home",
    )
    parser.add_argument(
        "--goal-plan", type=Path, help="Version-2 plan supplying approved joint and mount-motion bounds"
    )
    parser.add_argument(
        "--full-home", action="store_true", help="Require the contract home endpoint (legacy paths: 300 waypoints)"
    )
    parser.add_argument("--sdk-speed", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=4500)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--live", action="store_true", help="Stream measured joint feedback to stderr at up to 5 Hz")
    parser.add_argument("--telemetry-port", type=int, help="Publish measured home progress to a localhost viewer")
    parser.add_argument(
        "--telemetry-goal", type=float, nargs=7, metavar="VALUE", help="Display-only goal: X Y Z QX QY QZ QW"
    )
    parser.add_argument("--open-gripper", action="store_true", help="Approve one F100 opening to 100 if needed")
    args = parser.parse_args()
    if not args.full_home:
        parser.error("Only --full-home is supported; partial home trials have been removed")
    if args.current_start and args.goal_plan is None:
        parser.error("--current-start requires --full-home and --goal-plan")
    if not args.current_start and args.end_angles is None:
        parser.error("--end-angles is required unless --current-start is used")
    if args.telemetry_port is not None and not 1 <= args.telemetry_port <= 65535:
        parser.error("--telemetry-port must be in 1..65535")
    contract = json.loads((args.bundle / "contract.json").read_text())
    goal_plan = json.loads(args.goal_plan.read_text()) if args.goal_plan is not None else None
    motion_bounds = HomeMotionBounds(contract, goal_plan) if goal_plan is not None else None
    if args.telemetry_port is not None and args.telemetry_goal is None and goal_plan is not None:
        try:
            args.telemetry_goal = goal_plan["goals"][0]["pose_xyz_xyzw"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Goal plan must provide a first display pose for telemetry") from exc
    if (args.telemetry_port is None) != (args.telemetry_goal is None):
        parser.error("--telemetry-port requires --telemetry-goal or --goal-plan; --telemetry-goal requires a port")
    telemetry = (
        HomeTelemetry(None, contract["policy_sha256"], args.telemetry_goal, execute=args.execute)
        if args.telemetry_port is not None
        else None
    )
    preparation = None
    start, end = args.start_angles, args.end_angles
    if args.current_start:
        home = [round(v * 100) / 100 for v in contract_home_degrees(contract)]
        if end is not None and [round(v * 100) for v in end] != [round(v * 100) for v in home]:
            raise ValueError("Full-home endpoint must equal the contract home")
        end = home

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        with (
            TelemetryPublisher(args.telemetry_port) if args.telemetry_port is not None else nullcontext() as publisher,
            Pro450Interface(
                args.ip,
                args.port,
                writable=args.execute,
                timeout=0.1,
                connect_timeout=2,
                gripper_range=(0, 100) if args.execute and args.open_gripper else None,
            ) as robot,
        ):
            if telemetry is not None:
                telemetry.publisher = publisher

            def progress(row):
                if telemetry is not None:
                    telemetry(row)
                if args.live:
                    print(json.dumps(row), file=sys.stderr, flush=True)

            if args.execute:
                preparation = prepare_stream(robot, low_latency=True)
                print(json.dumps(preparation), file=sys.stderr, flush=True)
            report = run_direct_home(
                robot,
                contract,
                motion_bounds,
                execute=args.execute,
                sdk_speed=args.sdk_speed,
                open_gripper=args.open_gripper,
                progress=progress if args.live or telemetry is not None else None,
            )
            report["stream_preparation"] = preparation
            if telemetry is not None:
                telemetry.finish(report)
        print(json.dumps(report, indent=2), flush=True)
        if args.execute and not report["passed"]:
            parser.exit(1, "Home failed; no automatic retry or return was attempted.\n")
        return
    plan = make_plan(
        contract,
        start,
        end,
        sdk_speed=args.sdk_speed,
        full_home=args.full_home,
    )
    path = validate_home_path([plan["reviewed_start_deg"], *plan["waypoints_deg"]], full_home=args.full_home)
    if motion_bounds is not None:
        motion_bounds.validate_path(path)  # Refuse the path before opening its motion capability.

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    if args.live:
        print(json.dumps({"event": "process_started", "pid": os.getpid()}), file=sys.stderr, flush=True)

    with TelemetryPublisher(args.telemetry_port) if args.telemetry_port is not None else nullcontext() as publisher:
        if telemetry is not None:
            telemetry.publisher = publisher

        def progress(row):
            if telemetry is not None:
                telemetry(row)
            if args.live:
                print(json.dumps(row), file=sys.stderr, flush=True)

        with Pro450Interface(
            args.ip,
            args.port,
            timeout=0.1,
            connect_timeout=2,
            home_path=path if args.execute else None,
            allow_gripper_open=args.execute and args.open_gripper,
            home_speed=args.sdk_speed,
            home_full=args.execute and args.full_home,
        ) as robot:
            try:
                report = run_home(
                    robot,
                    plan,
                    execute=args.execute,
                    allow_gripper_open=args.open_gripper,
                    progress=progress if args.live or telemetry is not None else None,
                    motion_bounds=motion_bounds,
                )
            except HomeError as exc:
                exc.report["stream_preparation"] = preparation
                if telemetry is not None:
                    telemetry.finish(exc.report)
                print(json.dumps(exc.report, indent=2), flush=True)
                parser.exit(1, "Home path failed; no automatic retry or return was attempted.\n")
            except BaseException as exc:
                if telemetry is not None:
                    telemetry.finish(error=f"{type(exc).__name__}: {exc}")
                raise
            report["stream_preparation"] = preparation
            if telemetry is not None:
                telemetry.finish(report)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
