"""Bounded, sequential access to pymycobot 4.0.7's Pro450Client.

The SDK has long internal waits and a background socket reader. A separate process
lets us cancel a stalled request without leaving a thread that later sends old goals.
No connection or hardware operation is performed on import.
"""

import importlib.metadata
import io
import math
import multiprocessing
import signal
import time
from collections import Counter, deque
from contextlib import redirect_stdout, suppress

READ_METHODS = {
    "get_angles",
    "read_state",
    "get_fresh_mode",
    "get_vr_mode",
    "get_control_mode",
    "is_power_on",
    "get_system_version",
    "get_error_information",
    "get_robot_status",
    "get_pro_gripper_angle",
    "get_joint_min_angle",
    "get_joint_max_angle",
    "get_max_speed",
    "get_max_acc",
    "get_fusion_parameters",
    "get_filter_len",
}

# These diagnostics accept one bounded selector, never an arbitrary SDK operation.
READ_PARAMETER_RANGES = {
    "get_max_speed": range(2),
    "get_max_acc": range(2),
    "get_fusion_parameters": range(1, 5),
    "get_filter_len": range(1, 6),
}


def _check_robot_status(status):
    """Validate the complete Pro450 fault reply; the moving flag is informational."""
    # 4.0.7 returns 8 flags, then 12 entries: zero or a list of active fault bits.
    if not isinstance(status, list) or len(status) != 20:
        raise RuntimeError(f"Unexpected robot status format: {status!r}")
    if status[0] != 0 or any(value != 0 for value in status[2:8]) or any(status[8:]):
        raise RuntimeError(f"Robot reported a fault: {status!r}")


def _check_state_angles(values):
    if (
        not isinstance(values, list)
        or len(values) != 6
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
    ):
        raise ValueError(f"Invalid joint feedback: {values!r}")


def _sdk_call_completed(method, result):
    """Recognize successful returns used by the sequential streaming loop."""
    if method in {"send_angles", "stop", "set_fresh_mode", "set_vr_mode", "set_pro_gripper_angle"}:
        return type(result) is int and result == 1
    try:
        if method == "get_angles":
            _check_state_angles(result)
        elif method == "get_robot_status":
            _check_robot_status(result)
        else:
            return False
    except (ValueError, RuntimeError):
        return False
    return True


def _modbus_crc(data):
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def _f100_reply_matches(frame, expected):
    """Validate complete outer/nested frames, checksums, device and register."""
    if expected is None or len(frame) != 14 or bytes(frame[:4]) != b"\xfe\xfe\x0b\xb5":
        return False
    payload = bytes(frame[4:12])
    return (
        payload[:4] == expected
        and _modbus_crc(payload[:6]) == int.from_bytes(payload[6:], "little")
        and _modbus_crc(frame[:-2]) == int.from_bytes(frame[-2:], "big")
    )


def _install_f100_reply_validation(client):
    """Validate nested F100 replies without waiting for a lost packet or retrying.

    Invalid frames return the SDK's failure sentinel immediately. The caller may
    perform a bounded read retry or repeat the same absolute opening, within its
    original deadline. Explicit device failures remain distinguishable.
    """
    trace, frames_seen = deque(maxlen=12), deque(maxlen=16)
    parser, transaction = client._process_received, client.tool_serial_write_data

    def receive(buffer):
        frames, remain = parser(buffer)
        stamp = time.monotonic()
        frames_seen.extend((stamp, bytes(frame)) for frame in frames if frame[3] == 0xB5)
        return frames, remain

    def transfer(command):
        if len(command) != 8 or command[1] not in (3, 6):
            raise ValueError("Only F100 register transactions are supported")
        expected, started = bytes(command[:4]), time.monotonic()
        response = transaction(command)
        frames = [frame for stamp, frame in list(frames_seen) if stamp >= started]
        valid = (
            isinstance(response, (list, bytes, bytearray))
            and len(response) == 8
            and any(_f100_reply_matches(frame, expected) and list(frame[4:12]) == list(response) for frame in frames)
        )
        failure = None if valid else "invalid_reply" if frames else "missing_reply"
        if valid and command[1] == 6 and list(response[4:6]) != [0, 1]:
            failure = "device_rejected"
        for frame in frames:
            payload = frame[4:-2]
            if (
                len(payload) == 5
                and payload[:2] == bytes([command[0], command[1] | 0x80])
                and _modbus_crc(payload[:3]) == int.from_bytes(payload[3:], "little")
            ):
                failure = "device_rejected"
        trace.append(
            dict(
                started_monotonic_s=started,
                elapsed_ms=(time.monotonic() - started) * 1000,
                command=list(command),
                response=list(response) if isinstance(response, (bytes, bytearray)) else response,
                failure_kind=failure,
                frames=[frame.hex() for frame in frames],
            )
        )
        return response if valid else -1

    client._process_received, client.tool_serial_write_data = receive, transfer
    return trace


def _sdk_call(client, method, *args):
    """Call the pinned SDK, then retire only records from completed requests.

    4.0.7 retains acknowledged motion genres in ``write_command`` when sync_mode
    is False, and also retains unsolicited 0x5B arrival notifications. The worker
    makes one public call at a time; its only concurrent activity is the SDK reader.
    No socket bytes, acknowledgements or data replies are drained here.
    """
    before = None
    if hasattr(client, "write_command"):
        with client.lock:
            before = Counter(client.write_command)
    if method == "get_robot_status":
        # The SDK prints "Robot is moving" during normal motion. Preserve its
        # returned fault flags without corrupting the parent's JSON report.
        with redirect_stdout(io.StringIO()):
            result = getattr(client, method)(*args)
    else:
        result = getattr(client, method)(*args)
    if before is None or not _sdk_call_completed(method, result):
        return result
    # Match _mesg's lock_out -> lock order. The reader only takes lock, so it
    # cannot append while the lists are edited and no reverse lock cycle is added.
    with client.lock_out, client.lock:
        if client.sync_mode is not False:
            return result
        surplus = Counter(client.write_command) - before
        for index in range(len(client.write_command) - 1, -1, -1):
            genre = client.write_command[index]
            if surplus[genre] > 0:
                del client.write_command[index]
                surplus[genre] -= 1
        client.read_command[:] = [
            entry
            for entry in client.read_command
            if not (len(entry[0]) == 7 and entry[0][:2] == b"\xfe\xfe" and entry[0][2] == 4 and entry[0][3] == 0x5B)
        ]
    return result


def encode_sdk_angles(values):
    """Represent six centidegree targets without the SDK's float truncation drift."""
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 6
        or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 360 for v in values)
    ):
        raise ValueError("Expected six finite SDK angle targets")
    encoded = []
    for value in values:
        units = round(value * 100)
        if not math.isclose(value, units / 100, rel_tol=0, abs_tol=1e-8):
            raise ValueError("SDK angle targets must use centidegree precision")
        wire_value = units / 100
        for _ in range(4):
            if int(wire_value * 100) == units:
                break
            wire_value = math.nextafter(wire_value, math.inf if units > 0 else -math.inf)
        else:
            raise ValueError("Cannot encode the reviewed SDK angle exactly")
        encoded.append(wire_value)
    return encoded


def validate_home_path(path, *, full_home=False):
    """Validate the complete synchronized home path without connecting."""
    if full_home is not True:
        raise ValueError("Only full-home paths are supported; full_home must be the boolean True")
    if not isinstance(path, (list, tuple)) or not 2 <= len(path) <= 301:
        raise ValueError("Home requires a start and 1..300 full-home waypoints")
    result = []
    for pose in path:
        encode_sdk_angles(pose)
        units = tuple(round(v * 100) for v in pose)
        if any(abs(v) > limit for v, limit in zip(units, (16200, 12500, 15400, 16200, 16200, 16500), strict=True)):
            raise ValueError("Home path exceeds conservative SDK joint bounds")
        if result:
            if max(abs(a - b) for a, b in zip(units, result[-1], strict=True)) > 50:
                raise ValueError("Home waypoint steps must not exceed 0.5 degrees")
            if max(abs(a - b) for a, b in zip(units, result[0], strict=True)) > 15000:
                raise ValueError("Home path exceeds its 150-degree full-home envelope")
        result.append(units)
    count = max((abs(b - a) + 49) // 50 for a, b in zip(result[0], result[-1], strict=True))
    if count != len(result) - 1:
        raise ValueError("Full-home waypoint count must match the reviewed endpoint distance")
    for tick, pose in enumerate(result):
        expected = tuple(a + round((b - a) * tick / count) for a, b in zip(result[0], result[-1], strict=True))
        if pose != expected:
            raise ValueError("Full-home waypoints must follow the reviewed synchronized joint line")
    return tuple(tuple(v / 100 for v in pose) for pose in result)


def _authorize_operation(
    method,
    args,
    writable,
    home_path=None,
    allow_gripper_open=False,
    home_speed=1,
    gripper_range=None,
):
    if method == "read_state" and args:
        raise ValueError("read_state accepts no arguments or arbitrary operation lists")
    if method in READ_PARAMETER_RANGES and (
        len(args) != 1 or type(args[0]) is not int or args[0] not in READ_PARAMETER_RANGES[method]
    ):
        raise ValueError(f"Invalid read-only diagnostic selector for {method}: {args!r}")
    if method in READ_METHODS:
        return
    if home_path is not None:
        if method == "stop" and args == (1,) and type(args[0]) is int:
            return
        if (
            allow_gripper_open
            and method == "set_pro_gripper_angle"
            and args == (100, 14)
            and all(type(v) is int for v in args)
        ):
            return
        if (
            method == "send_angles"
            and len(args) == 2
            and type(args[1]) is int
            and type(home_speed) is int
            and 1 <= home_speed <= 10
            and args[1] == home_speed
            and isinstance(args[0], (list, tuple))
            and len(args[0]) == 6
            and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 360 for v in args[0])
            and tuple(int(v * 100) for v in args[0]) in {tuple(round(v * 100) for v in pose) for pose in home_path[1:]}
        ):
            return
    elif writable:
        if method in {"send_angles", "stop"}:
            return
        if method in {"set_fresh_mode", "set_vr_mode"} and args == (1,) and type(args[0]) is int:
            return
        if (
            method == "set_pro_gripper_angle"
            and gripper_range is not None
            and len(args) == 2
            and all(type(value) is int for value in args)
            and args[1] == 14
            and gripper_range[0] <= args[0] <= gripper_range[1]
        ):
            return
    raise ValueError(f"Disallowed robot operation: {method}")


class _HomePathGuard:
    """Worker-side sequence, time, mode and opening checks; counters precede each send."""

    def __init__(self, path, allow_gripper_open=False, sdk_speed=1, full_home=False):
        if type(sdk_speed) is not int or not 1 <= sdk_speed <= 10:
            raise ValueError("Home path capability permits SDK speed 1..10")
        self.path = validate_home_path(path, full_home=full_home)
        self.full_home = full_home
        self.budget_s = 300
        self.units = [tuple(round(v * 100) for v in pose) for pose in self.path]
        self.allow_gripper_open = allow_gripper_open
        self.sdk_speed = sdk_speed
        self.index = 0
        self.started = self.last_send = None
        self.initial_stop_ack = self.open_attempted = self.finished = self.gripper_open = False
        self.state = {}

    def before(self, method, args, now):
        _authorize_operation(
            method,
            args,
            False,
            home_path=self.path,
            allow_gripper_open=self.allow_gripper_open,
            home_speed=self.sdk_speed,
        )
        if method == "stop":
            self.started = now if self.started is None else self.started
            if self.index or self.open_attempted:
                self.finished = True  # A stop after any motion attempt ends this capability.
            return
        if method in READ_METHODS:
            return
        if self.finished or self.started is None or now - self.started >= self.budget_s or not self.initial_stop_ack:
            raise RuntimeError("Home path is stopped, expired, or lacks its initial stop acknowledgement")
        if any(
            self.state.get(key) != value
            for key, value in (("is_power_on", 1), ("get_control_mode", 0), ("get_fresh_mode", 1))
        ):
            raise RuntimeError("Home path requires existing power, position mode and refresh mode 1")
        if method == "set_pro_gripper_angle":
            if self.open_attempted or self.index:
                raise RuntimeError("Home path permits one gripper opening before arm motion")
            self.open_attempted = True
            self.gripper_open = False
        elif method == "send_angles":
            units = tuple(int(v * 100) for v in args[0])
            if not self.gripper_open:
                raise RuntimeError("F100 opening 95..100 must be read back before arm motion")
            if self.index >= len(self.path) - 1 or units != self.units[self.index + 1]:
                raise RuntimeError("Home path accepts only the next reviewed waypoint, once")
            if self.last_send is not None and now - self.last_send < 0.2 - 1e-9:
                raise RuntimeError("Home path waypoint interval must be at least 0.2 seconds")
            self.index += 1
            self.last_send = now

    def after(self, method, args, result):
        if method == "stop" and not self.index and not self.open_attempted:
            self.initial_stop_ack = result == 1
        elif method in {"is_power_on", "get_control_mode", "get_fresh_mode"}:
            self.state[method] = result
        elif method == "get_pro_gripper_angle" and args == (14,):
            self.gripper_open = type(result) in (int, float) and math.isfinite(result) and 95 <= result <= 100


def _sdk_worker(
    connection,
    host,
    port,
    writable,
    home_path=None,
    allow_gripper_open=False,
    home_speed=1,
    home_full=False,
    gripper_range=None,
    validate_f100_replies=True,
):
    client = None
    # The control process handles Ctrl+C and must retain this connection to stop.
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        version = importlib.metadata.version("pymycobot")
        if version != "4.0.7":
            raise RuntimeError(f"Expected pymycobot 4.0.7, found {version}")
        from pymycobot import Pro450Client

        client = Pro450Client(host, port)
        tool_trace = _install_f100_reply_validation(client) if validate_f100_replies else []
        # In 4.0.7 this changes only the Python wait mode; it sends no robot command.
        # Keep acknowledgement checking: do not use send_angles(_async=True).
        client.set_limit_switch(2, 0)
        connection.send((True, version))
        home_guard = None if home_path is None else _HomePathGuard(home_path, allow_gripper_open, home_speed, home_full)
        while True:
            method, args = connection.recv()
            if method is None:
                break  # Close the socket normally; this sends no robot command.
            _authorize_operation(
                method,
                args,
                writable,
                home_path,
                allow_gripper_open,
                home_speed,
                gripper_range,
            )
            if home_guard is not None:
                home_guard.before(method, args, time.monotonic())
            if method == "read_state":
                # A fixed read-only pair, not a generic batch API. The SDK prints
                # normal motion status; preserve the raw status for parent validation.
                with redirect_stdout(io.StringIO()):
                    status = _sdk_call(client, "get_robot_status")
                try:
                    _check_robot_status(status)
                except RuntimeError:
                    # Data faults must not close the connection needed to send stop.
                    # Do not perform the second read after an abnormal status.
                    values, received = None, None
                else:
                    values = _sdk_call(client, "get_angles")
                    received = time.monotonic()
                result = {"status": status, "angles": values, "received": received}
            else:
                result = _sdk_call(client, method, *args)
            if home_guard is not None:
                home_guard.after(method, args, result)
            if method in {"get_pro_gripper_angle", "set_pro_gripper_angle"} and (
                result == -1 or (tool_trace and tool_trace[-1]["failure_kind"])
            ):
                connection.send((True, result, list(tool_trace)))
            else:
                connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        with suppress(BrokenPipeError, EOFError):
            connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        if client is not None:
            client.close()
        connection.close()
        signal.signal(signal.SIGINT, previous_sigint)


def _stop_angles(values):
    if len(values) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Expected six finite SDK joint angles in degrees")
    return list(values)


def _observe_stop(
    robot,
    anchor,
    pre_stop,
    pre_stop_received,
    details,
    *,
    clock,
    sleep,
    monitor_all_joint_speeds=False,
    max_speed_deg_s=3,
    max_displacement_deg=0.15,
):
    """Observe bounded deceleration; a protocol ACK alone never establishes rest.

    The one-second budget starts at the stop request, including its ACK latency.
    A complete >=0.3-second window of >=6 replies must have <=0.02-degree range
    on every joint. Existing motion guards stay active, with an additional
    configurable displacement limit relative to the last pre-stop feedback.
    Host receipt times provide an observation bound, not a hardware guarantee.
    """
    if not 0 < max_speed_deg_s <= 30 or not 0 < max_displacement_deg <= 0.75:
        raise ValueError("Invalid stop observation limits")
    displacement_units = round(max_displacement_deg * 100)
    request_time = details["request_monotonic_s"]
    deadline = request_time + 1.0
    if (
        pre_stop is None
        or type(pre_stop_received) not in (int, float)
        or not math.isfinite(pre_stop_received)
        or pre_stop_received > request_time
    ):
        raise RuntimeError("No valid pre-stop feedback; physical stop remains unverified")

    def units(values):
        values = _stop_angles(values)
        result = [round(value * 100) for value in values]
        if any(
            not math.isclose(value, unit / 100, rel_tol=0, abs_tol=1e-8)
            for value, unit in zip(values, result, strict=True)
        ):
            raise ValueError("Unexpected precision in SDK joint feedback")
        return result

    initial, permitted = units(pre_stop), units(anchor)
    previous, previous_time = initial, pre_stop_received
    window = deque()

    def bounded_request(function):
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Stop observation exceeded its one-second budget")
        original_timeout = getattr(robot, "timeout", None)
        if original_timeout is not None:
            robot.timeout = min(original_timeout, remaining)
        try:
            return function()
        finally:
            if original_timeout is not None:
                robot.timeout = original_timeout

    while True:
        if clock() >= deadline:
            raise TimeoutError("Stop observation exceeded its one-second budget")
        row = {"requested_after_stop_s": clock() - request_time}
        details["samples"].append(row)
        try:
            bounded_request(robot.check_fault)
            q, received, latency = bounded_request(robot.read)
            row["raw_feedback"] = repr(q)
            row["raw_received_monotonic_s"] = repr(received)
            row["raw_read_latency_s"] = repr(latency)
            q = _stop_angles(q)
            measured = units(q)
            row["q_deg"] = q
            if not all(type(v) in (int, float) and math.isfinite(v) for v in (received, latency)) or latency < 0:
                raise ValueError("Invalid stop-feedback timing")
            row.update(received_after_stop_s=received - request_time, read_latency_ms=1000 * latency)
            details["last_feedback_deg"] = q
            if clock() > deadline or received > deadline:
                raise TimeoutError("Stop observation exceeded its one-second budget")
            excursion = [abs(a - b) for a, b in zip(measured, permitted, strict=True)]
            if excursion[0] > 75 or max(excursion[1:]) > displacement_units:
                raise RuntimeError("J1 left its monitoring envelope or another joint drifted during stopping")
            displacement = max(abs(a - b) for a, b in zip(measured, initial, strict=True))
            row["max_displacement_from_pre_stop_deg"] = displacement / 100
            if displacement > displacement_units:
                raise RuntimeError(
                    f"Stop displacement exceeded {max_displacement_deg:g} degrees; use the hardware E-stop"
                )
            dt = received - previous_time
            if not 0 < dt <= 0.1:
                raise TimeoutError("Stop-feedback gap exceeded 100 ms or timestamp did not increase")
            speeds = [abs(a - b) / (100 * dt) for a, b in zip(measured, previous, strict=True)]
            row["j1_speed_deg_s"] = speeds[0]
            row["joint_speeds_deg_s"] = speeds
            if (max(speeds) if monitor_all_joint_speeds else speeds[0]) > max_speed_deg_s + 1e-9:
                joint_label = "joint" if monitor_all_joint_speeds else "J1"
                raise RuntimeError(
                    f"Measured {joint_label} speed exceeded {max_speed_deg_s:g} degrees/s during stopping"
                )
            previous, previous_time = measured, received
            window.append((received, measured))
            while len(window) > 6 and received - window[1][0] >= 0.3:
                window.popleft()
            ranges = [max(item[1][j] for item in window) - min(item[1][j] for item in window) for j in range(6)]
            row["window_range_deg"] = [value / 100 for value in ranges]
            if len(window) >= 6 and received - window[0][0] >= 0.3 - 1e-9 and max(ranges) <= 2:
                details["stationary"] = True
                details["observed_stop_time_upper_bound_s"] = received - request_time
                return
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        sleep(min(0.05, max(0, deadline - clock())))


def prepare_stream(robot, *, sleep=time.sleep, low_latency=False):
    """Prepare a stationary arm, optionally selecting the short motion buffer for Reach."""
    state = {name: robot.call(name) for name in ("is_power_on", "get_control_mode", "get_fresh_mode")}
    if state["is_power_on"] != 1 or state["get_control_mode"] != 0 or state["get_fresh_mode"] not in (0, 1):
        raise RuntimeError(f"Robot is not ready for position streaming: {state}")
    samples = []
    for _ in range(4):
        angles, _, _ = robot.read_state(timeout=0.1)
        samples.append(angles)
        sleep(0.05)
    if any(max(abs(a - b) for a, b in zip(samples[0], row, strict=True)) > 0.03 for row in samples[1:]):
        raise RuntimeError("Robot moved during stream preparation; mode was not changed")
    opening = robot.call("get_pro_gripper_angle", 14, timeout=1.0)
    if type(opening) not in (int, float) or not math.isfinite(opening) or not 0 <= opening <= 100:
        raise RuntimeError("Prepare streaming requires valid F100 feedback")
    changed = state["get_fresh_mode"] != 1
    acknowledged = not changed or robot.call("set_fresh_mode", 1, timeout=5.0) == 1
    if robot.call("get_fresh_mode") != 1:
        raise RuntimeError("Fresh-target mode readback did not confirm mode 1")
    result = dict(
        mode="prepare_stream",
        error=None,
        mode_changed=changed,
        mode_write_acknowledged=acknowledged,
        fresh_mode=1,
        arm_motion_commands=0,
        gripper_motion_commands=0,
        measured_angles_deg=samples[-1],
        gripper_opening_sdk=opening,
    )
    if low_latency:
        before = robot.call("get_vr_mode")
        if type(before) is not int or before not in (0, 1):
            raise RuntimeError(f"Invalid VR buffer mode: {before!r}")
        # Pro450 protocol 0x7A: 1 selects the two-command buffer; SDK 4.0.7's docstring is reversed.
        if before != 1 and robot.call("set_vr_mode", 1, timeout=5.0) != 1:
            raise RuntimeError("VR buffer mode write was not acknowledged")
        if robot.call("get_vr_mode") != 1:
            raise RuntimeError("VR buffer mode readback did not confirm mode 1")
        result.update(vr_mode=1, vr_mode_changed=before != 1)
    robot.check_fault()
    return result


class NativePro450Interface:
    """Direct, single-connection public SDK calls, without parser/queue patches.

    Timeout arguments describe the caller's budget; SDK-native waits cannot be
    cancelled here. The deployment loop checks its deadline before any new target.
    """

    def __init__(self, host, port=4500, *, writable=False, gripper_range=None):
        from pymycobot import Pro450Client

        if importlib.metadata.version("pymycobot") != "4.0.7":
            raise RuntimeError("Native deployment requires pymycobot 4.0.7")
        self.host, self.port, self.writable, self.gripper_range = host, port, writable, gripper_range
        self.validate_f100_replies = False
        self.client = Pro450Client(host, port)
        if writable:
            self.client.set_limit_switch(2, 0)  # Public SDK setting: ACK, not arrival, completes send_angles.

    def call(self, method, *args, timeout=None):
        _authorize_operation(method, args, self.writable, gripper_range=self.gripper_range)
        if method == "get_pro_gripper_angle":
            return self.client.get_pro_gripper_angle(*args)
        if method == "set_pro_gripper_angle":
            return self.client.set_pro_gripper_angle(*args)
        return getattr(self.client, method)(*args)

    def read_state(self, *, timeout=None):
        before = time.monotonic()
        with redirect_stdout(io.StringIO()):
            status = self.client.get_robot_status()
        _check_robot_status(status)
        angles = self.client.get_angles()
        received = time.monotonic()
        _check_state_angles(angles)
        return angles, received, received - before

    def read(self):
        """Public joint read used by the shared stationary-start/home checks."""
        before = time.monotonic()
        angles = self.client.get_angles()
        received = time.monotonic()
        _check_state_angles(angles)
        return angles, received, received - before

    def check_fault(self):
        with redirect_stdout(io.StringIO()):
            _check_robot_status(self.client.get_robot_status())

    def check_ready(self):
        from .reach_home import read_f100_opening

        for method, expected in (("is_power_on", 1), ("get_control_mode", 0), ("get_fresh_mode", 1)):
            if self.call(method) != expected:
                raise RuntimeError(f"Native SDK requires {method}={expected}")
        self.startup_f100_reads = []
        opening = read_f100_opening(
            self, attempts=3, diagnostics=self.startup_f100_reads, context="policy-start F100 feedback"
        )
        if opening != 100:
            raise RuntimeError("Lift policy requires initial F100 opening 100; run lift_home.py --open-gripper")
        self.check_fault()

    def send(self, angles_deg, speed, timeout):
        _authorize_operation("send_angles", (angles_deg, speed), self.writable)
        if self.client.send_angles(angles_deg, speed) != 1:
            raise RuntimeError("Native send_angles was not acknowledged")

    def stop_and_observe(self, *, clock=time.monotonic, sleep=time.sleep):
        _authorize_operation("stop", (1,), self.writable)
        started = clock()
        result = dict(
            stop_acknowledged=False,
            post_stop_stationary=False,
            samples=[],
            error=None,
            stop_request_monotonic_s=started,
            sdk_native_waits=True,
        )
        try:
            result["stop_acknowledged"] = self.client.stop(1) == 1
            result["stop_ack_latency_ms"] = (clock() - started) * 1000
            if not result["stop_acknowledged"]:
                raise RuntimeError("Native SDK stop was not acknowledged")
            deadline = clock() + 2
            window = deque()
            while clock() < deadline:
                q, _, _ = self.read_state()
                stamp = clock()
                result["samples"].append(dict(q_deg=q, received_after_stop_s=stamp - started))
                window.append((stamp, q))
                while len(window) > 6 and stamp - window[1][0] >= 0.3:
                    window.popleft()
                if (
                    len(window) >= 6
                    and stamp - window[0][0] >= 0.3
                    and all(
                        max(v[1][j] for v in window) - min(v[1][j] for v in window) <= 0.02 + 1e-9 for j in range(6)
                    )
                ):
                    result["post_stop_stationary"] = True
                    break
                sleep(0.05)
            if not result["post_stop_stationary"]:
                raise TimeoutError("Native stop observation did not settle")
        except Exception as exc:
            result["error"] = str(exc)
        result["elapsed_s"] = clock() - started
        return result

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Pro450Interface:
    def __init__(
        self,
        host,
        port=4500,
        *,
        writable=False,
        timeout=0.1,
        connect_timeout=10.0,
        home_path=None,
        allow_gripper_open=False,
        home_speed=1,
        home_full=False,
        gripper_range=None,
        validate_f100_replies=True,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        if not math.isfinite(connect_timeout) or connect_timeout <= 0:
            raise ValueError("Connection timeout must be finite and positive")
        if type(home_full) is not bool or (home_full and home_path is None):
            raise ValueError("Full-home mode requires its own reviewed path capability")
        if gripper_range is not None:
            if (
                not writable
                or not isinstance(gripper_range, (list, tuple))
                or len(gripper_range) != 2
                or any(type(value) is not int for value in gripper_range)
                or not 0 <= gripper_range[0] < gripper_range[1] <= 100
            ):
                raise ValueError("Gripper actuation requires a writable calibrated SDK interval")
            gripper_range = tuple(gripper_range)
        if home_path is not None:
            if writable:
                raise ValueError("Home path requires its own bounded capability")
            home_path = validate_home_path(home_path, full_home=home_full)
        if type(allow_gripper_open) is not bool or (allow_gripper_open and home_path is None):
            raise ValueError("Gripper opening requires the separate home-path capability")
        if type(home_speed) is not int or not 1 <= home_speed <= 10:
            raise ValueError("Home path capability permits SDK speed 1..10")
        self.host, self.port, self.writable, self.timeout = host, port, writable, timeout
        if type(validate_f100_replies) is not bool:
            raise ValueError("validate_f100_replies must be boolean")
        self.validate_f100_replies = validate_f100_replies
        self.home_path, self.allow_gripper_open = home_path, allow_gripper_open
        self.home_speed = home_speed
        self.home_full = home_full
        self.gripper_range = gripper_range
        self.process = None
        self.connection = None
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_sdk_worker,
            args=(
                child,
                host,
                port,
                writable,
                home_path,
                allow_gripper_open,
                home_speed,
                home_full,
                gripper_range,
                validate_f100_replies,
            ),
            daemon=True,
        )
        self.process.start()
        child.close()
        try:
            self.version = self._receive(connect_timeout)
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"Pro450 SDK response exceeded {timeout:.4f} s")
        ok, result, *diagnostics = self.connection.recv()
        if diagnostics:
            self.tool_uart_trace = diagnostics[0]
        if not ok:
            raise RuntimeError(result)
        return result

    def call(self, method, *args, timeout=None):
        _authorize_operation(
            method,
            args,
            self.writable,
            getattr(self, "home_path", None),
            getattr(self, "allow_gripper_open", False),
            getattr(self, "home_speed", 1),
            getattr(self, "gripper_range", None),
        )
        if self.process is None or not self.process.is_alive():
            raise ConnectionError("Robot connection is closed")
        try:
            self.connection.send((method, args))
            return self._receive(self.timeout if timeout is None else timeout)
        except BaseException:
            self.close(force=True)  # No retry or grace period for a timed-out command.
            raise

    def read(self):
        before = time.monotonic()
        values = self.call("get_angles")
        after = time.monotonic()
        if (
            not isinstance(values, list)
            or len(values) != 6
            or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
        ):
            raise ValueError(f"Invalid joint feedback: {values!r}")
        # Host receipt time only: the SDK supplies no acquisition timestamp.
        return values, after, after - before

    def read_state(self, *, timeout=None):
        """Check faults and read angles in one IPC, retaining both SDK requests.

        Python's monotonic clock shares a system-wide origin across processes on
        this Linux runtime. The worker timestamps the angle reply before returning
        through IPC; this is still a host receipt time, not device acquisition time.
        Total latency includes both SDK reads and the complete IPC round trip.
        """
        before = time.monotonic()
        result = self.call("read_state", timeout=timeout)
        after = time.monotonic()
        if not isinstance(result, dict) or set(result) != {"status", "angles", "received"}:
            raise ValueError(f"Invalid read_state reply: {result!r}")
        # Validate outside call() so bad feedback retains the live stop connection.
        # Actual SDK/transport exceptions still follow call()'s cancellation path.
        _check_robot_status(result["status"])
        values, received = result["angles"], result["received"]
        _check_state_angles(values)
        if type(received) not in (int, float) or not math.isfinite(received) or not before <= received <= after:
            raise ValueError(f"Invalid worker receipt timestamp: {received!r}")
        return values, received, after - before

    def check_ready(self, gripper_id=14):
        for method, expected in (("is_power_on", 1), ("get_control_mode", 0), ("get_fresh_mode", 1)):
            value = self.call(method)
            if value != expected:
                raise RuntimeError(f"{method}={value}; required {expected}. Configure the robot before deployment.")
        opening = self.call("get_pro_gripper_angle", gripper_id, timeout=1.0)
        if not isinstance(opening, (int, float)) or not math.isfinite(opening) or not 95 <= opening <= 100:
            raise RuntimeError(f"F100 must already be open (95..100), got {opening}")
        self.check_fault()

    def check_fault(self):
        status = self.call("get_robot_status")
        _check_robot_status(status)

    def send(self, angles_deg, speed, timeout):
        result = self.call("send_angles", angles_deg, speed, timeout=timeout)
        if result != 1:
            raise RuntimeError(f"send_angles was not acknowledged: {result!r}")

    def stop(self):
        if not self.writable and getattr(self, "home_path", None) is None:
            raise RuntimeError("Read-only sessions cannot send stop commands")
        # Use a new connection when a stalled SDK process was cancelled.
        if self.process is None or not self.process.is_alive():
            with Pro450Interface(
                self.host,
                self.port,
                writable=self.writable,
                home_path=getattr(self, "home_path", None),
                allow_gripper_open=getattr(self, "allow_gripper_open", False),
                home_speed=getattr(self, "home_speed", 1),
                home_full=getattr(self, "home_full", False),
                timeout=1.0,
                connect_timeout=2.0,
            ) as emergency:
                result = emergency.call("stop", 1, timeout=1.0)
        else:
            result = self.call("stop", 1, timeout=1.0)
        if result != 1:
            raise RuntimeError(f"Stop was not acknowledged: {result!r}; use the hardware emergency stop")

    def stop_and_observe(self, *, clock=time.monotonic, sleep=time.sleep):
        """Request stop, then observe all six joints within a shared two-second I/O budget.

        A protocol ACK does not establish rest. Host feedback must cover at least
        0.3 seconds and six samples, with no joint varying by more than 0.02 degrees.
        Failed SDK process cleanup may take additional time; it is reported separately.
        Failures are returned so a caller can retain its original control exception.
        """
        started = clock()
        deadline = started + 2.0
        report = {
            "stop_acknowledged": False,
            "post_stop_stationary": False,
            "error": None,
            "stop_error": None,
            "observation_error": None,
            "samples": [],
            "observed_stop_time_upper_bound_s": None,
            "stop_request_monotonic_s": started,
            "stop_ack_monotonic_s": None,
            "stop_ack_latency_ms": None,
            "io_budget_s": 2.0,
            "stationary_window_s": 0.3,
            "stationary_range_deg": 0.02,
            "minimum_samples": 6,
            "reconnected_for_stop": False,
            "readonly_observation_connection": False,
            "cleanup_errors": [],
            "note": "Host receipt timestamps bound the observation; they do not certify hardware stopping time.",
        }
        owned = []
        stage = "stop"

        def remaining():
            value = deadline - clock()
            if value <= 0:
                raise TimeoutError("Stop observation exceeded its two-second I/O budget")
            return value

        def peer_connection(*, stop_capability=False):
            budget = remaining()
            options = {"timeout": min(0.2, budget), "connect_timeout": budget}
            if stop_capability:
                options.update(
                    writable=self.writable,
                    home_path=getattr(self, "home_path", None),
                    allow_gripper_open=getattr(self, "allow_gripper_open", False),
                    home_speed=getattr(self, "home_speed", 1),
                    home_full=getattr(self, "home_full", False),
                )
            peer = Pro450Interface(self.host, self.port, **options)
            owned.append(peer)
            remaining()  # A successful connection can still finish after the deadline.
            return peer

        def bounded(peer, function):
            original_timeout = peer.timeout
            peer.timeout = min(original_timeout, remaining())
            try:
                return function()
            finally:
                peer.timeout = original_timeout

        try:
            _authorize_operation(
                "stop",
                (1,),
                self.writable,
                getattr(self, "home_path", None),
                getattr(self, "allow_gripper_open", False),
                getattr(self, "home_speed", 1),
            )
            live = self.process is not None and self.process.is_alive()
            peer = self
            if not live:
                report["reconnected_for_stop"] = True
                peer = peer_connection(stop_capability=True)
            try:
                result = peer.call("stop", 1, timeout=min(1.0, remaining()))
            except (OSError, EOFError) as exc:
                if not live:
                    raise
                # The worker may die after is_alive(); only stop may be resent.
                report["initial_stop_transport_error"] = f"{type(exc).__name__}: {exc}"
                self.close(force=True)
                live = False
                report["reconnected_for_stop"] = True
                peer = peer_connection(stop_capability=True)
                result = peer.call("stop", 1, timeout=min(1.0, remaining()))
            acknowledged = clock()
            report["stop_ack_monotonic_s"] = acknowledged
            report["stop_ack_latency_ms"] = 1000 * (acknowledged - started)
            if result != 1:
                raise RuntimeError(f"Stop was not acknowledged: {result!r}")
            report["stop_acknowledged"] = True
            stage = "observation"
            remaining()
            if not live:
                peer.close()
                owned.remove(peer)
                report["readonly_observation_connection"] = True
                peer = peer_connection()
            previous_received = None
            window = deque()
            while True:
                bounded(peer, peer.check_fault)
                q, received, latency = bounded(peer, peer.read)
                if (
                    not isinstance(q, (list, tuple))
                    or len(q) != 6
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in q)
                ):
                    raise ValueError("Expected six finite stop-feedback angles")
                if (
                    type(received) not in (int, float)
                    or not math.isfinite(received)
                    or type(latency) not in (int, float)
                    or not math.isfinite(latency)
                    or latency < 0
                    or received < acknowledged
                    or received > clock()
                    or (previous_received is not None and received <= previous_received)
                ):
                    raise ValueError("Invalid or non-increasing stop-feedback timing")
                row = {"q_deg": list(q), "received_after_stop_s": received - started, "read_latency_ms": 1000 * latency}
                report["samples"].append(row)
                remaining()
                units = tuple(round(value * 100) for value in q)
                if any(
                    not math.isclose(value, unit / 100, rel_tol=0, abs_tol=1e-8)
                    for value, unit in zip(q, units, strict=True)
                ):
                    raise ValueError("Unexpected precision in SDK stop-feedback angles")
                previous_received = received
                window.append((received, units))
                while len(window) > 6 and received - window[1][0] >= 0.3:
                    window.popleft()
                ranges = [max(frame[1][j] for frame in window) - min(frame[1][j] for frame in window) for j in range(6)]
                row["window_range_deg"] = [value / 100 for value in ranges]
                if len(window) >= 6 and received - window[0][0] >= 0.3 - 1e-9 and max(ranges) <= 2:
                    report["post_stop_stationary"] = True
                    report["observed_stop_time_upper_bound_s"] = received - started
                    break
                sleep(min(0.05, remaining()))
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            report[f"{stage}_error"] = report["error"]
        finally:
            observed = clock()
            report["io_elapsed_s"] = observed - started
            for peer in reversed(owned):
                try:
                    peer.close()
                except BaseException as exc:
                    report["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
            report["cleanup_elapsed_s"] = clock() - observed
            report["elapsed_s"] = clock() - started
        return report

    def close(self, *, force=False):
        if self.process is not None:
            if not force and self.process.is_alive() and self.connection is not None:
                with suppress(BrokenPipeError, EOFError, OSError):
                    self.connection.send((None, ()))
                self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
