"""One-way, best-effort Reach feedback for a viewer on the same computer.

Packets never contain robot commands. A slow or absent viewer cannot block the
controller. Readers may select the newest state or consume a bounded history.
"""

import json
import math
import socket
import time

DEFAULT_PORT = 14550
MAX_PACKET_BYTES = 8192
MAX_POLL_PACKETS = 256


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_state(state):
    """Validate v1 feedback and optional raw policy I/O without changing it.

    policy_monotonic_s is the feedback receipt time used by that inference.
    (run_id, policy_step) identifies one sample, including when repeated at stop.
    """
    if not isinstance(state, dict) or type(state.get("schema")) is not int or state["schema"] != 1:
        raise ValueError("Expected a Reach telemetry schema-1 object")
    if state.get("type") != "reach_state":
        raise ValueError("Expected reach_state telemetry")
    for name in ("run_id", "policy_sha256"):
        value = state.get(name)
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError(f"Invalid telemetry {name}")
    for name in ("seq", "goal_index"):
        if type(state.get(name)) is not int or state[name] < 0:
            raise ValueError(f"Invalid telemetry {name}")
    if not _finite_number(state.get("monotonic_s")) or state["monotonic_s"] < 0:
        raise ValueError("Invalid telemetry monotonic_s")
    if "published_monotonic_s" in state and (
        not _finite_number(state["published_monotonic_s"]) or state["published_monotonic_s"] < state["monotonic_s"]
    ):
        raise ValueError("Invalid telemetry published_monotonic_s")
    if state.get("state") not in ("read_only", "running", "stopped", "fault"):
        raise ValueError("Invalid telemetry state")
    if type(state.get("sent")) is not bool:
        raise ValueError("Invalid telemetry sent flag")
    if "error" not in state or (state["error"] is not None and not isinstance(state["error"], str)):
        raise ValueError("Invalid telemetry error")
    for name, size in (("q_rad", 6), ("dq_rad_s", 6), ("target_rad", 6), ("goal_xyz_xyzw", 7)):
        values = state.get(name)
        if not isinstance(values, list) or len(values) != size or not all(_finite_number(value) for value in values):
            raise ValueError(f"Invalid telemetry {name}")
    policy_fields = ("policy_observation", "policy_action", "policy_step", "policy_monotonic_s")
    if any(name in state for name in policy_fields):
        if not all(name in state for name in policy_fields):
            raise ValueError("Telemetry policy sample requires observation, action, step and timestamp")
        for name, size in (("policy_observation", 43), ("policy_action", 6)):
            values = state[name]
            if not isinstance(values, list) or len(values) != size or not all(_finite_number(v) for v in values):
                raise ValueError(f"Invalid telemetry {name}")
        if type(state["policy_step"]) is not int or state["policy_step"] < 0:
            raise ValueError("Invalid telemetry policy_step")
        if (
            not _finite_number(state["policy_monotonic_s"])
            or not 0 <= state["policy_monotonic_s"] <= state["monotonic_s"]
        ):
            raise ValueError("Invalid telemetry policy_monotonic_s")
    quaternion = state["goal_xyz_xyzw"][3:]
    if not math.isclose(math.hypot(*quaternion), 1.0, abs_tol=1e-3):
        raise ValueError("Telemetry goal quaternion must be unit length in XYZW order")
    return state


def _open_socket(port, *, receiver):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Telemetry port must be an integer in 1..65535")
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.setblocking(False)
        if receiver:
            connection.bind(("127.0.0.1", port))
    except BaseException:
        connection.close()
        raise
    return connection


class TelemetryPublisher:
    """Nonblocking localhost publisher; send errors are counted, never retried."""

    def __init__(self, port=DEFAULT_PORT, *, validator=validate_state):
        self.validator = validator
        self._socket = _open_socket(port, receiver=False)
        self._address = ("127.0.0.1", port)
        self._closed = False
        self.sent = 0
        self.dropped = 0

    def publish(self, state):
        self.validator(state)
        payload = json.dumps(state, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > MAX_PACKET_BYTES:
            raise ValueError("Reach telemetry packet exceeds 8192 bytes")
        if self._closed:
            self.dropped += 1
            return False
        try:
            sent = self._socket.sendto(payload, self._address)
        except OSError:
            self.dropped += 1
            return False
        if sent != len(payload):
            self.dropped += 1
            return False
        self.sent += 1
        return True

    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TelemetryReceiver:
    """Read fresh states, rejecting reordered frames and retired run sessions.

    The publisher's timestamp is a host monotonic receipt time. Both processes
    must run on this computer. ``latest`` remains available after data goes stale;
    the viewer must label its age and must not extrapolate it as live motion.
    """

    def __init__(self, port=DEFAULT_PORT, *, max_age_s=1.0, clock=time.monotonic, validator=validate_state):
        self.validator = validator
        if not _finite_number(max_age_s) or max_age_s <= 0:
            raise ValueError("Telemetry max_age_s must be finite and positive")
        self._socket = _open_socket(port, receiver=True)
        self._closed = False
        self.max_age_s = max_age_s
        self.clock = clock
        self.latest = None
        self.dropped = 0
        self._retired_runs = set()

    def poll(self):
        """Return only the newest accepted packet, preserving the Reach API."""
        batch = self.poll_batch()
        return batch[-1] if batch else None

    def poll_batch(self, *, max_packets=MAX_POLL_PACKETS):
        """Return accepted packets in receive order, without discarding history.

        Age, session and sequence checks are identical to ``poll``. Consumers
        integrating physics must separately reject missing sequence numbers;
        this transport never invents or reorders measurements.
        """
        if type(max_packets) is not int or not 1 <= max_packets <= MAX_POLL_PACKETS:
            raise ValueError("Telemetry batch limit must be inside 1..256")
        if self._closed:
            return []
        batch = []
        for _ in range(max_packets):
            try:
                payload, address = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            except OSError:
                self.dropped += 1
                break
            if address[0] != "127.0.0.1" or len(payload) > MAX_PACKET_BYTES:
                self.dropped += 1
                continue
            try:
                state = self.validator(json.loads(payload))
            except (ValueError, TypeError, UnicodeError, RecursionError):
                self.dropped += 1
                continue
            now = self.clock()
            age = now - state["monotonic_s"]
            published_age = now - state["published_monotonic_s"] if "published_monotonic_s" in state else None
            terminal_update = (
                state["state"] in ("stopped", "fault")
                and self.latest is not None
                and state["run_id"] == self.latest["run_id"]
                and state["seq"] > self.latest["seq"]
                and published_age is not None
                and 0 <= published_age <= self.max_age_s
            )
            # A just-published stop failure may have no new joint feedback. Keep
            # its status without changing the measurement timestamp or treating
            # the frozen pose as current. Running frames still need fresh joints.
            if (
                age < 0
                or (age > self.max_age_s and not terminal_update)
                or (published_age is not None and published_age < 0)
                or state["run_id"] in self._retired_runs
            ):
                self.dropped += 1
                continue
            if self.latest is not None:
                same_run = state["run_id"] == self.latest["run_id"]
                if (
                    state["monotonic_s"] < self.latest["monotonic_s"]
                    or (same_run and state["seq"] <= self.latest["seq"])
                    or (not same_run and state["monotonic_s"] == self.latest["monotonic_s"])
                ):
                    self.dropped += 1
                    continue
                if not same_run:
                    # A restarted controller may reset seq, but a still-emitting
                    # older process cannot take the viewer back afterwards.
                    self._retired_runs.add(self.latest["run_id"])
            self.latest = state
            batch.append(state)
        return batch

    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
