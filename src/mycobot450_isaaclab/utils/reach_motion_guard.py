"""Explicit development guards; timestamps and velocities remain in real time.

This monitor never edits policy observations, targets, or controller state. Speed
is still a host-receipt finite difference, not a hardware acquisition measurement.
"""

import math
from collections import Counter, deque

import torch


class MotionGuard:
    """Separate following error, new-command lead, and sustained sampled speed."""

    FIELDS = {
        "version",
        "tracking_soft_rad",
        "new_target_soft_rad",
        "tracking_grace_s",
        "speed_soft_rad_s",
        "speed_window_s",
    }

    @classmethod
    def validate(cls, data):
        cfg = data.get("motion_guard")
        if not isinstance(cfg, dict) or set(cfg) != cls.FIELDS:
            raise ValueError("motion_guard requires exactly the documented version-1 fields")
        if type(cfg["version"]) is not int or cfg["version"] != 1:
            raise ValueError("motion_guard version must be integer 1")
        for name, value in [
            *((k, v) for k, v in cfg.items() if k != "version"),
            *(
                (k, data.get(k))
                for k in ("max_tracking_error_rad", "max_new_target_error_rad", "max_measured_velocity_rad_s")
            ),
        ]:
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid motion guard value: {name}")
        if not (
            cfg["tracking_soft_rad"] < data["max_tracking_error_rad"]
            and cfg["new_target_soft_rad"] < data["max_new_target_error_rad"]
            and cfg["speed_soft_rad_s"] < data["max_measured_velocity_rad_s"]
            and cfg["new_target_soft_rad"] >= cfg["tracking_soft_rad"]
            and data["max_new_target_error_rad"] >= data["max_tracking_error_rad"]
        ):
            raise ValueError("Soft limits must be below hard limits; new-target limits must allow command lead")
        lead = data.get("target_lead_limit_rad")
        gain = data.get("action_gain", 1.0)
        if type(gain) not in (float, int) or not math.isfinite(gain) or not 0 < gain <= 1:
            raise ValueError("action_gain must be finite and in (0, 1]")
        if lead is not None and (
            type(lead) not in (float, int) or not math.isfinite(lead) or not 0 < lead < cfg["tracking_soft_rad"]
        ):
            raise ValueError("target_lead_limit_rad must be positive and below tracking_soft_rad")

    def __init__(self, data):
        self.validate(data)
        self.target_lead_limit_rad = data.get("target_lead_limit_rad")
        self.action_gain = data.get("action_gain", 1.0)
        self.config = dict(data["motion_guard"])
        self.hard = {
            "tracking": data["max_tracking_error_rad"],
            "new_target": data["max_new_target_error_rad"],
            "speed": data["max_measured_velocity_rad_s"],
        }
        self.since = {k: [None] * 6 for k in ("tracking", "new_target")}
        self.active_speed = [False] * 6
        self.speed_segments = deque()
        self.speed_coverage = 0.0
        self.received = None
        self.target_received = None
        self.origin = None
        self.samples = 0
        self.events = deque(maxlen=64)
        self.counts = Counter()
        self.last = {}
        self.peaks = {
            k: 0.0 for k in ("raw_speed_rad_s", "mean_abs_speed_rad_s", "tracking_error_rad", "new_target_error_rad")
        }

    @staticmethod
    def _vector(value):
        value = torch.as_tensor(value)
        if value.shape != (6,) or value.dtype == torch.bool or not torch.isfinite(value).all():
            raise ValueError("Motion guard requires six finite joint values")
        return value.detach().to(dtype=torch.float64).tolist()

    def _event(self, kind, joint, value, limit, *, stop=False):
        event = {
            "kind": kind,
            "joint": joint + 1,
            "value": value,
            "limit": limit,
            "elapsed_s": self.received - self.origin,
            "stop": stop,
        }
        self.events.append(event)
        self.counts[kind] += 1
        return event

    def _stop(self, kind, joint, value, limit):
        self._event(kind, joint, value, limit, stop=True)
        raise RuntimeError(f"Motion guard {kind}: joint{joint + 1}, value={value:.6f}, limit={limit:.6f}")

    def _error(self, kind, q, target):
        error = [abs(a - b) for a, b in zip(target, q, strict=True)]
        self.last[f"{kind}_error_rad"] = error
        self.peaks[f"{kind}_error_rad"] = max(self.peaks[f"{kind}_error_rad"], max(error))
        hard = self.hard[kind]
        if max(error) > hard:
            joint = error.index(max(error))
            self._stop(f"{kind}_hard", joint, error[joint], hard)
        soft = self.config[f"{kind}_soft_rad"]
        warnings = []
        durations = [0.0] * 6
        self.last[f"{kind}_over_soft_s"] = durations
        for joint, value in enumerate(error):
            if value > soft:
                if self.since[kind][joint] is None:
                    self.since[kind][joint] = self.received
                    warnings.append(self._event(f"{kind}_soft", joint, value, soft))
                elapsed = self.received - self.since[kind][joint]
                durations[joint] = elapsed
                if elapsed + 1e-9 >= self.config["tracking_grace_s"]:
                    self._stop(f"{kind}_sustained", joint, value, soft)
            else:
                self.since[kind][joint] = None
        return warnings

    def check_feedback(self, q, dq, old_target, sample_dt, received):
        q, dq, old_target = (self._vector(v) for v in (q, dq, old_target))
        if (
            type(sample_dt) not in (int, float)
            or not math.isfinite(sample_dt)
            or not 0 < sample_dt <= 0.1
            or type(received) not in (int, float)
            or not math.isfinite(received)
            or (
                self.received is not None
                and (
                    received <= self.received
                    or not math.isclose(received - self.received, sample_dt, rel_tol=1e-5, abs_tol=1e-7)
                )
            )
        ):
            raise ValueError("Invalid motion guard feedback timestamp or sample interval")
        self.received = received
        self.samples += 1
        if self.origin is None:
            self.origin = received
        speed = [abs(v) for v in dq]
        self.last = {"sample_dt_s": sample_dt, "raw_speed_rad_s": speed}
        self.peaks["raw_speed_rad_s"] = max(self.peaks["raw_speed_rad_s"], max(speed))
        if max(speed) > self.hard["speed"]:
            joint = speed.index(max(speed))
            self._stop("speed_hard", joint, speed[joint], self.hard["speed"])
        self.speed_segments.append((sample_dt, speed))
        self.speed_coverage += sample_dt
        window = self.config["speed_window_s"]
        while self.speed_coverage > window and self.speed_segments:
            dt, values = self.speed_segments.popleft()
            excess = self.speed_coverage - window
            removed = min(dt, excess)
            self.speed_coverage -= removed
            if dt > removed:
                self.speed_segments.appendleft((dt - removed, values))
                break
        ready = self.speed_coverage + 1e-9 >= window
        coverage = math.fsum(dt for dt, _ in self.speed_segments)
        mean = [math.fsum(dt * v[j] for dt, v in self.speed_segments) / coverage for j in range(6)]
        self.last.update(
            speed_window_ready=ready, speed_window_coverage_s=self.speed_coverage, mean_abs_speed_rad_s=mean
        )
        soft = self.config["speed_soft_rad_s"]
        if ready:
            self.peaks["mean_abs_speed_rad_s"] = max(self.peaks["mean_abs_speed_rad_s"], max(mean))
        warnings = []
        for joint, value in enumerate(speed):
            if value > soft and not self.active_speed[joint]:
                warnings.append(self._event("speed_soft", joint, value, soft))
            self.active_speed[joint] = value > soft
        # Only absorb summation round-off at the exact window threshold.
        if ready and max(mean) > soft + 1e-12:
            joint = mean.index(max(mean))
            self._stop("speed_sustained", joint, mean[joint], soft)
        return warnings + self._error("tracking", q, old_target)

    def check_target(self, q, new_target, received):
        q, new_target = self._vector(q), self._vector(new_target)
        if (
            type(received) not in (int, float)
            or not math.isfinite(received)
            or self.received is None
            or received != self.received
            or received == self.target_received
        ):
            raise ValueError("New target requires one check against the current feedback timestamp")
        self.target_received = received
        return self._error("new_target", q, new_target)

    def snapshot(self):
        return dict(self.last)

    def report(self):
        return {
            "mode": "staged_development",
            "evaluated": self.samples > 0,
            "feedback_samples": self.samples,
            "config": self.config,
            "hard_limits": self.hard,
            "event_counts": dict(self.counts),
            "recent_events": list(self.events),
            "last_sample": self.snapshot(),
            "peaks": dict(self.peaks),
        }
