"""Viewer-only Reach reward estimates from actual policy telemetry.

These use the pre-action feedback in each packet, not Isaac Lab's post-step state.
Joint acceleration is a host-timestamp finite difference, not a sensor reading.
No training losses, complete episode returns, or unseen samples are invented.
"""

import math

from .reach_telemetry import validate_state


class ReachLiveMetrics:
    """Evaluate the current Reach reward configuration without stepping an environment."""

    FUNCTIONS = {
        "end_effector_position_tracking": "position_command_error",
        "end_effector_orientation_tracking": "orientation_command_error",
        "success": "settled_success",
        "action_rate": "action_rate_l2",
        "action_magnitude": "action_l2",
        "joint_vel": "joint_vel_l2",
        "position_fine": "position_command_error_tanh",
        "joint_acceleration": "joint_acc_l2",
    }
    metric_names = (
        "position_error_m",
        "orientation_error_rad",
        "settled",
        "observed_goal_time_s",
        "received_policy_samples",
        "estimated_step_reward",
    )

    def __init__(self, reward_manager, *, policy_dt, position_threshold=0.02, orientation_threshold=0.15):
        for name, value in (
            ("policy_dt", policy_dt),
            ("position_threshold", position_threshold),
            ("orientation_threshold", orientation_threshold),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Live metrics require a positive finite {name}")
        self.reward_names = tuple(reward_manager.active_terms)
        if len(self.reward_names) != len(self.FUNCTIONS) or set(self.reward_names) != set(self.FUNCTIONS):
            raise ValueError("Live metrics support only the eight configured Reach reward terms")
        self.weights = {}
        for name in self.reward_names:
            cfg = reward_manager.get_term_cfg(name)
            function = (
                cfg.func.rsplit(":", 1)[-1].rsplit(".", 1)[-1] if isinstance(cfg.func, str) else cfg.func.__name__
            )
            if function != self.FUNCTIONS[name]:
                raise ValueError(f"Unsupported live reward function for {name}: {function}")
            if type(cfg.weight) not in (int, float) or not math.isfinite(cfg.weight):
                raise ValueError(f"Invalid live reward weight for {name}")
            self.weights[name] = cfg.weight
        self.std = reward_manager.get_term_cfg("position_fine").params["std"]
        if type(self.std) not in (int, float) or not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("Live position_fine requires a positive finite std")
        self.policy_dt = policy_dt
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.rewards = {}
        self.metrics = {}
        self._previous = None
        self._goal = None
        self._goal_started = None
        self._samples = 0

    def update(self, state):
        """Accept one new inference sample; absent acceleration leaves rewards empty.

        Duplicate or older steps and home/terminal packets do not update metrics.
        A new run clears history; a new goal starts the *observed* goal timer.
        """
        validate_state(state)
        if state["state"] not in ("running", "read_only") or "policy_observation" not in state:
            return False
        run, step, stamp = state["run_id"], state["policy_step"], state["policy_monotonic_s"]
        previous = self._previous
        if previous is not None and run == previous[0] and step <= previous[1]:
            return False
        new_run = previous is None or run != previous[0]
        obs = state["policy_observation"]
        velocity, old_action = obs[6:12], obs[19:25]
        position_error = math.hypot(*obs[25:28])
        orientation_error = math.hypot(*obs[28:31])
        settled = (
            position_error < self.position_threshold
            and orientation_error < self.orientation_threshold
            and max(abs(v) for v in velocity) < 0.1
        )
        goal = (run, state["goal_index"])
        goal_started = stamp if goal != self._goal else self._goal_started
        samples = 1 if new_run else self._samples + 1
        metrics = {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "settled": float(settled),
            "observed_goal_time_s": max(0.0, stamp - goal_started),
            "received_policy_samples": samples,
        }
        rewards = {}
        if not new_run and step == previous[1] + 1 and 0 < stamp - previous[2] <= 3 * self.policy_dt:
            dt = stamp - previous[2]
            acceleration = [(v - old) / dt for v, old in zip(velocity, previous[3], strict=True)]
            # Training's RSL-RL wrapper clips before ActionManager computes these penalties.
            action = [max(-1.0, min(1.0, v)) for v in state["policy_action"]]
            values = {
                "end_effector_position_tracking": position_error,
                "end_effector_orientation_tracking": orientation_error,
                "success": float(settled),
                "action_rate": sum((v - old) ** 2 for v, old in zip(action, old_action, strict=True)),
                "action_magnitude": sum(v * v for v in action),
                "joint_vel": sum(v * v for v in velocity),
                "position_fine": 1 - math.tanh(position_error / self.std),
                "joint_acceleration": sum(v * v for v in acceleration),
            }
            rewards = {name: self.weights[name] * values[name] for name in self.reward_names}
            metrics["estimated_step_reward"] = sum(rewards.values()) * self.policy_dt
        if not all(math.isfinite(v) for v in (*metrics.values(), *rewards.values())):
            raise ValueError("Non-finite live reward estimate")
        self.rewards, self.metrics = rewards, metrics
        self._previous = (run, step, stamp, list(velocity))
        self._goal, self._goal_started, self._samples = goal, goal_started, samples
        return True
