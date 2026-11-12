"""Display real Reach joint feedback and its current goal in Isaac Sim.

With --ip it reads one startup joint snapshot and closes the read-only connection,
then receives local telemetry. It never loads a policy, advances physics, or sends
motion commands. Press Ctrl+C in the control
terminal to stop its controller; closing this viewer does not stop the robot.
"""

import argparse
import json
import math
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch

from mycobot450_isaaclab.utils.deploy import HardwareConfig
from mycobot450_isaaclab.utils.reach_live_metrics import ReachLiveMetrics
from mycobot450_isaaclab.utils.reach_policy import ReachController
from mycobot450_isaaclab.utils.reach_telemetry import TelemetryReceiver
from mycobot450_isaaclab.utils.robot_interface import Pro450Interface

TASK = "Mycobot450Isaaclab-Reach-Play"
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
POLICY_OBSERVATION_TERMS = (
    ("joint_pos", 6),
    ("joint_vel", 6),
    ("pose_command", 7),
    ("actions", 6),
    ("pose_error", 6),
    ("filtered_target", 6),
    ("target_velocity", 6),
)


class PolicyPlotData:
    """Actual deployment samples, separate from the display environment's managers."""

    def __init__(self, observation_manager, *, stale_after=1.0, contract=None, metrics=None):
        terms = observation_manager.active_terms.get("policy", [])
        dimensions = observation_manager.group_obs_term_dim.get("policy", [])
        expected_names = [name for name, _ in POLICY_OBSERVATION_TERMS]
        expected_dimensions = [(size,) for _, size in POLICY_OBSERVATION_TERMS]
        if terms != expected_names or [tuple(shape) for shape in dimensions] != expected_dimensions:
            raise ValueError("Live plots require the deployment's ordered seven-term, 43-dimensional observation")
        if contract is not None and (
            contract.get("observation_dim") != 43
            or contract.get("joint_order") != ARM_JOINTS
            or contract.get("quaternion_order") != "xyzw"
        ):
            raise ValueError("Live plot contract must match the six-joint, 43-dimensional Reach policy")
        self.policy_sha256 = contract.get("policy_sha256") if contract else None
        self.stale_after = stale_after
        self.state = None
        self.sample_key = None
        self.metrics = metrics

    def update(self, state, status, now):
        self.state, self.sample_key = state, None
        if state is None or status not in ("LIVE", "READ ONLY") or "policy_observation" not in state:
            return
        if self.policy_sha256 is not None and state["policy_sha256"] != self.policy_sha256:
            return
        age = now - state["policy_monotonic_s"]
        if 0 <= age <= self.stale_after:
            self.sample_key = (state["run_id"], state["policy_step"])
            if self.metrics is not None:
                self.metrics.update(state)


class PolicyTerms:
    """The small manager protocol consumed by Isaac Lab's official plot widgets."""

    num_envs = 1

    def __init__(self, data, *, observations=False):
        self.data, self.observations = data, observations
        self.active_terms = (
            [f"policy-{name}" for name, _ in POLICY_OBSERVATION_TERMS] if observations else ["arm_action"]
        )

    def get_active_iterable_terms(self, env_idx=0):
        if env_idx != 0:
            raise ValueError("Live policy plots support only the real robot at environment 0")
        if self.data.sample_key is None:
            return []
        if not self.observations:
            return [("arm_action", self.data.state["policy_action"])]
        values, terms, offset = self.data.state["policy_observation"], [], 0
        for name, size in POLICY_OBSERVATION_TERMS:
            terms.append((f"policy-{name}", values[offset : offset + size]))
            offset += size
        return terms


class EvaluationTerms:
    """Commands and feedback-derived scores, without stepping simulation managers."""

    num_envs = 1

    def __init__(self, data, kind):
        self.data, self.kind = data, kind
        self.active_terms = {
            "command_manager": ["ee_pose"],
            "reward_manager": list(data.metrics.reward_names),
            "episode": list(data.metrics.metric_names),
        }[kind]

    def get_active_iterable_terms(self, env_idx=0):
        if env_idx != 0:
            raise ValueError("Live evaluation supports only the real robot at environment 0")
        if self.data.sample_key is None:
            return []
        if self.kind == "command_manager":
            return [("ee_pose", self.data.state["policy_observation"][12:19])]
        # A complete score needs two consecutive samples; do not seed unknown acceleration with zero.
        if not self.data.metrics.rewards:
            return []
        values = self.data.metrics.rewards if self.kind == "reward_manager" else self.data.metrics.metrics
        return [(name, [values[name]]) for name in self.active_terms if name in values]


class PolicyPlots:
    """Isaac Lab window whose curves consume telemetry, never simulated actions."""

    def __init__(self, env, *, stale_after=1.0, contract=None):
        import omni.ui as ui

        from isaaclab.envs.ui import BaseEnvWindow
        from isaaclab.ui.widgets.manager_live_visualizer import ManagerLiveVisualizer, ManagerLiveVisualizerCfg

        metrics = ReachLiveMetrics(
            env.reward_manager,
            policy_dt=contract["policy_dt_s"] if contract else env.step_dt,
            position_threshold=env.cfg.commands.ee_pose.position_success_threshold,
            orientation_threshold=env.cfg.commands.ee_pose.orientation_success_threshold,
        )
        self.data = PolicyPlotData(env.observation_manager, stale_after=stale_after, contract=contract, metrics=metrics)

        class TelemetryManagerVisualizer(ManagerLiveVisualizer):
            def __init__(self, source, name, *, enabled=False):
                super().__init__(source, ManagerLiveVisualizerCfg(manager_name=name, debug_vis=enabled))
                self.last_key = None
                self.last_run = None
                self.sample_count = 0

            def _set_debug_vis_impl(self, enabled):
                super()._set_debug_vis_impl(enabled)
                # Widget constructors seed a point; only a received policy sample may advance the history.
                for plot in self._term_visualizers:
                    plot.clear()
                self.last_key = None

            def _debug_vis_callback(self, event):
                if not env.sim.is_playing():
                    return
                data = self._manager.data
                run = data.state["run_id"] if data.state is not None else None
                if run != self.last_run:
                    for plot in self._term_visualizers:
                        plot.clear()
                    self.last_key, self.last_run = None, run
                key = data.sample_key
                if key is None or (self.last_key is not None and key[1] <= self.last_key[1]):
                    return
                if not self._manager.get_active_iterable_terms():
                    return
                if not self._term_visualizers:
                    self._set_debug_vis_impl(True)
                super()._debug_vis_callback(event)
                self.last_key = key
                self.sample_count += 1

        self.visualizers = {
            "action_manager": TelemetryManagerVisualizer(PolicyTerms(self.data), "action_manager", enabled=True),
            "observation_manager": TelemetryManagerVisualizer(
                PolicyTerms(self.data, observations=True), "observation_manager"
            ),
            **{
                name: TelemetryManagerVisualizer(EvaluationTerms(self.data, name), name)
                for name in ("episode", "command_manager", "reward_manager")
            },
        }
        env.manager_visualizers = self.visualizers
        self.window = BaseEnvWindow(env, window_name="IsaacLab")
        with self.window.ui_window_elements["debug_vstack"]:
            self.window._visualize_manager(title="Commands", class_name="command_manager")
            self.window._visualize_manager(title="Rewards", class_name="reward_manager")
        self.window.ui_window_elements["reset_episode"].enabled = False
        for name in ("vis_markers_frame", "sim_frame", "viewer_frame"):
            self.window.ui_window_elements[name].collapsed = True
        with self.window.ui_window_elements["main_vstack"]:
            ui.Label(
                "Real policy telemetry. Actions: raw actor output, not joint angles. "
                "Observations: actual 43-value input before model normalization. "
                "Training Metrics: inference evaluation, not training loss. Rewards are estimates from "
                "current policy inputs and Reach weights; acceleration uses consecutive feedback samples. "
                "Goal time starts when this viewer first observes the goal. "
                "Curves advance only on new policy samples; no data during home or after stopping.",
                word_wrap=True,
            )
        self.visualizers["action_manager"].set_debug_vis(True)

    def update(self, state, status, now):
        self.data.update(state, status, now)

    def report(self):
        return {name: {"samples": vis.sample_count, "last_key": vis.last_key} for name, vis in self.visualizers.items()}

    def close(self):
        for visualizer in self.visualizers.values():
            visualizer.set_debug_vis(False)
        if self.window.ui_window is not None:
            self.window.ui_window.destroy()
            self.window.ui_window = None


def read_initial_state(ip, robot_port, bundle, hardware_config):
    """Read one angle reply using the controller's mapping, without loading policy.pt."""
    contract = json.loads((bundle / "contract.json").read_text())
    controller = ReachController(contract)
    hardware = HardwareConfig(json.loads(hardware_config.read_text()), controller)
    with Pro450Interface(ip, robot_port, writable=False, connect_timeout=2.0) as robot:
        degrees, received, _ = robot.read()
    q = hardware.to_sim(degrees)
    hardware.check_limits(q)
    return {
        "schema": 1,
        "type": "reach_state",
        "run_id": uuid.uuid4().hex,
        "seq": 0,
        "monotonic_s": received,
        "state": "read_only",
        "initial_snapshot": True,
        "q_rad": q.tolist(),
        "dq_rad_s": [0.0] * 6,  # One angle sample cannot measure velocity; this viewer does not integrate it.
        "target_rad": q.tolist(),
        "goal_xyz_xyzw": controller.fk(q).tolist(),
        "goal_index": 0,
        "sent": False,
        "policy_sha256": contract["policy_sha256"],
        "error": None,
    }


def display_status(state, *, now, stale_after):
    """Derive display freshness from the producer's system-monotonic timestamp."""
    if state is None:
        return "WAITING", "Waiting for real joint feedback on localhost."
    if state.get("initial_snapshot"):
        return "SNAPSHOT", "Startup angles read once; robot connection closed. Waiting for deploy telemetry."
    age = now - state["monotonic_s"]
    if age < 0 or age > stale_after:
        last_state = state["state"].upper()
        if state.get("error"):
            last_state += f": {state['error']}"
        return "STALE", f"Feedback is not current; last state: {last_state}. Pose is frozen."
    labels = {"running": "LIVE", "read_only": "READ ONLY", "stopped": "STOPPED", "fault": "FAULT"}
    return labels[state["state"]], state.get("error") or "Measured joints; displayed goal is the controller's goal."


class MirrorScene:
    """Write measured positions directly, retaining fixed-open gripper defaults."""

    def __init__(self, env, combine_frame_transforms):
        if env.num_envs != 1:
            raise ValueError("Live feedback requires exactly one display environment")
        self.env = env
        self.combine_frame_transforms = combine_frame_transforms
        self.robot = env.scene["robot"]
        self.joint_ids, names = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        self.joint_ids = self.joint_ids.torch if hasattr(self.joint_ids, "torch") else self.joint_ids
        if names != ARM_JOINTS:
            raise ValueError(f"Unexpected simulation joint order: {names}")
        self.joint_limits = self.robot.data.joint_pos_limits.torch[:, self.joint_ids].clone()
        self.command = env.command_manager.get_term("ee_pose")
        self.markers_visible = False
        self.command.set_debug_vis(False)
        default_q = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_position_to_sim_index(position=default_q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_q))
        env.sim.forward()

    def apply(self, state):
        position = torch.tensor([state["q_rad"]], device=self.env.device, dtype=torch.float32)
        velocity = torch.tensor([state["dq_rad_s"]], device=self.env.device, dtype=torch.float32)
        goal = torch.tensor([state["goal_xyz_xyzw"]], device=self.env.device, dtype=torch.float32)
        if not all(torch.isfinite(value).all() for value in (position, velocity, goal)):
            raise ValueError("Feedback cannot be represented as finite float32 state")
        if ((position < self.joint_limits[..., 0]) | (position > self.joint_limits[..., 1])).any():
            raise ValueError("Measured joints exceed the display model's limits")
        self.robot.write_joint_position_to_sim_index(position=position, joint_ids=self.joint_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=self.joint_ids)
        self.env.sim.forward()  # Update link transforms without integrating physics.
        self.command.pose_command_b[:] = goal
        position_w, quaternion_w = self.combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.command.pose_command_b[:, :3],
            self.command.pose_command_b[:, 3:],
        )
        self.command.pose_command_w[:, :3] = position_w
        self.command.pose_command_w[:, 3:] = quaternion_w

    def show_markers(self, visible):
        if visible != self.markers_visible:
            self.command.set_debug_vis(visible)
            self.markers_visible = visible

    def measured_display_joints(self):
        """Return the joint buffer, which by itself does not verify rendering."""
        return self.robot.data.joint_pos.torch[0, self.joint_ids].cpu().tolist()

    def physics_link_poses(self):
        """Read PhysX link poses directly instead of the write-through joint cache."""
        poses = self.robot.root_view.get_link_transforms().numpy()[0].tolist()
        return dict(zip(self.robot.backend_body_names, poses, strict=True))


class StatusDisplay:
    """A read-only Kit panel, plus console transitions for headless operation."""

    def __init__(self, *, headless):
        self.window = None
        self.label = None
        self.last_message = None
        self.ui = None
        self.docked = False
        self.dock_wait_frames = 0
        if not headless:
            import omni.ui as ui

            self.ui = ui
            # Stay hidden until a dock host exists; never float over the robot.
            self.window = ui.Window(
                "Reach real robot feedback",
                width=380,
                height=300,
                visible=False,
                dock_preference=ui.DockPreference.RIGHT_TOP,
            )
            with self.window.frame, ui.VStack(spacing=8):
                self.label = ui.Label("WAITING", word_wrap=True)
                ui.Label(
                    "Read-only display. Press Ctrl+C in the CONTROL terminal to stop its controller. "
                    "Closing this viewer does not stop the real robot. Use the physical E-stop when needed.",
                    word_wrap=True,
                )

    def update(self, status, detail, state, now):
        if self.window is not None and not self.docked:
            self.docked = self._dock_panel()
        lines = [status, detail]
        if state is not None:
            lines += [
                f"Feedback age: {max(0.0, now - state['monotonic_s']):.3f} s",
                "Measured joints 1-6 (deg): " + ", ".join(f"{math.degrees(v):.2f}" for v in state["q_rad"]),
            ]
            if not state.get("initial_snapshot"):
                lines.append(
                    f"Goal {state['goal_index']} XYZ (base, m): "
                    + ", ".join(f"{v:.3f}" for v in state["goal_xyz_xyzw"][:3])
                )
        if self.label is not None:
            self.label.text = "\n".join(lines)
        if (status, detail) != self.last_message:
            print(f"[LIVE VIEW] {status}: {detail}", flush=True)
            self.last_message = (status, detail)

    def _dock_panel(self):
        """Request deferred docking and confirm it after a normal render update."""
        ui = self.ui
        custom = ui.Workspace.get_window(self.window.title)
        if custom is None:
            return False
        for title in ("Property", "Stage"):
            target = ui.Workspace.get_window(title)
            if target is not None:
                if custom.dock_id and custom.dock_id == target.dock_id:
                    return True
                if self.dock_wait_frames >= 30:
                    self.window.visible = False
                    return False
                # Making a hidden window visible can reset its layout. Show it
                # first, then let OmniUI apply docking on a later UI update.
                if not self.window.visible:
                    self.window.visible = True
                self.window.deferred_dock_in(title, ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
                self.dock_wait_frames += 1
                return False
        return False  # Keep console status, rather than obscure the viewport.

    def close(self):
        if self.window is not None:
            self.window.destroy()


@torch.inference_mode()
def mirror_feedback(
    scene,
    receiver,
    display,
    *,
    steps=0,
    render_hz=30.0,
    stale_after=1.0,
    initial_state=None,
    plots=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Render newest feedback only; never catch up or extrapolate old packets."""
    count = updates = rejected = 0
    latest = None
    invalid_detail = None
    initial_physics_steps = scene.env.sim.get_physics_step_count()
    last_status = "WAITING"
    interrupted = False
    if initial_state is not None:
        scene.apply(initial_state)
        latest, updates = initial_state, 1
    try:
        while scene.env.sim.is_running() and (steps == 0 or count < steps):
            start = clock()
            state = receiver.poll()
            if state is not None:
                try:
                    scene.apply(state)
                except ValueError as exc:
                    invalid_detail = f"Rejected feedback: {exc}. Pose is frozen."
                    rejected += 1
                else:
                    latest = state
                    invalid_detail = None
                    updates += 1
            if invalid_detail:
                last_status, detail = "INVALID", invalid_detail
            else:
                last_status, detail = display_status(latest, now=clock(), stale_after=stale_after)
            scene.show_markers(latest is not None and last_status not in ("STALE", "INVALID", "SNAPSHOT"))
            display.update(last_status, detail, latest, clock())
            if plots is not None:
                plots.update(latest, last_status, clock())
            scene.env.sim.render()
            count += 1
            sleep(max(0.0, 1 / render_hz - (clock() - start)))
    except KeyboardInterrupt:
        interrupted = True
    return {
        "mode": "read_only_feedback_viewer",
        "interrupted": interrupted,
        "frames": count,
        "feedback_updates": updates,
        "rejected_feedback": rejected,
        "status": last_status,
        "physics_steps": scene.env.sim.get_physics_step_count() - initial_physics_steps,
        "last_run_id": latest["run_id"] if latest else None,
        "last_seq": latest["seq"] if latest else None,
        "last_q_rad": latest["q_rad"] if latest else None,
        "joint_buffer_q_rad": scene.measured_display_joints() if latest else None,
        "physics_link_poses_xyz_xyzw": scene.physics_link_poses() if latest else None,
        "robot_commands_sent": 0,
        "policy_plots": plots.report() if plots is not None else None,
        "note": (
            "Joint buffer and physics link poses are diagnostics, not proof of rendered mesh poses. "
            "Closing the viewer does not stop the real controller."
        ),
    }


def add_viewer_args(parser):
    parser.add_argument("--port", type=int, default=14550, help="Local telemetry UDP port (default: 14550).")
    parser.add_argument("--ip", help="Read one startup joint snapshot from this robot, then close the connection.")
    parser.add_argument("--robot-port", type=int, default=4500, help="Robot TCP port, separate from telemetry --port.")
    parser.add_argument("--bundle", type=Path, help="Contract directory for the startup snapshot; no policy is loaded.")
    parser.add_argument("--hardware-config", type=Path, help="Same joint mapping used by deploy.py.")
    parser.add_argument("--steps", type=int, default=0, help="Render frames; 0 runs until the viewer closes.")
    parser.add_argument("--render-hz", type=float, default=30.0)
    parser.add_argument("--stale-after", type=float, default=1.0, help="Mark feedback stale after this many seconds.")
    parser.add_argument("--report", type=Path, help="Optional new JSON diagnostic report, normally in outputs/.")
    # Isaac Lab 3's launcher still accepts this setting, but no longer adds its CLI flag.
    parser.add_argument("--headless", action="store_true", help="Run the Kit viewer without a visible window.")


def main():
    # Simulation imports are kept out of protocol and fake-scene tests.
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
    from isaaclab.utils import instantiate
    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_tasks.utils import parse_env_cfg

    import mycobot450_isaaclab.tasks  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    add_viewer_args(parser)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args, overrides = parser.parse_known_args()
    if args.steps < 0 or not 1 <= args.port <= 65535:
        parser.error("--steps must be nonnegative and --port must be in 1..65535")
    if args.ip and (args.bundle is None or args.hardware_config is None or not 1 <= args.robot_port <= 65535):
        parser.error("--ip requires --bundle, --hardware-config and --robot-port in 1..65535")
    if any(not math.isfinite(v) or v <= 0 for v in (args.render_hz, args.stale_after)):
        parser.error("--render-hz and --stale-after must be finite and positive")
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1, overrides=overrides)
    cfg.play_mode()
    cfg.terminations.time_out = None
    # The viewer supplies goals. Do not build or sample the Reach training bank.
    cfg.commands.ee_pose.class_type = UniformPoseCommand
    cfg.commands.ee_pose.debug_vis = False
    # Build the official window after replacing its plot sources with real telemetry.
    cfg.ui_window_class_type = None
    if args.headless:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        visualizers = cfg.sim.visualizer_cfgs
        visualizers = list(visualizers) if isinstance(visualizers, list) else ([visualizers] if visualizers else [])
        kit_visualizers = [item for item in visualizers if item.visualizer_type == "kit"]
        if not kit_visualizers:
            visualizers.append(KitVisualizerCfg(headless=True))
        for item in kit_visualizers:
            item.headless = True
        cfg.sim.visualizer_cfgs = visualizers
    # This mirror teleports articulation joints without a physics step. Native
    # PhysX Fabric can keep the rendered meshes at their original USD pose even
    # when PhysX link transforms are correct. Kit's own Fabric renderer remains
    # active: with native Fabric disabled it copies the current body transforms
    # and explicitly propagates them to visual descendants on each render.
    cfg.sim.use_fabric = False
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.report.open("x") if args.report else nullcontext(None) as report_file,
        TelemetryReceiver(args.port, max_age_s=args.stale_after) as receiver,
        launch_simulation(cfg, args),
    ):
        env = instantiate(cfg)
        display = plots = None
        try:
            display = StatusDisplay(headless=args.headless or not args.kit_visualizer)
            scene = MirrorScene(env, combine_frame_transforms)
            if not args.headless and args.kit_visualizer:
                contract = json.loads((args.bundle / "contract.json").read_text()) if args.bundle else None
                plots = PolicyPlots(env, stale_after=args.stale_after, contract=contract)
            initial_state = receiver.poll()
            if initial_state is None and args.ip:
                initial_state = read_initial_state(args.ip, args.robot_port, args.bundle, args.hardware_config)
            report = mirror_feedback(
                scene,
                receiver,
                display,
                steps=args.steps,
                render_hz=args.render_hz,
                stale_after=args.stale_after,
                initial_state=initial_state,
                plots=plots,
            )
            print(json.dumps(report, indent=2), flush=True)
            if report_file:
                json.dump(report, report_file, indent=2)
                report_file.write("\n")
                report_file.flush()  # Kit shutdown may exit before outer context managers unwind.
        except KeyboardInterrupt:
            print("\n[INFO] Viewer closed; the robot controller is a separate process.", flush=True)
        finally:
            if plots is not None:
                plots.close()
            if display is not None:
                display.close()
            env.close()


if __name__ == "__main__":
    main()
