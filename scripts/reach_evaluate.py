"""Evaluate held reach goals with fixed per-environment quotas; optionally export the policy."""

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import mycobot450_isaaclab.tasks  # noqa: F401
from mycobot450_isaaclab.tasks.reach.config.mycobot450.mycobot450_reach_env_cfg import ARM_JOINTS
from mycobot450_isaaclab.tasks.reach.mdp import GOAL_FAMILIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Omit to measure the zero-action baseline.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--goal_family", choices=("mixed", *GOAL_FAMILIES), default="mixed")
    parser.add_argument("--episode_seconds", type=float, default=12.0)
    parser.add_argument(
        "--trace", action="store_true", help="Include first-environment action/joint traces in the JSON."
    )
    parser.add_argument(
        "--switch_goals", action="store_true", help="Switch goals halfway through each trial without reset."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/reach/evaluation.json"))
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument(
        "--goals", type=Path, help="Replay goal poses from a prior evaluation JSON for paired comparison."
    )
    parser.add_argument(
        "--check_kinematics", action="store_true", help="Check sampled FK witnesses against the USD robot."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args, overrides = setup_preset_cli(parser)
    if args.num_envs < 1 or args.rounds < 1:
        parser.error("num_envs and rounds must be positive")
    minimum_duration = 2.0 if args.switch_goals else 1.0
    if not math.isfinite(args.episode_seconds) or args.episode_seconds <= minimum_duration:
        parser.error(
            f"episode_seconds must be finite and exceed {minimum_duration} seconds "
            "to measure a full one-second hold window after the final goal is set"
        )
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.export_dir and not args.checkpoint:
        parser.error("export_dir requires a checkpoint")
    if args.switch_goals and args.goals:
        parser.error("switch_goals and goals are separate evaluation modes")
    goals = None
    if args.goals:
        goals = json.loads(args.goals.read_text())["records"]
        if len(goals) != args.num_envs * args.rounds:
            parser.error("goals must contain exactly num_envs * rounds records")
    sys.argv = [sys.argv[0], *overrides]
    task = "Mycobot450Isaaclab-Reach-Play"
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_seconds
    cfg.seed = args.eval_seed
    # Build independent FK reservoirs rather than validating on training goals.
    cfg.commands.ee_pose.sampling_seed = args.eval_seed + 10000
    if args.goal_family != "mixed":
        cfg.commands.ee_pose.family_weights = tuple(float(name == args.goal_family) for name in GOAL_FAMILIES)
    # Each trial holds one goal for the full episode. Measure the last second,
    # stopping before auto-reset so terminal data cannot be replaced by reset states.
    cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    completed = False
    with launch_simulation(cfg, args):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        try:
            u = env.unwrapped
            robot = u.scene["robot"]
            command = u.command_manager.get_term("ee_pose")
            ids, names = robot.find_joints(ARM_JOINTS, preserve_order=True)
            ids = ids.torch if hasattr(ids, "torch") else ids
            assert names == ARM_JOINTS and u.action_manager.total_action_dim == 6
            observation_dim = env.get_observations()["policy"].shape[-1]
            kinematics_errors = {}
            if args.check_kinematics:
                env.reset()
                q = robot.data.default_joint_pos.torch.clone()
                q[:, ids] = command.goal_joints
                robot.write_joint_position_to_sim_index(position=q)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
                robot.set_joint_position_target_index(target=q)
                u.scene.write_data_to_sim()
                u.sim.step(render=False)
                u.scene.update(u.physics_dt)
                pos, quat = subtract_frame_transforms(
                    robot.data.root_pos_w.torch,
                    robot.data.root_quat_w.torch,
                    robot.data.body_pos_w.torch[:, command.body_idx],
                    robot.data.body_quat_w.torch[:, command.body_idx],
                )
                dp, dr = compute_pose_error(pos, quat, command.command[:, :3], command.command[:, 3:])
                kinematics_errors = {
                    "max_position_m": float(dp.norm(dim=-1).max()),
                    "max_orientation_rad": float(dr.norm(dim=-1).max()),
                }
                if kinematics_errors["max_position_m"] > 0.002 or kinematics_errors["max_orientation_rad"] > 0.02:
                    raise RuntimeError(f"URDF/USD kinematics mismatch: {kinematics_errors}")
            policy = None
            export_errors = {}
            if args.checkpoint:
                runner = create_rsl_rl_runner(env, agent)
                runner.load(str(args.checkpoint.resolve()))
                policy = runner.get_inference_policy(device=u.device)
                if args.export_dir:
                    runner.export_policy_to_jit(str(args.export_dir), filename="policy.pt")
                    runner.export_policy_to_onnx(str(args.export_dir), filename="policy.onnx")
                    # Verify exported policies on real environment observations before deployment.
                    import onnxruntime as ort

                    jit_policy = torch.jit.load(str(args.export_dir / "policy.pt"), map_location="cpu")
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    onnx_policy = ort.InferenceSession(
                        str(args.export_dir / "policy.onnx"),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    export_errors = {"torchscript_max_abs": 0.0, "onnx_max_abs": 0.0}
            hold_steps = math.ceil(1.0 / u.step_dt)
            records = []
            trace = []
            gripper_ids, _ = robot.find_joints("joint.*_(left|right)_joint")
            gripper_ids = gripper_ids.torch if hasattr(gripper_ids, "torch") else gripper_ids
            max_gripper_error = 0.0
            with torch.inference_mode():
                for trial in range(args.rounds):
                    # Model construction/export consumes RNG; reseed each trial so
                    # zero and learned policies receive exactly the same goals.
                    u.seed(args.eval_seed + trial)
                    obs, _ = env.reset()
                    families = [GOAL_FAMILIES[i] for i in command.goal_family.tolist()]
                    if goals is not None:
                        batch = goals[trial * args.num_envs : (trial + 1) * args.num_envs]
                        command.pose_command_b[:] = torch.tensor([r["goal_pose_b"] for r in batch], device=u.device)
                        families = [r.get("goal_family", "external") for r in batch]
                        u.obs_buf = u.observation_manager.compute()
                        obs = env.get_observations()
                    first_goal = command.command.clone()
                    if policy is not None:
                        policy.reset(torch.ones(args.num_envs, device=u.device, dtype=torch.bool))
                    held = torch.ones(args.num_envs, dtype=torch.bool, device=u.device)
                    settled = torch.ones_like(held)
                    reached = torch.zeros_like(held)
                    pos_sum = torch.zeros(args.num_envs, device=u.device)
                    rot_sum = torch.zeros_like(pos_sum)
                    measured = 0
                    motion = torch.zeros(args.num_envs, 4, device=u.device)
                    peak_target_speed = torch.zeros(args.num_envs, device=u.device)
                    previous_velocity = robot.data.joint_vel.torch[:, ids].clone()
                    previous_target = robot.data.joint_pos_target.torch[:, ids].clone()
                    steps = u.max_episode_length - 1
                    for step in range(steps):
                        if args.switch_goals and step == steps // 2:
                            command._resample_command(torch.arange(args.num_envs, device=u.device))
                            families = [GOAL_FAMILIES[i] for i in command.goal_family.tolist()]
                            u.obs_buf = u.observation_manager.compute()
                            obs = env.get_observations()
                            reached[:] = False
                        action = policy(obs) if policy is not None else torch.zeros(args.num_envs, 6, device=u.device)
                        obs, reward, dones, _ = env.step(action)
                        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                            raise RuntimeError("Non-finite observation or reward")
                        torch.testing.assert_close(
                            robot.data.joint_pos_target.torch[:, gripper_ids],
                            robot.data.default_joint_pos.torch[:, gripper_ids],
                        )
                        max_gripper_error = max(
                            max_gripper_error,
                            float(
                                (
                                    robot.data.joint_pos.torch[:, gripper_ids]
                                    - robot.data.default_joint_pos.torch[:, gripper_ids]
                                )
                                .abs()
                                .max()
                            ),
                        )
                        if dones.any():
                            raise RuntimeError("Unexpected reset during a held-goal trial")
                        velocity = robot.data.joint_vel.torch[:, ids]
                        acceleration = (velocity - previous_velocity) / u.step_dt
                        target = robot.data.joint_pos_target.torch[:, ids]
                        target_speed = (target - previous_target) / u.step_dt
                        if args.trace:
                            trace.append(
                                {
                                    "round": trial,
                                    "time_s": (step + 1) * u.step_dt,
                                    "action": action[0].clamp(-1, 1).tolist(),
                                    "target_rad": target[0].tolist(),
                                    "position_rad": robot.data.joint_pos.torch[0, ids].tolist(),
                                    "velocity_rad_s": velocity[0].tolist(),
                                }
                            )
                        if step > 0:  # exclude reset initialization of the control reference
                            peak_target_speed = torch.maximum(peak_target_speed, target_speed.abs().max(-1).values)
                        motion[:, 0] += velocity.square().mean(-1)
                        motion[:, 1] += acceleration.square().mean(-1)
                        motion[:, 2] += target_speed.square().mean(-1)
                        if step >= steps - hold_steps:
                            motion[:, 3] += velocity.square().mean(-1)
                        previous_velocity = velocity.clone()
                        previous_target = target.clone()
                        success = command.compute_success()
                        reached |= success
                        if step >= steps - hold_steps:
                            held &= success
                            settled &= success & (velocity.abs().max(-1).values < 0.1)
                            pos_sum += command.metrics["position_error"]
                            rot_sum += command.metrics["orientation_error"]
                            measured += 1
                    if args.export_dir:
                        reference = policy(obs).cpu()
                        for sample in range(min(args.num_envs, 8)):
                            x = obs["policy"][sample : sample + 1].cpu()
                            jit_action = jit_policy(x)
                            onnx_action = torch.from_numpy(
                                onnx_policy.run(None, {onnx_policy.get_inputs()[0].name: x.numpy()})[0]
                            )
                            expected = reference[sample : sample + 1]
                            torch.testing.assert_close(jit_action, expected, rtol=1e-4, atol=1e-5)
                            torch.testing.assert_close(onnx_action, expected, rtol=1e-4, atol=1e-5)
                            for key, value in (("torchscript_max_abs", jit_action), ("onnx_max_abs", onnx_action)):
                                export_errors[key] = max(export_errors[key], float((value - expected).abs().max()))
                    for i in range(args.num_envs):
                        records.append(
                            {
                                "round": trial,
                                "env": i,
                                "goal_family": families[i],
                                "held_success": bool(held[i]),
                                "settled_success": bool(settled[i]),
                                "joint_velocity_rms_rad_s": float((motion[i, 0] / steps).sqrt()),
                                "joint_acceleration_rms_rad_s2": float((motion[i, 1] / steps).sqrt()),
                                "target_speed_rms_rad_s": float((motion[i, 2] / steps).sqrt()),
                                "hold_joint_velocity_rms_rad_s": float((motion[i, 3] / measured).sqrt()),
                                "reached_success": bool(reached[i]),
                                "goal_pose_b": command.command[i].tolist(),
                                "first_goal_pose_b": first_goal[i].tolist(),
                                "peak_target_speed_rad_s": float(peak_target_speed[i]),
                                "position_error_m": float(pos_sum[i] / measured),
                                "orientation_error_rad": float(rot_sum[i] / measured),
                            }
                        )
            pos, quat = subtract_frame_transforms(
                robot.data.root_pos_w.torch,
                robot.data.root_quat_w.torch,
                robot.data.body_pos_w.torch[:, command.body_idx],
                robot.data.body_quat_w.torch[:, command.body_idx],
            )
            result = {
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "seed": args.eval_seed,
                "sampling_seed": cfg.commands.ee_pose.sampling_seed,
                "by_goal_family": {
                    family: {
                        "trials": len(group),
                        **{
                            name: sum(r[key] for r in group) / len(group)
                            for name, key in (
                                ("held_success_rate", "held_success"),
                                ("settled_success_rate", "settled_success"),
                                ("mean_position_error_m", "position_error_m"),
                                ("mean_orientation_error_rad", "orientation_error_rad"),
                                ("hold_joint_velocity_rms_rad_s", "hold_joint_velocity_rms_rad_s"),
                                ("joint_acceleration_rms_rad_s2", "joint_acceleration_rms_rad_s2"),
                            )
                        },
                    }
                    for family in sorted({r["goal_family"] for r in records})
                    if (group := [r for r in records if r["goal_family"] == family])
                },
                "kinematics_errors": kinematics_errors,
                "goals_source": str(args.goals) if args.goals else None,
                "switch_goals": args.switch_goals,
                "observation_dim": observation_dim,
                "motion_sample_dt": u.step_dt,
                "motion_metrics": {
                    name: sum(r[name] for r in records) / len(records)
                    for name in (
                        "joint_velocity_rms_rad_s",
                        "joint_acceleration_rms_rad_s2",
                        "target_speed_rms_rad_s",
                        "hold_joint_velocity_rms_rad_s",
                    )
                },
                "export_errors": export_errors,
                "max_gripper_error_rad": max_gripper_error,
                "trials": len(records),
                "definition": (
                    f"position < {cfg.commands.ee_pose.position_success_threshold} m AND "
                    f"orientation < {cfg.commands.ee_pose.orientation_success_threshold} rad "
                    f"throughout the final 1 s of a {cfg.episode_length_s} s trial; "
                    f"goal switch at halfway: {args.switch_goals}. Reached rate refers to the final goal."
                ),
                "held_success_rate": sum(r["held_success"] for r in records) / len(records),
                "settled_success_rate": sum(r["settled_success"] for r in records) / len(records),
                "settled_joint_speed_threshold_rad_s": 0.1,
                "reached_success_rate": sum(r["reached_success"] for r in records) / len(records),
                "mean_position_error_m": sum(r["position_error_m"] for r in records) / len(records),
                "mean_orientation_error_rad": sum(r["orientation_error_rad"] for r in records) / len(records),
                "peak_target_speed_rad_s": max(r["peak_target_speed_rad_s"] for r in records),
                "sampled_goal_position_min_m": [min(r["goal_pose_b"][i] for r in records) for i in range(3)],
                "sampled_goal_position_max_m": [max(r["goal_pose_b"][i] for r in records) for i in range(3)],
                "last_mount_pose_b": torch.cat((pos[0], quat[0])).tolist(),
                "joint_order": names,
                "home_rad": robot.data.default_joint_pos.torch[0, ids].tolist(),
                "step_dt": u.step_dt,
                "action_contract": {
                    "type": "filtered_joint_velocity_integrated_position_v1",
                    "observation_dim": observation_dim,
                    "velocity_scale_rad_s": cfg.actions.arm_action.scale,
                    "filter_time_constant_s": cfg.actions.arm_action.time_constant,
                    "physics_dt_s": u.physics_dt,
                    "policy_dt_s": u.step_dt,
                    "joint_target_limits_rad": cfg.actions.arm_action.target_limits,
                    "reset": "position target = measured joints; filtered velocity = 0; previous action = 0",
                    "observation_slices": {
                        "joint_position_minus_home": [0, 6],
                        "joint_velocity": [6, 12],
                        "goal_pose_xyz_xyzw": [12, 19],
                        "previous_clipped_action": [19, 25],
                        "pose_error_xyz_axis_angle": [25, 31],
                        "position_target_minus_home": [31, 37],
                        "filtered_target_velocity": [37, 43],
                    },
                    "normalization": "included in exported policy; input raw physical quantities",
                },
                "records": records,
            }
            if args.trace:
                result["trace"] = trace
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            if args.export_dir:
                contract = {**result["action_contract"], "joint_order": names, "home_rad": result["home_rad"]}
                (args.export_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k not in ("records", "trace")}, indent=2))
            completed = True
        finally:
            env.close()
    if not completed:
        raise RuntimeError("Reach evaluation failed; see simulator traceback above")


if __name__ == "__main__":
    main()
