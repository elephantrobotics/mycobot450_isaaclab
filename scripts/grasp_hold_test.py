# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop hold test: TCP above cube -> close F100 -> lift 10 cm.

Does not use a trained policy. Prints whether the cube follows the gripper.

    python scripts/grasp_hold_test.py --task Isaac-Lift-Cube-Mycobot-Pro450-Play-v0 --num_envs 4 --viz kit
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Scripted F100 grasp-and-hold test.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Mycobot-Pro450-Play-v0")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import mycobot450_isaaclab.tasks  # noqa: F401
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from mycobot450_isaaclab.assets.robots.mycobot_pro450 import (
    MYCOBOT_PRO450_EE_BODY_NAME,
    MYCOBOT_PRO450_GRIPPER_OPEN_POS,
    MYCOBOT_PRO450_GRIPPER_PINCH_POS,
    MYCOBOT_PRO450_TCP_OFFSET_POS,
)

HOVER_Z = 0.08
GRASP_Z = 0.015
LIFT_Z = 0.10
HOLD_Z_MIN = 0.06
HOLD_DIST_MAX = 0.08


def _tcp_pos_w(robot, body_id: int, offset_b: torch.Tensor) -> torch.Tensor:
    pose = robot.data.body_pose_w.torch[:, body_id]
    return pose[:, 0:3] + quat_apply(pose[:, 3:7], offset_b.expand(pose.shape[0], -1))


def _command_body_for_tcp(robot, body_id: int, tcp_w: torch.Tensor, offset_b: torch.Tensor) -> torch.Tensor:
    pose = robot.data.body_pose_w.torch[:, body_id]
    body_quat_w = pose[:, 3:7]
    body_pos_w = tcp_w - quat_apply(body_quat_w, offset_b.expand(pose.shape[0], -1))
    root = robot.data.root_pose_w.torch
    body_pos_b, body_quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], body_pos_w, body_quat_w)
    return torch.cat([body_pos_b, body_quat_b], dim=-1)


def main():
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.episode_length_s = 60.0
        env_cfg.commands.object_pose.debug_vis = True
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        unwrapped = env.unwrapped
        sim = unwrapped.sim
        scene = unwrapped.scene
        robot = scene["robot"]
        cube = scene["object"]
        sim_dt = sim.get_physics_dt()
        device = unwrapped.device
        num_envs = unwrapped.num_envs

        arm_cfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"], body_names=[MYCOBOT_PRO450_EE_BODY_NAME])
        grip_cfg = SceneEntityCfg("robot", joint_names=["joint2_left_joint"])
        arm_cfg.resolve(scene)
        grip_cfg.resolve(scene)
        body_id = arm_cfg.body_ids[0]
        ee_jacobi_idx = body_id - 1 if robot.is_fixed_base else body_id
        offset_b = torch.tensor(MYCOBOT_PRO450_TCP_OFFSET_POS, device=device, dtype=torch.float32)

        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=num_envs,
            device=device,
        )

        phases = [
            ("hover", int(2.0 / sim_dt), HOVER_Z, MYCOBOT_PRO450_GRIPPER_OPEN_POS),
            ("descend", int(2.0 / sim_dt), GRASP_Z, MYCOBOT_PRO450_GRIPPER_OPEN_POS),
            ("close", int(1.5 / sim_dt), GRASP_Z, MYCOBOT_PRO450_GRIPPER_PINCH_POS),
            ("lift", int(2.0 / sim_dt), GRASP_Z + LIFT_Z, MYCOBOT_PRO450_GRIPPER_PINCH_POS),
            ("hold", int(2.0 / sim_dt), GRASP_Z + LIFT_Z, MYCOBOT_PRO450_GRIPPER_PINCH_POS),
        ]
        phase_i = 0
        phase_step = 0
        cycle = 0
        cube_z_before_lift = None
        ik.reset()

        print("[INFO] Grasp-hold test: hover -> descend -> close -> lift 10 cm -> hold", flush=True)
        print(f"[INFO] TCP offset (f100 +X): {MYCOBOT_PRO450_TCP_OFFSET_POS}", flush=True)

        while True:
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break

            name, duration, height, grip_pos = phases[phase_i]
            if phase_step == 0 and name == "lift":
                cube_z_before_lift = cube.data.root_pos_w.torch[:, 2].clone()

            cube_pos = cube.data.root_pos_w.torch
            tcp_des_w = cube_pos.clone()
            tcp_des_w[:, 2] = cube_pos[:, 2] + height
            ik.set_command(_command_body_for_tcp(robot, body_id, tcp_des_w, offset_b))

            jacobi_joint_ids = [j + robot.num_base_dofs for j in arm_cfg.joint_ids]
            jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, jacobi_joint_ids]
            ee_pose_w = robot.data.body_pose_w.torch[:, body_id]
            root_pose_w = robot.data.root_pose_w.torch
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = ik.compute(
                ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos.torch[:, arm_cfg.joint_ids]
            )
            robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target_index(
                target=torch.full((num_envs, 1), grip_pos, device=device),
                joint_ids=grip_cfg.joint_ids,
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            phase_step += 1

            if phase_step < duration:
                continue

            tcp_w = _tcp_pos_w(robot, body_id, offset_b)
            tcp_err = torch.linalg.norm(tcp_w - tcp_des_w, dim=1)
            cube_tcp = torch.linalg.norm(cube_pos - tcp_w, dim=1)
            print(
                f"[phase {name}] tcp_err={tcp_err.mean().item():.3f} m  "
                f"cube-tcp={cube_tcp.mean().item():.3f} m  "
                f"cube_z={cube_pos[:, 2].mean().item():.3f} m",
                flush=True,
            )

            if name == "hold":
                cycle += 1
                lift = cube_pos[:, 2] - cube_z_before_lift
                held = (lift > HOLD_Z_MIN) & (cube_tcp < HOLD_DIST_MAX)
                print(
                    f"[hold] cycle {cycle}  held {int(held.sum())}/{num_envs}  "
                    f"rate={held.float().mean().item():.0%}  "
                    f"cube_lift={lift.mean().item():.3f} m  "
                    f"cube-tcp={cube_tcp.mean().item():.3f} m",
                    flush=True,
                )
                for i in range(num_envs):
                    print(
                        f"       env{i}: lift={lift[i].item():+.3f} m  "
                        f"cube-tcp={cube_tcp[i].item():.3f} m  "
                        f"{'FOLLOW' if held[i] else 'DROP'}",
                        flush=True,
                    )
                env.reset()
                ik.reset()
                cube_z_before_lift = None

            phase_i = (phase_i + 1) % len(phases)
            phase_step = 0

        env.close()


if __name__ == "__main__":
    main()
