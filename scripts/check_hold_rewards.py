# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity-check hold/throw rewards without launching the simulator."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

from mycobot450_isaaclab.assets.robots.mycobot_pro450 import MYCOBOT_PRO450_TCP_OFFSET_POS
from mycobot450_isaaclab.tasks.manager_based.mycobot_pro450_lift.mdp.rewards import (
    EeGoalAfterGrasp,
    EeRaisedAfterGrasp,
    ObjectEeDistanceCarry,
    ee_z_down,
    gripper_closed_away,
    object_ee_distance_aligned,
    object_is_lifted_and_held,
    object_pinch,
    object_pressed_against_arm,
    object_throwing,
    object_tips_on_cube,
)
from mycobot450_isaaclab.tasks.manager_based.mycobot_pro450_lift.lift_env_cfg import (
    MycobotRewardsCfg,
)


class _TensorData(SimpleNamespace):
    pass


def _quat_batch(xyzw: tuple[float, float, float, float], num_envs: int) -> torch.Tensor:
    quat = torch.tensor(xyzw, dtype=torch.float32).view(1, 1, 4).repeat(num_envs, 1, 1)
    return quat


def _env(cube_pos, cube_vel, ee_pos, gripper_q, close_ids=(0,), wrist_xyzw=(1.0, 0.0, 0.0, 0.0)) -> SimpleNamespace:
    """``wrist_xyzw`` default is 180 deg about X so Link6 +Z points at world -Z."""
    num_envs = cube_pos.shape[0]
    identity = torch.zeros(num_envs, 1, 4)
    identity[..., 3] = 1.0  # (x, y, z, w)
    obj = SimpleNamespace(
        data=_TensorData(
            root_pos_w=cube_pos,
            root_lin_vel_w=cube_vel,
        )
    )
    ee = SimpleNamespace(data=_TensorData(target_pos_w=ee_pos.unsqueeze(1), target_quat_w=identity))
    robot = SimpleNamespace(
        data=_TensorData(
            joint_pos=gripper_q,
            body_quat_w=_quat_batch(wrist_xyzw, num_envs),
        )
    )
    gripper_cfg = SimpleNamespace(name="robot", joint_ids=list(close_ids), body_ids=None)
    wrist_cfg = SimpleNamespace(name="robot", body_ids=[0])
    return SimpleNamespace(
        num_envs=num_envs,
        device=cube_pos.device,
        scene={"object": obj, "ee_frame": ee, "robot": robot},
        _gripper_cfg=gripper_cfg,
        _wrist_cfg=wrist_cfg,
    )


def _held_kwargs(env):
    return {
        "minimal_height": 0.10,
        "max_ee_object_dist": 0.05,
        "gripper_cfg": env._gripper_cfg,
        "gripper_open_pos": -1.012,
        "gripper_close_pos": 0.0,
        "closed_ratio": 0.45,
        "jaw_depth": 0.04,
        "jaw_width": 0.03,
        "jaw_height": 0.03,
    }


def main() -> None:
    rewards_cfg = MycobotRewardsCfg()
    assert rewards_cfg.lifting_object.func is lift_mdp.object_is_lifted
    assert rewards_cfg.lifting_object.weight == 15.0
    assert rewards_cfg.lifting_object.params["minimal_height"] == 0.10
    assert not hasattr(rewards_cfg, "pinch_weld")
    assert rewards_cfg.lifting_held.func is object_is_lifted_and_held
    assert rewards_cfg.lifting_held.weight == 4.0
    assert rewards_cfg.reaching_object.func is ObjectEeDistanceCarry
    assert rewards_cfg.aligning_down.func is ee_z_down
    assert rewards_cfg.aligning_down.weight == 1.0
    assert rewards_cfg.ee_raised.func is EeRaisedAfterGrasp
    assert rewards_cfg.ee_raised.weight == 4.0
    assert rewards_cfg.ee_goal_after_grasp.func is EeGoalAfterGrasp
    assert rewards_cfg.ee_goal_after_grasp.weight == 12.0
    assert rewards_cfg.gripper_closed_away.func is gripper_closed_away
    assert rewards_cfg.gripper_closed_away.weight == -3.0
    assert rewards_cfg.object_throwing.weight == -4.0
    assert rewards_cfg.object_throwing.params["airborne_height"] == 0.12
    assert rewards_cfg.object_pressed_against_arm.weight == -8.0
    assert rewards_cfg.reaching_object.weight == 1.0
    assert rewards_cfg.object_goal_tracking.weight == 16.0
    assert rewards_cfg.object_goal_tracking_fine_grained.weight == 5.0
    assert rewards_cfg.object_goal_tracking_l1.weight == 0.0
    assert rewards_cfg.action_rate.weight == -1.0e-2
    assert rewards_cfg.joint_vel.weight == -1.0e-2
    assert MYCOBOT_PRO450_TCP_OFFSET_POS == (0.127, 0.0, 0.0)
    assert rewards_cfg.object_tips_on_cube.weight == -3.0

    # env0: pinched in jaws and lifted
    # env1: flung away
    # env2: on table, gripper open
    # env3: body-pin — cube beside the TCP, gripper closed, off the table
    cube_pos = torch.tensor(
        [
            [0.20, 0.0, 0.12],
            [0.20, 0.0, 0.80],
            [0.20, 0.0, 0.055],
            [0.12, 0.0, 0.12],
        ]
    )
    cube_vel = torch.tensor(
        [
            [0.05, 0.0, 0.02],
            [0.0, 0.0, 2.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    ee_pos = torch.tensor(
        [
            [0.20, 0.0, 0.12],
            [0.20, 0.0, 0.12],
            [0.20, 0.0, 0.08],
            [0.20, 0.0, 0.12],
        ]
    )
    gripper_q = torch.tensor([[-0.10], [-0.10], [-1.012], [-0.10]])
    env = _env(cube_pos, cube_vel, ee_pos, gripper_q)
    kwargs = _held_kwargs(env)

    lift_held = object_is_lifted_and_held(env, **kwargs)
    lift = (cube_pos[:, 2] > 0.10).float()
    throw = object_throwing(env, airborne_height=0.12, vel_threshold=0.6, max_object_height=0.35, **kwargs)
    pinch = object_pinch(
        env,
        std=0.08,
        gripper_cfg=env._gripper_cfg,
        gripper_open_pos=-1.012,
        gripper_close_pos=0.0,
        wrist_cfg=env._wrist_cfg,
    )
    open_q = torch.full_like(gripper_q, -1.012)
    env_open = _env(cube_pos, cube_vel, ee_pos, open_q)
    pinch_open = object_pinch(
        env_open,
        std=0.08,
        gripper_cfg=env_open._gripper_cfg,
        gripper_open_pos=-1.012,
        gripper_close_pos=0.0,
        wrist_cfg=env_open._wrist_cfg,
    )
    reach = object_ee_distance_aligned(env, std=0.1, wrist_cfg=env._wrist_cfg)
    align = ee_z_down(env, asset_cfg=env._wrist_cfg)
    pin = object_pressed_against_arm(env, arm_cfg=env._gripper_cfg, **kwargs)
    tips = object_tips_on_cube(env)
    assert pinch[2].item() > 0.0, pinch
    assert pinch[0].item() > pinch[2].item()
    assert pinch[0].item() > pinch_open[0].item()
    assert abs(pinch[3].item() - pinch_open[3].item()) < 1e-5
    assert torch.all(align > 0.99)
    assert tips[0].item() == 0.0, tips
    assert tips[2].item() == 0.0, tips
    tips_park = _env(
        torch.tensor([[0.20, 0.04, 0.12]]),
        torch.zeros(1, 3),
        torch.tensor([[0.20, 0.0, 0.12]]),
        torch.tensor([[-0.10]]),
    )
    assert object_tips_on_cube(tips_park).item() == 1.0

    side_env = _env(cube_pos, cube_vel, ee_pos, gripper_q, wrist_xyzw=(0.0, 0.0, 0.0, 1.0))
    side_pinch = object_pinch(
        side_env,
        std=0.08,
        gripper_cfg=side_env._gripper_cfg,
        gripper_open_pos=-1.012,
        gripper_close_pos=0.0,
        wrist_cfg=side_env._wrist_cfg,
    )
    side_reach = object_ee_distance_aligned(side_env, std=0.1, wrist_cfg=side_env._wrist_cfg)
    side_align = ee_z_down(side_env, asset_cfg=side_env._wrist_cfg)
    assert torch.all(side_align < 1e-5)
    assert torch.all(side_pinch < pinch)
    assert torch.all(side_reach < reach)

    assert lift[0].item() == 1.0, lift
    assert lift[1].item() == 1.0, lift
    assert lift[2].item() == 0.0, lift
    assert lift[3].item() == 1.0, lift
    assert lift_held[0].item() == 1.0, lift_held
    assert lift_held[1].item() == 0.0, lift_held
    assert lift_held[2].item() == 0.0, lift_held
    assert lift_held[3].item() == 0.0, lift_held
    assert throw[0].item() == 0.0, throw
    assert throw[1].item() > 2.0, throw
    assert throw[2].item() == 0.0, throw
    assert throw[3].item() == 0.0, throw
    assert pin[0].item() == 0.0, pin
    assert pin[3].item() == 1.0, pin

    lift_w = rewards_cfg.lifting_object.weight
    held_w = rewards_cfg.lifting_held.weight
    throw_w = rewards_cfg.object_throwing.weight
    pin_w = rewards_cfg.object_pressed_against_arm.weight
    carry_score = lift[0] * lift_w + lift_held[0] * held_w + throw[0] * throw_w + pin[0] * pin_w
    fling_score = lift[1] * lift_w + lift_held[1] * held_w + throw[1] * throw_w + pin[1] * pin_w
    pin_score = lift[3] * lift_w + lift_held[3] * held_w + throw[3] * throw_w + pin[3] * pin_w
    assert carry_score > 0.0
    assert fling_score < carry_score
    assert pin_score < carry_score
    print("hold/throw reward check passed")
    print(f"  carry score={carry_score.item():.3f}  fling score={fling_score.item():.3f}  pin score={pin_score.item():.3f}")
    print(f"  lift terms={lift.tolist()}  throw terms={throw.tolist()}  pin terms={pin.tolist()}")
    print(f"  down pinch={pinch[0].item():.3f}  side pinch={side_pinch[0].item():.3f}")
    print(f"  down reach={reach[0].item():.3f}  side reach={side_reach[0].item():.3f}")


if __name__ == "__main__":
    main()
