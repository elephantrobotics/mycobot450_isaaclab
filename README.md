# MyCobot Pro450 Isaac Lab

External Isaac Lab project for the MyCobot Pro450 + F100. Two tasks live in separate folders and share the robot asset.

Python package name: `mycobot450_isaaclab` (install path `source/mycobot450_isaaclab`).

Robot USD stays in the sibling Isaac Sim repo `mycobot450_isaacsim`. This repository does not vendor `.usd` / `.usda` files.

## Tasks

| Gym ID | Folder | What it does |
|--------|--------|----------------|
| `Isaac-Reach-Mycobot-Pro450-v0` | `tasks/manager_based/mycobot_pro450_reach` | Official-style reach: move Link6 to a sampled pose, tool +Z down. No gripper action. |
| `Isaac-Reach-Mycobot-Pro450-Play-v0` | same | Play / visualize reach |
| `Isaac-Lift-Cube-Mycobot-Pro450-v0` | `tasks/manager_based/mycobot_pro450_lift` | Pick a cube and carry it |
| `Isaac-Lift-Cube-Mycobot-Pro450-Play-v0` | same | Play / visualize lift |
| `mycobot450_isaaclab` | lift | Alias for the lift train task |

Robot USD and actuators: `source/mycobot450_isaaclab/mycobot450_isaaclab/assets/robots/mycobot_pro450.py`.

## Requirements

- Isaac Lab 3.0 / Isaac Sim 6.0.1
- Combined robot USD: `~/mycobot450_isaacsim/USD/sim_450_f100/450_f100.usda`
- Optional: `ISAACLAB_PATH` if Isaac Lab is not cloned at `~/IsaacLab`
- Optional: `MYCOBOT_PRO450_USD_PATH` to override the USD file

## Install

```bash
source ~/env_isaaclab/bin/activate
python -m pip install -e source/mycobot450_isaaclab
```

## Reach (beginner)

Same MDP as `Isaac-Reach-Franka-v0`: joint-position arm, `ee_pose` command, position + orientation tracking. Workspace is `x 0.18–0.32`, `y ±0.12`, `z 0.18–0.32` so Link6 stays above the F100/table and away from the base. Tool +Z is down (`pitch = π`); yaw is `±π/4`. Position fine-tracking uses `std = 0.05` (tighter than Franka's `0.1`). Train 1500 iterations from scratch.

```bash
cd ~/mycobot450_isaaclab
python scripts/list_envs.py
python scripts/random_agent.py --task Isaac-Reach-Mycobot-Pro450-Play-v0 --num_envs 4 --viz kit
python -u scripts/rsl_rl/train.py --task Isaac-Reach-Mycobot-Pro450-v0 --headless --num_envs 4096
python -u scripts/rsl_rl/play.py --task Isaac-Reach-Mycobot-Pro450-Play-v0 --viz kit --num_envs 4 --real-time --checkpoint logs/rsl_rl/mycobot_pro450_reach/<run>/model_<N>.pt
```

TensorBoard:

```bash
python -m tensorboard.main --logdir logs/rsl_rl/mycobot_pro450_reach
```

## Lift (cube)

```bash
python -u scripts/rsl_rl/train.py --task Isaac-Lift-Cube-Mycobot-Pro450-v0 --headless --num_envs 4096
python -u scripts/rsl_rl/play.py --task Isaac-Lift-Cube-Mycobot-Pro450-Play-v0 --viz kit --num_envs 4 --real-time --checkpoint <path-to.pt>
```

## Notes

- Reach does not use the F100 as an action; fingers stay at the default opening.
- Lift still drives `joint2_left_joint`; F100 mimics follow in the USD.
- `.gitignore` hides `logs/`, `__pycache__/`, and USD files. `scripts/` and `source/` are project code and should be committed.
