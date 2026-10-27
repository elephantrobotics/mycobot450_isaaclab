# mycobot450_isaaclab

Isaac Lab reinforcement-learning tasks for the MyCobot Pro 450 with an F100 gripper.

## Requirements

- Isaac Lab **release/3.0.0**, checked out beside this repository at `../IsaacLab`.
- Isaac Sim **6.2.0** (package version `6.2.0.0` in `uv.lock`).
- Python **3.12**, **uv**, **Git LFS**, and an NVIDIA GPU supported by Isaac Sim.

Isaac Lab packages are editable dependencies pointing to the local source checkout.
The lockfile does not pin that checkout's Git revision; use the matching
`release/3.0.0` source when setting up another machine.

## Installation

Run these commands from this repository's root:

```bash
git lfs pull
uv sync --locked --extra isaacsim
```

Git LFS supplies the robot USD assets. Complete the NVIDIA Kit license prompt
when launching Isaac Sim for the first time.

## Tasks

| Task | Purpose |
| --- | --- |
| `Mycobot450Isaaclab-Reach` | Train the 450 + F100 end-effector pose-tracking policy. |
| `Mycobot450Isaaclab-Reach-Play` | Play a trained Reach policy with visible target frames. |

The template balance task has been removed. The existing cube-lift task remains
in the repository as a legacy grasping example.

## Reach

The six arm joints track the pose of `f100_base_link`, the F100 mounting frame,
while the gripper stays open. Targets include the original wrist orientations,
vertically downward poses, and forward-left/right downward poses. Targets are
sampled through forward kinematics within the robot's joint and workspace limits.

The task uses the table dimensions and robot placement from
`Isaac-Lift-Cable-Franka`. Joint-velocity actions are filtered and integrated into
position targets to keep motion smooth. The policy uses 43 observations and six
actions, running at 30 Hz with physics at 120 Hz.

Check that the environment starts (this does not train a policy):

```bash
uv run --extra isaacsim isaaclab zero_agent \
    --task Mycobot450Isaaclab-Reach --num_envs 4 --max_steps 120 \
    --viz none physics=isaacsim_physx
```

Train with RSL-RL (2048 environments by default; reduce `--num_envs` if needed):

```bash
uv run --extra isaacsim isaaclab train \
    --rl_library rsl_rl --task Mycobot450Isaaclab-Reach \
    --viz none physics=isaacsim_physx
```

Play the latest Reach checkpoint in the Kit window:

```bash
uv run --extra isaacsim isaaclab play \
    --rl_library rsl_rl --task Mycobot450Isaaclab-Reach-Play \
    --num_envs 1 --viz kit --real-time --checkpoint latest \
    physics=isaacsim_physx
```

Training logs and checkpoints are written to `logs/rsl_rl/reach_mycobot450_smooth/`.
To select a specific model, replace `latest` with its checkpoint path. Checkpoints
are generated locally and are not included in the repository. Use a checkpoint
trained with the current 43-observation velocity-action interface; older
position-action checkpoints are incompatible.

If a sourced ROS or Conda environment causes Python or shared-library conflicts,
prefix the command with `env -u PYTHONPATH -u LD_LIBRARY_PATH`. This only clears
those variables for that command; it is not required in a clean environment.
Keep `DISPLAY` set when opening the Kit window on the local desktop.

## Evaluation and tests

`scripts/` contains command-line tools for running simulations and writing results.
`tests/` contains automated regression checks run by pytest.

Evaluate a trained Reach checkpoint (replace `RUN` and `ITERATION`):

```bash
uv run --extra isaacsim python scripts/reach_evaluate.py \
    --checkpoint logs/rsl_rl/reach_mycobot450_smooth/RUN/model_ITERATION.pt \
    --num_envs 64 --rounds 4 --eval_seed 123 \
    --output outputs/reach/evaluation.json physics=isaacsim_physx
```

The evaluator samples goals independently of the training seed and reports pose
errors, success rates and joint-motion metrics. Each trial lasts 12 seconds by
default. Held success requires position error below 2 cm and orientation error
below 0.15 rad throughout the final second; settled success also requires every
joint speed to remain below 0.1 rad/s. Omit `--checkpoint` for a zero-action baseline.
Add `--switch_goals --episode_seconds 20` to check a transition between targets.
Use `--help` for goal replay, motion traces and optional policy export settings.
Generated reports belong under ignored `outputs/`.

Run the registration and Reach regression checks without launching Isaac Sim:

```bash
uv run --group dev pytest tests/test_registration.py tests/test_reach.py -q
```

The registration tests check task names and loadable configurations. The Reach
tests cover controller reset, smoothing, joint limits and target-pose sampling;
they complement the simulator evaluation above.

## Reach source layout

```text
src/mycobot450_isaaclab/tasks/reach/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── mycobot450/
│       ├── __init__.py                  # Train/play task registrations
│       ├── mycobot450_reach_env_cfg.py  # Robot, scene and environment settings
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py        # PPO settings
└── mdp/
    ├── __init__.py
    ├── __init__.pyi
    ├── actions.py                      # Smooth joint control
    ├── actions_cfg.py
    ├── commands.py                     # Forward kinematics and target sampling
    ├── commands_cfg.py
    ├── observations.py
    └── rewards.py
```

The environment inherits the generic `ReachEnvCfg` from Isaac Lab.
