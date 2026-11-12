# mycobot450_isaaclab

Reach reinforcement learning for the MyCobot Pro 450 with an F100 gripper,
using Isaac Lab for training and simulation and pymycobot for deployment.

The policy tracks `f100_base_link` with 43 observations and 6 actions at 30 Hz;
physics runs at 120 Hz. Deployment uses host CPU inference, with measured robot
motion displayed in Isaac Sim.

## Installation

Requires Python 3.12, uv, Git LFS, Isaac Lab 3.0, and an NVIDIA GPU supported by
Isaac Sim 6.2.0. Local dependency paths are configured in
[pyproject.toml](pyproject.toml). Run commands from the project directory.

```bash
git lfs install
git lfs pull
uv sync --locked --extra isaacsim
```

## Training and export

Use a new output directory for each experiment.

```bash
uv run --locked --extra isaacsim python scripts/reach_train.py \
  --output_dir outputs/reach/tutorial/training

uv run --locked --extra isaacsim python scripts/reach_evaluate.py \
  --checkpoint outputs/reach/tutorial/training/model_final.pt \
  --num_envs 128 --rounds 4 --eval_seed 2026 \
  --export_dir outputs/reach/tutorial/exported --output outputs/reach/tutorial/evaluation.json \
  physics=isaacsim_physx

uv run --script src/mycobot450_isaaclab/utils/export_reach.py \
  --checkpoint outputs/reach/tutorial/training/model_final.pt --policy outputs/reach/tutorial/exported/policy.pt \
  --output outputs/reach/tutorial/bundle
```

Defaults: 2048 environments, seed 42, 600 base-stage updates followed by 1000
mixed-orientation updates. Reduce `--num_envs` if GPU memory is limited.

Exports are saved as `policy.pt` and `policy.onnx`; deployment uses
`bundle/policy.pt` with `contract.json`. Training saves the selected checkpoint
as `training/model_final.pt` alongside its `params/` directory.

## Simulation

Check the exported policy, generate goals, and play the sequence:

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py self-check \
  --bundle outputs/reach/tutorial/bundle
```

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py sample-goals \
  --bundle outputs/reach/tutorial/bundle \
  --sampler default --seed 2026 --count 24 --interval 22 \
  --output outputs/reach/tutorial/goals.json
```

```bash
uv run --locked --extra isaacsim python scripts/reach_play.py \
  --policy outputs/reach/tutorial/bundle/policy.pt \
  --goal-plan outputs/reach/tutorial/goals.json \
  --time-scale 1 --target-update-decimation 4 \
  --velocity-source finite_difference \
  --steps 17160 --viz kit \
  physics=isaacsim_physx
```

The sequence has 26 segments of 22 seconds (572 seconds total), including home
at each end. Validation succeeds when the full run reports
`Goal-plan validation passed: True`. Ctrl+C stops playback early. Reports are
automatically saved to `outputs/reach/simulation/<timestamp>/report.json`.

The plan separates sampling bounds (`goal_workspace_m`) from motion bounds
(`motion_workspace_m`). Both default to X=0.12–0.35 m, Y=−0.20–0.20 m,
Z=0.20–0.46 m relative to `base_link`. Check physical clearance before expanding
motion bounds, then rerun simulation. Regenerate the plan when changing models.

## Real robot

After simulation passes, generate the hardware configuration:

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py init-config \
  --bundle outputs/reach/tutorial/bundle \
  --output outputs/reach/tutorial/hardware.json
```

- Verify joint mapping, limits, F100 ID, and speed on the robot before setting
  `calibration_verified: true`. Test command response and stopping first.
- Homing requires joint signs +1, zero offsets, and F100 ID 14. Confirm the home
  path; any extra clearance belongs in `home_motion_workspaces_m`.
- Power on the robot, start its control service, and select position mode.
  Empty the gripper, clear the arm, gripper, and cable paths, and remain beside
  the emergency stop. Workspace checks cover only the mounting frame.

### Terminal A: Isaac Sim viewer

From the project directory, use your robot's IP:

```bash
uv run --locked --extra isaacsim --with pymycobot==4.0.7 \
  python scripts/reach_live.py \
  --ip 192.168.0.232 --robot-port 4500 \
  --bundle outputs/reach/tutorial/bundle \
  --hardware-config outputs/reach/tutorial/hardware.json \
  --port 14550 physics=isaacsim_physx
```

Wait for `SNAPSHOT` before starting Terminal B. The viewer displays measured
feedback and sends no motion commands.

### Terminal B: home and run (real motion)

Open another terminal in the project directory and use the same robot IP:

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py run \
  --bundle outputs/reach/tutorial/bundle \
  --hardware-config outputs/reach/tutorial/hardware.json \
  --goal-plan outputs/reach/tutorial/goals.json \
  --ip 192.168.0.232 --port 4500 \
  --continuous --execute --allow-goal-jumps \
  --home --home-speed 10 --open-gripper \
  --command-hz 30 --time-scale 1 --duration 0 \
  --telemetry-port 14550
```

The robot opens the gripper, homes at SDK speed 10, then runs the policy at
30 Hz, cycling the goal plan until Ctrl+C or a protection/fault stop
(`--duration 0`). Homing failure blocks policy execution.

Stop with **Ctrl+C in Terminal B** and wait for confirmation. Closing the viewer
does not stop the robot. Faults stop execution without automatic retry or homing.
Logs are saved under `outputs/reach/deploy/`; check pose errors and
`final_second_tracking_success` in the report as well as `passed`.
