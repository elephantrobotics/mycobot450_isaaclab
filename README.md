# mycobot450_isaaclab

Reach reinforcement learning for the MyCobot Pro 450 with an F100 gripper,
using Isaac Lab for training and simulation and pymycobot for deployment.

The policy tracks `f100_base_link` with 43 observations and 6 actions at 30 Hz;
physics runs at 120 Hz. Deployment uses host CPU inference, with measured robot
motion displayed in Isaac Sim.

## Installation

Requires Python 3.12, uv, Git LFS, and an NVIDIA GPU supported by Isaac Sim 6.2.0.
Use Isaac Lab's `release/3.0.0` source branch in a sibling `IsaacLab` directory,
as configured in [pyproject.toml](pyproject.toml). Run the following commands
from the `mycobot450_isaaclab` directory. If the matching Isaac Lab checkout
already exists at `../IsaacLab`, skip the `git clone` command.

```bash
git lfs install
git clone --branch release/3.0.0 --single-branch https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab lfs pull
git lfs pull
uv sync --locked --extra isaacsim
```

`uv sync` installs the local Isaac Lab packages and this project into this
project's `.venv`; a separate Isaac Lab environment is not required.

If installation or training reports:

```text
error: The lockfile at `uv.lock` needs to be updated, but `--locked` was
       provided.
```

The local Isaac Lab checkout or dependency configuration no longer matches
`uv.lock`. To use the current checkout, update the lockfile from the project
directory, sync the environment, and then rerun the original command:

```bash
uv lock
uv sync --locked --extra isaacsim
```

## Training and export

Use a new output directory for each experiment. The default command runs all
three PPO stages with 2048 environments and seed 42: 600 base updates, 1000
mixed-orientation updates, then 200 transfer updates.

After mixed training, the script evaluates the last three checkpoints on fixed
and switching goals and passes the selected checkpoint to transfer. Transfer
keeps its policy and optimizer; resets have a 25% chance of using a previously
reached state when one is available. The output directory contains the final
`model_final.pt`, matching `params/`, and a `training.json` record of the stages
and checkpoint selection.

```bash
uv run --locked --extra isaacsim python scripts/reach_train.py \
  --output_dir outputs/reach/tutorial/run \
  --viz kit
```

Evaluate and export the final checkpoint:

```bash
uv run --locked --extra isaacsim python scripts/reach_evaluate.py \
  --checkpoint outputs/reach/tutorial/run/model_final.pt \
  --num_envs 128 --rounds 4 --eval_seed 2026 \
  --export_dir outputs/reach/tutorial/run/exported \
  --output outputs/reach/tutorial/run/evaluation.json \
  physics=isaacsim_physx
```

```bash
uv run --script src/mycobot450_isaaclab/utils/export_reach.py \
  --checkpoint outputs/reach/tutorial/run/model_final.pt \
  --policy outputs/reach/tutorial/run/exported/policy.pt \
  --output outputs/reach/tutorial/run/bundle
```

Exports include `policy.pt` and `policy.onnx`. Deployment uses the TorchScript
`bundle/policy.pt` and its `contract.json`.

## Hardware settings and goals

Generate the same settings for simulation and deployment. These commands create
files only; they do not connect to the robot. Existing files are not overwritten.

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py init-config \
  --bundle outputs/reach/tutorial/run/bundle \
  --profile streaming \
  --home-motion-workspace 0.05 0.20 -0.10 0 0.30 0.68 \
  --output outputs/reach/tutorial/run/hardware.json
```

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py sample-goals \
  --bundle outputs/reach/tutorial/run/bundle \
  --sampler default --seed 2026 --count 24 --interval 22 \
  --motion-workspace 0.08 0.41 -0.23 0.28 0.18 0.54 \
  --transition-after 3 \
  --output outputs/reach/tutorial/run/goals.json
```

The streaming profile uses SDK speed 15, action gain 0.25, a 0.1 rad target-lead
limit, and motion guards. The policy and command rate remain 30 Hz at time scale
1. Standard joint mapping needs no manual calibration flag.

Goals use the four training orientation families and the full sampling range:
X=0.12–0.35 m, Y=−0.20–0.20 m, Z=0.20–0.46 m relative to `base_link`.
The fixed seed generates 24 targets. Home at both ends and one Cartesian
transition between original goals 3 and 4 make 27 segments of 22 seconds.
The transition is tracked by the policy; its witness joints are not sent as a
motion command.

Before real motion, confirm clearance for the separate motion bounds:
X=0.08–0.41 m, Y=−0.23–0.28 m, Z=0.18–0.54 m. The extra homing-only box is
X=0.05–0.20 m, Y=−0.10–0 m, Z=0.30–0.68 m. These checks cover the gripper
mounting frame; the links, gripper and cables also need clearance. Different
hardware settings, spaces, or models require another validation run.

## Simulation

Load the same hardware configuration used for deployment:

```bash
uv run --locked --extra isaacsim python scripts/reach_play.py \
  --policy outputs/reach/tutorial/run/bundle/policy.pt \
  --goal-plan outputs/reach/tutorial/run/goals.json \
  --hardware-config outputs/reach/tutorial/run/hardware.json \
  --time-scale 1 --target-update-decimation 4 \
  --velocity-source finite_difference \
  --steps 17820 --viz kit \
  physics=isaacsim_physx
```

Wait for all 27 segments (594 simulated seconds) and
`Goal-plan validation passed: True`. Ctrl+C stops early and leaves validation
incomplete. Reports are saved to `outputs/reach/simulation/<timestamp>/report.json`.
Do not proceed to real motion after a failed validation.

## Real robot

Real-robot execution is paused while recurring `3-07-*` end-reception faults
are investigated. Do not run Terminal B below until the cause is resolved and
the affected setup passes validation again.

Power on the robot, start its control service, and select position mode. The F100
must be empty. Confirm the motion and homing clearance above, remain beside the
robot, and keep the emergency stop available.

**Terminal A — measured-state viewer:**

```bash
uv run --locked --extra isaacsim --with pymycobot==4.0.7 \
  python scripts/reach_live.py \
  --ip 192.168.0.232 --robot-port 4500 \
  --bundle outputs/reach/tutorial/run/bundle \
  --hardware-config outputs/reach/tutorial/run/hardware.json \
  --port 14550 \
  physics=isaacsim_physx
```

Wait for `SNAPSHOT`. The viewer reads the robot once, closes its robot connection,
then displays feedback from deployment. It sends no motion commands.

**Terminal B — home and run (real motion):**

Open another terminal in the project directory. Use the same robot IP in both
terminals.

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py run \
  --bundle outputs/reach/tutorial/run/bundle \
  --hardware-config outputs/reach/tutorial/run/hardware.json \
  --goal-plan outputs/reach/tutorial/run/goals.json \
  --ip 192.168.0.232 --port 4500 \
  --continuous --execute --response-trial --allow-goal-jumps \
  --home --home-speed 10 --open-gripper \
  --command-hz 30 --time-scale 1 --duration 0 \
  --telemetry-port 14550
```

The robot opens the gripper if needed, homes at SDK speed 10, then runs the
policy at 30 Hz with SDK speed 15. Homing failure prevents policy execution.
`--response-trial` enables supervised validation with motion guards; it does
not mark the hardware as verified.

The first goal holds the home pose for 22 seconds. All 27 segments then repeat
every 594 seconds until Ctrl+C, a guard trigger, or a fault. This is a fixed,
repeatable plan; the targets are not resampled each cycle. Terminal A should
show `LIVE` with measured joint motion and the current goal.

Press Ctrl+C in **Terminal B** first. Wait for `stop_acknowledged: true` and
`post_stop_stationary: true`, then close Terminal A. Closing the viewer alone
does not stop the robot. Logs and the final report are created automatically
under `outputs/reach/deploy/<timestamp>/`.
