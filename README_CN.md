# mycobot450_isaaclab

MyCobot Pro 450 + F100 夹爪的 Reach 强化学习项目，使用 Isaac Lab 训练和仿真，通过 pymycobot 部署到真机。

策略跟踪 `f100_base_link` 位姿，使用 43 维观测、6 维动作，策略频率为 30 Hz，物理仿真为 120 Hz。真机由主机 CPU 推理，Isaac Sim 显示实测运动。

## 安装

需要 Python 3.12、uv、Git LFS、Isaac Lab 3.0 和 Isaac Sim 6.2.0 支持的 NVIDIA GPU。本地依赖路径在 [pyproject.toml](pyproject.toml) 中配置。以下命令在项目目录执行。

```bash
git lfs install
git lfs pull
uv sync --locked --extra isaacsim
```

## 训练与导出

每次实验使用新的输出目录。

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

默认使用 2048 个环境、随机种子 42，基础阶段更新 600 次，混合朝向阶段更新 1000 次。显存不足时可减小 `--num_envs`。

导出文件为 `policy.pt` 和 `policy.onnx`，部署使用配套的 `bundle/policy.pt` 与 `contract.json`。训练会将选中的 checkpoint 保存为 `training/model_final.pt`，并自动复制配套的 `params/` 目录。

## 仿真

检查导出模型，生成目标并播放：

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

序列共 26 段，每段 22 秒，总计 572 秒，首尾各有一段 home。完整运行后显示 `Goal-plan validation passed: True` 表示通过。Ctrl+C 可提前停止。报告自动保存到 `outputs/reach/simulation/<时间戳>/report.json`。

目标计划分别设置采样边界 `goal_workspace_m` 和运动边界 `motion_workspace_m`，默认均为相对 `base_link` 的 X=0.12～0.35 m、Y=−0.20～0.20 m、Z=0.20～0.46 m。扩大运动边界前需确认实机净空，再重新仿真；更换模型后需重新生成计划。

## 真机部署

仿真通过后，生成硬件配置：

```bash
uv run --script src/mycobot450_isaaclab/utils/deploy.py init-config \
  --bundle outputs/reach/tutorial/bundle \
  --output outputs/reach/tutorial/hardware.json
```

- 在实机核对关节映射、限位、F100 ID 和速度后，才能设置 `calibration_verified: true`。先验证指令响应和停止行为。
- 归位要求关节方向 +1、零点偏移 0、F100 ID 14。确认归位路径，额外净空写入 `home_motion_workspaces_m`。
- 给机器人上电、启动控制服务并选择位置模式。清空夹爪，确认机械臂、夹爪和线缆的运动区域无障碍，全程在旁并备好急停。工作空间检查只覆盖安装座。

### 终端 A：Isaac Sim 显示

在项目目录执行，IP 改为实际机器人地址：

```bash
uv run --locked --extra isaacsim --with pymycobot==4.0.7 \
  python scripts/reach_live.py \
  --ip 192.168.0.232 --robot-port 4500 \
  --bundle outputs/reach/tutorial/bundle \
  --hardware-config outputs/reach/tutorial/hardware.json \
  --port 14550 physics=isaacsim_physx
```

看到 `SNAPSHOT` 后再启动终端 B。显示程序同步实测反馈，不发送运动指令。

### 终端 B：归位并运行（真实运动）

另开终端进入项目目录，使用相同的机器人 IP：

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

机器人先打开夹爪，以 SDK speed 10 归位，再以 30 Hz 持续循环目标计划，直到 Ctrl+C、保护触发或故障停止（`--duration 0`）；归位失败则不执行策略。

在**终端 B 按 Ctrl+C 停止**，等待停止确认。关闭显示窗口不会停止机器人。故障后不自动重试或归位。日志保存在 `outputs/reach/deploy/`；除 `passed` 外，还需查看报告中的位姿误差和 `final_second_tracking_success`。
