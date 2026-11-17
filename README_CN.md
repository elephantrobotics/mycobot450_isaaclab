# mycobot450_isaaclab

MyCobot Pro 450 + F100 夹爪的 Reach 强化学习项目，使用 Isaac Lab 训练和仿真，通过 pymycobot 部署到真机。

策略跟踪 `f100_base_link` 位姿，使用 43 维观测、6 维动作，策略频率为 30 Hz，物理仿真为 120 Hz。真机由主机 CPU 推理，Isaac Sim 显示实测运动。

## 安装

需要 Python 3.12、uv、Git LFS 和 Isaac Sim 6.2.0 支持的 NVIDIA GPU。使用 Isaac Lab 的 `release/3.0.0` 源码分支，放置在同级 `IsaacLab` 目录中，对应路径已在 [pyproject.toml](pyproject.toml) 中配置。以下命令在 `mycobot450_isaaclab` 目录执行。如果 `../IsaacLab` 已有对应的源码，请跳过 `git clone` 命令。

```bash
git lfs install
git clone --branch release/3.0.0 --single-branch https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab lfs pull
git lfs pull
uv sync --locked --extra isaacsim
```

`uv sync` 会将本地 Isaac Lab 包和本项目安装到本项目的 `.venv` 中，无需单独创建 Isaac Lab 环境。

如果安装或训练时报错：

```text
error: The lockfile at `uv.lock` needs to be updated, but `--locked` was
       provided.
```

说明本地 Isaac Lab 源码或依赖配置与 `uv.lock` 不再一致。若继续使用当前源码，请在项目目录更新锁文件并同步环境，然后重新执行原命令：

```bash
uv lock
uv sync --locked --extra isaacsim
```

## 训练与导出

每次实验使用新的输出目录。默认命令自动完成三个 PPO 阶段，采用 2048 个环境、种子 42：600 次 base 更新、1000 次 mixed 更新，再进行 200 次 transfer 更新。

mixed 训练后，脚本自动用固定目标和切换目标评估最后三个 checkpoint，将选中的模型传给 transfer。transfer 保留其策略和优化器；存在可用的已到达姿态时，重置有 25% 的概率使用这些姿态。输出目录包含最终的 `model_final.pt`、对应的 `params/`，以及记录各阶段和 checkpoint 选择结果的 `training.json`。

```bash
uv run --locked --extra isaacsim python scripts/reach_train.py \
  --output_dir outputs/reach/tutorial/run
```

评估并导出最终的 checkpoint：

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

导出包含 `policy.pt` 和 `policy.onnx`。部署使用 TorchScript 模型 `bundle/policy.pt` 及对应的 `contract.json`。

## 硬件配置与目标

为仿真和部署生成同一套配置。以下命令只创建文件，不连接机器人，也不覆盖已有文件。

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

streaming 配置使用 SDK speed 15、动作增益 0.25、目标超前限制 0.1 rad，并启用运动保护。策略与指令发送仍为 30 Hz、1 倍时间尺度。标准关节映射无需手动设置校准标志。

目标采用训练中的四类姿态与完整采样范围：相对 `base_link` 的 X=0.12～0.35 m、Y=−0.20～0.20 m、Z=0.20～0.46 m。固定种子生成 24 个目标，加上首尾 home 和原第 3、4 个目标之间的一个笛卡尔过渡目标，共 27 段，每段 22 秒。过渡目标仍由策略跟踪，不会把其见证关节角直接作为运动指令发送。

真机运动前需确认独立的运动保护区域无障碍：X=0.08～0.41 m、Y=−0.23～0.28 m、Z=0.18～0.54 m。仅用于归位的额外区域为 X=0.05～0.20 m、Y=−0.10～0 m、Z=0.30～0.68 m。程序检查的是夹爪安装座，连杆、夹爪和线缆也必须有净空。更换硬件配置、空间或模型后需重新验证。

## 仿真

加载与真机部署相同的硬件配置：

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

等待完整运行 27 段（594 秒仿真时间），并显示 `Goal-plan validation passed: True`。Ctrl+C 可提前停止，但此时验证不完整。报告自动保存到 `outputs/reach/simulation/<时间戳>/report.json`。验证失败后不要继续执行真机运动。

## 真机部署

目前正在排查反复出现的 `3-07-*` 末端接收异常，暂停真机执行。故障原因解决、对应硬件配置重新验证通过前，不要运行下方终端 B 的运动指令。

给机器人上电、启动控制服务并选择位置模式。F100 夹爪必须为空。确认上述运动与归位区域有净空，全程在旁观察并备好急停。

**终端 A：显示实测姿态。**

```bash
uv run --locked --extra isaacsim --with pymycobot==4.0.7 \
  python scripts/reach_live.py \
  --ip 192.168.0.232 --robot-port 4500 \
  --bundle outputs/reach/tutorial/run/bundle \
  --hardware-config outputs/reach/tutorial/run/hardware.json \
  --port 14550 \
  physics=isaacsim_physx
```

等待显示 `SNAPSHOT`。显示程序读取一次实机角度后关闭机器人连接，随后显示部署程序发来的反馈，不发送运动指令。

**终端 B：归位并运行（真实运动）。**

另开终端进入项目目录，两个终端使用相同的机器人 IP。

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

机器人按需打开夹爪，以 SDK speed 10 归位，再以 30 Hz、SDK speed 15 运行策略。归位失败则不执行策略。`--response-trial` 启用带运动保护的有人值守验证，不会把硬件标记为已验证。

第一个目标保持初始姿态 22 秒，全部 27 段每 594 秒重复一轮，直到 Ctrl+C、保护触发或故障停止。这是可重复的固定计划，每轮不会重新采样目标。终端 A 应显示 `LIVE`、实测关节运动和当前目标。

先在**终端 B** 按 Ctrl+C，等待 `stop_acknowledged: true` 和 `post_stop_stationary: true`，再关闭终端 A。只关闭显示窗口不会停止真机。日志和最终报告自动生成在 `outputs/reach/deploy/<时间戳>/`。
