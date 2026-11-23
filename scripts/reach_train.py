"""Train Reach through base, mixed, checkpoint selection, and reached-state transfer."""

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "reach_mycobot450_staged"


def prepare_continuation(source: Path, destination: Path, exploration_std: float = 0.15) -> int:
    """Keep learned weights/normalizers, reset exploration, and discard the old optimizer."""
    if not math.isfinite(exploration_std) or not 0.005 <= exploration_std <= 1.0:
        raise ValueError("exploration_std must be finite and inside [0.005, 1.0]")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    actor = checkpoint["actor_state_dict"]
    if actor["mlp.0.weight"].shape != (128, 43) or actor["mlp.6.weight"].shape != (6, 64):
        raise ValueError("Expected the current 43-input, six-action Reach policy")
    std = actor["distribution.std_param"]
    if std.shape != (6,):
        raise ValueError("Expected six scalar Gaussian standard deviations")
    if "critic_state_dict" not in checkpoint or not isinstance(checkpoint.get("iter"), int):
        raise ValueError("Expected a complete RSL-RL actor/critic training checkpoint")
    actor["distribution.std_param"] = torch.full_like(std, exploration_std)
    checkpoint.pop("optimizer_state_dict", None)
    # Exclusive creation prevents overwriting a user checkpoint, including the source.
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return checkpoint["iter"]


def training_command(*, num_envs, seed, iterations, timestamp, stage, checkpoint=None, viz="none"):
    """Use the upstream training CLI; only the documented stage settings differ."""
    if stage not in ("base", "mixed", "transfer"):
        raise ValueError("stage must be base, mixed or transfer")
    if (checkpoint is not None) != (stage != "base"):
        raise ValueError("Mixed and transfer stages require a checkpoint")
    learning_rate = 0.001 if stage == "base" else 0.0003
    if stage == "transfer":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        learning_rate = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
        if type(saved.get("iter")) is not int or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Transfer requires a complete PPO checkpoint with a valid optimizer learning rate")
    command = [
        sys.executable,
        "-m",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "Mycobot450Isaaclab-Reach",
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--max_iterations",
        str(iterations),
        "--experiment_name",
        EXPERIMENT,
        "--run_timestamp",
        timestamp,
        "--run_name",
        stage,
        "--viz",
        viz,
    ]
    if checkpoint is not None:
        command += [
            "--checkpoint",
            str(checkpoint),
            "--resume_optimizer_state" if stage == "transfer" else "--no-resume_optimizer_state",
        ]
    command += [
        "physics=isaacsim_physx",
        f"env.commands.ee_pose.legacy_online_sampling={'true' if stage == 'base' else 'false'}",
        f"env.commands.ee_pose.family_weights={'[1.0,0.0,0.0,0.0]' if stage == 'base' else '[0.25,0.25,0.25,0.25]'}",
        f"env.rewards.end_effector_orientation_tracking.weight={-0.5 if stage == 'base' else -1.5}",
        f"agent.algorithm.learning_rate={learning_rate}",
        f"agent.actor.distribution_cfg.init_std={0.15 if stage == 'mixed' else 0.3}",
    ]
    if stage == "transfer":
        command += [
            "env.events.reset_robot_joints.params.reached_state_probability=0.25",
            "agent.algorithm.schedule=fixed",
        ]
    return command


def select_checkpoint(checkpoints: list[Path], output: Path) -> tuple[Path, list[dict]]:
    """Rank late checkpoints using fixed and switching goals, separate from the final test seeds."""
    output.mkdir()
    scores = []
    for checkpoint in checkpoints:
        rates = []
        for mode in ("fixed", "switching"):
            report = output / f"{checkpoint.stem}_{mode}.json"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/reach_evaluate.py"),
                "--checkpoint",
                str(checkpoint),
                "--num_envs",
                "128",
                "--rounds",
                "2",
                "--eval_seed",
                "2025",
                "--output",
                str(report),
                "physics=isaacsim_physx",
            ]
            if mode == "switching":
                command += ["--switch_goals", "--episode_seconds", "20"]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            rates.append(json.loads(report.read_text())["settled_success_rate"])
        scores.append({"checkpoint": str(checkpoint), "fixed": rates[0], "switching": rates[1]})
    best = max(scores, key=lambda item: (item["fixed"] + item["switching"], item["fixed"]))
    return Path(best["checkpoint"]), scores


def git_revision(directory):
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "base", "mixed", "transfer"), default="all")
    parser.add_argument("--checkpoint", type=Path, help="Required with --stage mixed or transfer.")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_iterations", type=int, default=600)
    parser.add_argument("--mixed_iterations", type=int, default=1000)
    parser.add_argument("--transfer_iterations", type=int, default=200)
    parser.add_argument("--viz", choices=("none", "kit"), default="none", help="Visualizer for all training stages.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/reach/training"))
    args = parser.parse_args()
    if (args.stage in ("mixed", "transfer")) != (args.checkpoint is not None):
        parser.error("--checkpoint is required only with --stage mixed or transfer")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if min(args.num_envs, args.base_iterations, args.mixed_iterations, args.transfer_iterations) < 1:
        parser.error("num_envs and iteration counts must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}. Choose a new --output_dir to keep previous models.")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    logs = REPO_ROOT / "logs/rsl_rl" / EXPERIMENT
    base_dir, mixed_dir, transfer_dir = (logs / f"{timestamp}_{stage}" for stage in ("base", "mixed", "transfer"))
    base_checkpoint = (
        args.checkpoint.resolve() if args.checkpoint else base_dir / f"model_{args.base_iterations - 1}.pt"
    )
    warmstart = output / "stage2_start.pt"
    stages = ("base", "mixed", "transfer") if args.stage == "all" else (args.stage,)
    output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "recipe_version": 7,
        "stage": args.stage,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "base_iterations": args.base_iterations,
        "mixed_iterations": args.mixed_iterations,
        "transfer_iterations": args.transfer_iterations,
        "base_checkpoint": str(base_checkpoint),
        "initial_exploration_std_stage2": 0.15,
        "resume_optimizer_state_stage2": False,
        "selection_seed": 2025,
        "project_revision": git_revision(REPO_ROOT),
        "isaaclab_revision": git_revision(REPO_ROOT.parent / "IsaacLab"),
        "torch_version": str(torch.__version__),
        "commands": [],
        "completed_stages": [],
    }
    manifest_path = output / "training.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    selected = base_checkpoint
    try:
        for stage in stages:
            if stage == "base":
                source, final_checkpoint = None, base_checkpoint
            elif stage == "mixed":
                iteration = prepare_continuation(base_checkpoint, warmstart)
                source = warmstart
                if args.stage == "all" and iteration != args.base_iterations - 1:
                    raise RuntimeError("Unexpected base checkpoint iteration")
                # RSL-RL resumes its loop at the saved iteration, not saved iteration + 1.
                final_checkpoint = mixed_dir / f"model_{iteration + args.mixed_iterations - 1}.pt"
                manifest.update(
                    continuation_source_iteration=iteration,
                    continuation_source_sha256=hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
                    final_checkpoint=str(final_checkpoint),
                )
            elif stage == "transfer":
                source = selected
                saved = torch.load(source, map_location="cpu", weights_only=True)
                iteration = saved["iter"]
                final_checkpoint = transfer_dir / f"model_{iteration + args.transfer_iterations - 1}.pt"
                manifest.update(
                    continuation_source_iteration=iteration,
                    continuation_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    resume_optimizer_state_transfer=True,
                    reached_state_probability=0.25,
                    final_checkpoint=str(final_checkpoint),
                )
            # Transfer's selected checkpoint must exist before its optimizer settings can be read.
            command = training_command(
                num_envs=args.num_envs,
                seed=args.seed,
                iterations=getattr(args, f"{stage}_iterations"),
                timestamp=timestamp,
                stage=stage,
                checkpoint=source,
                viz=args.viz,
            )
            manifest["commands"].append(command)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"\n=== Reach stage: {stage} ===\n{shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if not final_checkpoint.is_file():
                raise RuntimeError(f"Training ended before its final checkpoint was saved: {final_checkpoint}")
            selected = final_checkpoint
            if stage == "mixed":
                print("\n=== Selecting mixed-stage checkpoint ===", flush=True)
                candidates = sorted(mixed_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))[-3:]
                selected, scores = select_checkpoint(candidates, output / "selection")
                manifest.update(mixed_selected_checkpoint=str(selected), selection_scores=scores)
            manifest["completed_stages"].append(stage)
            manifest.update(final_checkpoint=str(final_checkpoint), selected_checkpoint=str(selected))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        filename = "model_base.pt" if args.stage == "base" else "model_final.pt"
        if args.stage == "base":
            print("Base stage complete. Evaluate legacy goals before choosing the mixed-stage starting policy.")
        shutil.copy2(selected, output / filename)
        shutil.copytree(selected.parent / "params", output / "params")
        manifest["status"] = "complete"
        print(f"\nTraining complete. Play/evaluate this checkpoint: {output / filename}", flush=True)
    except BaseException:
        manifest["status"] = "incomplete"
        raise
    finally:
        warmstart.unlink(missing_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
