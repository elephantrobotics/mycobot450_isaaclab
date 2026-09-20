# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register this project's tasks, then run Isaac Lab RSL-RL training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mycobot450_isaaclab.tasks  # noqa: F401


def _isaaclab_rl_dir() -> Path:
    """Locate Isaac Lab's ``scripts/reinforcement_learning`` directory."""
    roots: list[Path] = []
    env_path = os.environ.get("ISAACLAB_PATH")
    if env_path:
        roots.append(Path(env_path))
    roots.append(Path.home() / "IsaacLab")
    for root in roots:
        rl_dir = root / "scripts" / "reinforcement_learning"
        if (rl_dir / "rsl_rl" / "train_rsl_rl.py").is_file():
            return rl_dir
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        "Isaac Lab RSL-RL train script not found. Set ISAACLAB_PATH to the IsaacLab clone. "
        f"Searched: {searched}"
    )


_RL_DIR = _isaaclab_rl_dir()
_TRAIN = _RL_DIR / "rsl_rl" / "train_rsl_rl.py"

sys.path.insert(0, str(_RL_DIR))
sys.path.insert(0, str(_TRAIN.parent))

import train_rsl_rl  # noqa: E402


def _force_line_buffered_stdio() -> None:
    """Keep iteration banners visible when Kit replaces or block-buffers stdout."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except Exception:
            continue


def _echo_iteration_when_stdout_is_not_a_tty() -> None:
    """Agent/background shells are not a TTY; Kit then swallows RSL-RL's print()."""
    from rsl_rl.utils.logger import Logger

    original_log = Logger.log

    def log(self, *args, **kwargs):
        original_log(self, *args, **kwargs)
        if sys.stdout.isatty():
            return
        it = kwargs.get("it")
        total_it = kwargs.get("total_it")
        if it is None or total_it is None:
            return
        line = f"Learning iteration {it}/{total_it}"
        print(line, flush=True)
        print(line, file=sys.stderr, flush=True)

    Logger.log = log


if __name__ == "__main__":
    _force_line_buffered_stdio()
    _echo_iteration_when_stdout_is_not_a_tty()
    train_rsl_rl.run(sys.argv[1:])
