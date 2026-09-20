# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register this project's tasks, then run Isaac Lab RSL-RL play."""

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
        if (rl_dir / "rsl_rl" / "play_rsl_rl.py").is_file():
            return rl_dir
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        "Isaac Lab RSL-RL play script not found. Set ISAACLAB_PATH to the IsaacLab clone. "
        f"Searched: {searched}"
    )


_RL_DIR = _isaaclab_rl_dir()
_PLAY = _RL_DIR / "rsl_rl" / "play_rsl_rl.py"

sys.path.insert(0, str(_RL_DIR))
sys.path.insert(0, str(_PLAY.parent))

import play_rsl_rl  # noqa: E402

if __name__ == "__main__":
    play_rsl_rl.main()
