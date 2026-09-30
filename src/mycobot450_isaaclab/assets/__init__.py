# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paths and configurations for project-owned assets."""

from pathlib import Path

MYCOBOT450_ISAACLAB_ASSETS_DIR = Path(__file__).resolve().parent / "data"
"""Path to project-owned asset data."""

from .robots.mycobot_pro450 import MYCOBOT_PRO450_CFG, MYCOBOT_PRO450_HIGH_PD_CFG  # noqa: E402, F401
