# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paths and configurations for project-owned assets."""

from pathlib import Path

RLFORHEXAPOD_ASSETS_DIR = Path(__file__).resolve().parent
"""Path to project-owned asset data."""

# Complete robot exported from the current URDF (including the sensor payload).
ROBOT_USDA = RLFORHEXAPOD_ASSETS_DIR / "robot" / "full" / "robot" / "robot.usda"
OBSTACLES_DIR = RLFORHEXAPOD_ASSETS_DIR / "obstacles"
