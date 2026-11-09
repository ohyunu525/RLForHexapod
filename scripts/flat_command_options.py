"""Command difficulty options for the 67-observation flat walking task.

This module deliberately has no simulator dependency so launchers can validate
arguments before creating a simulator or loading a checkpoint.
"""

from __future__ import annotations

import math


TASK = "Hexapod-Walk-Commands"
PRESETS = {
    "standard": {"lin_vel_x": (-0.1, 0.2), "lin_vel_y": (-0.1, 0.1),
                 "yaw_rate": (-0.5, 0.5), "command_interval": (10.0, 10.0)},
    "hard": {"lin_vel_x": (-0.15, 0.30), "lin_vel_y": (-0.15, 0.15),
             "yaw_rate": (-0.75, 0.75), "command_interval": (3.0, 5.0)},
    "expert": {"lin_vel_x": (-0.2, 0.4), "lin_vel_y": (-0.2, 0.2),
               "yaw_rate": (-1.0, 1.0), "command_interval": (1.5, 3.0)},
    "extreme": {"lin_vel_x": (-0.25, 0.50), "lin_vel_y": (-0.25, 0.25),
                "yaw_rate": (-1.25, 1.25), "command_interval": (1.0, 2.0)},
}
RANGE_KEYS = ("lin_vel_x", "lin_vel_y", "yaw_rate", "command_interval")
OPTION_KEYS = ("command_difficulty", *RANGE_KEYS, "standing_fraction")


def add_command_arguments(parser):
    """Add optional overrides; None distinguishes an omitted option on old tasks."""
    parser.add_argument("--command-difficulty", choices=tuple(PRESETS), default=None,
                        help="Flat command preset (Commands task default: hard)")
    for key, description in (
        ("lin_vel_x", "Forward/backward velocity range in m/s"),
        ("lin_vel_y", "Lateral velocity range in m/s"),
        ("yaw_rate", "Turning velocity range in rad/s"),
        ("command_interval", "Seconds between velocity-command changes"),
    ):
        parser.add_argument("--" + key.replace("_", "-"), nargs=2, type=float,
                            default=None, metavar=("MIN", "MAX"), help=description)
    parser.add_argument("--standing-fraction", type=float, default=None,
                        help="Probability of a zero-velocity command (default: 0.30)")


def _range(key, values):
    try:
        if len(values) != 2:
            raise ValueError
        low, high = (float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must contain two finite numbers") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{key} requires finite MIN <= MAX")
    if key == "command_interval" and low <= 0:
        raise ValueError("command_interval must be strictly positive")
    return [low, high]


def _validated_settings(settings):
    difficulty = settings.get("command_difficulty")
    if difficulty not in PRESETS:
        raise ValueError(f"command_difficulty must be one of {tuple(PRESETS)}")
    result = {"command_difficulty": difficulty}
    for key in RANGE_KEYS:
        result[key] = _range(key, settings.get(key))
    try:
        fraction = float(settings.get("standing_fraction"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("standing_fraction must be finite and in [0, 1]") from exc
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("standing_fraction must be finite and in [0, 1]")
    result["standing_fraction"] = fraction
    return result


def resolve_command_settings(args):
    """Resolve a serializable preset plus explicit overrides, without mutation."""
    explicit = {key: getattr(args, key, None) for key in OPTION_KEYS}
    if args.task != TASK:
        if any(value is not None for value in explicit.values()):
            raise ValueError(f"Command difficulty options require --task {TASK}")
        return None
    difficulty = explicit["command_difficulty"] or "hard"
    if difficulty not in PRESETS:
        raise ValueError(f"command_difficulty must be one of {tuple(PRESETS)}")
    settings = {"command_difficulty": difficulty, **PRESETS[difficulty], "standing_fraction": 0.30}
    settings.update({key: value for key, value in explicit.items() if value is not None})
    return _validated_settings(settings)


def apply_command_settings(env_cfg, settings):
    """Update only velocity commands, after all settings pass validation."""
    if settings is None:
        return
    validated = _validated_settings(settings)
    command = env_cfg.commands.base_velocity
    if command.heading_command is not False:
        raise ValueError("Command difficulty requires heading_command=False")
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("Control-step duration must be finite and positive")
    if validated["command_interval"][0] < control_dt:
        raise ValueError("command_interval MIN must be at least one control step")
    command.ranges.lin_vel_x = tuple(validated["lin_vel_x"])
    command.ranges.lin_vel_y = tuple(validated["lin_vel_y"])
    command.ranges.ang_vel_z = tuple(validated["yaw_rate"])
    command.resampling_time_range = tuple(validated["command_interval"])
    command.rel_standing_envs = validated["standing_fraction"]
