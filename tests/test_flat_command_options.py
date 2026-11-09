"""Flat command presets and validation without an Isaac Lab runtime."""

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flat_command_options as options


def arguments(*tokens, task=options.TASK):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=task)
    options.add_command_arguments(parser)
    return parser.parse_args(tokens)


def config():
    return SimpleNamespace(
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                heading_command=False, debug_vis=True,
                ranges=SimpleNamespace(lin_vel_x=(-.1, .2), lin_vel_y=(-.1, .1),
                                       ang_vel_z=(-.5, .5), heading=(-3.14, 3.14)),
                resampling_time_range=(10., 10.), rel_standing_envs=.3),
            base_height=SimpleNamespace(height_range=(.17, .24), resampling_time_range=(10., 10.))),
        sim=SimpleNamespace(dt=.005), decimation=4, episode_length_s=20.,
        observations=SimpleNamespace(width=67), rewards={"unchanged": 1.0})


class FlatCommandOptionsTests(unittest.TestCase):
    def test_parser_preserves_omitted_options(self):
        args = arguments()
        for key in options.OPTION_KEYS:
            self.assertIsNone(getattr(args, key))

    def test_new_task_defaults_to_hard_and_is_json_serializable(self):
        args = arguments()
        original = copy.deepcopy(vars(args))
        result = options.resolve_command_settings(args)
        self.assertEqual(result, {"command_difficulty": "hard", "lin_vel_x": [-.15, .30],
                                  "lin_vel_y": [-.15, .15], "yaw_rate": [-.75, .75],
                                  "command_interval": [3., 5.], "standing_fraction": .3})
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(vars(args), original)

    def test_all_presets_and_fresh_results(self):
        for name, expected in options.PRESETS.items():
            with self.subTest(preset=name):
                result = options.resolve_command_settings(arguments("--command-difficulty", name))
                for key, value in expected.items():
                    self.assertEqual(result[key], list(value))
                self.assertEqual(result["standing_fraction"], .3)
                result["lin_vel_x"][0] = -999
                fresh = options.resolve_command_settings(arguments("--command-difficulty", name))
                self.assertEqual(fresh["lin_vel_x"], list(expected["lin_vel_x"]))

    def test_overrides_replace_only_requested_fields(self):
        result = options.resolve_command_settings(arguments(
            "--command-difficulty", "standard", "--lin-vel-x", "-.2", ".35",
            "--lin-vel-y", "-.12", ".12", "--yaw-rate", "-.9", ".9",
            "--command-interval", "2", "4", "--standing-fraction", ".2"))
        self.assertEqual(result, {"command_difficulty": "standard", "lin_vel_x": [-.2, .35],
                                  "lin_vel_y": [-.12, .12], "yaw_rate": [-.9, .9],
                                  "command_interval": [2., 4.], "standing_fraction": .2})
        result = options.resolve_command_settings(arguments("--yaw-rate", "0", "0"))
        self.assertEqual(result["yaw_rate"], [0., 0.])
        self.assertEqual(result["lin_vel_x"], [-.15, .30])

    def test_old_tasks_return_none_and_reject_every_explicit_option(self):
        self.assertIsNone(options.resolve_command_settings(arguments(task="Hexapod-Walk-WspStop")))
        # Namespaces created by older callers need not define the new fields.
        self.assertIsNone(options.resolve_command_settings(SimpleNamespace(task="Hexapod-Walk-WSP")))
        for key in options.OPTION_KEYS:
            args = arguments(task="Hexapod-Walk-WSP")
            setattr(args, key, "standard" if key == "command_difficulty" else 0.)
            with self.subTest(option=key), self.assertRaisesRegex(ValueError, "require --task"):
                options.resolve_command_settings(args)

    def test_invalid_ranges_rejected_without_changing_arguments(self):
        for key in options.RANGE_KEYS:
            for value in ((1., -1.), (float("nan"), 1.), (0., float("inf")),
                          (float("-inf"), 0.), (1.,), None, "bad"):
                if value is None:
                    continue  # Omitted overrides are valid.
                args = arguments()
                setattr(args, key, value)
                with self.subTest(option=key, value=value), self.assertRaises(ValueError):
                    options.resolve_command_settings(args)
                self.assertIs(getattr(args, key), value)
        for value in ((0., 1.), (-1., 1.)):
            args = arguments()
            args.command_interval = value
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                options.resolve_command_settings(args)

    def test_standing_fraction_boundaries_and_nonfinite(self):
        for fraction in (0., 1.):
            self.assertEqual(options.resolve_command_settings(arguments(
                "--standing-fraction", str(fraction)))["standing_fraction"], fraction)
        for fraction in (-.01, 1.01, float("nan"), float("inf"), float("-inf")):
            args = arguments()
            args.standing_fraction = fraction
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                options.resolve_command_settings(args)

    def test_programmatic_unknown_preset_is_rejected(self):
        args = arguments()
        args.command_difficulty = "unknown"
        with self.assertRaisesRegex(ValueError, "command_difficulty"):
            options.resolve_command_settings(args)

    def test_apply_changes_only_velocity_command_settings(self):
        cfg = config()
        expected = copy.deepcopy(cfg)
        velocity = expected.commands.base_velocity
        velocity.ranges.lin_vel_x = (-.15, .30)
        velocity.ranges.lin_vel_y = (-.15, .15)
        velocity.ranges.ang_vel_z = (-.75, .75)
        velocity.resampling_time_range = (3., 5.)
        options.apply_command_settings(cfg, options.resolve_command_settings(arguments()))
        self.assertEqual(cfg, expected)
        other_instance = config()
        self.assertEqual(other_instance.commands.base_velocity.ranges.lin_vel_x, (-.1, .2))

    def test_apply_none_is_noop_and_one_control_step_is_allowed(self):
        cfg = config()
        original = copy.deepcopy(cfg)
        options.apply_command_settings(cfg, None)
        self.assertEqual(cfg, original)
        result = options.resolve_command_settings(arguments("--command-interval", ".02", ".02"))
        options.apply_command_settings(cfg, result)
        self.assertEqual(cfg.commands.base_velocity.resampling_time_range, (.02, .02))

    def test_apply_invalid_settings_never_partially_mutates_config(self):
        valid = options.resolve_command_settings(arguments())
        for key, value in (("yaw_rate", [1., -1.]), ("command_interval", [.001, 1.]),
                           ("command_interval", [0., 1.]), ("standing_fraction", float("nan")),
                           ("command_difficulty", "unknown")):
            cfg = config()
            original = copy.deepcopy(cfg)
            settings = {**valid, key: value}
            with self.subTest(option=key, value=value), self.assertRaises(ValueError):
                options.apply_command_settings(cfg, settings)
            self.assertEqual(cfg, original)

    def test_heading_control_and_invalid_timestep_are_rejected_before_mutation(self):
        settings = options.resolve_command_settings(arguments())
        cfg = config()
        cfg.commands.base_velocity.heading_command = True
        original = copy.deepcopy(cfg)
        with self.assertRaisesRegex(ValueError, "heading_command=False"):
            options.apply_command_settings(cfg, settings)
        self.assertEqual(cfg, original)
        for dt in (0., -1., float("inf")):
            cfg = config()
            cfg.sim.dt = dt
            original = copy.deepcopy(cfg)
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "Control-step"):
                options.apply_command_settings(cfg, settings)
            self.assertEqual(cfg, original)


if __name__ == "__main__":
    unittest.main()
