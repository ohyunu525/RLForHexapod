"""Test trace provenance and paired comparison gates without a simulator."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("run_flat_wsp", Path(__file__).resolve().parents[1]
                                            / "scripts/run_flat_wsp.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def report():
    metrics = {key: 0.0 for key in (
        "xy_tracking_rmse_m_s", "yaw_tracking_rmse_rad_s", "vertical_velocity_rms_m_s", "tilt_rms_rad",
        "height_tracking_rmse_m", "failed_episodes", "all_tibia_air_fraction", "fewer_than_three_tibia_contacts_fraction",
    )}
    metrics["mean_forward_speed_body_m_s"] = .15
    metrics["per_leg"] = {f"L{i}": {"joint_excursion_p95_p05_rad": [.2, .2, .2],
                                    "joint_mean_rad": [0., .4, -2.], "action_clip_fraction": [0., 0., .7],
                                    "tibia_contact_fraction": .5} for i in range(1, 7)}
    return {"status": "completed", "task": "Hexapod-Walk", "seed": 73, "num_envs": 4,
            "steps_requested": 750, "steps_completed": 750, "command": {"vx": .15},
            "step_dt_s": .02, "joint_names": ["example"], "asset_usd_sha256": "asset",
            "asset_manifest_sha256": "manifest", "environment_source_sha256": "source",
            "initial_state_sha256": "initial", "normalization_epsilon": .01, "metrics": metrics}


class ComparisonTests(unittest.TestCase):
    def test_incomparable_initial_state_is_rejected(self):
        original, candidate = report(), report()
        candidate["initial_state_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "initial_state"):
            runner.compare_pair(original, candidate)

    def test_failures_and_speed_regressions_are_not_hidden_by_joint_motion(self):
        original = report()
        candidate = copy.deepcopy(original)
        candidate["metrics"]["failed_episodes"] = 1
        candidate["metrics"]["mean_forward_speed_body_m_s"] = .1
        candidate["metrics"]["per_leg"]["L1"]["joint_excursion_p95_p05_rad"][2] = .6
        result = runner.compare_pair(original, candidate)
        self.assertIn("more_failed_episodes", result["regression_flags"])
        self.assertIn("forward_speed_lost_more_than_five_percent", result["regression_flags"])

    def test_truncated_rollout_is_rejected(self):
        original, candidate = report(), report()
        original["steps_completed"] = candidate["steps_completed"] = 10
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runner.compare_pair(original, candidate)

    def test_merge_requires_exact_checkpoint_and_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "trace.npz"
            np.savez(source, obs=np.zeros((3, 2, 67), dtype=np.float32), source_sha256="original",
                     joint_names=np.array(["L1_coxa"]), normalization_epsilon=.01)
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runner.merge_traces([source], root / "bad.npz", "other")
            runner.merge_traces([source], root / "merged.npz", "original")
            with np.load(root / "merged.npz", allow_pickle=False) as merged:
                self.assertEqual(merged["obs"].shape, (6, 67))
                self.assertEqual(merged["source_sha256"].item(), "original")


if __name__ == "__main__":
    unittest.main()
