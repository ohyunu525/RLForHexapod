"""CPU-only checks for the experimental final-head edit, without claiming physics."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "flat_wsp.py"
SPEC = importlib.util.spec_from_file_location("flat_wsp_under_test", MODULE)
wsp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wsp)


def source_state():
    generator = torch.Generator().manual_seed(42)
    state = {
        "obs_normalizer._mean": torch.zeros(1, 67),
        "obs_normalizer._std": torch.ones(1, 67),
        "obs_normalizer._var": torch.ones(1, 67),
        "obs_normalizer.count": torch.tensor(123.),
        "distribution.log_std_param": torch.zeros(18),
    }
    state["obs_normalizer._mean"][0, 44] = -1.
    state["obs_normalizer._mean"][0, 48] = .2
    for index, output_size, input_size in [(0, 256, 67), (2, 256, 256), (4, 128, 256), (6, 18, 128)]:
        state[f"mlp.{index}.weight"] = torch.randn(output_size, input_size, generator=generator) * .03
        state[f"mlp.{index}.bias"] = torch.zeros(output_size)
    state[wsp.HEAD_BIAS][wsp.TIBIA_ROWS] = -1.3
    return state


class FlatWspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actor_matches_explicit_policy_and_verified_epsilon(self):
        state = source_state()
        state["obs_normalizer._std"][:, :3] = 0
        actor = wsp.FlatActor(state)
        obs = torch.randn(7, 67, generator=torch.Generator().manual_seed(8)) * .1
        expected = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + .01)
        for index in (0, 2, 4):
            expected = torch.nn.functional.elu(torch.nn.functional.linear(
                expected, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]))
        torch.testing.assert_close(actor.features(obs), expected, atol=0, rtol=0)
        expected = torch.nn.functional.linear(expected, state[wsp.HEAD_WEIGHT], state[wsp.HEAD_BIAS])
        torch.testing.assert_close(actor(obs), expected, atol=0, rtol=0)
        self.assertFalse(actor.training)

    def test_tail_is_continuous_monotonic_and_releases_negative_clipping(self):
        values = torch.tensor([-100., -2., -1., -.800001, -.8, -.799999, 0., 2.], dtype=torch.float64)
        result = wsp.negative_tail_target(values, .2)
        self.assertTrue(bool(torch.all(result[1:] >= result[:-1])))
        self.assertTrue(bool(torch.all(result[:3] > -1)))
        self.assertAlmostEqual(result[3].item(), result[4].item(), places=5)
        torch.testing.assert_close(result[4:], values[4:], atol=0, rtol=0)

    def test_candidate_preserves_every_unselected_tensor_and_row(self):
        state = source_state()
        original = copy.deepcopy(state)
        obs, evidence = wsp.synthetic_probes(state, samples_per_case=16)
        candidate, report = wsp.fit_candidate(state, obs)
        self.assertTrue(wsp.same(state, original))
        self.assertTrue(wsp.verify_changes(state, candidate)["only_tibia_final_layer_rows_changed"])
        changed = [key for key in state if not wsp.same(state[key], candidate[key])]
        self.assertEqual(changed, [wsp.HEAD_WEIGHT, wsp.HEAD_BIAS])
        self.assertLess(report["target_raw_error_after_rms"], report["target_raw_error_before_rms"])
        self.assertFalse(evidence["observed_trajectories"])
        self.assertTrue(torch.isfinite(wsp.FlatActor(candidate)(obs)).all())

    def test_all_scope_explicitly_releases_coxa_femur_both_signs(self):
        state = source_state()
        state[wsp.HEAD_BIAS][0::3] = -1.3
        state[wsp.HEAD_BIAS][1::3] = 1.3
        state[wsp.HEAD_BIAS][17] = 1.3  # Positive tibia saturation is deliberately outside the target scope.
        obs, _ = wsp.synthetic_probes(state, samples_per_case=20)
        default, default_report = wsp.fit_candidate(state, obs)
        explicit_default, _ = wsp.fit_candidate(state, obs, joint_scope="tibia")
        self.assertTrue(wsp.same(default, explicit_default))
        self.assertEqual(default_report["allowed_action_rows"], wsp.TIBIA_ROWS)
        candidate, report = wsp.fit_candidate(state, obs, joint_scope="all")
        self.assertEqual(report["allowed_action_rows"], list(range(18)))
        self.assertEqual(report["joint_scope"], "all")
        self.assertGreater(report["originally_positive_clipped_targets"], 0)
        self.assertGreater(report["originally_negative_clipped_targets"], 0)
        self.assertIn(0, report["changed_action_rows"])
        self.assertIn(1, report["changed_action_rows"])
        self.assertNotIn(17, report["changed_action_rows"])
        for key in (wsp.HEAD_WEIGHT, wsp.HEAD_BIAS):
            self.assertTrue(torch.equal(state[key][17], candidate[key][17]))
        self.assertFalse(report["preservation"]["only_tibia_final_layer_rows_changed"])
        self.assertTrue(wsp.verify_changes(state, candidate, range(18))["only_selected_final_layer_rows_changed"])
        with self.assertRaisesRegex(AssertionError, "unselected"):
            wsp.verify_changes(state, candidate)  # Default verification cannot silently widen.
        for key in state:
            if key not in (wsp.HEAD_WEIGHT, wsp.HEAD_BIAS):
                self.assertTrue(wsp.same(state[key], candidate[key]))
        self.assertLess(report["target_raw_error_after_rms"], report["target_raw_error_before_rms"])
        before = wsp.FlatActor(state)(obs).clamp(-1, 1)
        after = wsp.FlatActor(candidate)(obs).clamp(-1, 1)
        self.assertLessEqual((after - before).abs().max().item(), .2 + 1e-6)
        self.assertLessEqual(report["protected_applied_delta"]["max_abs"], .1 + 1e-6)

    def test_synthetic_probes_are_deterministic_and_include_protections(self):
        state = source_state()
        obs, _ = wsp.synthetic_probes(state, samples_per_case=8)
        repeat, _ = wsp.synthetic_probes(state, samples_per_case=8)
        self.assertTrue(torch.equal(obs, repeat))
        self.assertTrue(wsp.forward_mask(obs).any())
        self.assertTrue((obs[:, 45:48] == 0).all(dim=1).any())
        self.assertTrue((obs[:, 47].abs() > .1).any())
        torch.testing.assert_close(obs[:-2, 42:45].norm(dim=1), torch.ones(len(obs) - 2))

    def test_trace_checks_source_shape_finiteness_and_joint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            metadata = {"source_sha256": "verified", "joint_names": wsp.JOINT_NAMES,
                        "normalization_epsilon": .01}
            good = np.zeros((3, 2, 67), dtype=np.float32)
            np.savez(path, observations=good, metadata=json.dumps(metadata))
            obs, evidence = wsp.load_trace(path, "verified")
            self.assertEqual(tuple(obs.shape), (6, 67))
            self.assertEqual(evidence["original_shape"], [3, 2, 67])
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                wsp.load_trace(path, "wrong")
            for value in (np.zeros((3, 66), np.float32), np.full((3, 2, 67), np.nan, np.float32)):
                np.savez(path, observations=value, metadata=json.dumps(metadata))
                with self.assertRaises(ValueError):
                    wsp.load_trace(path, "verified")
            metadata["joint_names"] = list(reversed(wsp.JOINT_NAMES))
            np.savez(path, observations=good, metadata=json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "joint order"):
                wsp.load_trace(path, "verified")

    def test_merged_trace_array_metadata_and_epsilon_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            fields = {"obs": np.zeros((5, 67), np.float32), "source_sha256": np.array("verified"),
                      "joint_names": np.array(wsp.JOINT_NAMES), "normalization_epsilon": np.array(.01)}
            np.savez(path, **fields)
            obs, evidence = wsp.load_trace(path, "verified")
            self.assertEqual(tuple(obs.shape), (5, 67))
            self.assertEqual(evidence["temporal_structure"], "unordered calibration rows")
            fields["normalization_epsilon"] = np.array(1e-8)
            np.savez(path, **fields)
            with self.assertRaisesRegex(ValueError, "normalization_epsilon"):
                wsp.load_trace(path, "verified")
            fields.pop("normalization_epsilon")
            np.savez(path, **fields)
            with self.assertRaisesRegex(ValueError, "normalization_epsilon"):
                wsp.load_trace(path, "verified")

    def test_backoff_enforces_measured_applied_action_caps(self):
        state = source_state()
        state[wsp.HEAD_WEIGHT] *= 20
        state[wsp.HEAD_BIAS][wsp.TIBIA_ROWS] = -2.
        obs, _ = wsp.synthetic_probes(state, samples_per_case=20)
        candidate, report = wsp.fit_candidate(state, obs, ridge=1e-6)
        before = wsp.FlatActor(state)(obs).clamp(-1, 1)
        after = wsp.FlatActor(candidate)(obs).clamp(-1, 1)
        self.assertLessEqual((after - before).abs().max().item(), .2 + 1e-6)
        self.assertLessEqual(report["protected_applied_delta"]["max_abs"], .1 + 1e-6)
        self.assertTrue(all(0 <= value <= 1 for value in report["calibration_backoff_factors"].values()))

    def test_explicit_caps_preserve_defaults_and_bound_broader_candidate(self):
        state = source_state()
        state[wsp.HEAD_WEIGHT] *= 20
        state[wsp.HEAD_BIAS][0::3] = -2.
        state[wsp.HEAD_BIAS][1::3] = 2.
        state[wsp.HEAD_BIAS][2::3] = -2.
        obs, _ = wsp.synthetic_probes(state, samples_per_case=20)
        implicit, implicit_report = wsp.fit_candidate(state, obs, ridge=1e-6, joint_scope="all")
        explicit, explicit_report = wsp.fit_candidate(
            state, obs, ridge=1e-6, joint_scope="all", protected_cap=.1, max_applied_change=.2)
        self.assertTrue(wsp.same(implicit, explicit))
        self.assertTrue(implicit_report["uses_default_caps"])
        self.assertTrue(explicit_report["uses_default_caps"])
        broader, report = wsp.fit_candidate(state, obs, ridge=1e-6, joint_scope="all", strength=.4,
                                           protected_cap=.5, max_applied_change=.8)
        self.assertFalse(wsp.same(implicit, broader))
        self.assertEqual(report["protected_cap"], .5)
        self.assertEqual(report["max_applied_change"], .8)
        self.assertFalse(report["uses_default_caps"])
        self.assertLessEqual(report["protected_applied_delta"]["max_abs"], .5 + 1e-6)
        before = wsp.FlatActor(state)(obs).clamp(-1, 1)
        after = wsp.FlatActor(broader)(obs).clamp(-1, 1)
        self.assertLessEqual((after - before).abs().max().item(), .8 + 1e-6)
        self.assertEqual(report["maximum_observed_applied_change"], (after - before).abs().max().item())
        self.assertNotIn("protected_applied_delta_at_most_010", report["offline_criteria"])
        self.assertNotIn("all_applied_delta_at_most_strength", report["offline_criteria"])
        _, zero_report = wsp.fit_candidate(state, obs, protected_cap=0)
        self.assertLessEqual(zero_report["protected_applied_delta"]["max_abs"], 1e-6)

    def test_bad_actor_observations_and_hyperparameters_are_rejected(self):
        state = source_state()
        obs, _ = wsp.synthetic_probes(state, samples_per_case=4)
        for strength in (0, -.1, .5, float("nan")):
            with self.assertRaises(ValueError):
                wsp.fit_candidate(state, obs, strength=strength)
        for ridge in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                wsp.fit_candidate(state, obs, ridge=ridge)
        with self.assertRaisesRegex(ValueError, "joint_scope"):
            wsp.fit_candidate(state, obs, joint_scope="automatic")
        for cap in (-.01, 1.01, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "protected_cap"):
                wsp.fit_candidate(state, obs, protected_cap=cap)
        for cap in (0, -.01, 2.01, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "max_applied_change"):
                wsp.fit_candidate(state, obs, max_applied_change=cap)
        for malformed in (torch.zeros(3, 66), torch.full((3, 67), float("nan")), torch.zeros(1, 67)):
            with self.assertRaises(ValueError):
                wsp.fit_candidate(state, malformed)
        wrong = copy.deepcopy(state)
        wrong["obs_normalizer._std"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            wsp.FlatActor(wrong)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            wsp.fit_candidate(state, torch.zeros(3, 67))

    def test_roundtrip_preserves_checkpoint_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.pt", root / "candidate.pt"
            content = {"actor_state_dict": source_state(), "critic_state_dict": {"unchanged": torch.tensor([3.])},
                       "optimizer_state_dict": {"step": 9}, "iter": 13250, "infos": None}
            torch.save(content, source)
            original_hash = wsp.sha256(source)
            with patch.object(wsp, "SOURCE_SHA256", original_hash):
                report = wsp.main(["--checkpoint", str(source), "--output", str(output), "--synthetic-probes"])
                edited = torch.load(output, weights_only=True)
                for key in content:
                    if key != "actor_state_dict":
                        self.assertTrue(wsp.same(content[key], edited[key]))
                self.assertEqual(wsp.sha256(source), original_hash)
                self.assertFalse(report["physics_validated"])
                self.assertFalse(report["policy_adopted"])
                self.assertIsNone(report["performance_claim"])
                self.assertEqual(report["joint_scope"], "tibia")
                with self.assertRaises(FileExistsError):
                    wsp.main(["--checkpoint", str(source), "--output", str(output), "--synthetic-probes"])
                expanded = root / "expanded.pt"
                all_report = wsp.main(["--checkpoint", str(source), "--output", str(expanded),
                                       "--synthetic-probes", "--joint-scope", "all", "--strength", ".4",
                                       "--protected-cap", ".5", "--max-applied-change", ".8"])
                self.assertEqual(all_report["joint_scope"], "all")
                self.assertEqual(all_report["allowed_action_rows"], list(range(18)))
                self.assertEqual(all_report["fit"]["protected_cap"], .5)
                self.assertEqual(all_report["fit"]["max_applied_change"], .8)
                changed = torch.load(expanded, weights_only=True)["actor_state_dict"]
                self.assertEqual(all_report["changed_action_rows"], wsp.verify_changes(
                    content["actor_state_dict"], changed, range(18))["changed_action_rows"])


if __name__ == "__main__":
    unittest.main()
