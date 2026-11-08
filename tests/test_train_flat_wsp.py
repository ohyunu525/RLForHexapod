"""Actor-only initialization regressions without Isaac Lab or GPU training."""

import copy
import importlib.util
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_flat_wsp", ROOT / "scripts/train_flat_wsp.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = .01
        self.register_buffer("_mean", torch.zeros(1, 67))
        self.register_buffer("_var", torch.ones(1, 67))
        self.register_buffer("_std", torch.ones(1, 67))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))


class Actor(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.obs_normalizer = Normalizer()
        self.mlp = nn.Sequential(nn.Linear(67, 8), nn.ELU(), nn.Linear(8, 18))
        self.distribution = nn.Module()
        self.distribution.register_parameter("log_std_param", nn.Parameter(torch.full((18,), math.log(std))))


class Runner:
    def __init__(self, std=.1):
        actor = Actor(std)
        critic = nn.Sequential()
        critic.add_module("obs_normalizer", Normalizer())
        critic.add_module("mlp", nn.Linear(67, 1))
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)
        self.alg = SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer, get_policy=lambda: actor)
        self.current_learning_iteration = 0


def checkpoint():
    source = Actor(.3)
    with torch.no_grad():
        source.obs_normalizer._mean.fill_(.2)
        source.obs_normalizer._std.fill_(.5)
        source.obs_normalizer._var.fill_(.25)
        source.obs_normalizer.count.fill_(4281000000)
    return {"actor_state_dict": source.state_dict(), "critic_state_dict": {"sentinel": "must not load"},
            "optimizer_state_dict": {"sentinel": "must not load"}, "iter": 13250, "infos": "old run"}


class ActorWarmstartTests(unittest.TestCase):
    def test_refine_preserves_model_1999_mean_and_moments_with_fresh_exploration(self):
        for std in (.08, .06, .05):
            with self.subTest(std=std):
                runner = Runner(std)
                source = checkpoint()
                source["iter"] = 1999
                source["actor_state_dict"]["obs_normalizer.count"].fill_(4922888448)
                proof = training.actor_only_warmstart(runner, source, expected_std=std)
                self.assertTrue(proof["actor_mean_and_normalizer_exact"])
                self.assertTrue(proof["critic_equals_fresh_initialization"])
                self.assertEqual(proof["actor_normalizer_count"], 4922888448)
                self.assertEqual(proof["source_iteration_not_loaded"], 1999)
                self.assertEqual(proof["optimizer_state_entries"], 0)
                self.assertEqual(proof["initial_learning_iteration"], 0)
                self.assertTrue(torch.allclose(torch.tensor(proof["exploration_std"]), torch.full((18,), std)))

    def test_mean_and_normalizer_loaded_without_stale_ppo_state(self):
        runner = Runner()
        source = checkpoint()
        critic = copy.deepcopy(runner.alg.critic.state_dict())
        proof = training.actor_only_warmstart(runner, source)
        self.assertTrue(proof["actor_mean_and_normalizer_exact"])
        self.assertEqual(proof["source_iteration_not_loaded"], 13250)
        self.assertEqual(proof["initial_learning_iteration"], 0)
        self.assertEqual(proof["optimizer_state_entries"], 0)
        self.assertEqual(proof["critic_normalizer_count"], 0)
        self.assertEqual(proof["actor_normalizer_count"], 4281000000)
        self.assertTrue(torch.allclose(runner.alg.actor.distribution.log_std_param.exp(), torch.full((18,), .1)))
        for name, value in source["actor_state_dict"].items():
            if name != training.STD_KEY:
                self.assertTrue(torch.equal(runner.alg.actor.state_dict()[name], value), name)
        for name, value in critic.items():
            self.assertTrue(torch.equal(value, runner.alg.critic.state_dict()[name]), name)

    def test_bad_tensor_rejected_before_any_actor_mutation(self):
        for fault in ("nonfinite", "missing", "wrong_shape"):
            with self.subTest(fault=fault):
                runner = Runner()
                before = copy.deepcopy(runner.alg.actor.state_dict())
                source = checkpoint()
                if fault == "nonfinite":
                    source["actor_state_dict"]["mlp.2.bias"][0] = float("nan")
                elif fault == "missing":
                    del source["actor_state_dict"]["obs_normalizer.count"]
                else:
                    source["actor_state_dict"]["mlp.2.bias"] = torch.zeros(17)
                with self.assertRaises(ValueError):
                    training.actor_only_warmstart(runner, source)
                for name, value in before.items():
                    self.assertTrue(torch.equal(value, runner.alg.actor.state_dict()[name]), name)

    def test_nonfresh_runner_is_rejected(self):
        runner = Runner()
        runner.current_learning_iteration = 13250
        with self.assertRaisesRegex(ValueError, "iteration zero"):
            training.actor_only_warmstart(runner, checkpoint())
        runner.current_learning_iteration = 0
        parameter = next(runner.alg.actor.parameters())
        runner.alg.optimizer.state[parameter]["step"] = 1
        with self.assertRaisesRegex(ValueError, "empty fresh optimizer"):
            training.actor_only_warmstart(runner, checkpoint())

    def test_normalization_epsilon_and_new_exploration_contract_enforced(self):
        runner = Runner()
        runner.alg.actor.obs_normalizer.eps = 1e-8
        with self.assertRaisesRegex(ValueError, "epsilon"):
            training.actor_only_warmstart(runner, checkpoint())
        runner.alg.actor.obs_normalizer.eps = .01
        with torch.no_grad():
            runner.alg.actor.distribution.log_std_param.fill_(math.log(.3))
        with self.assertRaisesRegex(ValueError, "Fresh exploration std"):
            training.actor_only_warmstart(runner, checkpoint())


class TrainingTaskDefaultsTests(unittest.TestCase):
    def arguments(self, task):
        return SimpleNamespace(task=task, checkpoint=None, checkpoint_sha256=None,
                               num_envs=None, max_iterations=None, run_name=None, initial_std=None)

    def test_original_wsp_defaults_remain_unchanged_and_refine_uses_pinned_model_1999(self):
        wsp = training.apply_task_defaults(self.arguments("Hexapod-Walk-WSP"))
        self.assertEqual((wsp.num_envs, wsp.max_iterations, wsp.initial_std), (1024, 3000, .1))
        self.assertEqual(wsp.checkpoint, training.CHECKPOINT)
        self.assertIsNone(wsp.checkpoint_sha256)
        refine = training.apply_task_defaults(self.arguments("Hexapod-Walk-Refine"))
        self.assertEqual((refine.num_envs, refine.max_iterations, refine.initial_std), (5120, 1000, .08))
        self.assertEqual(refine.checkpoint.name, "model_1999.pt")
        self.assertEqual(refine.checkpoint_sha256, training.REFINE_SHA256)
        self.assertEqual(refine.run_name, "flat_refine")

    def test_explicit_values_preserved_and_unsupported_or_invalid_inputs_rejected(self):
        args = self.arguments("Hexapod-Walk-Refine")
        args.num_envs, args.max_iterations, args.initial_std = 64, 5, .06
        args.checkpoint, args.checkpoint_sha256 = Path("explicit.pt"), "AB" * 32
        result = training.apply_task_defaults(args)
        self.assertEqual((result.num_envs, result.max_iterations, result.initial_std), (64, 5, .06))
        self.assertEqual(result.checkpoint, Path("explicit.pt"))
        self.assertEqual(result.checkpoint_sha256, "ab" * 32)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            training.apply_task_defaults(self.arguments("Hexapod-Walk"))
        for std in (0., -1., float("nan"), float("inf")):
            args = self.arguments("Hexapod-Walk-Refine")
            args.initial_std = std
            with self.assertRaisesRegex(ValueError, "initial_std"):
                training.apply_task_defaults(args)
        args = self.arguments("Hexapod-Walk-Refine")
        args.checkpoint_sha256 = "not-a-checkpoint-hash"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            training.apply_task_defaults(args)

    def test_wsp_stop_requires_explicit_pinned_candidate_and_lineage(self):
        args = self.arguments("Hexapod-Walk-WspStop")
        with self.assertRaisesRegex(ValueError, "explicit checkpoint"):
            training.apply_task_defaults(args)
        args = self.arguments("Hexapod-Walk-WspStop")
        args.checkpoint, args.checkpoint_sha256 = Path("candidate.pt"), "ab" * 32
        args.wsp_provenance = Path("candidate.json")
        args = training.apply_task_defaults(args)
        self.assertEqual((args.num_envs, args.max_iterations, args.initial_std), (5120, 2000, .05))


class WspLineageTests(unittest.TestCase):
    def test_last_two_layers_scope_rejects_early_layer_changes_and_false_prefix_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            parent_path, candidate_path, report_path = (directory / name for name in ("model_999.pt", "candidate.pt", "candidate.json"))
            original = checkpoint()
            original["iter"] = 999
            state = original["actor_state_dict"]
            state["mlp.6.weight"] = state.pop("mlp.2.weight")
            state["mlp.6.bias"] = state.pop("mlp.2.bias")
            for layer in (2, 4):
                state[f"mlp.{layer}.weight"] = torch.eye(8)
                state[f"mlp.{layer}.bias"] = torch.zeros(8)
            torch.save(original, parent_path)
            source_hash = training.sha256(parent_path)
            edited = copy.deepcopy(original)
            edited["actor_state_dict"]["mlp.4.weight"] += .01
            edited["actor_state_dict"]["mlp.6.bias"] += .01
            torch.save(edited, candidate_path)
            candidate_hash = training.sha256(candidate_path)
            report = {"source_checkpoint": str(parent_path), "candidate": str(candidate_path),
                      "source_sha256": source_hash, "candidate_sha256": candidate_hash,
                      "source_sha256_unchanged": True, "normalizer_preserved": True,
                      "upstream_preserved": True, "runtime_inputs_added": False,
                      "normalization_epsilon": .01, "edit_scope": "last_two_layers", "method": "nonlinear",
                      "prefix_preserved": True, "preserved_prefix_layers": [0, 2],
                      "changed_actor_tensors": ["mlp.4.weight", "mlp.6.bias"]}

            def verify(metadata=report, candidate=edited):
                report_path.write_text(json.dumps(metadata), encoding="utf-8")
                return training.verify_wsp_lineage(report_path, candidate_path, candidate_hash, candidate)

            with patch.object(training, "WSP_STOP_PARENT_SHA256", source_hash):
                proof = verify()
                self.assertEqual(proof["edit_scope"], "last_two_layers")
                self.assertEqual(proof["preserved_prefix_layers"], [0, 2])
                self.assertIn("mlp.2.weight", proof["preserved_actor_tensors"])
                self.assertNotIn("mlp.4.weight", proof["preserved_actor_tensors"])
                for key, value in (("prefix_preserved", False), ("preserved_prefix_layers", [0]), ("method", "linear")):
                    with self.subTest(claim=key):
                        false_claim = copy.deepcopy(report)
                        false_claim[key] = value
                        with self.assertRaisesRegex(ValueError, "preserved prefix"):
                            verify(false_claim)
                for changed_key in ("mlp.0.weight", "mlp.2.bias", "obs_normalizer._mean"):
                    with self.subTest(changed_key=changed_key):
                        invalid = copy.deepcopy(edited)
                        invalid["actor_state_dict"][changed_key] += .01
                        with self.assertRaisesRegex(ValueError, "last-two-layers-only"):
                            verify(candidate=invalid)
                default_scope = copy.deepcopy(report)
                del default_scope["edit_scope"]
                with self.assertRaisesRegex(ValueError, "final-layer-only"):
                    verify(default_scope)

    def test_actual_head_only_candidate_is_required_despite_claimed_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            parent_path, candidate_path, report_path = (directory / name for name in ("model_999.pt", "candidate.pt", "candidate.json"))
            original = checkpoint()
            original["iter"] = 999
            # A minimal fixture uses the production final-layer key names.
            original["actor_state_dict"]["mlp.6.weight"] = original["actor_state_dict"].pop("mlp.2.weight")
            original["actor_state_dict"]["mlp.6.bias"] = original["actor_state_dict"].pop("mlp.2.bias")
            torch.save(original, parent_path)
            source_hash = training.sha256(parent_path)
            edited = copy.deepcopy(original)
            edited["actor_state_dict"]["mlp.6.bias"] += .01
            torch.save(edited, candidate_path)
            candidate_hash = training.sha256(candidate_path)
            report = {"source_checkpoint": str(parent_path), "candidate": str(candidate_path),
                      "source_sha256": source_hash, "candidate_sha256": candidate_hash,
                      "source_sha256_unchanged": True, "normalizer_preserved": True,
                      "upstream_preserved": True, "runtime_inputs_added": False,
                      "normalization_epsilon": .01, "changed_actor_tensors": ["mlp.6.bias"]}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(training, "WSP_STOP_PARENT_SHA256", source_hash):
                proof = training.verify_wsp_lineage(report_path, candidate_path, candidate_hash, edited)
                self.assertTrue(proof["normalizer_exact_to_parent"])
                self.assertEqual(proof["actual_changed_actor_tensors"], ["mlp.6.bias"])
                # A dishonest report cannot hide a changed normalizer tensor.
                edited["actor_state_dict"]["obs_normalizer._mean"] += .01
                with self.assertRaisesRegex(ValueError, "final-layer-only"):
                    training.verify_wsp_lineage(report_path, candidate_path, candidate_hash, edited)
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    training.verify_wsp_lineage(report_path, candidate_path, "ff" * 32, edited)


if __name__ == "__main__":
    unittest.main()
