"""Actor-only initialization regressions without Isaac Lab or GPU training."""

import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

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
    def __init__(self):
        actor = Actor(.1)
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


if __name__ == "__main__":
    unittest.main()
