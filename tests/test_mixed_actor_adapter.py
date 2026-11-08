"""CPU adapter invariance, freshness and learnability checks without Isaac Lab."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mixed_actor_adapter as adapter
from train_mixed_terrain import validate_initial_episode_ages


class Normalizer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.eps = .01
        self.register_buffer("_mean", torch.zeros(1, width))
        self.register_buffer("_var", torch.ones(1, width))
        self.register_buffer("_std", torch.ones(1, width))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))


class Actor(nn.Module):
    def __init__(self, width, std=.05):
        super().__init__()
        self.obs_normalizer = Normalizer(width)
        self.mlp = nn.Sequential(nn.Linear(width, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 18))
        self.distribution = nn.Module()
        self.distribution.register_parameter("log_std_param", nn.Parameter(torch.full((18,), math.log(std))))

    def forward(self, obs):
        return self.mlp((obs - self.obs_normalizer._mean) / (self.obs_normalizer._std + .01))


def source():
    torch.manual_seed(73)
    actor = Actor(67, .02)
    with torch.no_grad():
        actor.obs_normalizer._mean.uniform_(-.5, .5)
        actor.obs_normalizer._var.fill_(.25)
        actor.obs_normalizer._std.fill_(.5)
        actor.obs_normalizer.count.fill_(5905928448)
    return actor.state_dict()


def runner():
    actor = Actor(95)
    critic = nn.Sequential()
    critic.add_module("obs_normalizer", Normalizer(95))
    critic.add_module("mlp", nn.Linear(95, 1))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    return SimpleNamespace(current_learning_iteration=0,
                           alg=SimpleNamespace(get_policy=lambda: actor, critic=critic, optimizer=optimizer))


class MixedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prefix_is_exact_and_arbitrary_new_features_do_not_change_initial_mean(self):
        original = source()
        before = copy.deepcopy(original)
        expanded = adapter.adapt_flat_actor(original)
        proof = adapter.frozen_input_equivalence(original, expanded, torch.randn(7, 67))
        self.assertEqual(proof["physical_prefixes"], 7)
        self.assertEqual(proof["normalizer_count_retained"], 5905928448)
        self.assertLess(proof["max_abs_action_error"], 5e-5)
        self.assertTrue(torch.equal(expanded["mlp.0.weight"][:, :67], original["mlp.0.weight"]))
        self.assertEqual(torch.count_nonzero(expanded["mlp.0.weight"][:, 67:]).item(), 0)
        for key in original:
            self.assertTrue(torch.equal(original[key], before[key]), key)
        expanded["mlp.0.weight"][0, 67] = .001
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            adapter.frozen_input_equivalence(original, expanded)

    def test_real_loading_keeps_fresh_critic_optimizer_iteration_and_exploration(self):
        state = source()
        fresh = runner()
        critic_before = copy.deepcopy(fresh.alg.critic.state_dict())
        checkpoint = {"actor_state_dict": state, "iter": 1999, "critic_state_dict": {"old": 1},
                      "optimizer_state_dict": {"old": 1}}
        proof = adapter.initialize_mixed_runner(fresh, checkpoint, torch.randn(4, 67))
        self.assertEqual(proof["actor_only_load"]["source_iteration_not_loaded"], 1999)
        self.assertEqual(proof["actor_only_load"]["initial_learning_iteration"], 0)
        self.assertEqual(proof["actor_only_load"]["critic_normalizer_count"], 0)
        self.assertEqual(len(fresh.alg.optimizer.state), 0)
        self.assertTrue(torch.allclose(fresh.alg.get_policy().distribution.log_std_param.exp(), torch.full((18,), .05)))
        for key, value in critic_before.items():
            self.assertTrue(torch.equal(value, fresh.alg.critic.state_dict()[key]), key)

    def test_invalid_source_is_rejected_before_mutating_it(self):
        for fault in ("shape", "dtype", "nonfinite", "missing"):
            with self.subTest(fault=fault):
                state = source()
                if fault == "shape": state["mlp.0.weight"] = torch.zeros(256, 68)
                elif fault == "dtype": state["obs_normalizer.count"] = torch.tensor(5905928448.)
                elif fault == "nonfinite": state["mlp.6.bias"][0] = float("inf")
                else: del state["obs_normalizer._var"]
                before = copy.deepcopy(state)
                with self.assertRaises(ValueError): adapter.adapt_flat_actor(state)
                for key in state: self.assertTrue(torch.equal(state[key], before[key]), key)

    def test_zero_new_columns_are_trainable_in_ordinary_ppo_network(self):
        actor = Actor(95)
        actor.load_state_dict(adapter.adapt_flat_actor(source()))
        obs = torch.randn(12, 95)
        actor(obs).square().mean().backward()
        self.assertGreater(float(actor.mlp[0].weight.grad[:, 67:].abs().sum()), 0.)

    def test_mixed_rollout_rejects_global_random_ages_or_preexpired_flat_episodes(self):
        fresh = torch.zeros(4, dtype=torch.long)
        proof = validate_initial_episode_ages(SimpleNamespace(init_at_random_ep_len=False), fresh)
        self.assertTrue(proof["initial_episode_ages_zero"])
        with self.assertRaisesRegex(ValueError, "disable global random"):
            validate_initial_episode_ages(SimpleNamespace(init_at_random_ep_len=True), fresh)
        with self.assertRaisesRegex(ValueError, "zero episode ages"):
            validate_initial_episode_ages(SimpleNamespace(init_at_random_ep_len=False), torch.tensor([1100, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
