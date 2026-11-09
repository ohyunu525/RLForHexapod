"""Exercise training orchestration and real CPU warmstart without Isaac or PPO.

Simulator/runner factories are doubles; these checks do not establish physical
runtime compatibility or improved walking performance.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_flat_command_options import config as command_config
from test_train_flat_wsp import Runner, checkpoint, training


class Config(SimpleNamespace):
    def to_dict(self):
        return {key: value.to_dict() if isinstance(value, Config) else value
                for key, value in vars(self).items()}


class RuntimeDouble:
    def __init__(self):
        self.env_cfg = command_config()
        self.env_cfg.sim.device = "cpu"
        self.agent_cfg = Config(
            experiment_name="hexapod_walk_commands", class_name="OnPolicyRunner",
            actor=Config(hidden_dims=[256, 256, 128], activation="elu",
                         distribution_cfg=Config(init_std=.05, std_range=(.01, 1.))),
            algorithm=Config(learning_rate=.0001), clip_actions=1.,
            num_steps_per_env=24, save_interval=50, init_at_random_ep_len=True)
        self.runner = None
        self.cfg_at_make = None
        self.make = Mock(side_effect=self._make)
        self.close = Mock()
        self.launch = Mock(side_effect=lambda cfg, args: contextlib.nullcontext())
        names = ("gymnasium", "isaaclab", "isaaclab.app", "isaaclab_rl",
                 "isaaclab_rl.entrypoints", "isaaclab_rl.entrypoints.common", "isaaclab_rl.rsl_rl",
                 "isaaclab_tasks", "isaaclab_tasks.utils", "RLforHexapod", "RLforHexapod.tasks")
        self.modules = {name: ModuleType(name) for name in names}
        for name, module in self.modules.items():
            module.__path__ = []
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(self.modules[parent], child, module)
        self.modules["gymnasium"].make = self.make
        self.modules["isaaclab.app"].launch_simulation = self.launch
        common = self.modules["isaaclab_rl.entrypoints.common"]
        common.apply_env_overrides = lambda args, cfg: None
        common.close_env = self.close
        common.dump_train_configs = Mock()
        common.write_run_manifest = Mock()
        rsl = self.modules["isaaclab_rl.rsl_rl"]
        rsl.RslRlVecEnvWrapper = lambda env, clip_actions: env
        rsl.create_rsl_rl_runner = self._runner
        rsl.handle_deprecated_rsl_rl_cfg = lambda cfg, version: cfg
        self.modules["isaaclab_tasks.utils"].resolve_task_config = (
            lambda task, entry, overrides: (self.env_cfg, self.agent_cfg))

    def _make(self, task, cfg):
        self.cfg_at_make = copy.deepcopy(cfg)
        return SimpleNamespace(task=task)

    def _runner(self, wrapped, cfg, log_dir):
        self.runner = self.pending_runner
        self.fresh_actor = copy.deepcopy(self.runner.alg.actor.state_dict())

        def learn(num_learning_iterations, init_at_random_ep_len):
            self.runner.current_learning_iteration = num_learning_iterations - 1
            torch.save({"iter": self.runner.current_learning_iteration}, Path(log_dir) / "model_1.pt")

        self.runner.learn = Mock(side_effect=learn)
        return self.runner


def arguments(**overrides):
    values = dict(task=training.COMMAND_TASK, checkpoint=None, checkpoint_sha256=None,
                  num_envs=None, max_iterations=None, run_name=None, initial_std=None,
                  seed=42, device=None, logger="tensorboard", wsp_provenance=None, verify_only=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class FlatCommandTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "pre_stair.pt"
        self.checkpoint = checkpoint()
        self.checkpoint["iter"] = 1999
        torch.save(self.checkpoint, self.source)
        self.source_hash = training.sha256(self.source)
        self.runtime = RuntimeDouble()

    def run_training(self, **overrides):
        args = arguments(checkpoint=self.source, num_envs=4, max_iterations=2, **overrides)
        training.apply_task_defaults(args)
        # Adam lazily imports Torch libraries. Create it before patch.dict so
        # restoring sys.modules cannot unload their Python registration state.
        self.runtime.pending_runner = Runner(args.initial_std)
        with patch.dict(sys.modules, self.runtime.modules), patch.object(training, "ROOT", self.root), \
                patch.object(training.importlib.metadata, "version", return_value="5.5.1"), \
                contextlib.redirect_stdout(io.StringIO()):
            return training.run(args)

    def test_pinned_pre_stair_defaults_and_custom_source_do_not_mix(self):
        default = training.apply_task_defaults(arguments())
        self.assertEqual(default.checkpoint, training.PRE_STAIR_CHECKPOINT)
        self.assertEqual(default.checkpoint.name, "model_1999.pt")
        self.assertIn("hexapod_walk_wsp_stop", default.checkpoint.parts)
        self.assertEqual(default.checkpoint_sha256,
                         "725c9696e82ef25e4b13e20c648562d17be6c79a6c1bc6ebd81936fbf2590527")
        self.assertEqual((default.num_envs, default.max_iterations, default.initial_std), (5120, 2000, .05))
        self.assertEqual(default.command_settings["command_difficulty"], "hard")
        explicit = training.apply_task_defaults(arguments(checkpoint=self.source))
        self.assertEqual(explicit.checkpoint, self.source)
        self.assertIsNone(explicit.checkpoint_sha256)
        explicit_hash = training.apply_task_defaults(arguments(checkpoint=self.source, checkpoint_sha256="AB" * 32))
        self.assertEqual(explicit_hash.checkpoint_sha256, "ab" * 32)

    def test_wsp_stop_still_requires_provenance(self):
        for extra in ({}, {"checkpoint": self.source, "checkpoint_sha256": self.source_hash}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "wsp_provenance"):
                training.apply_task_defaults(arguments(task="Hexapod-Walk-WspStop", **extra))
        old = training.apply_task_defaults(arguments(task="Hexapod-Walk-WspStop", checkpoint=self.source,
                                                     checkpoint_sha256=self.source_hash, wsp_provenance=Path("proof.json")))
        self.assertIsNone(old.command_settings)

    def test_verify_only_loads_exact_actor_and_records_settings_without_learning(self):
        log_dir = self.run_training(verify_only=True, checkpoint_sha256=self.source_hash)
        self.runtime.runner.learn.assert_not_called()
        self.runtime.close.assert_called_once()
        status = json.loads((log_dir / "training_status.json").read_text())
        proof = json.loads((log_dir / "initialization.json").read_text())
        self.assertEqual(status["status"], "verified_initialization_without_ppo")
        self.assertEqual(status["ppo_updates_performed"], 0)
        self.assertTrue(status["source_sha256_unchanged"])
        self.assertEqual(proof["source_sha256"], self.source_hash)
        self.assertEqual(proof["expected_source_sha256"], self.source_hash)
        self.assertEqual(proof["source_checkpoint"], str(self.source.resolve()))
        self.assertEqual(proof["command_settings"], status["command_settings"])
        self.assertTrue(proof["actor_mean_and_normalizer_exact"])
        self.assertTrue(proof["critic_equals_fresh_initialization"])
        self.assertEqual((proof["initial_learning_iteration"], proof["optimizer_state_entries"]), (0, 0))
        self.assertEqual(proof["source_iteration_not_loaded"], 1999)
        for key, value in self.checkpoint["actor_state_dict"].items():
            if key != training.STD_KEY:
                self.assertTrue(torch.equal(self.runtime.runner.alg.actor.state_dict()[key], value), key)
        self.assertFalse(list(log_dir.glob("model_*.pt")))

    def test_training_applies_commands_before_environment_creation_and_uses_separate_output(self):
        log_dir = self.run_training(command_difficulty="expert", lin_vel_x=[-.12, .32],
                                    command_interval=[2., 4.], standing_fraction=.2)
        velocity = self.runtime.cfg_at_make.commands.base_velocity
        self.assertEqual(velocity.ranges.lin_vel_x, (-.12, .32))
        self.assertEqual(velocity.ranges.lin_vel_y, (-.2, .2))
        self.assertEqual(velocity.ranges.ang_vel_z, (-1., 1.))
        self.assertEqual(velocity.resampling_time_range, (2., 4.))
        self.assertEqual(velocity.rel_standing_envs, .2)
        self.assertEqual(self.runtime.cfg_at_make.commands.base_height, command_config().commands.base_height)
        self.runtime.runner.learn.assert_called_once_with(num_learning_iterations=2, init_at_random_ep_len=True)
        self.assertEqual(log_dir.parent, self.root / "logs/rsl_rl/hexapod_walk_commands")
        status = json.loads((log_dir / "training_status.json").read_text())
        proof = json.loads((log_dir / "initialization.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["saved_checkpoints"], [str(log_dir / "model_1.pt")])
        self.assertEqual(status["command_settings"], proof["command_settings"])
        self.assertEqual(proof["command_settings"]["command_interval"], [2., 4.])
        self.assertEqual(training.sha256(self.source), self.source_hash)

    def test_invalid_commands_and_substep_interval_do_not_create_environment(self):
        for overrides in ({"lin_vel_x": [.3, -.1]}, {"command_interval": [.001, .002]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.run_training(**overrides)
            self.runtime.make.assert_not_called()
            self.runtime.launch.assert_not_called()
            self.assertIsNone(self.runtime.runner)

    def test_source_hash_mismatch_is_rejected_before_simulation(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_training(checkpoint_sha256="ab" * 32)
        self.runtime.make.assert_not_called()
        self.runtime.launch.assert_not_called()

    def test_mixed_actor_shape_cannot_start_flat_training(self):
        self.checkpoint["actor_state_dict"]["mlp.0.weight"] = torch.zeros(8, 95)
        torch.save(self.checkpoint, self.source)
        with self.assertRaisesRegex(ValueError, "Incompatible actor tensor"):
            self.run_training()
        self.runtime.runner.learn.assert_not_called()
        self.runtime.close.assert_called_once()
        for key, value in self.runtime.fresh_actor.items():
            self.assertTrue(torch.equal(self.runtime.runner.alg.actor.state_dict()[key], value), key)
        status_path, = self.root.glob("logs/rsl_rl/hexapod_walk_commands/*/training_status.json")
        status = json.loads(status_path.read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["saved_checkpoints"], [])
        self.assertFalse((status_path.parent / "initialization.json").exists())


if __name__ == "__main__":
    unittest.main()
