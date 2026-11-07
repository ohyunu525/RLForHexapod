"""No simulator: first-episode accounting, recurrent reset, failure denominator."""
import importlib.util
from pathlib import Path
import pickle
import types
import unittest
from unittest.mock import patch
import tempfile
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1] / "remote_eval_2850"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("eval_worker")
launcher = load("run_eval")


class Policy:
    def __init__(self):
        self.resets = []

    def __call__(self, obs):
        return torch.zeros(3, 18)

    def reset(self, done):
        self.resets.append(done.clone())


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.stair_tracker = types.SimpleNamespace(move=torch.zeros(3, dtype=torch.long), ended_move=torch.zeros(3, dtype=torch.long), final_advance=15)
        data = types.SimpleNamespace(root_state_w=torch.zeros(3, 13), joint_pos=torch.zeros(3, 18), joint_vel=torch.zeros(3, 18))
        self.scene = {"robot": types.SimpleNamespace(data=data)}
        self.termination_manager = self
        self.active_terms = ["bad_ori", "time_out"]
        self.step_index = 0
        self.flags = {}

    def get_term(self, name):
        return self.flags[name]

    def step(self, action):
        # Env0 falls on step2, then reaches top in a subsequent episode: must stay failure.
        # Env1 first reaches top on step2, ends on step3. Env2 reaches top only when ending step4.
        self.step_index += 1
        rows = {
            1: ([1, 1, 1], [0, 0, 0], [False, False, False]),
            2: ([0, 15, 2], [3, 0, 0], [True, False, False]),
            3: ([16, 0, 3], [3, 17, 0], [False, True, False]),
            4: ([16, 1, 0], [3, 17, 15], [False, False, True]),
        }
        moves, ended, dones = rows[self.step_index]
        self.stair_tracker.move = torch.tensor(moves)
        self.stair_tracker.ended_move = torch.tensor(ended)
        self.flags = {"bad_ori": torch.tensor([self.step_index == 2, False, False]),
                      "time_out": torch.tensor([False, self.step_index == 3, self.step_index == 4])}
        return None, None, torch.tensor(dones), {}


class EvaluationTests(unittest.TestCase):
    def test_cuda_tagged_original_loads_on_cpu_for_evaluation(self):
        checkpoint = ROOT / "model_2850.pt"

        class LoadingRunner:
            def __init__(self):
                self.calls = []

            def load(self, path, *, map_location):
                self.calls.append((path, map_location))
                return torch.load(path, map_location=map_location, weights_only=True)

        runner = LoadingRunner()
        loaded = worker.load_for_evaluation(runner, checkpoint, torch.device("cpu"))
        self.assertEqual(runner.calls, [(str(checkpoint), "cpu")])
        self.assertEqual(loaded["iter"], 2850)
        for group in ("student_state_dict", "teacher_state_dict"):
            self.assertTrue(all(value.device.type == "cpu" for value in loaded[group].values()))
        if not torch.cuda.is_available():
            # PyTorch 2.2 wraps this CUDA restore error in UnpicklingError under weights_only.
            with self.assertRaisesRegex((RuntimeError, pickle.UnpicklingError), "Attempting to deserialize object on a CUDA device"):
                torch.load(checkpoint, weights_only=True)

    def test_first_episode_and_reset(self):
        policy = Policy()
        data = worker.collect(FakeEnv(), policy, None, 4)
        self.assertEqual(data["top_step"], [-1, 2, 4])
        self.assertEqual(data["length_steps"], [2, 3, 4])
        self.assertEqual(data["final_phase"], [3, 17, 15])
        self.assertEqual(len(policy.resets), 4)
        self.assertTrue(torch.equal(policy.resets[1], torch.tensor([True, False, False])))
        report = worker.summarize(data, 0.02, 0.08, 17)
        self.assertEqual(report["reached_top_count"], 2)
        self.assertEqual(report["settled_success_count"], 1)
        self.assertEqual(report["fall_count"], 1)
        self.assertEqual(report["capped_top_time_per_episode_s"], [0.08, 0.04, 0.08])
        self.assertAlmostEqual(report["all_episode_capped_top_time_mean_s"], 0.2 / 3)

    def test_incomplete_not_reported_as_complete(self):
        with self.assertRaises(RuntimeError):
            worker.collect(FakeEnv(), Policy(), None, 3)

    def test_phase1_start_rejected(self):
        env = FakeEnv()
        env.stair_tracker.move[0] = 1
        with self.assertRaises(ValueError):
            worker.collect(env, Policy(), None, 4)

    def test_no_top_denominator(self):
        data = {"top_step": [-1, -1], "final_phase": [3, 10], "termination_names": ["bad_ori", "time_out"],
                "termination_flags": [[True, False], [False, True]]}
        result = worker.summarize(data, 0.02, 45.0, 17)
        self.assertIsNone(result["successful_top_time_mean_s"])
        self.assertEqual(result["all_episode_capped_top_time_mean_s"], 45.0)
        self.assertEqual(result["episodes"], 2)

    def test_aggregate_rejects_initial_state_mismatch(self):
        a = {"seed": 42, "role": "original", "episodes": {"initial_state": [0]}}
        b = {"seed": 42, "role": "wsp", "episodes": {"initial_state": [1]}}
        with self.assertRaisesRegex(ValueError, "Initial physical"):
            launcher.aggregate([a, b])

    def test_aggregate_requires_pair(self):
        with self.assertRaisesRegex(ValueError, "Incomplete seed"):
            launcher.aggregate([{"seed": 42, "role": "original"}])


    def test_default_cuda_16_and_results_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = base / "bundle"
            bundle.mkdir()
            results = base / "evaluation_results" / "wsp2850"
            argv = ["run_eval.py", "--python", sys.executable, "--output-root", str(results)]
            with patch.object(launcher, "__file__", str(bundle / "run_eval.py")), patch.object(launcher, "verify", return_value="verified"), patch.object(sys, "argv", argv), patch.object(launcher.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)) as run:
                with self.assertRaisesRegex(RuntimeError, "Evaluation failed"):
                    launcher.main()
            state = json.loads(next(results.glob("*/run.json")).read_text())
            self.assertEqual(state["settings"]["num_envs"], 16)
            self.assertEqual(state["settings"]["device"], "cuda:0")
            self.assertEqual(state["settings"]["seeds"], [42, 43, 44])
            self.assertEqual(state["status"], "failed")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(command[command.index("--num_envs") + 1], "16")


if __name__ == "__main__":
    unittest.main()
