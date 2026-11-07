"""Isolated evaluation diagnostics: terminal snapshots and first-episode filtering."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SPEC = importlib.util.spec_from_file_location(
    "eval_stairs", Path(__file__).resolve().parents[1] / "scripts" / "eval_stairs.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class FakeEnv:
    active_terms = ["bad_ori", "time_out"]

    def __init__(self):
        self.step_index = 0
        self.tracker = SimpleNamespace(
            move=torch.tensor([1, 2]), ended_move=torch.zeros(2, dtype=torch.long), final_advance=15,
        )
        self.flags = {}

    def get_term(self, name):
        return self.flags[name]

    def step(self, action):
        self.step_index += 1
        first = self.step_index == 1
        # Snapshot is written before transition/reset. tracker.move is then overwritten by reset.
        self.tracker.diagnostics = {
            "move": torch.tensor([2, 3] if first else [8, 4]),
            "feet_up": torch.tensor([False, True]),
            "body_counts": torch.tensor([False, True]),
            "move_done": torch.tensor([False, False]),
            "all_kept": torch.tensor([True, False]),
            "move_ready_time": torch.zeros(2),
            "just_done": torch.zeros(2, dtype=torch.bool),
            "body_x": torch.tensor([0.4, 0.6]),
            "body_vx": torch.tensor([0.2, 0.1]),
            "feet_contact": torch.ones(2, 6, dtype=torch.bool),
            "foot_force_z": torch.tensor([[2., 3., 4., 5., 6., 7.], [1., 2., 3., 4., 5., 6.]]),
            "contact_forces_w": torch.tensor([
                [[2., 0., 0.], [0., 0., 2.], [0., 0., 10.], [0., 0., 20.]],
                [[0., 2., 0.], [0., 0., 0.], [0., 0., 10.], [0., 0., 0.]],
            ]),
            # A smaller horizontal event must survive selection of the strongest (vertical) sample.
            "horizontal_contact_force": torch.tensor([[2., 0., 0., 5.], [2., 0., 0., 0.]]),
        }
        self.tracker.move = torch.tensor([0, 3] if first else [8, 0])
        self.tracker.ended_move = torch.tensor([2, 0] if first else [2, 4])
        dones = torch.tensor([first, not first])
        self.flags = {"bad_ori": dones, "time_out": torch.zeros(2, dtype=torch.bool)}
        return None, None, dones, {}


class Policy:
    def __call__(self, obs):
        return torch.zeros(2, 18)

    def reset(self, dones):
        pass


class StairDiagnosticsTests(unittest.TestCase):
    body_names = ["body", "coxa", "tibia", "tibia_2"]

    def collect(self, max_envs=4):
        stream = io.StringIO()
        recorder = evaluation.StairDiagnostics(stream, self.body_names, 0.02, max_envs=max_envs)
        env = FakeEnv()
        data = evaluation.collect_first_episodes(env, Policy(), None, env.tracker, env, 2, recorder)
        return data, [json.loads(line) for line in stream.getvalue().splitlines()]

    def test_terminal_snapshot_and_first_episode_only(self):
        data, records = self.collect()
        self.assertEqual(data["ended"].tolist(), [2, 4])
        steps = records[1:]
        self.assertEqual([(r["step"], r["env_id"]) for r in steps], [(1, 0), (1, 1), (2, 1)])
        # Final diagnostic phase is the pre-reset snapshot, not the new episode's phase zero.
        self.assertEqual([r["move"] for r in steps], [2, 3, 4])
        self.assertEqual([r["episode_done"] for r in steps], [True, False, True])
        self.assertEqual(steps[0]["termination_terms"], ["bad_ori"])
        self.assertFalse(steps[0]["feet_up"])
        self.assertGreater(steps[0]["body_vx"], 0)
        self.assertEqual(steps[0]["foot_force_z"], [2., 3., 4., 5., 6., 7.])

    def test_contact_candidates_keep_simultaneous_links(self):
        _, records = self.collect()
        first, second, final = records[1:]
        self.assertEqual(first["first_sampled_undesired_contact_links"], ["body", "coxa"])
        self.assertEqual(first["tibia_horizontal_contact_links"], ["tibia_2"])
        self.assertEqual(first["first_sampled_tibia_horizontal_contact_links"], ["tibia_2"])
        self.assertEqual(first["contact_forces_w"][3], [0., 0., 20.])
        self.assertEqual(first["horizontal_contact_force"][3], 5.)
        self.assertEqual(second["first_sampled_undesired_contact_links"], ["body"])
        self.assertEqual(final["undesired_contact_links"], ["body"])
        self.assertEqual(final["first_sampled_undesired_contact_links"], [])
        self.assertEqual(records[0]["contact_body_names"], self.body_names)

    def test_env_limit_stops_after_selected_first_episode(self):
        _, records = self.collect(max_envs=1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["env_id"], 0)

    def test_nonfinite_state_writes_valid_json_and_missing_snapshot_fails(self):
        env = FakeEnv()
        _, _, dones, _ = env.step(None)
        env.tracker.diagnostics["body_vx"][0] = float("nan")
        stream = io.StringIO()
        recorder = evaluation.StairDiagnostics(stream, self.body_names, 0.02)
        recorder.record(1, env.tracker.diagnostics, dones, torch.zeros(2, dtype=torch.bool), env)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertIsNone(records[1]["body_vx"])
        with self.assertRaisesRegex(RuntimeError, "before reset"):
            recorder.record(2, None, dones, torch.zeros(2, dtype=torch.bool), env)


if __name__ == "__main__":
    unittest.main()
