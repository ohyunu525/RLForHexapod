"""Rollout accounting regressions; these do not simulate robot physics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_flat_wsp", ROOT / "scripts/eval_flat_wsp.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class Scene(dict):
    pass


class Terminations:
    def __init__(self):
        self.active_terms = ["time_out", "bad_ori", evaluation.PROBE_TERM]
        self.flags = {name: torch.zeros(1, dtype=torch.bool) for name in self.active_terms}

    def get_term(self, name):
        return self.flags[name]

    def get_term_cfg(self, name):
        return SimpleNamespace(time_out=name == "time_out")


def identity_rotation(quat, vector):
    if not torch.equal(quat, torch.tensor([0., 0., 0., 1.]).expand_as(quat)):
        raise AssertionError("Test fixture only supports identity")
    return vector


def environment():
    pose = torch.tensor([[0., 0., .205, 0., 0., 0., 1.]])
    values = {"root_pose_w": pose, "root_vel_w": torch.zeros(1, 6),
              "joint_pos": torch.zeros(1, 18), "joint_vel": torch.zeros(1, 18),
              "body_quat_w": torch.tensor([0., 0., 0., 1.]).expand(1, 6, 4).clone(),
              "body_pos_w": torch.zeros(1, 6, 3)}
    data = SimpleNamespace(**{key: SimpleNamespace(torch=value) for key, value in values.items()})
    sensor = SimpleNamespace(data=SimpleNamespace(
        current_air_time=SimpleNamespace(torch=torch.zeros(1, 6)),
        net_forces_w=SimpleNamespace(torch=torch.tensor([0., 0., 2.]).expand(1, 6, 3).clone())))
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.sensors = {"contact_forces": sensor}
    return SimpleNamespace(scene=scene, num_envs=1, device="cpu", termination_manager=Terminations())


def make_probe():
    probe = evaluation.RolloutProbe(torch, identity_rotation)
    probe.body_ids = list(range(6))
    probe.sensor_ids = list(range(6))
    return probe


class FlatEvaluationTests(unittest.TestCase):
    def test_reset_jump_excluded_and_failure_frame_counted(self):
        env = environment()
        probe = make_probe()
        pose = env.scene["robot"].data.root_pose_w.torch
        pose[0, 0] = 2.0
        probe.begin(env, torch.zeros(1, 67), torch.full((1, 18), -2.0))
        pose[0, 0] = 2.1
        env.termination_manager.flags["bad_ori"][0] = True
        probe.capture(env)
        # In-place reset must not change captured failure states.
        pose[0, 0] = 0.0
        env.termination_manager.flags["bad_ori"].zero_()
        probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))
        pose[0, 0] = .2
        probe.capture(env)
        arrays = probe.arrays()
        result = evaluation.summarize_trace(arrays, 1.0, .15, .205)
        self.assertAlmostEqual(result["progress_world_x_m_per_env"][0], .3, places=6)
        self.assertEqual(result["failed_episodes"], 1)
        self.assertEqual(result["timeout_episodes"], 0)
        self.assertEqual(arrays["episode_id"].tolist(), [[0], [1]])
        self.assertEqual(result["per_leg"]["L1"]["action_clip_fraction"], [.5, .5, .5])
        self.assertAlmostEqual(arrays["root_pose"][0, 0, 0], 2.1, places=6)

    def test_timeout_is_separate_from_failure(self):
        env = environment()
        probe = make_probe()
        probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))
        env.termination_manager.flags["time_out"][0] = True
        probe.capture(env)
        result = evaluation.summarize_trace(probe.arrays(), .02, 0., .205)
        self.assertEqual(result["failed_episodes"], 0)
        self.assertEqual(result["timeout_episodes"], 1)

    def test_nonfinite_failure_saved_but_never_scored(self):
        env = environment()
        probe = make_probe()
        probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))
        env.scene["robot"].data.joint_pos.torch[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite pre-reset"):
            probe.capture(env)
        self.assertEqual(len(probe.frames), 1)
        self.assertTrue(np.isnan(probe.arrays()["joint_pos"][0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "cannot be scored"):
            evaluation.summarize_trace(probe.arrays(), .02, 0., .205)

    def test_contract_and_callback_pairing_enforced(self):
        env = environment()
        probe = make_probe()
        with self.assertRaisesRegex(ValueError, "67 observations"):
            probe.begin(env, torch.zeros(1, 68), torch.zeros(1, 18))
        with self.assertRaisesRegex(RuntimeError, "matching policy step"):
            probe.capture(env)
        probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))
        with self.assertRaisesRegex(RuntimeError, "Previous step"):
            probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))

    def test_body_forward_speed_uses_xyzw_orientation(self):
        env = environment()
        probe = make_probe()
        probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))
        probe.capture(env)
        trace = probe.arrays()
        # +90 degree yaw means body-forward equals world +y.
        trace["root_pose"][..., 3:7] = [0., 0., 2**-.5, 2**-.5]
        trace["root_vel"][..., :3] = [0., .15, 0.]
        result = evaluation.summarize_trace(trace, .02, .15, .205)
        self.assertAlmostEqual(result["mean_forward_speed_body_m_s"], .15, places=6)
        self.assertLess(result["forward_tracking_rmse_m_s"], 1e-6)

    def test_rigid_body_yaw_is_not_foot_excursion(self):
        env = environment()
        probe = make_probe()
        probe.begin(env, torch.zeros(1, 67), torch.zeros(1, 18))
        probe.capture(env)
        trace = {key: np.repeat(value, 50, axis=0) for key, value in probe.arrays().items()}
        yaw = np.linspace(0., np.pi, 50)
        trace["root_pose"][:, 0, 3:7] = np.column_stack((
            np.zeros(50), np.zeros(50), np.sin(yaw / 2), np.cos(yaw / 2)))
        # Each tip is rigidly attached at body x=.2, y=.1 while the whole body yaws.
        offset = np.column_stack((.2 * np.cos(yaw) - .1 * np.sin(yaw),
                                  .2 * np.sin(yaw) + .1 * np.cos(yaw), np.full(50, -.205)))
        trace["foot_tip_pos_w"] = trace["root_pose"][..., None, :3] + offset[:, None, None, :]
        trace["foot_tip_pos_w"] = np.repeat(trace["foot_tip_pos_w"], 6, axis=2)
        result = evaluation.summarize_trace(trace, .02, 0., .205)
        for metrics in result["per_leg"].values():
            self.assertLess(metrics["tip_forward_excursion_p95_p05_m"], 1e-6)

    def test_reset_jumps_do_not_add_joint_travel_or_excursion(self):
        # Two monotonic extensions separated by a large reset jump.
        q = np.array([0., .1, .2, -4., -3.9, -3.8])[:, None, None]
        episodes = np.array([0, 0, 0, 1, 1, 1])[:, None]
        motion = evaluation.joint_motion(q, episodes, .02)
        self.assertAlmostEqual(motion["positive_travel_rad"][0, 0], .4)
        self.assertEqual(motion["negative_travel_rad"][0, 0], 0.)
        self.assertEqual(motion["bidirectional_travel_rad"][0, 0], 0.)
        self.assertEqual(motion["direction_reversals"][0, 0], 0)
        self.assertAlmostEqual(evaluation.episode_weighted_excursion(q, episodes)[0], .18)
        self.assertAlmostEqual(motion["measured_time_s"][0], .08)

    def test_hysteresis_counts_flexion_extension_without_counting_jitter(self):
        q = np.array([0., .05, .1, .0999, .1001, .05, 0., .05, .1])[:, None, None]
        episodes = np.zeros((len(q), 1), dtype=np.int64)
        motion = evaluation.joint_motion(q, episodes, .02)
        self.assertEqual(motion["direction_reversals"][0, 0], 2)
        self.assertGreater(motion["bidirectional_travel_rad"][0, 0], .09)
        # Opposite one-way motions in separate episodes are not repeated motion.
        q = np.array([0., .1, .2, .2, .1, 0.])[:, None, None]
        episodes = np.array([0, 0, 0, 1, 1, 1])[:, None]
        motion = evaluation.joint_motion(q, episodes, .02)
        self.assertEqual(motion["direction_reversals"][0, 0], 0)
        self.assertEqual(motion["bidirectional_travel_rad"][0, 0], 0.)


if __name__ == "__main__":
    unittest.main()
