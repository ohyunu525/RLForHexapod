"""Regression checks for orientation and pre-reset smoke diagnostics, without Isaac Lab."""

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("smoke_walk", ROOT / "scripts" / "smoke_walk.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)
ROLL_180_XYZW = (1.0, 0.0, 0.0, 0.0)
PROBE_TERM = "smoke_raw_state_finite"


def known_rotation_only(quat, vector):
    """Independent fixture oracle: identity and an X half-turn only, never a general mock rotation."""
    identity = (quat == torch.tensor(IDENTITY_XYZW, dtype=quat.dtype, device=quat.device)).all(dim=-1)
    half_turn = (quat == torch.tensor(ROLL_180_XYZW, dtype=quat.dtype, device=quat.device)).all(dim=-1)
    if not (identity | half_turn).all().item():
        raise AssertionError("The test rotation oracle accepts only identity and X-axis 180 degrees")
    flipped = vector * torch.tensor((1.0, -1.0, -1.0), dtype=vector.dtype, device=vector.device)
    return torch.where(half_turn.unsqueeze(-1), flipped, vector)


def configured_initial_rotation():
    """Read the actual configuration expression without importing simulator-dependent modules."""
    path = ROOT / "src" / "RLforHexapod" / "tasks" / "walk" / "config" / "hexapod" / "env_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "HEXAPOD_CFG" for target in node.targets)
    )
    initial = next(keyword.value for keyword in assignment.value.keywords if keyword.arg == "init_state")
    rotation = next(keyword.value for keyword in initial.keywords if keyword.arg == "rot")
    return ast.literal_eval(rotation)


class Scene(dict):
    pass


class Terminations:
    def __init__(self, count):
        self.active_terms = ["time_out", "bad_ori", "base_low", PROBE_TERM]
        self.flags = {name: torch.zeros(count, dtype=torch.bool) for name in self.active_terms}

    def get_term(self, name):
        return self.flags[name]

    def get_term_cfg(self, name):
        return SimpleNamespace(time_out=name == "time_out")


def make_env(*quaternions):
    count = len(quaternions)
    pose = torch.zeros(count, 7)
    pose[:, 2] = 0.3
    pose[:, 3:7] = torch.tensor(quaternions)
    gravity = torch.tensor([[0.0, 0.0, -1.0 if quat == IDENTITY_XYZW else 1.0] for quat in quaternions])
    data = SimpleNamespace(**{
        name: SimpleNamespace(torch=value) for name, value in {
            "root_pose_w": pose,
            "root_vel_w": torch.zeros(count, 6),
            "joint_pos": torch.zeros(count, 18),
            "joint_vel": torch.zeros(count, 18),
            "projected_gravity_b": gravity,
        }.items()
    })
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(count, 3)
    return SimpleNamespace(scene=scene, num_envs=count, device="cpu", termination_manager=Terminations(count))


class SmokeWalkTests(unittest.TestCase):
    def test_actual_configuration_is_upright_and_forward(self):
        env = make_env(configured_initial_rotation())
        report = {}
        probe = smoke.StateProbe(report, torch, known_rotation_only)
        probe.check(env, sampling="immediately after initial reset")
        probe.initial_orientation(env)
        self.assertNotIn("first_failure", report)
        orientation = report["initial_orientation"]
        self.assertEqual(orientation["quaternion_order"], "xyzw")
        self.assertEqual(orientation["body_x_world"], [[1.0, 0.0, 0.0]])
        self.assertEqual(orientation["body_z_world"], [[0.0, 0.0, 1.0]])
        self.assertEqual(orientation["projected_gravity_b"], [[0.0, 0.0, -1.0]])
        self.assertEqual(orientation["tilt_rad"], [0.0])

    def test_old_scalar_first_tuple_faces_forward_but_fails_upright_check(self):
        env = make_env(ROLL_180_XYZW)
        report = {}
        probe = smoke.StateProbe(report, torch, known_rotation_only)
        values = smoke.orientation_tensors(env.scene["robot"].data, torch, known_rotation_only)
        self.assertEqual(values["body_x_world"].tolist(), [[1.0, 0.0, 0.0]])
        self.assertEqual(values["body_z_world"].tolist(), [[0.0, 0.0, -1.0]])
        self.assertAlmostEqual(values["tilt_rad"].item(), math.pi, places=6)
        with self.assertRaisesRegex(smoke.SmokeFailure, "body \\+z upward"):
            probe.initial_orientation(env)
        self.assertEqual(report["first_failure"]["snapshot"]["quaternion_xyzw"], [[1.0, 0.0, 0.0, 0.0]])

    def test_first_termination_snapshot_survives_reset_and_later_failures(self):
        env = make_env(ROLL_180_XYZW, IDENTITY_XYZW)
        data = env.scene["robot"].data
        data.root_pose_w.torch[0, 0] = 2.5
        data.root_vel_w.torch[0, 0] = 0.4
        env.termination_manager.flags["bad_ori"][0] = True
        report = {}
        probe = smoke.StateProbe(report, torch, known_rotation_only)
        probe.step = 7
        probe.capture_termination_failure(env, PROBE_TERM)
        first = copy.deepcopy(report["first_failure"])
        self.assertEqual(first["step"], 7)
        self.assertEqual(first["terms"], ["bad_ori"])
        self.assertEqual(first["snapshot"]["sampling"], "before automatic reset")
        self.assertEqual(first["snapshot"]["root_pose_w"][0][0], 2.5)
        # Simulate in-place automatic reset of simulator buffers after the probe ran.
        data.root_pose_w.torch.zero_()
        data.root_pose_w.torch[:, 6] = 1.0
        data.root_vel_w.torch.zero_()
        data.projected_gravity_b.torch[:, 2] = -1.0
        env.termination_manager.flags["bad_ori"].zero_()
        env.termination_manager.flags["base_low"][1] = True
        probe.step = 8
        probe.capture_termination_failure(env, PROBE_TERM)
        with self.assertRaises(smoke.SmokeFailure):
            probe.fail("Later caller failure")
        self.assertEqual(report["first_failure"], first)
        self.assertEqual(first["snapshot"]["body_z_world"], [[0.0, 0.0, -1.0]])

    def test_timeout_and_probe_term_are_not_stability_failures(self):
        env = make_env(IDENTITY_XYZW, IDENTITY_XYZW)
        env.termination_manager.flags["time_out"][:] = True
        env.termination_manager.flags[PROBE_TERM][:] = True
        report = {}
        smoke.StateProbe(report, torch, known_rotation_only).capture_termination_failure(env, PROBE_TERM)
        self.assertNotIn("first_failure", report)

    def test_nan_gravity_or_pose_preserves_json_null_snapshot(self):
        for field, column in (("projected_gravity_b", 2), ("root_pose_w", 0)):
            with self.subTest(field=field):
                env = make_env(IDENTITY_XYZW)
                getattr(env.scene["robot"].data, field).torch[0, column] = float("nan")
                report = {}
                probe = smoke.StateProbe(report, torch, known_rotation_only)
                probe.step = 3
                with self.assertRaisesRegex(smoke.SmokeFailure, "Non-finite raw simulator state"):
                    probe.check(env)
                first = report["first_failure"]
                self.assertEqual(first["invalid_envs_by_field"], {field: [0]})
                self.assertIsNone(first["snapshot"][field][0][column])
                parsed = json.loads(json.dumps(report, allow_nan=False))
                self.assertIsNone(parsed["first_failure"]["snapshot"][field][0][column])
                if field == "projected_gravity_b":
                    self.assertIsNone(first["snapshot"]["tilt_rad"][0])

    def test_fixture_oracle_rejects_unverified_rotations(self):
        with self.assertRaisesRegex(AssertionError, "only identity"):
            known_rotation_only(torch.tensor([[0.0, 0.0, 1.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
