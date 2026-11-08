"""Behavioral support/hopping reward checks without an Isaac Lab installation.

These exercise the real reward/state code with controlled contacts and poses;
they do not claim that a policy trained with these rewards will walk correctly.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


MDP = Path(__file__).resolve().parents[1] / "src/RLforHexapod/tasks/walk/mdp"
NAMES = ["tibia", *[f"tibia_{i}" for i in range(2, 7)]]


def load_rewards():
    names = ["isaaclab", "isaaclab.managers", "isaaclab.utils", "isaaclab.utils.math",
             "_wsp_reward_test", "_wsp_reward_test.commands", "_wsp_reward_test.observations"]
    stubs = {name: ModuleType(name) for name in names}
    stubs["_wsp_reward_test"].__path__ = [str(MDP)]

    class Entity:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)

    class Term:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    managers = stubs["isaaclab.managers"]
    managers.SceneEntityCfg, managers.ManagerTermBase, managers.RewardTermCfg = Entity, Term, object
    stubs["_wsp_reward_test.commands"].base_height = lambda env, asset: asset.data.root_vel_w.torch[:, 2] * 0

    def offsets(body_names, device):
        return torch.tensor([[-.004, -.147, -.021] if name == "tibia" else [.004, .147, -.021]
                             for name in body_names], device=device)

    stubs["_wsp_reward_test.observations"].tip_offsets = offsets

    def apply_xyzw(quat, vector):
        cross = 2 * torch.cross(quat[..., :3], vector, dim=-1)
        return vector + quat[..., 3:] * cross + torch.cross(quat[..., :3], cross, dim=-1)

    stubs["isaaclab.utils.math"].quat_apply = apply_xyzw
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("_wsp_reward_test.rewards", MDP / "rewards.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, Entity


reward, Entity = load_rewards()


class Scene(dict):
    pass


def data(**values):
    return SimpleNamespace(**{key: SimpleNamespace(torch=value) for key, value in values.items()})


class Fixture:
    def __init__(self, count=1, dt=.02):
        self.forces = torch.zeros(count, 7, 3)
        self.positions = torch.zeros(count, 7, 3)
        self.positions[:, 1:, 2] = .031  # rotated tip offset -21mm gives +10mm.
        quat = torch.zeros(count, 7, 4)
        quat[..., 3] = 1
        self.velocity = torch.zeros(count, 6)
        robot = SimpleNamespace(body_names=["body", *NAMES], data=data(
            body_pos_w=self.positions, body_quat_w=quat, root_vel_w=self.velocity))
        sensor = SimpleNamespace(body_names=[*NAMES, "body"], data=data(net_forces_w=self.forces))
        scene = Scene(robot=robot)
        scene.sensors, scene.env_origins = {"contact_forces": sensor}, torch.zeros(count, 3)
        self.command = torch.tensor([[.15, 0., 0.]]).repeat(count, 1)
        self.env = SimpleNamespace(scene=scene, device="cpu", num_envs=count, step_dt=dt,
                                   common_step_counter=0, extras={},
                                   command_manager=SimpleNamespace(get_command=lambda _: self.command))
        # Different asset/sensor indices catch accidental reuse of the other ordering.
        self.params = {"sensor_cfg": Entity("contact_forces", body_ids=list(range(6))),
                       "asset_cfg": Entity("robot", body_ids=list(range(1, 7)))}
        self.term = reward.wsp_support_participation(SimpleNamespace(params=self.params), self.env)

    def contact(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        self.forces.zero_()
        self.forces[:, :6, 2] = 4 * values

    def step(self, values):
        self.contact(values)
        self.env.common_step_counter += 1
        return self.term(self.env, **self.params).clone()

    def total_cost(self):
        return (.6 * self.term(self.env, **self.params)
                + .8 * reward.wsp_all_feet_airborne(self.env, **self.params)
                + .25 * reward.wsp_support_deficit(self.env, **self.params))


class WspSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_alternating_tripods_beat_locked_feet_unused_leg_and_hopping(self):
        fixture = Fixture(count=4)
        total = torch.zeros(4)
        odd = torch.tensor([1, 0, 1, 0, 1, 0])
        for step in range(250):
            tripod = odd if (step // 10) % 2 == 0 else 1 - odd
            unused = tripod.clone()
            unused[5] = 0
            hop = torch.full((6,), int((step // 10) % 2 == 0))
            fixture.step(torch.stack([tripod, torch.ones(6), unused, hop]))
            if step >= 150:
                total += fixture.total_cost()
        self.assertLess(total[0].item(), 1e-5)
        self.assertGreater(total[1].item(), 40.)  # all six locked while moving
        self.assertGreater(total[2].item(), 25.)  # one leg never contributes
        self.assertGreater(total[3].item(), 45.)  # duty alone cannot excuse hops

    def test_horizontal_force_or_elevated_knee_is_not_foot_support(self):
        fixture = Fixture()
        fixture.contact([1] * 6)
        self.assertEqual(reward.wsp_all_feet_airborne(fixture.env, **fixture.params).item(), 0.)
        fixture.forces[:, :6] = torch.tensor([100., 0., 0.])
        self.assertEqual(reward.wsp_all_feet_airborne(fixture.env, **fixture.params).item(), 1.)
        fixture.contact([1] * 6)
        fixture.positions[:, 1:, 2] = .121  # distal tips are 100mm up despite shaft load
        self.assertEqual(reward.wsp_all_feet_airborne(fixture.env, **fixture.params).item(), 1.)
        fixture.positions[:, 1:, 2] = .031
        fixture.env.scene.env_origins[:, 2] = 2.
        fixture.positions[:, 1:, 2] += 2.
        self.assertEqual(reward.wsp_all_feet_airborne(fixture.env, **fixture.params).item(), 0.)

    def test_stationary_has_no_swing_requirement_but_turning_does(self):
        fixture = Fixture(count=2)
        fixture.command[0] = 0
        fixture.command[1] = torch.tensor([0., 0., .25])
        for _ in range(200):
            cost = fixture.step(torch.ones(2, 6))
        self.assertEqual(cost[0].item(), 0.)
        self.assertGreater(cost[1].item(), .9)
        # Stopping clears the moving history; restarting does not inherit a lock penalty.
        fixture.command.zero_()
        self.assertTrue(torch.equal(fixture.step(torch.ones(2, 6)), torch.zeros(2)))
        fixture.command[:, 0] = .15
        self.assertTrue(torch.equal(fixture.step(torch.ones(2, 6)), torch.zeros(2)))

    def test_partial_reset_logs_ended_episode_and_preserves_other_history(self):
        fixture = Fixture(count=2)
        for _ in range(100):
            fixture.step([[1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0]])
        retained = fixture.term.duty[1].clone()
        fixture.env.extras["log"] = {}  # Actual _reset_idx clears this before RewardManager.reset.
        fixture.term.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(fixture.term.duty[0], torch.full((6,), .5)))
        self.assertTrue(torch.equal(fixture.term.duty[1], retained))
        self.assertAlmostEqual(fixture.env.extras["log"]["Metrics/WSP/support_duty_L6"].item(), 0.)
        self.assertAlmostEqual(fixture.env.extras["log"]["Metrics/WSP/support_count"].item(), 5., places=5)
        self.assertEqual(fixture.term._time[0].item(), 0.)
        self.assertAlmostEqual(fixture.term._time[1].item(), 2., places=5)
        fixture.term.reset(torch.tensor([], dtype=torch.long))
        self.assertTrue(all(torch.isfinite(v) for v in fixture.env.extras["log"].values()))

    def test_repeated_calls_do_not_double_age_or_double_log(self):
        fixture = Fixture()
        first = fixture.step([0, 0, 0, 0, 0, 0])
        duty = fixture.term.duty.clone()
        for _ in range(10):
            torch.testing.assert_close(fixture.term(fixture.env, **fixture.params), first)
        self.assertTrue(torch.equal(fixture.term.duty, duty))
        self.assertAlmostEqual(fixture.term._time.item(), .02, places=7)
        with patch.object(reward, "tip_offsets", side_effect=AssertionError("geometry should be cached")):
            reward.wsp_all_feet_airborne(fixture.env, **fixture.params)
            fixture.step([1, 0, 1, 0, 1, 0])

    def test_duty_window_is_time_based_not_control_step_count(self):
        a, b = Fixture(dt=.02), Fixture(dt=.01)
        for _ in range(100):
            a.step([1, 0, 1, 0, 1, 0])
        for _ in range(200):
            b.step([1, 0, 1, 0, 1, 0])
        torch.testing.assert_close(a.term.duty, b.term.duty, atol=2e-6, rtol=0)

    def test_nonfinite_vertical_velocity_is_counted_without_poisoning_metrics(self):
        fixture = Fixture()
        fixture.velocity[:, 2] = float("nan")
        fixture.step([1, 0, 1, 0, 1, 0])
        fixture.term.reset()
        log = fixture.env.extras["log"]
        self.assertEqual(log["Metrics/WSP/invalid_vertical_velocity_fraction"].item(), 1.)
        self.assertEqual(log["Metrics/WSP/vertical_velocity_rms"].item(), 0.)
        self.assertTrue(all(torch.isfinite(value) for value in log.values()))


if __name__ == "__main__":
    unittest.main()
