"""Command-aware gait refinement checks; no simulator needed for reward math.

The optional config test runs with real Isaac Lab configuration classes. Physics
and learning outcomes still require the independent rollout/training checks.
"""

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


_spec = importlib.util.spec_from_file_location(
    "_refine_support_fixture", Path(__file__).with_name("test_wsp_support_rewards.py"))
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)
reward, Entity = _support.reward, _support.Entity


class Fixture(_support.Fixture):
    def __init__(self, count=1, dt=.02):
        super().__init__(count, dt)
        self.pose = torch.zeros(count, 7)
        self.pose[:, 6] = 1.
        self.joint_velocity = torch.zeros(count, 18)
        self.env.scene["robot"].data.root_pose_w = SimpleNamespace(torch=self.pose)
        self.env.scene["robot"].data.joint_vel = SimpleNamespace(torch=self.joint_velocity)
        self.yaw = reward.refine_yaw_tracking_l2(SimpleNamespace(params={}), self.env)

    def yaw_step(self):
        self.env.common_step_counter += 1
        return self.yaw(self.env).clone()


class RefineRewardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_yaw_penalty_tracks_commanded_turn_instead_of_suppressing_turning(self):
        f = Fixture(count=4)
        f.command[:, 2] = torch.tensor([0., .25, -.25, .25])
        f.velocity[:, 5] = torch.tensor([.2, .25, -.25, -.25])
        torch.testing.assert_close(f.yaw_step(), torch.tensor([.04, 0., 0., .25]))
        # Mean yaw is zero, but an alternating oscillation must still cost.
        f.command.zero_()
        f.velocity[:, 5] = torch.tensor([.2, -.2, .2, -.2])
        torch.testing.assert_close(f.yaw_step(), torch.full((4,), .04))

    def test_yaw_is_measured_in_body_frame_even_when_tilted(self):
        f = Fixture()
        # 90 degree roll rotates local +z into world -y (XYZW quaternion).
        f.pose[:, 3:7] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        f.command[:, 2] = .25
        f.velocity[:, 4] = -.25
        self.assertLess(f.yaw_step().item(), 1e-12)
        f.velocity[:, 4] = 0.
        f.velocity[:, 5] = .25  # World z is not this tilted body's yaw axis.
        self.assertAlmostEqual(f.yaw_step().item(), .25**2, places=6)

    def test_stand_motion_cost_is_command_gated_and_cannot_be_escaped_by_moving(self):
        f = Fixture(count=5)
        f.command[:] = torch.tensor([[0., 0., 0.], [.0251, 0., 0.], [0., 0., .1001],
                                     [0., .1, 0.], [0., 0., -.25]])
        f.joint_velocity[:] = 3.
        expected = torch.tensor([1., 0., 0., 0., 0.])
        torch.testing.assert_close(reward.refine_stand_joint_velocity_l2(f.env), expected)
        f.velocity[:] = 10.  # Real base motion must not turn off a stand command.
        f.contact([0] * 6)  # Nor may lifting all feet evade the motion cost.
        torch.testing.assert_close(reward.refine_stand_joint_velocity_l2(f.env), expected)

    def test_stationary_cost_respects_any_resting_pose_and_scales_quadratically(self):
        f = Fixture(count=3)
        f.command.zero_()
        f.joint_velocity[:] = torch.tensor([0., 3., -6.])[:, None]
        torch.testing.assert_close(reward.refine_stand_joint_velocity_l2(f.env), torch.tensor([0., 1., 4.]))
        f.pose[:, :3] = torch.tensor([100., -20., .17])
        torch.testing.assert_close(reward.refine_stand_joint_velocity_l2(f.env), torch.tensor([0., 1., 4.]))
        with self.assertRaises(ValueError):
            reward.refine_stand_joint_velocity_l2(f.env, velocity_scale=0.)

    def test_metrics_are_conditional_and_subset_reset_preserves_other_episodes(self):
        f = Fixture(count=2)
        f.command[0] = 0.
        f.joint_velocity[:] = torch.tensor([3., 9.])[:, None]
        f.velocity[:, 5] = torch.tensor([.2, .4])
        for _ in range(10):
            f.yaw_step()
        f.env.extras["log"] = {}
        f.yaw.reset(torch.tensor([0]))
        log = f.env.extras["log"]
        self.assertAlmostEqual(log["Metrics/Refine/yaw_error_rms"].item(), .2, places=6)
        self.assertAlmostEqual(log["Metrics/Refine/stationary_joint_velocity_rms"].item(), 3., places=5)
        self.assertEqual(log["Metrics/Refine/stationary_fraction"].item(), 1.)
        self.assertEqual(f.yaw._time[0].item(), 0.)
        self.assertAlmostEqual(f.yaw._time[1].item(), .2, places=6)
        f.yaw.reset(torch.tensor([1]))
        self.assertEqual(log["Metrics/Refine/stationary_joint_velocity_rms"].item(), 0.)
        self.assertEqual(log["Metrics/Refine/stationary_fraction"].item(), 0.)
        self.assertAlmostEqual(log["Metrics/Refine/yaw_error_rms"].item(), .4, places=6)
        f.yaw.reset(torch.tensor([], dtype=torch.long))
        self.assertTrue(all(torch.isfinite(x) for x in log.values()))

    def test_repeated_calls_do_not_duplicate_episode_metrics(self):
        f = Fixture()
        f.command.zero_()
        f.velocity[:, 5] = .2
        expected = f.yaw_step()
        for _ in range(10):
            torch.testing.assert_close(f.yaw(f.env), expected)
        self.assertAlmostEqual(f.yaw._time.item(), .02, places=7)
        self.assertAlmostEqual(f.yaw._yaw_error2_time.item(), .04*.02, places=7)

    def test_invalid_state_is_flagged_without_poisoning_episode_logs(self):
        f = Fixture()
        f.command.zero_()
        f.velocity[:, 5] = float("nan")
        f.joint_velocity[:] = float("inf")
        f.yaw_step()
        f.yaw.reset()
        self.assertEqual(f.env.extras["log"]["Metrics/Refine/invalid_state_fraction"].item(), 1.)
        self.assertTrue(all(torch.isfinite(x) for x in f.env.extras["log"].values()))

    def test_real_configs_are_isolated_and_keep_velocity_and_support_rewards(self):
        try:
            import isaaclab  # noqa: F401
        except ImportError:
            self.skipTest("Isaac Lab is required for real config construction")
        from RLforHexapod.tasks.walk.config.hexapod.env_cfg import HexapodEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.wsp_env_cfg import HexapodWspEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.walk_refine_env_cfg import HexapodRefineEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.agents.rsl_rl_ppo_cfg import (
            HexapodWspPPORunnerCfg, HexapodRefinePPORunnerCfg)
        original, wsp, refined = HexapodEnvCfg(), HexapodWspEnvCfg(), HexapodRefineEnvCfg()
        for name in ("track_lin_vel", "track_ang_vel_z_exp", "track_base_height", "support_participation",
                     "all_feet_airborne", "sparse_support", "lin_vel_z_l2", "action_rate_l2"):
            self.assertEqual(getattr(wsp.rewards, name).weight, getattr(refined.rewards, name).weight)
        self.assertFalse(hasattr(wsp.rewards, "yaw_error_l2"))
        self.assertFalse(hasattr(original.rewards, "stand_joint_velocity"))
        refined.rewards.track_lin_vel.params["std"] = .987
        self.assertEqual(wsp.rewards.track_lin_vel.params["std"], .1)
        self.assertEqual(original.rewards.track_lin_vel.params["std"], .1)
        refined.commands.base_velocity.ranges.lin_vel_x = (-9., 9.)
        self.assertEqual(wsp.commands.base_velocity.ranges.lin_vel_x, (-.1, .2))
        old_runner, new_runner = HexapodWspPPORunnerCfg(), HexapodRefinePPORunnerCfg()
        self.assertEqual(new_runner.algorithm.learning_rate, 5e-5)
        self.assertEqual(new_runner.actor.distribution_cfg.std_range, (.03, .15))
        new_runner.algorithm.learning_rate = 9.
        new_runner.actor.hidden_dims[0] = 1
        self.assertEqual(old_runner.algorithm.learning_rate, 1e-4)
        self.assertEqual(old_runner.actor.hidden_dims, [256, 256, 128])


if __name__ == "__main__":
    unittest.main()
