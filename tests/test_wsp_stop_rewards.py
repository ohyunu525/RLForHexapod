"""Behavior and isolation checks for the new WspStop reward configuration.

Reward math uses controlled states, not physical simulations. The final test
requires actual Isaac Lab config classes and is skipped if they are unavailable.
"""

import ast
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


_spec = importlib.util.spec_from_file_location(
    "_wsp_stop_refine_fixture", Path(__file__).with_name("test_walk_refine_rewards.py"))
_refine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_refine)
reward, Fixture = _refine.reward, _refine.Fixture
ROOT = Path(__file__).resolve().parents[1]
JOINTS = tuple(f"L{leg}_{part}" for leg in range(1, 7) for part in ("coxa", "femur", "tibia"))


def posture_fixture(count=1, names=JOINTS, ids=None, table=None):
    f = Fixture(count=count)
    f.command.zero_()
    f.height = torch.full((count, 1), .205)
    f.position = torch.zeros(count, 18)
    asset = f.env.scene["robot"]
    asset.joint_names = list(names)
    asset.data.joint_pos = SimpleNamespace(torch=f.position)
    f.env.command_manager.get_command = lambda name: f.height if name == "base_height" else f.command
    f.posture_params = {
        "anchor_heights": (.17, .205, .24),
        "anchor_joint_positions_rad": table if table is not None else ((0.,)*18, (.5,)*18, (1.,)*18),
        "anchor_joint_names": JOINTS,
        "asset_cfg": _refine._support.Entity("robot", joint_ids=list(range(18)) if ids is None else ids),
    }
    f.posture = reward.wsp_stop_stand_posture_l2(SimpleNamespace(params=f.posture_params), f.env)
    return f


class WspStopRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_moving_commands_do_not_receive_joint_motion_penalty_or_masked_nan(self):
        f = Fixture(count=6)
        f.command[:] = torch.tensor([[.026, 0., 0.], [0., -.026, 0.], [0., 0., .101],
                                     [0., 0., -.25], [.15, 0., 0.], [0., 0., 0.]])
        f.joint_velocity[:] = 3.
        f.joint_velocity[0] = float("nan")
        f.joint_velocity[1] = float("inf")
        f.joint_velocity[2] = -100.
        cost = reward.refine_stand_joint_velocity_l2(f.env)
        torch.testing.assert_close(cost, torch.tensor([0., 0., 0., 0., 0., 1.]))
        self.assertTrue(torch.isfinite(cost).all())

    def test_stationary_gate_uses_full_requested_planar_motion_and_requested_yaw(self):
        f = Fixture(count=5)
        f.command[:] = torch.tensor([[.02, .02, 0.], [0., 0., -.11],
                                     [.01, .01, .05], [0., 0., 0.], [-.15, 0., 0.]])
        f.joint_velocity[:] = 3.
        f.velocity[:] = 20.
        f.contact([0] * 6)
        torch.testing.assert_close(reward.refine_stand_joint_velocity_l2(f.env),
                                   torch.tensor([0., 0., 1., 1., 0.]))

    def test_static_supported_stop_has_no_swing_requirement_but_moving_lock_does(self):
        f = Fixture(count=2)
        f.command[0] = 0.
        f.command[1] = torch.tensor([.15, 0., 0.])
        f.joint_velocity.zero_()
        for _ in range(200):
            cost = f.step([1] * 6)
        self.assertEqual(cost[0].item(), 0.)
        self.assertGreater(cost[1].item(), .8)
        self.assertTrue(torch.equal(reward.refine_stand_joint_velocity_l2(f.env), torch.zeros(2)))

    def test_correct_turn_has_no_yaw_error_or_stand_motion_cost(self):
        f = Fixture(count=3)
        f.command[:] = torch.tensor([[0., 0., .25], [0., 0., -.25], [0., 0., 0.]])
        f.velocity[:, 5] = torch.tensor([.25, -.25, .20])
        f.joint_velocity[:] = 3.
        torch.testing.assert_close(f.yaw_step(), torch.tensor([0., 0., .04]))
        torch.testing.assert_close(reward.refine_stand_joint_velocity_l2(f.env), torch.tensor([0., 0., 1.]))

    def test_posture_height_interpolation_endpoint_clamp_and_exact_cost_scale(self):
        f = posture_fixture(count=6)
        f.height[:, 0] = torch.tensor([.1, .17, .1875, .205, .24, .3])
        f.position[:] = torch.tensor([0., 0., .25, .5, 1., 1.])[:, None]
        torch.testing.assert_close(f.posture(f.env, **f.posture_params), torch.zeros(6), atol=1e-10, rtol=0.)
        f.position += .35
        torch.testing.assert_close(f.posture(f.env, **f.posture_params), torch.ones(6), atol=2e-6, rtol=0.)

    def test_posture_moving_gate_and_nonfinite_state_return_finite_costs(self):
        f = posture_fixture(count=5)
        f.command[:] = torch.tensor([[.15, 0., 0.], [0., 0., -.25], [0., 0., 0.],
                                     [0., 0., 0.], [.01, .01, .05]])
        f.position[:] = .5
        f.position[0] = float("nan")
        f.height[1] = float("inf")
        f.position[2] = float("nan")
        f.height[3] = float("nan")
        f.velocity[:] = 10.  # Actual motion does not disable a stationary command.
        cost = f.posture(f.env, **f.posture_params)
        torch.testing.assert_close(cost, torch.tensor([0., 0., 1., 1., 0.]))
        self.assertTrue(torch.isfinite(cost).all())

    def test_posture_respects_resolved_joint_order_and_rejects_invalid_tables(self):
        target = tuple(float(x)/100 for x in range(18))
        table = (target, target, target)
        f = posture_fixture(names=JOINTS[::-1], ids=list(range(17, -1, -1)), table=table)
        f.position[:] = torch.tensor(target[::-1])
        self.assertEqual(f.posture(f.env, **f.posture_params).item(), 0.)
        with self.assertRaisesRegex(ValueError, "Resolved posture joint order"):
            posture_fixture(names=JOINTS[::-1], table=table)
        for field, value in (("anchor_heights", (.17, .17, .24)),
                             ("anchor_joint_positions_rad", ((float("nan"),)*18,)*3),
                             ("anchor_joint_positions_rad", ((0.,)*17,)*3),
                             ("anchor_joint_names", JOINTS[::-1])):
            with self.subTest(field=field):
                changed = dict(f.posture_params)
                changed[field] = value
                with self.assertRaises(ValueError):
                    reward.wsp_stop_stand_posture_l2(SimpleNamespace(params=changed), f.env)

    def test_embedded_nine_height_anchors_match_reviewed_urdf_tip_geometry(self):
        path = ROOT / "src/RLforHexapod/tasks/walk/config/hexapod/walk_wsp_stop_env_cfg.py"
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse(path.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in ("WSP_STOP_ANCHOR_HEIGHTS", "WSP_STOP_ANCHOR_POSITIONS")}
        heights, q = values["WSP_STOP_ANCHOR_HEIGHTS"], np.asarray(values["WSP_STOP_ANCHOR_POSITIONS"])
        self.assertEqual(q.shape, (9, 18))
        self.assertIsInstance(heights, tuple)
        self.assertEqual((heights[0], heights[4], heights[-1]), (.17, .205, .24))
        urdf = ROOT / "src/RLforHexapod/assets/urdf/robot.urdf"
        self.assertEqual(hashlib.sha256(urdf.read_bytes()).hexdigest(),
                         "34e8d5d2c20542b23f0fce285029944ad65910d10afa8113a7b76b56067bf23f")
        # Reuse exact URDF FK from the production offline WSP tool. No outputs
        # files or simulator are needed to validate the embedded geometry.
        spec = importlib.util.spec_from_file_location(
            "_posture_fk_check", ROOT / "scripts/wsp_stop_refine.py")
        geometry = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            spec.loader.exec_module(geometry)
        skeleton = geometry.Kinematics(urdf)
        for height, joints in zip(heights, q):
            points = np.concatenate([skeleton.fk_jacobian(leg, joints[leg*3:leg*3+3])[0]
                                     for leg in range(6)])
            np.testing.assert_allclose(points[:, 2]+height, .012, atol=2e-8)
            actions = (joints-np.tile(np.deg2rad([0., 20., -110.]), 6))/np.tile([.5, 1., 1.], 6)
            self.assertLess(float(np.abs(actions).max()), .98)

    def test_real_new_task_is_isolated_and_changes_only_approved_controls(self):
        try:
            import isaaclab  # noqa: F401
        except ImportError:
            self.skipTest("Isaac Lab is required for config isolation checks")
        import gymnasium as gym
        from RLforHexapod.tasks.walk.config.hexapod.env_cfg import HexapodEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.wsp_env_cfg import HexapodWspEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.walk_refine_env_cfg import HexapodRefineEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.walk_wsp_stop_env_cfg import HexapodWspStopEnvCfg
        from RLforHexapod.tasks.walk.config.hexapod.agents.rsl_rl_ppo_cfg import (
            HexapodRefinePPORunnerCfg, HexapodWspStopPPORunnerCfg)
        original, wsp, previous, new = (HexapodEnvCfg(), HexapodWspEnvCfg(),
                                        HexapodRefineEnvCfg(), HexapodWspStopEnvCfg())
        self.assertEqual(gym.spec("Hexapod-Walk-WspStop").kwargs["env_cfg_entry_point"].split(":")[-1],
                         "HexapodWspStopEnvCfg")
        self.assertEqual(new.rewards.stand_joint_velocity.weight, -1.)
        self.assertEqual(new.rewards.stand_joint_velocity.params["velocity_scale"], 3.)
        self.assertEqual(new.rewards.yaw_error_l2.weight, -3.)
        self.assertEqual(new.rewards.lin_vel_z_l2.weight, -8.)
        self.assertEqual(new.commands.base_velocity.rel_standing_envs, .3)
        self.assertEqual(new.rewards.stand_posture.weight, -.25)
        self.assertEqual(new.rewards.stand_posture.params["angle_scale"], .35)
        self.assertEqual(new.rewards.stand_posture.params["anchor_joint_names"], JOINTS)
        self.assertEqual(len(new.rewards.stand_posture.params["anchor_heights"]), 9)
        self.assertTrue(new.rewards.stand_posture.params["asset_cfg"].preserve_order)
        self.assertFalse(hasattr(previous.rewards, "stand_posture"))
        for name in ("track_lin_vel", "track_ang_vel_z_exp", "track_base_height", "support_participation",
                     "all_feet_airborne", "sparse_support", "action_rate_l2", "joint_acc_l2", "joint_torques_l2"):
            old_term, new_term = getattr(previous.rewards, name), getattr(new.rewards, name)
            self.assertEqual(old_term.weight, new_term.weight)
            self.assertIs(old_term.func, new_term.func)
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            self.assertEqual(getattr(new.commands.base_velocity.ranges, name),
                             getattr(previous.commands.base_velocity.ranges, name))
        self.assertEqual(new.commands.base_height.height_range, previous.commands.base_height.height_range)
        self.assertEqual(new.commands.base_velocity.resampling_time_range, previous.commands.base_velocity.resampling_time_range)
        self.assertEqual(new.decimation, previous.decimation)
        self.assertEqual(new.sim.dt, previous.sim.dt)
        self.assertEqual(new.episode_length_s, previous.episode_length_s)
        self.assertEqual(new.scene.robot.spawn.usd_path, previous.scene.robot.spawn.usd_path)
        self.assertEqual(new.actions.joint_pos.scale, previous.actions.joint_pos.scale)
        new.commands.base_velocity.ranges.lin_vel_x = (-9., 9.)
        new.commands.base_height.height_range = (-9., 9.)
        new.rewards.track_lin_vel.params["std"] = 9.
        for config in (original, wsp, previous):
            self.assertEqual(config.commands.base_velocity.rel_standing_envs, .1)
            self.assertEqual(config.commands.base_velocity.ranges.lin_vel_x, (-.1, .2))
            self.assertEqual(config.commands.base_height.height_range, (.17, .24))
            self.assertEqual(config.rewards.track_lin_vel.params["std"], .1)
        self.assertEqual(previous.rewards.stand_joint_velocity.weight, -.2)
        self.assertEqual(previous.rewards.yaw_error_l2.weight, -1.)
        self.assertEqual(previous.rewards.lin_vel_z_l2.weight, -4.)
        fresh = HexapodWspStopEnvCfg()
        self.assertEqual(fresh.commands.base_velocity.ranges.lin_vel_x, (-.1, .2))
        self.assertEqual(fresh.commands.base_velocity.rel_standing_envs, .3)
        old_ppo, new_ppo = HexapodRefinePPORunnerCfg(), HexapodWspStopPPORunnerCfg()
        self.assertEqual(new_ppo.experiment_name, "hexapod_walk_wsp_stop")
        self.assertEqual(new_ppo.max_iterations, 2000)
        self.assertEqual(new_ppo.algorithm.learning_rate, 3e-5)
        self.assertEqual(new_ppo.algorithm.clip_param, .1)
        self.assertEqual(new_ppo.algorithm.entropy_coef, .0005)
        self.assertEqual(new_ppo.actor.distribution_cfg.init_std, .05)
        self.assertEqual(new_ppo.actor.distribution_cfg.std_range, (.02, .1))
        new_ppo.actor.hidden_dims[0] = 1
        new_ppo.algorithm.learning_rate = 9.
        self.assertEqual(old_ppo.actor.hidden_dims, [256, 256, 128])
        self.assertEqual(old_ppo.algorithm.learning_rate, 5e-5)
        self.assertEqual(old_ppo.actor.distribution_cfg.init_std, .08)


if __name__ == "__main__":
    unittest.main()
