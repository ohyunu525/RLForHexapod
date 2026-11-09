"""Exercise real tracker transitions with tensors, without launching Isaac Lab."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


MDP = Path(__file__).resolve().parents[1] / "src/RLforHexapod/tasks/walk/mdp"


def load_tracker():
    """Keep simulator import stubs local; use the real terrain and tracker code."""
    stubs = {
        name: ModuleType(name)
        for name in (
            "isaaclab", "isaaclab.managers", "isaaclab.utils",
            "isaaclab.utils.configclass", "isaaclab.utils.math", "_tracker_test_mdp",
        )
    }
    stubs["_tracker_test_mdp"].__path__ = [str(MDP)]
    managers = stubs["isaaclab.managers"]
    managers.ManagerTermBase = managers.ObservationTermCfg = managers.SceneEntityCfg = object
    stubs["isaaclab.utils.configclass"].configclass = lambda cls: cls
    # These transition tests support only identity rotation. Reject a mismatched
    # quaternion convention instead of silently treating a flipped pose as identity.
    def identity_quat_apply(quat, vector):
        expected = torch.zeros_like(quat)
        expected[..., 3] = 1.0  # Isaac Lab 3.0: xyzw.
        if not torch.allclose(quat, expected):
            raise AssertionError("Tracker fixture requires an xyzw identity quaternion")
        return vector

    stubs["isaaclab.utils.math"].quat_apply = identity_quat_apply
    with patch.dict(sys.modules, stubs):
        modules = []
        for stem in ("observations", "stair_reward", "mixed_stair_geometry"):
            name = f"_tracker_test_mdp.{stem}"
            spec = importlib.util.spec_from_file_location(name, MDP / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


observations, reward, geometry = load_tracker()


def tensor_data(**values):
    return SimpleNamespace(**{name: SimpleNamespace(torch=value) for name, value in values.items()})


class Scene(dict):
    pass


class Fixture:
    def __init__(self, *, count=1, move=1, hold=0.0):
        pose = torch.zeros(count, 7)
        pose[:, 0], pose[:, 2], pose[:, 6] = 0.2, 0.2, 1.0
        quat = torch.zeros(count, 6, 4)
        quat[..., 3] = 1.0
        self.data = tensor_data(
            root_pose_w=pose, root_vel_w=torch.zeros(count, 6),
            body_pos_w=torch.zeros(count, 6, 3), body_quat_w=quat,
            joint_pos=torch.zeros(count, 2), joint_vel=torch.zeros(count, 2),
        )
        robot = SimpleNamespace(
            data=self.data,
            find_bodies=lambda names, **kwargs: (list(range(6)), names),
            find_joints=lambda names, **kwargs: ([0, 1], names),
        )
        self.forces = torch.zeros(count, 7, 3)
        self.forces[:, :6, 2] = 2.0
        self.air = torch.zeros(count, 7)
        self.history = self.forces[:, None].repeat(1, 3, 1, 1)
        sensor = SimpleNamespace(
            data=tensor_data(
                current_air_time=self.air, net_normal_forces_w=self.forces,
                net_normal_forces_w_history=self.history,
            ),
            find_bodies=lambda names, **kwargs: (list(range(6)), names),
        )
        scene = Scene(robot=robot)
        scene.sensors = {"contact_forces": sensor}
        scene.env_origins = torch.zeros(count, 3)
        gait = reward.StairGaitCfg()
        gait.move_contact_hold_time = hold
        self.env = SimpleNamespace(
            cfg=SimpleNamespace(stairs=observations.StairsGeometryCfg(), stair_gait=gait),
            scene=scene, device="cpu", num_envs=count, step_dt=0.02, common_step_counter=0,
        )
        self.tracker = reward.StairTracker(self.env)
        self.tracker.move[:] = move
        self.put_feet(range(6), x=0.1, z=gait.tip_height)

    def put_feet(self, feet, *, x=0.55, z=0.173):
        feet = list(feet)
        position = torch.tensor([x, 0.0, z]) - self.tracker.tip_offset[feet]
        self.data.body_pos_w.torch[:, feet] = position

    def tick(self):
        self.env.common_step_counter += 1
        self.tracker.update()


class StairTrackerTests(unittest.TestCase):
    def test_front_approach_starts_lift_before_the_body_reaches_its_target(self):
        f = Fixture(move=0)
        f.env.cfg.stair_gait.front_approach_distance = .04
        f.data.root_pose_w.torch[:, 0] = .26
        f.put_feet([reward.FRONT[0]], x=.465, z=.08)
        f.air[:, reward.FRONT[0]] = .02
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)
        self.assertTrue(f.tracker.just_done.item())
        # Reaching the riser only opens the lift phase. Placement still requires
        # both front feet to contact the actual upper tread.
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)

    def test_front_approach_rejects_distant_or_lateral_feet_and_missing_support(self):
        for condition in ("distant", "lateral", "rear_airborne"):
            with self.subTest(condition=condition):
                f = Fixture(move=0)
                f.env.cfg.stair_gait.front_approach_distance = .04
                f.data.root_pose_w.torch[:, 0] = .26
                f.put_feet([reward.FRONT[0]], x=.465, z=.08)
                if condition == "distant":
                    f.put_feet([reward.FRONT[0]], x=.45, z=.08)
                elif condition == "lateral":
                    f.data.body_pos_w.torch[:, reward.FRONT[0], 1] += .6
                else:
                    f.put_feet([reward.REAR[0]], x=.1, z=.10)
                f.tick()
                self.assertEqual(f.tracker.move.item(), 0)

    def test_front_approach_does_not_skip_the_final_advance(self):
        f = Fixture(move=3 * reward.NUM_PHASES)
        f.env.cfg.stair_gait.front_approach_distance = .04
        f.data.root_pose_w.torch[:, 0] = 1.15
        f.put_feet(range(6), x=1.2, z=.493)
        f.put_feet([reward.FRONT[0]], x=1.365, z=.55)
        f.tick()
        self.assertEqual(f.tracker.move.item(), f.tracker.final_advance)

    def test_legacy_body_target_remains_required_without_front_approach_option(self):
        f = Fixture(move=0)
        f.data.root_pose_w.torch[:, 0] = .26
        f.put_feet(reward.FRONT, x=.465, z=.013)
        f.tick()
        self.assertEqual(f.tracker.move.item(), 0)

    def test_mixed_geometry_updates_signed_treads_and_keeps_other_rows(self):
        f = Fixture(count=3)
        f.env.scene.env_origins[:, 2] = torch.tensor([2., 3., 4.])
        writes = []
        blocks = SimpleNamespace(
            data=tensor_data(default_body_pose=torch.zeros(3, 4, 7)),
            write_body_pose_to_sim_index=lambda **kwargs: writes.append(kwargs),
        )
        f.env.scene["stair_blocks"] = blocks
        ids = torch.tensor([0, 2])
        geometry.set_mixed_stair_heights(f.env, ids, torch.tensor([.04, -.16]))

        torch.testing.assert_close(observations.env_stair_step_height(f.env), torch.tensor([.04, .16, -.16]))
        x = torch.tensor([0., .55, .85, 1.15]).repeat(3, 1)
        tops = observations.env_stair_terrain_height(f.env, x, torch.zeros_like(x))
        torch.testing.assert_close(tops, torch.tensor([[0., .04, .08, .12],
                                                      [0., .16, .32, .48],
                                                      [.48, .32, .16, 0.]]))
        torch.testing.assert_close(observations.env_stair_terrain_height(f.env, x, torch.full_like(x, 4.)),
                                   torch.zeros_like(tops))
        actual = writes[0]["body_poses"][..., 2] - f.env.scene.env_origins[ids, 2, None] + .25
        torch.testing.assert_close(actual, torch.where(tops[ids] > 0, tops[ids], -.02))
        torch.testing.assert_close(writes[0]["env_ids"], ids)

    def test_descent_last_tread_requires_a_foot_inside_the_stair_lane(self):
        move = 2 * reward.NUM_PHASES + 1
        f = Fixture(move=move)
        f.env.mixed_stair_heights = torch.tensor([-.16])
        f.put_feet(range(6), x=.85, z=.173)
        f.put_feet(reward.FRONT, x=1.15, z=.013)
        f.data.body_pos_w.torch[:, reward.FRONT, 1] += .6
        f.tick()
        self.assertEqual(f.tracker.move.item(), move)
        f.put_feet(reward.FRONT, x=1.15, z=.013)
        f.tick()
        self.assertEqual(f.tracker.move.item(), move + 1)

    def test_descending_reset_spawns_robot_on_raised_approach(self):
        f = Fixture(count=2)
        f.env.mixed_stair_heights = torch.tensor([-.16, .04])
        f.env.stair_tracker = f.tracker
        f.env.scene.env_origins[:, 2] = torch.tensor([2., 3.])
        robot = f.env.scene["robot"]
        robot.num_joints = 2
        robot.data.default_root_pose = SimpleNamespace(torch=f.data.root_pose_w.torch.clone())
        writes = []
        robot.write_root_pose_to_sim_index = lambda **kwargs: writes.append(kwargs)

        reward.reset_stair_state(f.env, torch.arange(2), ((0., 0.),), buffer_ratio=0.)

        torch.testing.assert_close(writes[0]["root_pose"][:, 2], torch.tensor([2.68, 3.2]))

    def test_pre_lift_forward_distance_is_never_paid_after_lifting(self):
        f = Fixture()
        f.tick()
        f.data.root_pose_w.torch[:, 0] = 0.3
        f.tick()
        self.assertAlmostEqual(f.tracker.progress.item(), 0.0)
        # The fixture's tilted tibias earn no lift shaping, isolating body progress.
        f.put_feet(reward.FRONT, x=0.1, z=0.20)
        f.tick()
        self.assertAlmostEqual(f.tracker.progress.item(), 0.0)
        f.data.root_pose_w.torch[:, 0] = 0.32
        f.tick()
        self.assertAlmostEqual(f.tracker.progress.item(), 0.04, places=6)

    def test_retreat_from_previously_reached_advance_target_blocks_transition(self):
        f = Fixture(move=0, hold=1.0)  # MOVE dwell must not delay an ADVANCE.
        f.data.root_pose_w.torch[:, 0] = 0.3
        f.put_feet([0], x=0.1, z=0.1)
        f.tick()
        self.assertEqual(f.tracker.move.item(), 0)
        f.data.root_pose_w.torch[:, 0] = 0.2
        f.put_feet([0], x=0.1, z=0.013)
        f.tick()
        self.assertAlmostEqual(f.tracker.best_body_x.item(), 0.3, places=6)
        self.assertEqual(f.tracker.move.item(), 0)
        f.data.root_pose_w.torch[:, 0] = 0.3
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)

    def test_airborne_support_blocks_move_even_when_near_ground(self):
        f = Fixture()
        f.put_feet(reward.FRONT)
        f.air[:, 1] = 0.02
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)
        self.assertEqual(f.tracker.standing_feet_off.item(), 1)
        f.air[:, 1] = 0.0
        f.tick()
        self.assertEqual(f.tracker.move.item(), 2)

    def test_advance_allows_near_ground_airborne_feet_without_move_dwell(self):
        f = Fixture(move=0, hold=1.0)
        f.data.root_pose_w.torch[:, 0] = 0.3
        f.air[:, :6] = 0.02
        f.forces[:, :6, 2] = 0.0
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)
        self.assertEqual(f.tracker.standing_feet_off.item(), 0)

    def test_target_contact_keeps_existing_air_time_semantics(self):
        f = Fixture()
        f.put_feet(reward.FRONT)
        f.forces[:, reward.FRONT, 0] = 5.0
        f.forces[:, reward.FRONT, 2] = 0.0
        f.tick()
        # Force direction is diagnostic only; original contact and geometry suffice.
        self.assertEqual(f.tracker.move.item(), 2)
        self.assertEqual(f.tracker.new_placed.item(), 2)
        self.assertEqual(f.tracker.new_touches.item(), 2)

    def test_both_moving_feet_need_target_geometry_and_air_time_contact(self):
        for x, z, air_time in ((0.51, 0.173, 0.0), (0.55, 0.25, 0.0), (0.55, 0.173, 0.02)):
            with self.subTest(x=x, z=z, air_time=air_time):
                f = Fixture()
                f.put_feet(reward.FRONT)
                f.put_feet([reward.FRONT[0]], x=x, z=z)
                f.air[:, reward.FRONT[0]] = air_time
                f.tick()
                self.assertEqual(f.tracker.move.item(), 1)
                self.assertEqual(f.tracker.new_placed.item(), 1)
                f.put_feet(reward.FRONT)
                f.air[:, reward.FRONT] = 0.0
                f.tick()
                self.assertEqual(f.tracker.move.item(), 2)

    def test_optional_hold_restarts_when_support_is_interrupted(self):
        f = Fixture(hold=0.06)
        f.put_feet(reward.FRONT)
        f.tick()
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)
        f.air[:, 1] = 0.02
        f.tick()
        self.assertEqual(f.tracker.move_ready_time.item(), 0.0)
        f.air[:, 1] = 0.0
        f.tick()
        f.tick()
        self.assertEqual(f.tracker.move.item(), 1)
        f.tick()
        self.assertEqual(f.tracker.move.item(), 2)

    def test_partial_reset_preserves_other_hold_and_once_per_step_guard(self):
        f = Fixture(count=2, hold=0.06)
        f.put_feet(reward.FRONT)
        f.tick()
        f.tick()
        f.tracker.reset(torch.tensor([0]), torch.tensor([1]))
        torch.testing.assert_close(f.tracker.move_ready_time, torch.tensor([0.0, 0.04]))
        f.tracker.update()  # Same global step: must not count either environment twice.
        torch.testing.assert_close(f.tracker.move_ready_time, torch.tensor([0.0, 0.04]))
        f.tick()
        self.assertEqual(f.tracker.move.tolist(), [1, 2])
        torch.testing.assert_close(f.tracker.move_ready_time, torch.tensor([0.02, 0.0]))

    def test_next_moving_pair_must_accumulate_its_own_hold(self):
        f = Fixture(hold=0.04)
        f.put_feet(reward.FRONT)
        f.tick()
        f.tick()
        self.assertEqual(f.tracker.move.item(), 2)
        self.assertEqual(f.tracker.move_ready_time.item(), 0.0)
        f.put_feet(reward.MIDDLE)
        f.tick()
        self.assertEqual(f.tracker.move.item(), 2)
        f.tick()
        self.assertEqual(f.tracker.move.item(), 3)

    def test_diagnostic_snapshot_captures_previous_phase_and_survives_reset(self):
        f = Fixture()
        f.put_feet(reward.FRONT)
        f.tracker.diagnostics_enabled = True
        f.data.root_vel_w.torch[:, 0] = 0.25
        f.history[0, 0, 6] = torch.tensor([4.0, 0.0, 0.0])
        f.history[0, 2, 6] = torch.tensor([2.0, 0.0, 0.0])
        f.tick()
        self.assertEqual(f.tracker.move.item(), 2)
        snapshot = f.tracker.diagnostics
        f.tracker.reset(torch.tensor([0]), torch.tensor([0]))
        f.data.root_vel_w.torch.zero_()
        f.forces.zero_()
        f.history.zero_()
        self.assertIs(f.tracker.diagnostics, snapshot)
        self.assertEqual(snapshot["move"].item(), 1)
        self.assertTrue(snapshot["just_done"].item())
        self.assertTrue(snapshot["move_done"].item())
        self.assertTrue(snapshot["all_kept"].item())
        self.assertAlmostEqual(snapshot["body_vx"].item(), 0.25)
        torch.testing.assert_close(snapshot["foot_force_z"], torch.full((1, 6), 2.0))
        torch.testing.assert_close(snapshot["contact_forces_w"][0, 6], torch.tensor([4.0, 0.0, 0.0]))
        self.assertEqual(f.tracker.ended_move.item(), 2)


if __name__ == "__main__":
    unittest.main()
