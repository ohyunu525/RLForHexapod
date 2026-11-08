"""Offline WSP safeguards and URDF geometry, without claims about learned gait."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import wsp_stop_refine as wsp


class WspStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.kin = wsp.Kinematics(ROOT/"src/RLforHexapod/assets/urdf/robot.urdf")

    def test_analytical_jacobian_matches_finite_differences_for_all_mounts(self):
        q = np.deg2rad([[5., 65., -140.], [-8., 47., -130.]])
        for leg in range(6):
            _, jac = self.kin.fk_jacobian(leg, q)
            for joint in range(3):
                d = np.zeros_like(q); d[:, joint] = 1e-6
                finite = (self.kin.fk_jacobian(leg, q+d)[0]-self.kin.fk_jacobian(leg, q-d)[0])/2e-6
                np.testing.assert_allclose(jac[:, :, joint], finite, atol=1e-8, rtol=0)

    def test_entire_height_grid_is_reachable_with_unclipped_standing_targets(self):
        for height in np.linspace(.17, .24, 29):
            q, report = self.kin.anchor(height)
            self.assertLess(np.max(np.abs((q-wsp.DEFAULT)/wsp.SCALE)), .98)
            tips = np.asarray(report["tip_positions_body_m"])
            np.testing.assert_allclose(tips[:, 2]+height, .012, atol=1e-7)
            self.assertTrue(np.isfinite(report["foot_load_only_joint_torque_estimate_Nm"]).all())
        with self.assertRaises(ValueError): self.kin.anchor(.15)

    def test_yaw_correction_sign_and_stance_mask(self):
        q, _ = self.kin.anchor(.205)
        obs = np.zeros((2, 67)); obs[:, :18] = q-wsp.DEFAULT
        obs[:, 41] = .2
        quat = np.tile([0.,0.,0.,1.], (2,1))
        support = np.ones((2,6), bool); support[1] = False
        delta = wsp.damping_delta(self.kin, obs, quat, support, .05, 0.)
        np.testing.assert_array_equal(delta[1], np.zeros(18))
        for leg in range(6):
            sl = slice(3*leg, 3*leg+3)
            tip,jac = self.kin.fk_jacobian(leg, q[sl])
            dp = jac[0]@(delta[0,sl]*wsp.SCALE[sl])
            self.assertGreater(dp@np.array([-tip[0,1], tip[0,0], 0.]), 0.)
        obs[:,47] = .2  # Correct commanded turn must not receive yaw damping.
        np.testing.assert_allclose(wsp.damping_delta(self.kin,obs,quat,support,.05,0.), 0., atol=1e-12)

    def test_vertical_feedback_uses_world_up_even_for_tilted_body(self):
        q, _ = self.kin.anchor(.205)
        quat = np.array([[np.sin(.2),0.,0.,np.cos(.2)]])
        up_b = wsp.world_up_in_body(quat)
        obs = np.zeros((1,67)); obs[:, :18] = q-wsp.DEFAULT
        obs[:,36:39] = .05*up_b
        delta = wsp.damping_delta(self.kin,obs,quat,np.ones((1,6),bool),0.,.05)
        for leg in range(6):
            sl = slice(3*leg, 3*leg+3)
            _,jac = self.kin.fk_jacobian(leg,q[sl])
            dp = jac[0]@(delta[0,sl]*wsp.SCALE[sl])
            self.assertGreater(dp@up_b[0],0.)
            self.assertLess(np.linalg.norm(dp),.0021)

    def test_stop_target_has_no_cycle_and_brakes_both_velocity_signs(self):
        q,_ = self.kin.anchor(.205)
        obs = np.zeros((3,67)); obs[:,48]=.205
        obs[:,18:36] = np.array([0.,3.,-3.])[:,None]
        result = wsp.stop_targets(obs,{.205:q},.01)
        np.testing.assert_allclose(result[0],(q-wsp.DEFAULT)/wsp.SCALE)
        self.assertTrue((result[1]<result[0]).all())
        self.assertTrue((result[2]>result[0]).all())

    def test_reflection_is_an_involution_in_raw_coordinates(self):
        raw=np.arange(134,dtype=float).reshape(2,67)
        np.testing.assert_array_equal(wsp.mirror_obs(wsp.mirror_obs(raw)),raw)

    def test_protected_head_fit_does_not_change_features_or_normalizer(self):
        torch.manual_seed(3)
        network=torch.nn.Sequential(torch.nn.Linear(67,256),torch.nn.ELU(),torch.nn.Linear(256,256),
                                    torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,18))
        state={"mlp."+k:v for k,v in network.state_dict().items()}
        state.update({"obs_normalizer._mean":torch.zeros(1,67),"obs_normalizer._std":torch.ones(1,67)})
        actor=wsp.FlatActor(state);rng=np.random.default_rng(3);obs=rng.normal(0,.2,(200,67));raw=wsp.outputs(actor,obs)
        blocks=[dict(name="move",obs=obs[:100],raw=raw[:100],target=raw[:100],moving=True,weight=1.),
                dict(name="stop",obs=obs[100:],raw=raw[100:],target=raw[100:]+.5,moving=False,weight=4.)]
        edited,fit=wsp.fit_head(state,blocks,.001,.08)
        rows,ok=wsp.evaluate_blocks(state,edited,blocks,.08)
        self.assertTrue(ok)
        self.assertFalse(fit["heldout_used_for_fit_or_backoff"])
        self.assertLessEqual(rows[0]["applied_delta_max"],.08)
        self.assertLessEqual(rows[0]["raw_delta_rms"],.08/3)
        for key in state:
            if key not in wsp.HEAD: self.assertTrue(torch.equal(state[key],edited[key]))


if __name__ == "__main__": unittest.main()
