"""Frozen-prefix and training-only selection guards for nonlinear WSP."""
import sys
import unittest
import json
import tempfile
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import wsp_stop_nonlinear as nonlinear


def source():
    torch.manual_seed(7)
    net=torch.nn.Sequential(torch.nn.Linear(67,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),
                            torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,18))
    state={"mlp."+k:v.clone() for k,v in net.state_dict().items()}
    state.update({"obs_normalizer._mean":torch.zeros(1,67),"obs_normalizer._std":torch.ones(1,67)})
    return state


class NonlinearWspTests(unittest.TestCase):
    def test_joint_selection_permits_bounded_stop_tradeoff_but_never_guard_failure(self):
        args=argparse.Namespace(selection_objective="joint",stop_rmse_max_factor=1.15,moving_weight=20.)
        initial=dict(training_guard_passed=True,weighted_stop_target_rmse=.05,weighted_moving_target_loss=.002)
        repaired=dict(training_guard_passed=True,weighted_stop_target_rmse=.054,weighted_moving_target_loss=.0005)
        self.assertLess(nonlinear.selection_value(repaired,initial,args),nonlinear.selection_value(initial,initial,args,candidate=False))
        self.assertEqual(nonlinear.selection_value(initial,initial,args),float("inf"))
        repaired["weighted_stop_target_rmse"]=.058
        self.assertEqual(nonlinear.selection_value(repaired,initial,args),float("inf"))
        repaired["weighted_stop_target_rmse"]=.04;repaired["training_guard_passed"]=False
        self.assertEqual(nonlinear.selection_value(repaired,initial,args),float("inf"))

    def test_extra_moving_distills_source_on_student_states_and_excludes_heldout(self):
        producer="b"*64
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"forward15.npz"
            obs=np.zeros((8,1,67),dtype=np.float32);obs[...,45]=.15;obs[...,48]=.205
            obs[:,0,0]=np.arange(8)
            reference={"asset_usd_sha256":"usd","asset_manifest_sha256":"manifest",
                       "environment_source_sha256":"environment","step_dt_s":.02}
            def write():
                np.savez(path,obs=obs,terminated=np.zeros((8,1),bool),
                         checkpoint_sha256=np.array(producer),source_sha256=np.array(producer),
                         normalization_epsilon=np.array(.01),joint_names=np.array(nonlinear.base.JOINT_NAMES),
                         command_actual=obs[...,45:48],actions_raw=np.full((8,1,18),-7.))
                path.with_suffix(".json").write_text(json.dumps({"status":"completed",
                    "checkpoint_sha256":producer,"trace_sha256":nonlinear.base.sha256(path),"steps_requested":8,
                    "command":{"vx":.15,"vy":0.,"yaw_rate":0.,"height":.205},**reference}),encoding="utf-8")
            write()
            actor=lambda x:x[:,:1].expand(-1,18)
            train,test,provenance=nonlinear.load_extra_moving(path,producer,actor,8.,reference)
            np.testing.assert_array_equal(train["target"][:,0],np.arange(6))
            np.testing.assert_array_equal(test["target"][:,0],np.arange(6,8))
            np.testing.assert_array_equal(train["raw"],train["target"])
            self.assertTrue(train["moving"]);self.assertFalse(provenance["source_action_replay_asserted"])
            self.assertEqual(provenance["recording_policy_sha256"],producer)
            obs[-1,0,45]=0.;write()
            with self.assertRaises(ValueError):nonlinear.load_extra_moving(path,producer,actor,8.,reference)

    def test_broad_stop_neighborhood_preserves_moving_rows_and_braking_labels(self):
        kin=nonlinear.base.Kinematics(nonlinear.base.ROOT/"src/RLforHexapod/assets/urdf/robot.urdf")
        actor=lambda x:torch.zeros((len(x),18))
        args=argparse.Namespace(seed=999,stop_strength=1.,stop_weight=1.,moving_weight=1.,stand_velocity_damping=.01)
        legacy=nonlinear.base.synthetic_anchors(actor,kin,args,{})
        anchors={}
        broad=nonlinear.synthetic_stop_neighborhoods(actor,kin,args,anchors,[.1,1.,.08,.1,.02],4.,256)
        for before,after in zip(legacy,broad):
            if after["moving"]:
                for key in ("obs","raw","target"):np.testing.assert_array_equal(before[key],after[key])
                self.assertEqual(before["weight"],after["weight"])
            else:
                obs=after["obs"];self.assertEqual(len(obs),256)
                np.testing.assert_array_equal(obs[:,45:48],0.)
                np.testing.assert_array_equal(obs[0,18:42],0.)
                np.testing.assert_allclose(after["target"],nonlinear.base.stop_targets(obs,anchors,.01))
                self.assertTrue(np.isfinite(obs).all());self.assertLessEqual(np.abs(obs[:,49:67]).max(),1.)
                self.assertGreater(obs[1:,18:36].std(),.9)
                self.assertAlmostEqual(after["weight"],before["weight"]*4.)

    def test_extra_stop_records_its_true_producer_and_rejects_moving_commands(self):
        kin=nonlinear.base.Kinematics(nonlinear.base.ROOT/"src/RLforHexapod/assets/urdf/robot.urdf")
        producer="a"*64
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"stand.npz"
            obs=np.zeros((8,1,67),dtype=np.float32);obs[...,48]=.205
            reference={"asset_usd_sha256":"usd","asset_manifest_sha256":"manifest",
                       "environment_source_sha256":"environment","step_dt_s":.02}
            def write():
                np.savez(path,obs=obs,terminated=np.zeros((8,1),bool),
                         checkpoint_sha256=np.array(producer),source_sha256=np.array(producer),
                         normalization_epsilon=np.array(.01),joint_names=np.array(nonlinear.base.JOINT_NAMES),
                         actions_raw=np.full((8,1,18),7.))  # Deliberately NOT source999 actions.
                report={"status":"completed","checkpoint_sha256":producer,
                        "trace_sha256":nonlinear.base.sha256(path),"steps_requested":8,
                        "command":{"vx":0.,"vy":0.,"yaw_rate":0.,"height":.205},**reference}
                path.with_suffix(".json").write_text(json.dumps(report),encoding="utf-8")
            write()
            actor=lambda x:torch.zeros((len(x),18))
            train,test,provenance=nonlinear.load_extra_stop(path,producer,actor,kin,{},4.,reference)
            self.assertEqual(len(train["obs"]),6);self.assertEqual(len(test["obs"]),2)
            self.assertEqual(provenance["recording_policy_sha256"],producer)
            self.assertFalse(provenance["source_action_replay_asserted"])
            np.testing.assert_array_equal(train["raw"],0.)
            with self.assertRaises(ValueError):nonlinear.load_extra_stop(path,"b"*64,actor,kin,{},4.,reference)
            obs[0,0,45]=.15;write()
            with self.assertRaises(ValueError):nonlinear.load_extra_stop(path,producer,actor,kin,{},4.,reference)

    def test_moving_damping_targets_do_not_replace_source_deviation_guards(self):
        source=torch.zeros(2,18);target=torch.full((2,18),.5)
        fit,guard=nonlinear.moving_objective(target,source,target,.1)
        self.assertEqual(fit.item(),0.)
        self.assertGreater(guard.item(),0.)

    def test_cached_prefix_tail_reproduces_source_and_export_only_changes_allowed_tensors(self):
        torch.set_num_threads(1);s=source();actor=nonlinear.base.FlatActor(s)
        obs=np.random.default_rng(7).normal(size=(20,67)).astype(np.float32)
        prefix=nonlinear.prefix_features(actor,obs);model=nonlinear.EditableTail(s)
        with torch.inference_mode():
            torch.testing.assert_close(model(prefix),actor(torch.from_numpy(obs)),atol=1e-7,rtol=1e-5)
        with torch.no_grad():model.hidden.bias.add_(.01);model.head.weight.mul_(.99)
        edited=model.export(s)
        for key in s:
            if key not in nonlinear.EDIT_KEYS:self.assertTrue(torch.equal(s[key],edited[key]))
        self.assertFalse(torch.equal(s["mlp.4.bias"],edited["mlp.4.bias"]))

    def test_guard_rejects_moving_distortion_even_if_stop_error_improves(self):
        s=source();actor=nonlinear.base.FlatActor(s);model=nonlinear.EditableTail(s)
        obs=np.random.default_rng(8).normal(size=(20,67)).astype(np.float32);raw=nonlinear.base.outputs(actor,obs)
        blocks=[dict(name="move",obs=obs[:10],raw=raw[:10],target=raw[:10],moving=True,weight=1.),
                dict(name="stop",obs=obs[10:],raw=raw[10:],target=raw[10:]+.5,moving=False,weight=1.)]
        d=nonlinear.prepare(s,blocks);before=nonlinear.training_score(model,d,.1)
        self.assertFalse(d["prefix"].is_inference())
        model(d["prefix"]).square().mean().backward()
        self.assertIsNotNone(model.hidden.weight.grad)
        with torch.no_grad():model.head.bias.add_(.5)
        after=nonlinear.training_score(model,d,.1)
        self.assertTrue(before["training_guard_passed"])
        self.assertFalse(after["training_guard_passed"])
        self.assertLess(after["weighted_stop_target_rmse"],before["weighted_stop_target_rmse"])


if __name__=="__main__":unittest.main()
