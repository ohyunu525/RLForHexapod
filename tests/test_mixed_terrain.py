"""Cohort isolation and real StairTracker checks without an Isaac runtime."""
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

import torch
from test_stair_tracker import Fixture,observations,reward

ROOT=Path(__file__).resolve().parents[1]
MDP=ROOT/"src/RLforHexapod/tasks/walk/mdp"


def load_mixed():
    names=("isaaclab","isaaclab.envs","isaaclab.envs.mdp","isaaclab.managers","isaaclab.utils",
           "isaaclab.utils.math","_mixed_unit","_mixed_unit.commands")
    stubs={name:ModuleType(name) for name in names}
    stubs["_mixed_unit"].__path__=[str(MDP)]
    stubs["_mixed_unit.observations"]=observations
    stubs["_mixed_unit.stair_reward"]=reward
    class Term:
        def __init__(self,cfg,env):self.cfg,self._env=cfg,env
        def reset(self,env_ids=None):pass
    class Entity:
        def __init__(self):self.resolved=False
        def resolve(self,scene):self.resolved=True
    class Velocity:
        def _resample_command(self,ids):self.command[ids]=-.33
        def _update_command(self):self.command[self.is_standing_env]=0.
    class Height:
        def _resample_command(self,ids):self.height_command[ids]=.19
    stubs["isaaclab.managers"].ManagerTermBase=Term
    stubs["isaaclab.managers"].SceneEntityCfg=Entity
    stubs["isaaclab.envs.mdp"].UniformVelocityCommand=Velocity
    stubs["_mixed_unit.commands"].BaseHeightCommand=Height
    stubs["isaaclab.utils.math"].quat_apply=lambda q,v:v
    with patch.dict(sys.modules,stubs):
        spec=importlib.util.spec_from_file_location("_mixed_unit.mixed_terrain",MDP/"mixed_terrain.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,Term,Entity


mixed,Term,Entity=load_mixed()


def fixture(fraction=.5):
    f=Fixture(count=4,move=0)
    f.env.cfg.flat_fraction=fraction
    mixed.mixed_masks(f.env)
    f.tracker=reward.StairTracker(f.env)
    f.env.stair_tracker=f.tracker
    f.env.extras={"log":{}}
    f.env.command_manager=SimpleNamespace(get_command=lambda _:torch.zeros(4,3))
    f.env.termination_manager=SimpleNamespace(terminated=torch.zeros(4,dtype=torch.bool),
                                            time_outs=torch.zeros(4,dtype=torch.bool))
    f.data.root_pose_w.torch[:2,1]=4.
    f.data.root_pos_w=SimpleNamespace(torch=f.data.root_pose_w.torch[:,:3])
    f.env.scene["robot"].body_names=reward.FOOT_BODIES
    f.put_feet(range(6),x=.1,z=.013)
    return f


class MixedTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_cohorts_and_stair_tracker_records_never_accept_flat_envs(self):
        f=fixture()
        f.data.root_pose_w.torch[:,0]=.3
        for _ in range(2):f.tick()
        self.assertEqual(f.tracker.move.tolist(),[0,0,1,1])
        self.assertEqual(f.tracker.best_body_x[:2].tolist(),[0.,0.])
        self.assertTrue((f.tracker.best_z[:2]==0).all())
        self.assertTrue((f.tracker.vertical_score[:2]==0).all())
        self.assertTrue((f.env.stair_states[1][:,1]==0).all())
        reward.save_stair_states(f.env,torch.arange(4),torch.tensor([99,99,2,2]))
        self.assertNotIn(99,f.env.stair_states)
        f.tracker.reset(torch.arange(4),torch.full((4,),5))
        self.assertEqual(f.tracker.move.tolist(),[0,0,5,5])

    def test_commands_remain_coherent_after_resampling_and_settle(self):
        f=fixture();c=mixed.MixedVelocityCommand()
        c._env=f.env;c.command=torch.zeros(4,3);c.command[:2]=.2
        c.is_standing_env=torch.ones(4,dtype=torch.bool);c.is_heading_env=torch.ones(4,dtype=torch.bool)
        c._resample_command(torch.arange(4))
        torch.testing.assert_close(c.command[:2],torch.full((2,3),-.33))
        torch.testing.assert_close(c.command[2:],torch.tensor([[.1,0,0],[.1,0,0]]))
        f.tracker.move[3]=f.tracker.settle_move;c._update_command()
        torch.testing.assert_close(c.command[3],torch.zeros(3))
        self.assertAlmostEqual(c.command[2,0].item(),.1,places=6)
        h=mixed.MixedHeightCommand();h._env=f.env;h.height_command=torch.zeros(4,1)
        h._resample_command(torch.arange(4))
        torch.testing.assert_close(h.height_command[:,0],torch.tensor([.19,.19,.205,.205]))

    def test_flat_context_is_zero_even_if_inactive_context_contains_nan(self):
        f=fixture();term=mixed.mixed_stair_context(SimpleNamespace(),f.env)
        f.data.body_pos_w.torch[:2]=float("nan")
        result=term(f.env)
        self.assertEqual(tuple(result.shape),(4,28))
        self.assertTrue((result[:2]==0).all());self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[2:,12],torch.full((2,),.16))
        f.data.root_vel_w.torch[:2]=float("nan")
        f.tick()
        self.assertTrue((f.tracker.progress[:2]==0).all())
        self.assertTrue((f.tracker.settle_score[:2]==0).all())
        self.assertTrue((f.tracker.best_z[:2]==0).all())

    def test_lane_bounds_are_local_and_do_not_timeout_stairs(self):
        f=fixture();f.env.scene.env_origins[:,1]=torch.tensor([12.,24.,36.,48.])
        p=f.data.root_pose_w.torch;p[:,:3]+=f.env.scene.env_origins
        p[0,1]=f.env.scene.env_origins[0,1]+2.49
        p[2,1]=f.env.scene.env_origins[2,1]+2.49
        self.assertEqual(mixed.mixed_flat_boundary(f.env).tolist(),[True,False,False,False])
        f.env.episode_length_buf=torch.tensor([1000,999,2250,2250])
        self.assertEqual(mixed.mixed_flat_timeout(f.env).tolist(),[True,False,False,False])

    def test_termination_cross_mask_and_empty_cohort_reset(self):
        f=fixture()
        function=lambda env:torch.ones(4,dtype=torch.bool)
        self.assertEqual(mixed.mixed_termination(f.env,"flat",function,{}).tolist(),[True,True,False,False])
        self.assertEqual(mixed.mixed_termination(f.env,"stair",function,{}).tolist(),[False,False,True,True])
        f=fixture(0.)
        source=SimpleNamespace(value=SimpleNamespace(func=lambda env:torch.ones(4),weight=1.,params={}))
        cfg=SimpleNamespace(params={"group":"flat","source_rewards":source},weight=1.)
        term=mixed.mixed_reward_bundle(cfg,f.env);term.reset(torch.arange(4))
        self.assertEqual(f.env.extras["log"],{})

    def test_reward_mask_scaling_nested_entities_and_filtered_reset(self):
        f=fixture();resets=[]
        class Stateful(Term):
            def __init__(self,cfg,env):
                super().__init__(cfg,env)
                if not cfg.params["asset_cfg"].resolved:raise AssertionError("Entity was not resolved")
            def __call__(self,env,asset_cfg):return torch.tensor([1.,2.,float("nan"),9.])
            def reset(self,ids):resets.append(ids.tolist())
        source=SimpleNamespace(test=SimpleNamespace(func=Stateful,weight=2.,params={"asset_cfg":Entity()}))
        cfg=SimpleNamespace(params={"group":"flat","source_rewards":source},weight=.1)
        term=mixed.mixed_reward_bundle(cfg,f.env)
        torch.testing.assert_close(term(f.env,source,"flat"),torch.tensor([2.,4.,0.,0.]))
        term.reset(torch.tensor([0,2,3]))
        self.assertEqual(resets,[[0]])
        self.assertAlmostEqual(f.env.extras["log"]["Metrics/Mixed/flat_reward_per_s"].item(),.2,places=6)
        term.reset(torch.tensor([2,3]));self.assertEqual(resets,[[0]])
        self.assertFalse(source.test.params["asset_cfg"].resolved)

    def test_pure_group_modes(self):
        for fraction,expected in ((0.,0),(1.,4),(.49,1)):
            f=fixture(fraction);self.assertEqual(int(f.env.mixed_flat_mask.sum()),expected)

    def test_fixed_stair_usd_and_analytic_surface_match(self):
        geometry=observations.StairsGeometryCfg()
        for height in (.04,.08,.12,.16):
            path=ROOT/f"src/RLforHexapod/assets/obstacles/stairs_{round(height*1000)}mm.usda"
            text=path.read_text()
            scales=[tuple(map(float,v.split(','))) for v in re.findall(r'xformOp:scale = \(([^)]+)\)',text)]
            positions=[tuple(map(float,v.split(','))) for v in re.findall(r'xformOp:translate = \(([^)]+)\)',text)]
            geometry.step_height=height
            for i,(scale,pos) in enumerate(zip(scales,positions)):
                x=torch.tensor([geometry.x0+pos[0]]);y=torch.tensor([0.])
                analytical=observations.stairs_terrain_height(geometry,x,y).item()
                self.assertAlmostEqual(analytical,pos[2]+scale[2]/2,places=6)
            self.assertLessEqual(max(s[1]/2 for s in scales),.5)
            # Even a conservative one-metre radial envelope leaves one metre
            # between the minimum allowed flat body y and the stair side.
            self.assertGreaterEqual(2.5-1.-.5,1.)


if __name__=="__main__":unittest.main()
