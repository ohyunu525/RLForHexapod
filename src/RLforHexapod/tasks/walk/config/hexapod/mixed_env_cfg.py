"""One privileged 95-input actor, fixed flat/stair cohorts in identical worlds.

Each world contains the same staircase. Flat robots use a distant floor lane;
this is not automatic terrain detection or an evaluated flat-to-stair transition.
The initial training stage is 40mm. Height changes are explicit new stages,
never an undocumented automatic curriculum. Old task configurations are intact.
"""
from copy import deepcopy
import math

from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import RLforHexapod.tasks.walk.mdp as mdp
from RLforHexapod.assets import OBSTACLES_DIR
from .env_cfg import (HexapodEnvCfg, ObservationsCfg, StairsSceneCfg, EventsCfg,
                      StairsRewardsCfg, TerminationsCfg, StairsTerminationsCfg)
from .walk_wsp_stop_env_cfg import WspStopRewardsCfg, WspStopCommandsCfg

MIXED_OBSERVATION_SIZE=95
MIXED_POLICY_TERMS=("joint_pos","joint_vel","base_lin_vel","base_ang_vel","projected_gravity",
                    "command","height_command","last_action","stair_context")
MIXED_CONTEXT_TERMS=(("feet_height",6),("terrain_under_feet",6),("step_height",1),
                     ("phase",6),("feet_target",6),("body_target",3))
SUPPORTED_STAIR_HEIGHTS=(.04,.08,.12,.16)


@configclass
class MixedObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        stair_context=ObsTerm(func=mdp.mixed_stair_context)
    policy:PolicyCfg=PolicyCfg()


@configclass
class MixedCommandsCfg(WspStopCommandsCfg):
    base_velocity=deepcopy(WspStopCommandsCfg().base_velocity)
    base_velocity.class_type=mdp.MixedVelocityCommand
    base_height=deepcopy(WspStopCommandsCfg().base_height)
    base_height.class_type=mdp.MixedHeightCommand


@configclass
class MixedEventsCfg(EventsCfg):
    mixed_reset=EventTerm(func=mdp.mixed_reset,mode="reset",
        params={"flat_lane_y":4.,"stair_x_ranges":((-0.1,0.1),(0.30,0.32)),"buffer_ratio":.5})


def _stairs_rewards():
    cfg=StairsRewardsCfg()
    cfg.termination_penalty.params["term_keys"]=["stair_base_low","stair_bad_ori","stair_off_stairs",
                                                "stair_sim_blowup","sim_nan"]
    return cfg


@configclass
class MixedRewardsCfg:
    # All original source weights are inside these bundles, applied once.
    # .1 preserves stair-internal ratios; it does not guarantee balanced PPO gradients.
    flat=RewTerm(func=mdp.mixed_reward_bundle,weight=1.,
                 params={"group":"flat","source_rewards":WspStopRewardsCfg()})
    stair=RewTerm(func=mdp.mixed_reward_bundle,weight=.1,
                  params={"group":"stair","source_rewards":_stairs_rewards()})


def _done(source,group):
    return DoneTerm(func=mdp.mixed_termination,time_out=source.time_out,
                    params={"group":group,"function":source.func,"parameters":deepcopy(source.params)})


@configclass
class MixedTerminationsCfg:
    time_out=deepcopy(TerminationsCfg().time_out)  # Global stair horizon45s.
    flat_time_out=DoneTerm(func=mdp.mixed_flat_timeout,time_out=True,params={"seconds":20.})
    flat_lane_boundary=DoneTerm(func=mdp.mixed_flat_boundary,time_out=True,
                               params={"x_limit":4.,"y_min":2.5,"y_max":5.5})
    flat_base_low=_done(TerminationsCfg().base_low,"flat")
    flat_bad_ori=_done(TerminationsCfg().bad_ori,"flat")
    stair_base_low=DoneTerm(func=mdp.mixed_stair_base_low,params={"minimum_height":.05})
    stair_bad_ori=_done(StairsTerminationsCfg().bad_ori,"stair")
    stair_off_stairs=_done(StairsTerminationsCfg().off_stairs,"stair")
    stair_sim_blowup=_done(StairsTerminationsCfg().sim_blowup,"stair")
    stair_success=_done(StairsTerminationsCfg().success,"stair")
    sim_nan=deepcopy(TerminationsCfg().sim_nan)


@configclass
class HexapodMixedEnvCfg(HexapodEnvCfg):
    scene:StairsSceneCfg=StairsSceneCfg(num_envs=4,env_spacing=12.)
    observations:MixedObservationsCfg=MixedObservationsCfg()
    events:MixedEventsCfg=MixedEventsCfg()
    rewards:MixedRewardsCfg=MixedRewardsCfg()
    terminations:MixedTerminationsCfg=MixedTerminationsCfg()
    commands:MixedCommandsCfg=MixedCommandsCfg()
    stairs:mdp.StairsGeometryCfg=mdp.StairsGeometryCfg(x0=.5,step_height=.04)
    stair_gait:mdp.StairGaitCfg=mdp.StairGaitCfg()
    flat_fraction:float=.5
    stair_height:float=.04
    stair_reward_scale:float=.1

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s=45.
        self.configure_mixed_terrain()

    def configure_mixed_terrain(self,flat_fraction=None,stair_height=None,stair_reward_scale=None):
        """Call after CLI overrides and before creating the environment."""
        fraction=self.flat_fraction if flat_fraction is None else float(flat_fraction)
        height=self.stair_height if stair_height is None else float(stair_height)
        scale=self.stair_reward_scale if stair_reward_scale is None else float(stair_reward_scale)
        if not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError("flat_fraction outside[0,1]")
        if not any(abs(height-h)<1e-9 for h in SUPPORTED_STAIR_HEIGHTS):raise ValueError("Unsupported fixed stair height")
        if not math.isfinite(scale) or scale<=0:raise ValueError("stair_reward_scale must be positive")
        self.flat_fraction,self.stair_height,self.stair_reward_scale=fraction,height,scale
        self.stairs.step_height=height
        self.scene.stairs.spawn.usd_path=str(OBSTACLES_DIR/f"stairs_{round(height*1000)}mm.usda")
        self.scene.stairs.init_state.pos=(self.stairs.x0,0.,0.)
        self.scene.env_spacing=12.
        self.rewards.stair.weight=scale
        return self
