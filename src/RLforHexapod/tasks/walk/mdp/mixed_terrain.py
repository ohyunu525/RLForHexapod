"""Opt-in fixed-cohort flat/stair task helpers; no runtime action override."""
from copy import deepcopy
import inspect
import math

import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .commands import BaseHeightCommand
from .mixed_stair_geometry import MAX_RANDOM_STAIR_HEIGHT, set_mixed_stair_heights
from .observations import env_stair_step_height, env_stair_terrain_height, tip_offsets
from .stair_reward import get_tracker, reset_stair_state, stair_phase_obs, stair_feet_target_obs, stair_body_target_obs


MIXED_CURRICULUM_HEIGHTS = (0.04, 0.06, 0.08, 0.12, 0.16)
CURRICULUM_WINDOW = 100
CURRICULUM_MIN_SUCCESSES = 80


class MixedStairCurriculum:
    """Evaluate batches enrolled at episode start so fast successes cannot bias promotion."""

    def __init__(self, height, state=None):
        self.stage = MIXED_CURRICULUM_HEIGHTS.index(height)
        self.outcomes = []
        self.pending = 0
        self.last_success_rate = None
        if state is not None:
            if not isinstance(state, dict):
                raise ValueError("Invalid mixed stair curriculum checkpoint state")
            stage = state.get("stage")
            outcomes = state.get("outcomes")
            if (type(stage) is not int or not 0 <= stage < len(MIXED_CURRICULUM_HEIGHTS)
                    or not isinstance(outcomes, list) or len(outcomes) >= CURRICULUM_WINDOW
                    or any(type(outcome) is not bool for outcome in outcomes)):
                raise ValueError("Invalid mixed stair curriculum checkpoint state")
            self.stage = stage
            self.outcomes.extend(outcomes)

    @property
    def height(self):
        return MIXED_CURRICULUM_HEIGHTS[self.stage]

    @property
    def available_trials(self):
        return CURRICULUM_WINDOW - len(self.outcomes) - self.pending

    def record(self, outcomes, ended_count):
        self.pending -= ended_count
        self.outcomes.extend(outcomes)
        if len(self.outcomes) == CURRICULUM_WINDOW:
            successes = sum(self.outcomes)
            self.last_success_rate = successes / CURRICULUM_WINDOW
            if successes >= CURRICULUM_MIN_SUCCESSES and self.stage < len(MIXED_CURRICULUM_HEIGHTS) - 1:
                self.stage += 1
            self.outcomes.clear()


def _mixed_curriculum(env):
    if not getattr(env.cfg, "height_curriculum", False):
        return None
    if not hasattr(env, "mixed_stair_curriculum"):
        env.mixed_stair_curriculum = MixedStairCurriculum(
            env.cfg.stair_height, getattr(env.cfg, "stair_curriculum_state", None)
        )
    return env.mixed_stair_curriculum


def mixed_curriculum_state(env):
    """Return checkpoint-safe stage and window data for the next reset."""
    curriculum = _mixed_curriculum(env)
    if curriculum is None:
        return None
    return {"stage": curriculum.stage, "outcomes": list(curriculum.outcomes)}


def _reset_mixed_stair_heights(env, stair_ids):
    curriculum = _mixed_curriculum(env)
    randomize = getattr(env.cfg, "random_stair_height", False)
    if curriculum is None:
        heights = torch.empty(len(stair_ids), device=env.device).uniform_(
            -MAX_RANDOM_STAIR_HEIGHT, MAX_RANDOM_STAIR_HEIGHT
        )
    else:
        if not hasattr(env, "mixed_stair_trial"):
            env.mixed_stair_trial = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        trials = env.mixed_stair_trial
        ended = stair_ids[trials[stair_ids]]
        if len(ended):
            # The visualizer can mark manually reset rows as terminated after
            # Isaac Lab builds its automatic-reset mask. Only use that mask.
            done = (env.episode_length_buf[ended] > 0) & env.reset_buf[ended]
            completed = ended[done]
            manager = env.termination_manager
            success = manager.get_term("stair_success")[completed] & ~manager.terminated[completed]
            curriculum.record(success.tolist(), len(ended))
            trials[ended] = False

        heights = torch.full((len(stair_ids),), curriculum.height, device=env.device)
        fixed_ids = stair_ids
        if randomize:
            randomized = torch.rand(len(stair_ids), device=env.device) < 0.5
            sampled = torch.empty_like(heights).uniform_(-MAX_RANDOM_STAIR_HEIGHT, MAX_RANDOM_STAIR_HEIGHT)
            heights = torch.where(randomized, sampled, heights)
            fixed_ids = stair_ids[~randomized]
        available = curriculum.available_trials
        if available and len(fixed_ids):
            if len(fixed_ids) > available:
                sampled_ids = torch.randperm(len(fixed_ids), device=env.device)[:available]
                fixed_ids = fixed_ids[sampled_ids]
            trials[fixed_ids] = True
            curriculum.pending += len(fixed_ids)
    set_mixed_stair_heights(env, stair_ids, heights)


def mixed_masks(env):
    """Deterministic first-floor(N*fraction) flat cohort, immutable for a run."""
    if not hasattr(env, "mixed_flat_mask"):
        fraction=float(env.cfg.flat_fraction)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("flat_fraction must be finite in [0,1]")
        count=math.floor(env.num_envs*fraction)
        env.mixed_flat_mask=torch.arange(env.num_envs,device=env.device)<count
        env.mixed_stair_mask=~env.mixed_flat_mask
        env.stair_active_mask=env.mixed_stair_mask
    return env.mixed_flat_mask,env.mixed_stair_mask


def mixed_ids(env, env_ids, group):
    mask=mixed_masks(env)[0 if group=="flat" else 1]
    if group not in ("flat","stair"):raise ValueError("Unknown mixed group")
    ids=torch.arange(env.num_envs,device=env.device)
    if env_ids is not None:ids=ids[env_ids]
    return ids[mask[ids]]


def mixed_reset(env, env_ids, flat_lane_y=4., stair_x_ranges=((-0.1,0.1),(0.30,0.32)), buffer_ratio=.5):
    """Run after normal scene reset; only stair IDs enter stair reset/buffers."""
    flat_ids=mixed_ids(env,env_ids,"flat")
    stair_ids=mixed_ids(env,env_ids,"stair")
    asset=env.scene["robot"]
    if len(flat_ids):
        pose=asset.data.default_root_pose.torch[flat_ids].clone()
        pose[:,:3]+=env.scene.env_origins[flat_ids]
        pose[:,1]+=flat_lane_y
        asset.write_root_pose_to_sim_index(root_pose=pose,env_ids=flat_ids)
    # Initialize tracker even in pure-flat validation; active mask prevents all
    # flat progress, buffer writes and stair statistics.
    get_tracker(env)
    if len(stair_ids):
        dynamic = getattr(env.cfg, "height_curriculum", False) or getattr(env.cfg, "random_stair_height", False)
        if dynamic:
            _reset_mixed_stair_heights(env, stair_ids)
            # Replay states belong to a particular geometry and skip some of
            # the course. Dynamic episodes always start at the first approach.
            reset_stair_state(env, stair_ids, stair_x_ranges[:1], buffer_ratio=0.0)
        else:
            reset_stair_state(env, stair_ids, stair_x_ranges, buffer_ratio)


class MixedVelocityCommand(UniformVelocityCommand):
    def _coherent_stair_commands(self):
        _,stair=mixed_masks(self._env)
        tracker=get_tracker(self._env)
        climbing=stair & (tracker.move<tracker.settle_move)
        self.command[stair]=0.
        self.command[climbing,0]=.10
        # The superclass samples these flags; they cannot override this cohort.
        if hasattr(self,"is_standing_env"):self.is_standing_env[stair]=False
        if hasattr(self,"is_heading_env"):self.is_heading_env[stair]=False

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        self._coherent_stair_commands()

    def _update_command(self):
        super()._update_command()
        self._coherent_stair_commands()


class MixedHeightCommand(BaseHeightCommand):
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        self.height_command[mixed_masks(self._env)[1],0]=.205

    def _update_metrics(self):
        pose=self.robot.data.root_pos_w.torch-self._env.scene.env_origins
        terrain=env_stair_terrain_height(self._env,pose[:,0],pose[:,1])
        self._error_sum+=(self.height_command[:,0]-(pose[:,2]-terrain)).abs()
        self._step_count+=1.


class mixed_stair_context(ManagerTermBase):
    """28 legacy teacher features, with literal finite zeros for flat rows."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        mixed_masks(env)
        self.asset=env.scene["robot"]
        self.ids,_=self.asset.find_bodies("tibia.*")
        if len(self.ids)!=6:raise ValueError("Mixed context requires six tibia bodies")
        self.offsets=tip_offsets([self.asset.body_names[i] for i in self.ids],env.device)

    def __call__(self,env):
        d=self.asset.data
        quat=d.body_quat_w.torch[:,self.ids]
        tips=d.body_pos_w.torch[:,self.ids]+quat_apply(quat,self.offsets.expand(env.num_envs,-1,-1))
        tips=tips-env.scene.env_origins[:,None,:]
        ground=env_stair_terrain_height(env,tips[...,0],tips[...,1])
        body_z=d.root_pos_w.torch[:,2:3]-env.scene.env_origins[:,2:3]
        height=env_stair_step_height(env).unsqueeze(1)
        context=torch.cat((tips[...,2]-ground,body_z-ground,height,
                           stair_phase_obs(env),stair_feet_target_obs(env),stair_body_target_obs(env)),dim=1)
        if context.shape!=(env.num_envs,28):raise ValueError("Mixed stair context must have 28 columns")
        return torch.where(mixed_masks(env)[1][:,None],context,torch.zeros_like(context))


def _resolve_entities(value,scene):
    if isinstance(value,SceneEntityCfg):value.resolve(scene)
    elif isinstance(value,dict):
        for item in value.values():_resolve_entities(item,scene)
    elif isinstance(value,(list,tuple)):
        for item in value:_resolve_entities(item,scene)


class mixed_reward_bundle(ManagerTermBase):
    """Reuse existing reward definitions, with cohort-filtered stateful resets.

    The outer RewardManager multiplies the returned weighted rate by step_dt.
    Source term weights are applied once here; the outer scale is flat1/stair.1.
    """
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.group=cfg.params["group"]
        self.mask=mixed_masks(env)[0 if self.group=="flat" else 1]
        if self.group not in ("flat","stair"):raise ValueError("Unknown mixed reward group")
        self.terms=[]
        source=deepcopy(cfg.params["source_rewards"])
        for name,term in vars(source).items():
            if name.startswith("_") or term is None or not hasattr(term,"weight") or term.weight==0:continue
            _resolve_entities(term.params,env.scene)
            function=term.func(term,env) if inspect.isclass(term.func) and issubclass(term.func,ManagerTermBase) else term.func
            self.terms.append((name,term,function))
        if not self.terms:raise ValueError("Mixed reward bundle has no source terms")
        self.total=torch.zeros(env.num_envs,device=env.device)
        self.duration=torch.zeros_like(self.total)
        self.motion_integral=torch.zeros(env.num_envs,4,device=env.device)
        self.failed=torch.zeros_like(self.total)
        self.truncated=torch.zeros_like(self.total)
        self.lane_exits=torch.zeros_like(self.total)
        self.scale=float(cfg.weight)

    def __call__(self,env,source_rewards,group):
        reward=torch.zeros(env.num_envs,device=env.device)
        for _,cfg,function in self.terms:
            value=function(env,**cfg.params)
            if value.shape!=reward.shape:raise ValueError("Mixed reward must return one value per env")
            reward+=torch.where(self.mask,value,torch.zeros_like(value))*cfg.weight
        self.total+=torch.nan_to_num(reward,nan=0.,posinf=0.,neginf=0.)*self.scale*env.step_dt
        self.duration+=self.mask.float()*env.step_dt
        # Cohort-specific diagnostics; generic command-manager averages combine
        # the two tasks and must not be interpreted as flat-only tracking.
        data=env.scene["robot"].data
        velocity=data.root_vel_w.torch
        inverse=data.root_pose_w.torch[:,3:7].clone();inverse[:,:3]*=-1.
        linear_b=quat_apply(inverse,velocity[:,:3])
        angular_b=quat_apply(inverse,velocity[:,3:6])
        command=env.command_manager.get_command("base_velocity")
        sample=torch.stack((linear_b[:,0],(linear_b[:,:2]-command[:,:2]).square().sum(1),
                            (angular_b[:,2]-command[:,2]).square(),velocity[:,2].square()),dim=1)
        sample=torch.nan_to_num(sample,nan=0.,posinf=0.,neginf=0.)
        self.motion_integral+=torch.where(self.mask[:,None],sample,0.)*env.step_dt
        self.failed+=env.termination_manager.terminated.float()*self.mask
        self.truncated+=env.termination_manager.time_outs.float()*self.mask
        self.lane_exits+=mixed_flat_boundary(env).float()*self.mask
        return reward

    def reset(self,env_ids=None):
        ids=mixed_ids(self._env,env_ids,self.group)
        if len(ids)==0:return
        for _,_,function in self.terms:
            if isinstance(function,ManagerTermBase):function.reset(ids)
        log=self._env.extras.setdefault("log",{})
        if self.group == "stair":
            curriculum = _mixed_curriculum(self._env)
            if curriculum is not None:
                log["Metrics/Mixed/stair_curriculum_height_m"] = curriculum.height
                log["Metrics/Mixed/stair_curriculum_window_count"] = len(curriculum.outcomes)
                log["Metrics/Mixed/stair_curriculum_success_rate"] = curriculum.last_success_rate or 0.0
        log[f"Metrics/Mixed/{self.group}_reward_per_s"]=(self.total[ids]/self.duration[ids].clamp_min(self._env.step_dt)).mean()
        log[f"Metrics/Mixed/{self.group}_reset_count"]=torch.tensor(float(len(ids)),device=self._env.device)
        means=(self.motion_integral[ids]/self.duration[ids,None].clamp_min(self._env.step_dt)).mean(0)
        for key,value in (("forward_speed",means[0]),("xy_tracking_rmse",means[1].sqrt()),
                          ("yaw_tracking_rmse",means[2].sqrt()),("vertical_speed_rms",means[3].sqrt()),
                          ("failed_episode_fraction",self.failed[ids].mean()),
                          ("truncated_episode_fraction",self.truncated[ids].mean()),
                          ("lane_exit_fraction",self.lane_exits[ids].mean())):
            log[f"Metrics/Mixed/{self.group}_{key}"]=value
        self.total[ids]=0.;self.duration[ids]=0.
        self.motion_integral[ids]=0.;self.failed[ids]=0.;self.truncated[ids]=0.;self.lane_exits[ids]=0.


def mixed_termination(env,group,function,parameters):
    mask=mixed_masks(env)[0 if group=="flat" else 1]
    result=function(env,**parameters)
    return torch.where(mask,result,torch.zeros_like(result))


def mixed_flat_boundary(env,x_limit=4.,y_min=2.5,y_max=5.5):
    p=env.scene["robot"].data.root_pos_w.torch-env.scene.env_origins
    outside=(p[:,0].abs()>x_limit)|(p[:,1]<y_min)|(p[:,1]>y_max)
    return outside & mixed_masks(env)[0]


def mixed_flat_timeout(env,seconds=20.):
    return (env.episode_length_buf*env.step_dt>=seconds) & mixed_masks(env)[0]


def mixed_stair_base_low(env,minimum_height=.05):
    p=env.scene["robot"].data.root_pos_w.torch-env.scene.env_origins
    terrain=env_stair_terrain_height(env,p[:,0],p[:,1])
    return ((p[:,2]-terrain)<minimum_height) & mixed_masks(env)[1]
