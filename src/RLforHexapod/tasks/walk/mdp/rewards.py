import math

import torch
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply

from .commands import base_height
from .observations import tip_offsets


def track_base_height_exp(env, command_name: str, std: float,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 높이 명령 추종: exp(-오차² / std²). 1 = 명령 높이와 같음."""
    target = env.command_manager.get_command(command_name)[:, 0]
    error = target - base_height(env, env.scene[asset_cfg.name])
    return torch.exp(-torch.square(error) / std**2)


class base_height_progress(ManagerTermBase):
    """에피소드 중 base 최고 높이를 갱신한 만큼만 보상 (출렁임 악용 방지)."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.max_z = torch.zeros(env.num_envs, device=env.device)
        self.need_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.need_init[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        z = self.asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        self.max_z = torch.where(self.need_init, z, self.max_z)
        self.need_init[:] = False
        gain = torch.clamp(z - self.max_z, min=0.0)
        self.max_z = torch.maximum(self.max_z, z)
        return gain / env.step_dt   # m/s 단위로 정규화


def feet_stumble(env, sensor_cfg: SceneEntityCfg, ratio: float = 4.0) -> torch.Tensor:
    """발에 걸린 수평 힘이 수직 힘보다 훨씬 크면 = 계단 수직면을 찬 것."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    hit = torch.norm(f[..., :2], dim=-1) > ratio * torch.abs(f[..., 2])
    return torch.any(hit, dim=1).float()


def feet_long_air_time(env, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """발이 threshold(초)보다 오래 떠 있으면 넘은 시간만큼 벌점 (공중에서 흔들리는 다리 억제)."""
    air = env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(air - threshold, min=0.0), dim=1)


class feet_clearance(ManagerTermBase):
    """스윙 중인 발끝을 target_height까지 높이 들수록 보상 (가장 낮은 발 기준 높이)."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        # tibia 원점은 무릎 관절 → 발끝 = 원점 + 회전 × offset (observations.py의 TIP_OFFSET)
        self.tip_offset = tip_offsets([self.asset.body_names[i] for i in asset_cfg.body_ids], env.device)

    def __call__(self, env, asset_cfg: SceneEntityCfg, command_name: str,
                 target_height: float = 0.05, tanh_mult: float = 2.0):
        data = self.asset.data
        quat = data.body_quat_w.torch[:, asset_cfg.body_ids]
        offset_w = quat_apply(quat, self.tip_offset.expand(quat.shape[0], -1, -1))
        tip_z = data.body_pos_w.torch[:, asset_cfg.body_ids, 2] + offset_w[..., 2]
        # 발끝 속도 = 무릎 속도 + ω × r
        tip_vel = data.body_lin_vel_w.torch[:, asset_cfg.body_ids] + torch.cross(
            data.body_ang_vel_w.torch[:, asset_cfg.body_ids], offset_w, dim=-1)
        # 가장 낮은 발(지지 발) 기준 높이 → 평지/계단 모두 동작, target 이상은 보상 없음
        height = torch.clamp(tip_z - tip_z.min(dim=1, keepdim=True).values, max=target_height)
        # 수평으로 움직이는 발(스윙)만 인정 → 정지한 지지 발은 0
        swing = torch.tanh(tanh_mult * torch.norm(tip_vel[..., :2], dim=-1))
        reward = torch.sum(height * swing, dim=1) / target_height
        # 서 있으라는 명령(속도 ≈ 0)일 때는 보상 없음
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward


def _wsp_distal_support(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                        force_threshold: float = 1.0, min_tip_height: float = -0.01,
                        max_tip_height: float = 0.03) -> torch.Tensor:
    """Flat-ground support proxy: upward tibia force AND distal tip near the floor.

    A tibia sensor alone cannot distinguish knee/shaft contact from the foot.
    The height condition rejects an elevated foot with a loaded knee, but this
    remains a proxy rather than a contact-point or support-polygon measurement.
    Both SceneEntityCfg selections must contain the same six legs in L1..L6 order.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    force_z = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    if force_z.shape[1] != 6 or quat.shape[1] != 6:
        raise ValueError("WSP support rewards require exactly six ordered tibia bodies")
    # Three reward terms share this geometry; do not allocate/copy it every step.
    names = tuple(asset.body_names[i] for i in asset_cfg.body_ids)
    cache = getattr(env, "_wsp_tip_offset_cache", None)
    if cache is None:
        cache = env._wsp_tip_offset_cache = {}
    key = (asset_cfg.name, names, str(quat.device), quat.dtype)
    if key not in cache:
        cache[key] = tip_offsets(list(names), quat.device).to(dtype=quat.dtype)
    offsets = cache[key].expand(env.num_envs, -1, -1)
    tip_z = (asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
             + quat_apply(quat, offsets)[..., 2] - env.scene.env_origins[:, 2:3])
    return ((force_z > force_threshold) & (tip_z >= min_tip_height) & (tip_z <= max_tip_height))


def wsp_all_feet_airborne(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                          force_threshold: float = 1.0, min_tip_height: float = -0.01,
                          max_tip_height: float = 0.03) -> torch.Tensor:
    """Unit cost when no distal support is detected, including standing commands."""
    support = _wsp_distal_support(env, sensor_cfg, asset_cfg, force_threshold, min_tip_height, max_tip_height)
    return (~support.any(dim=1)).float()


def wsp_support_deficit(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                        minimum_support: int = 3, force_threshold: float = 1.0,
                        min_tip_height: float = -0.01, max_tip_height: float = 0.03) -> torch.Tensor:
    """Bounded [0, 1] cost below three supporting feet; no bonus for six-foot lock."""
    if not 1 <= minimum_support <= 6:
        raise ValueError("minimum_support must be between one and six")
    support = _wsp_distal_support(env, sensor_cfg, asset_cfg, force_threshold, min_tip_height, max_tip_height)
    return ((minimum_support - support.sum(dim=1)).clamp_min(0).float() / minimum_support).square()


class wsp_support_participation(ManagerTermBase):
    """Penalize a leg that never supports OR never swings while commanded to move.

    Each leg's support duty follows a time-based exponential window. No phase is
    prescribed: tripod, ripple and other alternating gaits can satisfy the duty
    band. Airborne and sparse-support terms separately reject synchronous hops.
    Standing clears the moving-duty history and imposes no swing requirement.
    The neutral initial duty prevents an artificial six-foot demand after reset.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._env = env
        tau = cfg.params.get("time_constant_s", 0.6)
        low, high = cfg.params.get("minimum_duty", 0.2), cfg.params.get("maximum_duty", 0.85)
        if tau <= 0 or not 0 < low < high < 1:
            raise ValueError("Require positive duty time constant and 0 < minimum_duty < maximum_duty < 1")
        self.duty = torch.full((env.num_envs, 6), .5, device=env.device)
        self._cached_cost = torch.zeros(env.num_envs, device=env.device)
        self._last_step = None
        self._support_time = torch.zeros_like(self.duty)
        self._moving_support_time = torch.zeros_like(self.duty)
        self._time = torch.zeros(env.num_envs, device=env.device)
        self._moving_time = torch.zeros_like(self._time)
        self._air_time = torch.zeros_like(self._time)
        self._count_time = torch.zeros_like(self._time)
        self._vz2_time = torch.zeros_like(self._time)
        self._invalid_vz_time = torch.zeros_like(self._time)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        if self._time[ids].numel() == 0:
            return
        # _reset_idx creates extras['log'] immediately before RewardManager.reset.
        # Writing here survives that clear; writing only in __call__ would not.
        env = self._env
        duration = self._time[ids].clamp_min(env.step_dt)
        moving_duration = self._moving_time[ids].clamp_min(env.step_dt)
        valid = (self._moving_time[ids] > 0).float()
        moving_count = valid.sum().clamp_min(1.)
        log = env.extras.setdefault("log", {})
        for leg in range(6):
            log[f"Metrics/WSP/support_duty_L{leg+1}"] = (self._support_time[ids, leg] / duration).mean()
            log[f"Metrics/WSP/moving_support_duty_L{leg+1}"] = (
                self._moving_support_time[ids, leg] / moving_duration * valid).sum() / moving_count
        log["Metrics/WSP/all_air_fraction"] = (self._air_time[ids] / duration).mean()
        log["Metrics/WSP/support_count"] = (self._count_time[ids] / duration).mean()
        log["Metrics/WSP/vertical_velocity_rms"] = (self._vz2_time[ids] / duration).mean().sqrt()
        log["Metrics/WSP/invalid_vertical_velocity_fraction"] = (self._invalid_vz_time[ids] / duration).mean()
        log["Metrics/WSP/moving_fraction"] = (self._moving_time[ids] / duration).mean()
        self.duty[ids] = .5
        self._cached_cost[ids] = 0.
        for buffer in (self._support_time, self._moving_support_time, self._time,
                       self._moving_time, self._air_time, self._count_time, self._vz2_time, self._invalid_vz_time):
            buffer[ids] = 0.

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                 command_name: str = "base_velocity", time_constant_s: float = 0.6,
                 minimum_duty: float = 0.2, maximum_duty: float = 0.85,
                 linear_command_threshold: float = 0.025, yaw_command_threshold: float = 0.1,
                 force_threshold: float = 1.0, min_tip_height: float = -0.01,
                 max_tip_height: float = 0.03) -> torch.Tensor:
        # Diagnostic/repeated evaluation in one step must not age the window twice.
        step = env.common_step_counter
        if step == self._last_step:
            return self._cached_cost
        support = _wsp_distal_support(env, sensor_cfg, asset_cfg, force_threshold, min_tip_height, max_tip_height)
        command = env.command_manager.get_command(command_name)
        moving = ((command[:, :2].square().sum(dim=1) > linear_command_threshold**2)
                  | (command[:, 2].abs() > yaw_command_threshold))
        decay = math.exp(-env.step_dt / time_constant_s)
        updated = decay * self.duty + (1 - decay) * support.float()
        self.duty[:] = torch.where(moving[:, None], updated, .5)
        missing_support = ((minimum_duty - self.duty).clamp_min(0) / minimum_duty).square()
        missing_swing = ((self.duty - maximum_duty).clamp_min(0) / (1 - maximum_duty)).square()
        per_leg = missing_support + missing_swing
        # Mean alone lets one unused leg disappear among five useful legs.
        self._cached_cost[:] = .5 * (per_leg.mean(dim=1) + per_leg.max(dim=1).values) * moving.float()
        dt = env.step_dt
        self._support_time += support.float() * dt
        self._moving_support_time += (support & moving[:, None]).float() * dt
        self._time += dt
        self._moving_time += moving.float() * dt
        self._air_time += (~support.any(dim=1)).float() * dt
        self._count_time += support.sum(dim=1).float() * dt
        vz2 = env.scene[asset_cfg.name].data.root_vel_w.torch[:, 2].square()
        self._invalid_vz_time += (~torch.isfinite(vz2)).float() * dt
        self._vz2_time += torch.nan_to_num(vz2, nan=0., posinf=0., neginf=0.) * dt
        self._last_step = step
        return self._cached_cost


def _refine_stationary_command(env, command_name: str, linear_command_threshold: float,
                               yaw_command_threshold: float) -> torch.Tensor:
    """Gate by requested motion, never by actual speed that a policy could exploit."""
    command = env.command_manager.get_command(command_name)
    return ((command[:, :2].square().sum(dim=1) <= linear_command_threshold**2)
            & (command[:, 2].abs() <= yaw_command_threshold))


def refine_stand_joint_velocity_l2(
    env, command_name: str = "base_velocity", velocity_scale: float = 3.0,
    linear_command_threshold: float = 0.025, yaw_command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cost of continuing to march after a stationary command, with no pose target.

    Mean rather than sum keeps the scale independent of the 18-joint count.
    A turn-in-place or translational command above the thresholds disables this
    cost completely; the existing support rewards still protect walking legs.
    No resting pose is prescribed. Height adjustment still trades this modest
    motion cost against the unchanged height-tracking reward.
    """
    if velocity_scale <= 0:
        raise ValueError("velocity_scale must be positive")
    stationary = _refine_stationary_command(
        env, command_name, linear_command_threshold, yaw_command_threshold)
    velocity = env.scene[asset_cfg.name].data.joint_vel.torch
    cost = velocity.square().mean(dim=1) / velocity_scale**2
    return torch.where(stationary, cost, torch.zeros_like(cost))


class refine_yaw_tracking_l2(ManagerTermBase):
    """Yaw-rate error cost and episode diagnostics for the refinement task.

    Body-frame yaw matches the existing angular tracking reward. Penalizing
    error, rather than angular speed itself, permits the requested turns while
    charging alternating positive/negative yaw error during straight walking.
    Diagnostics stay on the device and are logged at RewardManager.reset, after
    the environment clears extras['log']. No runtime phase or controller state
    is added to observations or actions.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._last_step = None
        self._cached_cost = torch.zeros(env.num_envs, device=env.device)
        self._time = torch.zeros_like(self._cached_cost)
        self._yaw_error2_time = torch.zeros_like(self._time)
        self._stationary_time = torch.zeros_like(self._time)
        self._stationary_joint_velocity2_time = torch.zeros_like(self._time)
        self._invalid_time = torch.zeros_like(self._time)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        if self._time[ids].numel() == 0:
            return
        duration = self._time[ids].clamp_min(self._env.step_dt)
        stationary_duration = self._stationary_time[ids].clamp_min(self._env.step_dt)
        had_stationary_command = (self._stationary_time[ids] > 0).float()
        count = had_stationary_command.sum().clamp_min(1.)
        log = self._env.extras.setdefault("log", {})
        log["Metrics/Refine/yaw_error_rms"] = (self._yaw_error2_time[ids] / duration).mean().sqrt()
        log["Metrics/Refine/stationary_joint_velocity_rms"] = (
            (self._stationary_joint_velocity2_time[ids] / stationary_duration
             * had_stationary_command).sum() / count).sqrt()
        log["Metrics/Refine/stationary_fraction"] = (self._stationary_time[ids] / duration).mean()
        log["Metrics/Refine/invalid_state_fraction"] = (self._invalid_time[ids] / duration).mean()
        for buffer in (self._cached_cost, self._time, self._yaw_error2_time, self._stationary_time,
                       self._stationary_joint_velocity2_time, self._invalid_time):
            buffer[ids] = 0.

    def __call__(self, env, command_name: str = "base_velocity",
                 linear_command_threshold: float = 0.025, yaw_command_threshold: float = 0.1,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        step = env.common_step_counter
        if self._last_step == step:
            return self._cached_cost
        data = env.scene[asset_cfg.name].data
        angular_velocity_w = data.root_vel_w.torch[:, 3:6]
        local_up = torch.zeros_like(angular_velocity_w)
        local_up[:, 2] = 1.
        body_up_w = quat_apply(data.root_pose_w.torch[:, 3:7], local_up)
        yaw_rate_b = (angular_velocity_w * body_up_w).sum(dim=1)
        error = yaw_rate_b - env.command_manager.get_command(command_name)[:, 2]
        self._cached_cost[:] = error.square()
        stationary = _refine_stationary_command(
            env, command_name, linear_command_threshold, yaw_command_threshold)
        joint_velocity2 = data.joint_vel.torch.square().mean(dim=1)
        dt = env.step_dt
        self._time += dt
        self._stationary_time += stationary.float() * dt
        self._invalid_time += (~(torch.isfinite(self._cached_cost)
                                & torch.isfinite(joint_velocity2))).float() * dt
        self._yaw_error2_time += torch.nan_to_num(self._cached_cost, nan=0., posinf=0., neginf=0.) * dt
        self._stationary_joint_velocity2_time += torch.where(
            stationary, torch.nan_to_num(joint_velocity2, nan=0., posinf=0., neginf=0.),
            torch.zeros_like(joint_velocity2)) * dt
        self._last_step = step
        return self._cached_cost


class wsp_stop_stand_posture_l2(ManagerTermBase):
    """Stationary-only deviation from an immutable height-conditioned rest table.

    The table is cached on the simulation device once. It is a reward target,
    never an action override. Moving commands receive exactly zero cost even
    when their joint/height state is non-finite; invalid stationary state gets
    a finite unit cost, while the existing physics termination handles it.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        expected = tuple(f"L{leg}_{part}" for leg in range(1, 7) for part in ("coxa", "femur", "tibia"))
        if tuple(params["anchor_joint_names"]) != expected:
            raise ValueError("Posture anchors require exact L1..L6 coxa/femur/tibia joint order")
        heights = torch.as_tensor(params["anchor_heights"], dtype=torch.float64)
        anchors = torch.as_tensor(params["anchor_joint_positions_rad"], dtype=torch.float64)
        if (heights.ndim != 1 or heights.numel() < 2 or anchors.shape != (heights.numel(), 18)
                or not torch.isfinite(heights).all() or not torch.isfinite(anchors).all()
                or not (heights[1:] > heights[:-1]).all()):
            raise ValueError("Posture table must contain finite increasing heights and corresponding 18-joint rows")
        asset_cfg = params["asset_cfg"]
        self._asset = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        indices = (list(range(len(self._asset.joint_names)))[joint_ids] if isinstance(joint_ids, slice)
                   else [int(index) for index in joint_ids])
        if tuple(self._asset.joint_names[index] for index in indices) != expected:
            raise ValueError("Resolved posture joint order differs from the anchor table")
        self._joint_ids = torch.as_tensor(indices, dtype=torch.long, device=env.device)
        dtype = self._asset.data.joint_pos.torch.dtype
        self._height_min, self._height_max = float(heights[0]), float(heights[-1])
        self._heights = heights.to(device=env.device, dtype=dtype).clone()
        self._anchors = anchors.to(device=env.device, dtype=dtype).clone()

    def __call__(self, env, anchor_heights, anchor_joint_positions_rad, anchor_joint_names,
                 asset_cfg: SceneEntityCfg, command_name: str = "base_velocity",
                 height_command_name: str = "base_height", angle_scale: float = 0.35,
                 linear_command_threshold: float = 0.025, yaw_command_threshold: float = 0.1) -> torch.Tensor:
        if not math.isfinite(angle_scale) or angle_scale <= 0:
            raise ValueError("Posture angle_scale must be positive and finite")
        height = env.command_manager.get_command(height_command_name)[:, 0]
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if q.shape != (env.num_envs, 18) or height.shape != (env.num_envs,):
            raise ValueError("Posture reward requires 18 joint positions and one height command per environment")
        valid = torch.isfinite(height) & torch.isfinite(q).all(dim=1)
        h = torch.nan_to_num(height, nan=self._height_min, posinf=self._height_max, neginf=self._height_min)
        h = h.clamp(self._height_min, self._height_max).contiguous()
        upper = torch.searchsorted(self._heights, h, right=True).clamp(1, len(self._heights)-1)
        lower = upper-1
        blend = ((h-self._heights[lower]) / (self._heights[upper]-self._heights[lower]))[:, None]
        target = self._anchors[lower] + blend*(self._anchors[upper]-self._anchors[lower])
        cost = ((q-target)/angle_scale).square().mean(dim=1)
        cost = torch.where(valid, torch.nan_to_num(cost, nan=1., posinf=1., neginf=1.), torch.ones_like(cost))
        stationary = _refine_stationary_command(
            env, command_name, linear_command_threshold, yaw_command_threshold)
        return torch.where(stationary, cost, torch.zeros_like(cost))

