import torch
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply

from .commands import base_height
from .observations import feet_contact, tip_offsets


class tibia_vertical(ManagerTermBase):
    """땅에 닿은 다리의 tibia 수직도 평균 (N,). 다리마다 exp(-기울기² / std²), 1 = 수직. 닿은 다리가 없으면 0.

    tibia가 수직이면 지지력이 tibia 축으로 들어가 무릎 토크가 작고, 몸 높이가 femur 각도만으로 정해진다.
    체중을 받을 때만 의미가 있으므로 공중의 다리는 세지 않는다: 발을 들 때 tibia가 기우는 것은 벌하지 않는다.
    asset_cfg와 sensor_cfg는 다리 순서가 같아야 한다 (둘 다 FOOT_BODIES, preserve_order=True).
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        # 무릎→발끝 벡터 (observations.py의 TIP_OFFSET). z는 다리 평면 밖 성분이라 뺀다:
        # 그대로 쓰면 어떤 관절각에서도 약 8° 기울어 수직 점수가 만점에 닿지 못한다
        self.tibia_axis = tip_offsets([self.asset.body_names[i] for i in asset_cfg.body_ids], env.device)
        self.tibia_axis[..., 2] = 0.0

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, std: float, threshold: float = 1.0):
        quat = self.asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        axis_w = quat_apply(quat, self.tibia_axis.expand(quat.shape[0], -1, -1))
        tilt = torch.atan2(torch.norm(axis_w[..., :2], dim=-1), -axis_w[..., 2])  # 수직에서 기운 각도 [rad]
        score = torch.exp(-torch.square(tilt) / std**2)
        contact = feet_contact(env, sensor_cfg, threshold)
        return torch.sum(score * contact, dim=1) / contact.sum(dim=1).clamp(min=1.0)


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


def feet_air_time_error(env, command_name: str, sensor_cfg: SceneEntityCfg, target: float, std: float,
                        command_threshold: float = 0.05) -> torch.Tensor:
    """착지한 발의 공중 시간이 target에서 벗어난 정도의 합 (음수 weight로 쓴다). weight = 착지 한 번당 최대 벌점.

    착지마다 1 - exp(-(공중 시간 - target)² / std²): target과 같으면 0, 멀수록 1.
    양수 보상으로 주면 착지가 잦을수록 점수가 늘어 종종걸음이 되므로 벌점으로 준다.
    서 있으라는 명령(병진 + 회전 속도 ≈ 0)일 때는 계산하지 않는다.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    landed = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids].float()
    air_time = sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    error = 1.0 - torch.exp(-torch.square(air_time - target) / std**2)
    cmd = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) > command_threshold
    # 보상 관리자가 weight × step_dt를 곱하므로, 착지 한 번당 weight가 되도록 step_dt로 나눈다
    return torch.sum(error * landed, dim=1) * moving / env.step_dt


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

