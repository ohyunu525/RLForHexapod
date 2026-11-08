import torch
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply

from .commands import base_height
from .observations import tip_offsets


def tibia_vertical_score(knees: torch.Tensor, tips: torch.Tensor) -> torch.Tensor:
    """tibia가 지면에 수직인 정도 (..., 다리). 1 = 무릎 바로 아래 발끝, 0 = 수평 이상.
    무릎→발끝 벡터가 아래를 향하는 cos 값이다. 걷기·계단 teacher가 같은 정의를 쓴다 (증류 때 자세 규칙을 맞추기 위해)."""
    tibia = tips - knees
    return (-tibia[..., 2] / tibia.norm(dim=-1).clamp(min=1e-6)).clamp(0.0, 1.0)


class tibia_vertical(ManagerTermBase):
    """여섯 다리 tibia 수직도의 평균 (N,). weight = 초당 최대 점수.

    tibia가 수직이면 지지력이 tibia 축으로 들어가 무릎·femur 토크가 작고, 몸 높이가 femur 각도만으로 정해진다.
    몸을 낮출 때 다리를 바깥으로 벌려 버티다 넘어지는 것을 막기 위한 항목이다.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        # tibia 원점은 무릎 관절 → 발끝 = 원점 + 회전 × offset (observations.py의 TIP_OFFSET)
        self.tip_offset = tip_offsets([self.asset.body_names[i] for i in asset_cfg.body_ids], env.device)

    def __call__(self, env, asset_cfg: SceneEntityCfg):
        data = self.asset.data
        quat = data.body_quat_w.torch[:, asset_cfg.body_ids]
        knees = data.body_pos_w.torch[:, asset_cfg.body_ids]
        tips = knees + quat_apply(quat, self.tip_offset.expand(quat.shape[0], -1, -1))
        return tibia_vertical_score(knees, tips).mean(dim=1)


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

