"""계단 teacher용 특권 관측: 계단 형상을 알고 있다는 가정으로 해석적으로 계산."""

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply


@configclass
class StairsGeometryCfg:
    """stairs_*.usda의 형상 (env 원점 기준). USD를 바꾸면 여기도 같이 바꿀 것."""

    x0: float = 0.5            # 첫 단 시작 x (= StairsSceneCfg의 stairs 위치)
    step_height: float = 0.16  # 단 높이
    tread: float = 0.3         # 단 깊이 (마지막 단 제외)
    num_steps: int = 3
    top_length: float = 0.8    # 마지막 단(꼭대기) 깊이
    width: float = 1.0


def stairs_terrain_height(stairs: StairsGeometryCfg, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """env 원점 기준 (x, y) 지점의 지형 높이. 계단 밖은 바닥(0)."""
    xr = x - stairs.x0
    length = (stairs.num_steps - 1) * stairs.tread + stairs.top_length
    on_stairs = (xr >= 0.0) & (xr < length) & (torch.abs(y) <= stairs.width / 2)
    # i번째 단(0부터) 윗면 높이 = (i + 1) × 단 높이, 마지막 단은 top_length 끝까지
    k = torch.clamp(torch.floor(xr / stairs.tread) + 1, max=stairs.num_steps)
    return torch.where(on_stairs, k * stairs.step_height, torch.zeros_like(xr))


def stairs_next_riser_distance(stairs: StairsGeometryCfg, x: torch.Tensor, max_dist: float) -> torch.Tensor:
    """x 앞쪽의 가장 가까운 단 수직면(riser)까지 거리. 계단 앞에서는 = 계단까지 거리, 꼭대기 위는 max_dist."""
    risers = stairs.x0 + stairs.tread * torch.arange(stairs.num_steps, device=x.device)
    d = risers - x.unsqueeze(-1)
    d = torch.where(d >= 0.0, d, torch.full_like(d, max_dist))
    return torch.clamp(d.min(dim=-1).values, max=max_dist)


# tibia 원점(무릎) → 발끝 offset (tibia 프레임, URDF tibiatip 위치). L1("tibia")만 y 방향이 반대
TIP_OFFSET_L1 = (-0.004, -0.147, -0.021)
TIP_OFFSET = (0.004, 0.147, -0.021)


def tip_offsets(body_names: list[str], device) -> torch.Tensor:
    return torch.tensor([TIP_OFFSET_L1 if n == "tibia" else TIP_OFFSET for n in body_names], device=device)


class _FeetTipTerm(ManagerTermBase):
    """발끝 위치가 필요한 관측의 공통 부분: 바디별 발끝 offset을 한 번만 만들어 둠."""

    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.tip_offset = tip_offsets([self.asset.body_names[i] for i in asset_cfg.body_ids], env.device)

    def tips_local(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """발끝 위치 (env 원점 기준) (N, B, 3)."""
        quat = self.asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        offset_w = quat_apply(quat, self.tip_offset.expand(quat.shape[0], -1, -1))
        tips_w = self.asset.data.body_pos_w.torch[:, asset_cfg.body_ids] + offset_w
        return tips_w - env.scene.env_origins.unsqueeze(1)


class front_feet_to_stairs(_FeetTipTerm):
    """1. 앞발 끝에서 앞쪽 다음 단 수직면까지 x 거리 (N, 앞발 수)."""

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg, max_dist: float = 1.0):
        tips = self.tips_local(env, asset_cfg)
        return stairs_next_riser_distance(stairs, tips[..., 0], max_dist)


class feet_height_above_terrain(_FeetTipTerm):
    """2-a. 각 발끝이 바로 아래 지형에서 떠 있는 높이 (N, 6). 딛고 있으면 ≈ 0."""

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg):
        tips = self.tips_local(env, asset_cfg)
        return tips[..., 2] - stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])


class terrain_height_under_feet(_FeetTipTerm):
    """2-b. 각 발끝 아래 지형이 몸통보다 얼마나 낮은지 (N, 6). 발 밑에 단이 있으면 값이 작아짐."""

    def __call__(self, env, asset_cfg: SceneEntityCfg, stairs: StairsGeometryCfg):
        tips = self.tips_local(env, asset_cfg)
        base_z = self.asset.data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3]
        return base_z - stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])


def stairs_step_height(env, stairs: StairsGeometryCfg) -> torch.Tensor:
    """3. 단 높이 (N, 1). 지금은 모든 env가 같은 계단 → 계단 높이 커리큘럼 때 env별 값으로 교체."""
    return torch.full((env.num_envs, 1), stairs.step_height, device=env.device)
