import torch
from isaaclab.managers import SceneEntityCfg


def out_of_stairs_width(env, half_width: float, center_y: float = 0.0,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 중심이 계단 폭(env 원점 기준 center_y ± half_width) 밖으로 나가면 종료."""
    asset = env.scene[asset_cfg.name]
    # env 원점 기준 옆(y) 위치
    y = asset.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]
    return torch.abs(y - center_y) > half_width
