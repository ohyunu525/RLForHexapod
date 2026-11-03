import math

import torch
from isaaclab.managers import SceneEntityCfg


def set_joint_pos_limits(env, env_ids, limits_deg: dict[str, tuple[float, float]],
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """관절 패턴별 위치 한계(도)를 시뮬레이션에 덮어씀. startup 이벤트용이라 env_ids는 쓰지 않고 전체 env에 적용."""
    asset = env.scene[asset_cfg.name]
    for pattern, (lower, upper) in limits_deg.items():
        joint_ids, _ = asset.find_joints(pattern)
        limits = torch.tensor([math.radians(lower), math.radians(upper)], device=env.device)
        # limits : (N, 관절 수, 2)
        asset.write_joint_position_limit_to_sim_index(
            limits=limits.repeat(env.num_envs, len(joint_ids), 1), joint_ids=joint_ids)
