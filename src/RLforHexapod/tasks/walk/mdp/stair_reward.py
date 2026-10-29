import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from .observations import tip_offsets, stairs_terrain_height

# asset_cfg 발 순서: tibia(L1), tibia_2 ~ tibia_6 (preserve_order=True)
FRONT, MIDDLE, REAR = [0, 5], [1, 4], [2, 3]
NUM_PHASES = 5  # 0: 전진, 1: 앞발 들기, 2/3/4: 앞/중간/뒷발 1단 이상, 5: 완료


def feet_tip_pos(env, asset_cfg, tip_offset):
    """발끝 위치 (env 원점 기준) (N, B, 3) = 무릎 위치 + 회전 × offset."""
    data = env.scene[asset_cfg.name].data
    quat = data.body_quat_w.torch[:, asset_cfg.body_ids]
    offset_w = quat_apply(quat, tip_offset.expand(quat.shape[0], -1, -1))
    return data.body_pos_w.torch[:, asset_cfg.body_ids] + offset_w - env.scene.env_origins.unsqueeze(1)


def get_phase(env):
    """env별 현재 phase (NUM_PHASES 참고). obs/보상/리셋/종료가 같이 씀."""
    if not hasattr(env, "stair_phase"):
        env.stair_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env.stair_phase


def stair_phase_obs(env, num_phases: int = NUM_PHASES):
    return F.one_hot(get_phase(env).clamp(max=num_phases - 1), num_phases).float()


def stair_phase_reached(env, phase: int):
    return get_phase(env) >= phase


def reset_stair_phase(env, env_ids, x_ranges):
    """phase를 랜덤으로 뽑고 그 phase의 시작 x 위치로 로봇을 옮김 (reset_scene_to_default 다음에 실행)."""
    if isinstance(env_ids, slice):  # 첫 env.reset()은 slice(None)으로 호출됨
        env_ids = torch.arange(env.num_envs, device=env.device)
    p = torch.randint(len(x_ranges), (len(env_ids),), device=env.device)
    get_phase(env)[env_ids] = p
    lo, hi = torch.tensor(x_ranges, device=env.device)[p].unbind(-1)
    asset = env.scene["robot"]
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, 0] += lo + (hi - lo) * torch.rand(len(env_ids), device=env.device)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


class stair_phase_reward(ManagerTermBase):
    """최고기록 갱신량 보상 + phase 완료 시 bonus 1회.
    0: 앞발 x, 1: 앞발 높이, 2/3/4: 앞/중간/뒷발 높이 + 몸통 x."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        names = env.scene[asset_cfg.name].body_names
        self.offset = tip_offsets([names[i] for i in asset_cfg.body_ids], env.device)
        self.best_x = torch.zeros(env.num_envs, device=env.device)
        self.best_bx = torch.zeros(env.num_envs, device=env.device)
        self.best_z = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
        self.need_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.need_init[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, asset_cfg, stairs, lift: float = 0.03, contact_tol: float = 0.03, bonus: float = 1.0):
        approach_x, lift_z = stairs.x0 - 0.05, stairs.step_height + lift
        tips = feet_tip_pos(env, asset_cfg, self.offset)
        x = tips[:, FRONT, 0].min(dim=1).values.clamp(max=approach_x)  # 두 앞발 중 뒤처진 발
        z = tips[..., 2].clamp(max=lift_z)
        bx = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
        # 1단 이상을 딛고 있음: 발 아래 지형 ≥ 단 높이, 지형 위 contact_tol 이내
        terrain = stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])
        on = (terrain >= stairs.step_height - 1e-3) & (tips[..., 2] - terrain < contact_tol)

        self.best_x = torch.where(self.need_init, x, self.best_x)
        self.best_bx = torch.where(self.need_init, bx, self.best_bx)
        self.best_z = torch.where(self.need_init.unsqueeze(1), z, self.best_z)
        self.need_init[:] = False
        gx = (x - self.best_x).clamp(min=0)
        gbx = (bx - self.best_bx).clamp(min=0)
        gz = (z - self.best_z).clamp(min=0)
        self.best_x = torch.maximum(self.best_x, x)
        self.best_bx = torch.maximum(self.best_bx, bx)
        self.best_z = torch.maximum(self.best_z, z)

        phase = get_phase(env)
        gain = gx * (phase == 0) + gz[:, FRONT].sum(1) * (phase == 1)
        done = ((phase == 0) & (self.best_x >= approach_x)) | ((phase == 1) & (self.best_z[:, FRONT] >= lift_z).all(1))
        for k, feet in zip((2, 3, 4), (FRONT, MIDDLE, REAR)):
            gain += (gz[:, feet].sum(1) + gbx) * (phase == k)
            done |= (phase == k) & on[:, feet].all(1)

        # 새 phase는 지금부터 다시 기록
        self.best_z[done] = z[done]
        self.best_bx[done] = bx[done]
        phase += done.long()
        return (gain + bonus * done) / env.step_dt
