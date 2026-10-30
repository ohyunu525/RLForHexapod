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


def save_phase_states(env, env_ids, phases, size: int = 1000):
    """env_ids의 현재 상태를 phase별 버퍼에 추가 (위치는 env 원점 기준, 꽉 차면 오래된 것부터 버림)."""
    if not hasattr(env, "stair_states"):
        env.stair_states = {}
    d = env.scene["robot"].data
    pose = d.root_pose_w.torch[env_ids].clone()
    pose[:, :3] -= env.scene.env_origins[env_ids]
    s = torch.cat([pose, d.root_vel_w.torch[env_ids], d.joint_pos.torch[env_ids], d.joint_vel.torch[env_ids]], 1)
    for k in phases.unique().tolist():
        env.stair_states[k] = torch.cat([env.stair_states.get(k, s[:0]), s[phases == k]])[-size:]


def reset_stair_phase(env, env_ids, x_ranges, buffer_ratio: float = 0.5):
    """시작 phase를 뽑아 그 시작 상태로 옮김 (reset_scene_to_default 다음에 실행).
    phase 0/1: 기본 자세 + x_ranges 위치, buffer_ratio 비율로 phase 2~4: 저장된 상태 (비어 있으면 phase 0)."""
    if isinstance(env_ids, slice):  # 첫 env.reset()은 slice(None)으로 호출됨
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, buf = len(env_ids), getattr(env, "stair_states", {})
    p = torch.randint(len(x_ranges), (n,), device=env.device)
    use_buf = torch.rand(n, device=env.device) < buffer_ratio
    p = torch.where(use_buf, torch.randint(len(x_ranges), NUM_PHASES, (n,), device=env.device), p)
    for k in range(len(x_ranges), NUM_PHASES):
        if k not in buf:
            p[p == k] = 0
    get_phase(env)[env_ids] = p
    asset = env.scene["robot"]

    hand = p < len(x_ranges)
    ids = env_ids[hand]
    if len(ids):
        lo, hi = torch.tensor(x_ranges, device=env.device)[p[hand]].unbind(-1)
        pose = asset.data.default_root_pose.torch[ids].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        pose[:, 0] += lo + (hi - lo) * torch.rand(len(ids), device=env.device)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)

    J = asset.num_joints
    for k, states in buf.items():
        ids = env_ids[p == k]
        if len(ids) == 0:
            continue
        s = states[torch.randint(len(states), (len(ids),), device=env.device)]
        s[:, :3] += env.scene.env_origins[ids]
        asset.write_root_pose_to_sim_index(root_pose=s[:, :7], env_ids=ids)
        asset.write_root_velocity_to_sim_index(root_velocity=s[:, 7:13], env_ids=ids)
        asset.write_joint_position_to_sim_index(position=s[:, 13:13 + J], env_ids=ids)
        asset.write_joint_velocity_to_sim_index(velocity=s[:, 13 + J:], env_ids=ids)


class stair_phase_reward(ManagerTermBase):
    """최고기록 갱신량 보상 + phase 완료 시 bonus 1회.
    0: 앞발 x, 1: 앞발 높이, 2/3: 앞/중간발 높이 + 몸통 x, 4: 뒷발 높이 + 뒷발 x."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        names = env.scene[asset_cfg.name].body_names
        self.offset = tip_offsets([names[i] for i in asset_cfg.body_ids], env.device)
        self.best_x = torch.zeros(env.num_envs, device=env.device)
        self.best_bx = torch.zeros(env.num_envs, device=env.device)
        self.best_rx = torch.zeros(env.num_envs, device=env.device)
        self.best_z = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
        self.need_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_phase = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.need_init[env_ids] = True
        self._env.extras["log"]["Metrics/stair_phase"] = self.last_phase[env_ids].mean()  # 끝난 에피소드의 도달 phase

    def __call__(self, env, asset_cfg, stairs, lift: float = 0.03, contact_tol: float = 0.03, bonus: float = 1.0,
                 hold: float = 1.0):
        approach_x, lift_z = stairs.x0 - 0.05, stairs.step_height + lift
        tips = feet_tip_pos(env, asset_cfg, self.offset)
        x = tips[:, FRONT, 0].min(dim=1).values.clamp(max=approach_x)  # 두 앞발 중 뒤처진 발
        z = tips[..., 2].clamp(max=lift_z)
        bx = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
        rx = tips[:, REAR, 0].min(dim=1).values.clamp(max=stairs.x0 + 0.05)  # 두 뒷발 중 뒤처진 발 (단 위 5cm까지)
        # 1단 이상을 딛고 있음: 발 아래 지형 ≥ 단 높이, 지형 위 contact_tol 이내
        terrain = stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])
        on = (terrain >= stairs.step_height - 1e-3) & (tips[..., 2] - terrain < contact_tol)

        self.best_x = torch.where(self.need_init, x, self.best_x)
        self.best_bx = torch.where(self.need_init, bx, self.best_bx)
        self.best_rx = torch.where(self.need_init, rx, self.best_rx)
        self.best_z = torch.where(self.need_init.unsqueeze(1), z, self.best_z)
        self.need_init[:] = False
        gx = (x - self.best_x).clamp(min=0)
        gbx = (bx - self.best_bx).clamp(min=0)
        grx = (rx - self.best_rx).clamp(min=0)
        gz = (z - self.best_z).clamp(min=0)
        self.best_x = torch.maximum(self.best_x, x)
        self.best_bx = torch.maximum(self.best_bx, bx)
        self.best_rx = torch.maximum(self.best_rx, rx)
        self.best_z = torch.maximum(self.best_z, z)

        phase = get_phase(env)
        gain = gx * (phase == 0) + gz[:, FRONT].sum(1) * (phase == 1)
        done = ((phase == 0) & (self.best_x >= approach_x)) | ((phase == 1) & (self.best_z[:, FRONT] >= lift_z).all(1))
        # 앞/중간발은 몸통 x, 뒷발은 뒷발 x (몸을 밀지 않고 다리를 접어 6발을 한 단에)
        for k, feet, gfwd in zip((2, 3, 4), (FRONT, MIDDLE, REAR), (gbx, gbx, grx)):
            gain += (gz[:, feet].sum(1) + gfwd) * (phase == k)
            done |= (phase == k) & on[:, feet].all(1)

        # 새 phase는 지금부터 다시 기록
        self.best_z[done] = z[done]
        self.best_bx[done] = bx[done]
        self.best_rx[done] = rx[done]
        phase += done.long()
        self.last_phase = phase.float()
        # phase 2~4에 막 들어선 상태 저장 → 리셋 때 시작 상태로 사용
        save = done & (phase >= 2) & (phase < NUM_PHASES)
        if save.any():
            save_phase_states(env, save.nonzero().squeeze(-1), phase[save])
        # 유지 보상: 완료한 다리 쌍이 단 위에 있으면 매 step
        held = sum(on[:, feet].all(1) * (phase > k) for k, feet in zip((2, 3, 4), (FRONT, MIDDLE, REAR)))
        return (gain + bonus * done) / env.step_dt + hold * held
