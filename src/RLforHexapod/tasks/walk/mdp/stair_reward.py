import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from .observations import tip_offsets, stairs_terrain_height

# asset_cfg 발 순서: tibia(L1), tibia_2 ~ tibia_6 (preserve_order=True)
FRONT, MIDDLE, REAR = [0, 5], [1, 4], [2, 3]
# phase 표 (종류, 대상 발). approach: 앞발 x 전진, lift: 발 높이 들기, place: 한 단 위에 딛기
# 번호 = 1 + 2 × 쌍 + (놓기면 1). 순서는 고정이 아니라 놓기를 마칠 때마다 가장 낮은 쌍(같으면 앞)을 고름
PAIRS = [FRONT, MIDDLE, REAR]  # 같은 단이면 앞쪽 우선 (argmin은 첫 번째를 고름)
PHASES = [("approach", FRONT), ("lift", FRONT), ("place", FRONT),
          ("lift", MIDDLE), ("place", MIDDLE), ("lift", REAR), ("place", REAR)]
NUM_PHASES = len(PHASES)  # = 완료 phase 번호 (모든 쌍이 꼭대기 단)
OBS_PHASES = 12  # obs one-hot 크기 고정 → phase를 늘려도 체크포인트 유지


def feet_tip_pos(env, asset_cfg, tip_offset):
    """발끝 위치 (env 원점 기준) (N, B, 3) = 무릎 위치 + 회전 × offset."""
    data = env.scene[asset_cfg.name].data
    quat = data.body_quat_w.torch[:, asset_cfg.body_ids]
    offset_w = quat_apply(quat, tip_offset.expand(quat.shape[0], -1, -1))
    return data.body_pos_w.torch[:, asset_cfg.body_ids] + offset_w - env.scene.env_origins.unsqueeze(1)


def pair_levels(stairs, terrain):
    """쌍별 단 번호 (N, 3): 두 발 중 낮은 발 아래의 단 (바닥 = 0)."""
    lvl = (terrain / stairs.step_height).round()
    return torch.stack([lvl[:, p].min(1).values for p in PAIRS], 1)


def get_phase(env):
    """env별 현재 phase (NUM_PHASES 참고). obs/보상/리셋/종료가 같이 씀."""
    if not hasattr(env, "stair_phase"):
        env.stair_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env.stair_phase


def stair_phase_obs(env, num_phases: int = OBS_PHASES):
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
    """PHASES 표에 따라 최고기록 갱신량 보상 + phase 완료 시 bonus 1회.
    approach: 앞발 x, lift: 발 높이, place: 몸통 x (뒷발은 뒷발 x + 몸통 x × rear_body_fwd).
    모든 phase 공통: 매 step 몸통 x 위치 × body_fwd (올라갈 단 수직면 + fwd_ahead에서 포화).
    높이/위치 기준은 모두 target(옮기는 쌍이 올라갈 단 번호) 기준."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        names = env.scene[asset_cfg.name].body_names
        self.offset = tip_offsets([names[i] for i in asset_cfg.body_ids], env.device)
        self.best_x = torch.zeros(env.num_envs, device=env.device)
        self.best_bx = torch.zeros(env.num_envs, device=env.device)
        self.best_rx = torch.zeros(env.num_envs, device=env.device)
        self.best_z = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
        self.target = torch.ones(env.num_envs, device=env.device)
        self.need_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_phase = torch.zeros(env.num_envs, device=env.device)
        self.last_level = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.need_init[env_ids] = True
        self._env.extras["log"]["Metrics/stair_phase"] = self.last_phase[env_ids].mean()  # 끝난 에피소드의 phase
        self._env.extras["log"]["Metrics/stair_level"] = self.last_level[env_ids].mean()  # 끝난 에피소드의 쌍별 평균 단

    def __call__(self, env, asset_cfg, stairs, lift: float = 0.03, contact_tol: float = 0.03, bonus: float = 1.0,
                 hold: float = 1.0, rear_body_fwd: float = 0.5, support: float = 0.5, support_margin: float = 0.05,
                 body_fwd: float = 1.0, fwd_ahead: float = 0.18):
        tips = feet_tip_pos(env, asset_cfg, self.offset)
        terrain = stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])
        levels = pair_levels(stairs, terrain)
        phase = get_phase(env)
        # 리셋 직후 target: phase 0 → 1단, 들기 → 지금 쌍의 단 + 1, 놓기 → 이미 올라간 단
        lvl = levels.gather(1, ((phase - 1) // 2).clamp(0, 2).unsqueeze(1)).squeeze(1)
        start = torch.where(phase == 0, 1.0, lvl + ((phase % 2 == 1) & (phase < NUM_PHASES)))
        self.target = torch.where(self.need_init, start, self.target)

        top = (self.target * stairs.step_height).unsqueeze(1)  # 올라갈 단 높이 (N, 1)
        riser = stairs.x0 + (self.target - 1) * stairs.tread   # 올라갈 단 수직면 x
        approach_x = stairs.x0 - 0.05
        x = tips[:, FRONT, 0].min(dim=1).values.clamp(max=approach_x)  # 두 앞발 중 뒤처진 발
        z = torch.minimum(tips[..., 2], top + lift)
        bx = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
        rx = torch.minimum(tips[:, REAR, 0].min(dim=1).values, riser + 0.05)  # 두 뒷발 중 뒤처진 발 (단 위 5cm까지)
        # 올라갈 단 이상을 딛고 있음: 발 아래 지형 ≥ 그 단 높이, 지형 위 contact_tol 이내
        contact = tips[..., 2] - terrain < contact_tol
        on = (terrain >= top - 1e-3) & contact

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

        gain = torch.zeros(env.num_envs, device=env.device)
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for k, (kind, feet) in enumerate(PHASES):
            now = phase == k
            if kind == "approach":
                gain += gx * now
                done |= now & (self.best_x >= approach_x)
            # elif kind == "lift":  # 발마다 한 번씩 lift_z 이상 (동시에 들 필요 없음)
                # gain += gz[:, feet].sum(1) * now
                # done |= now & (self.best_z[:, feet] >= lift_z).all(1)
            else:  # place: 뒷발은 몸을 밀지 않고 다리를 접어 단 안쪽으로
                gain += gz[:, feet].sum(1) * now
                gain += (grx + rear_body_fwd * gbx if feet is REAR else gbx) * now
                done |= now & on[:, feet].all(1)
        

        # 새 phase는 지금부터 다시 기록
        self.best_z[done] = z[done]
        self.best_bx[done] = bx[done]
        self.best_rx[done] = rx[done]
        # 다음 phase: 들기 완료 → 같은 쌍 놓기, 전진/놓기 완료 → 가장 낮은 쌍(같으면 앞)의 들기, 모든 쌍이 꼭대기 → 성공
        lowest = levels.min(1).values
        pick = done & (phase % 2 == 0)
        nxt = torch.where(phase % 2 == 0, 1 + 2 * levels.argmin(1), phase + 1)
        nxt = nxt.masked_fill((phase % 2 == 0) & (lowest >= stairs.num_steps), NUM_PHASES)
        self.target = torch.where(pick, lowest + 1, self.target)
        phase[:] = torch.where(done, nxt, phase)
        self.last_phase = phase.float()
        self.last_level = levels.mean(1)
        # phase 2 이상에 막 들어선 상태 저장 → 리셋 때 시작 상태로 사용
        save = done & (phase >= 2) & (phase < NUM_PHASES)
        if save.any():
            save_phase_states(env, save.nonzero().squeeze(-1), phase[save])
        # 유지 보상: 1단 이상을 딛고 있는 다리 쌍마다 매 step
        on_step = (terrain >= stairs.step_height - 1e-3) & contact
        held = sum(on_step[:, p].all(1).float() for p in PAIRS)
        # 뒷발 phase(lift, place): 몸통(무게중심)이 중간발보다 support_margin 앞 → 뒷발을 들어도 안 넘어짐 (매 step, 상한 1)
        margin = (bx - tips[:, MIDDLE, 0].mean(1)) / support_margin
        stable = margin.clamp(0.0, 1.0) * (phase >= NUM_PHASES - 2) * (phase < NUM_PHASES)
        # 몸통 전진: 앞에 있을수록 매 step (hold와 같은 크기로 경쟁), 올라갈 단 수직면 + fwd_ahead에서 포화 → 앞발이 다음 수직면에 안 닿게
        fwd = torch.minimum(bx, riser + fwd_ahead).clamp(min=0.0)
        return (gain + bonus * done) / env.step_dt + hold * held + support * stable + body_fwd * fwd
