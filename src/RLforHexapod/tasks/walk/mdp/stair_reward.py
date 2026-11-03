import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from .observations import tip_offsets, stairs_terrain_height

# ===== 다리 쌍 =====
# asset_cfg 발 순서: tibia(L1), tibia_2 ~ tibia_6 (preserve_order=True)
FRONT, MIDDLE, REAR = [0, 5], [1, 4], [2, 3]
PAIRS = [FRONT, MIDDLE, REAR]  # 같은 단이면 앞쪽 우선 (argmin은 첫 번째를 고름)

# ===== phase =====
# 흐름: APPROACH → MOVE_* 반복 → SETTLE → DONE
# MOVE_*는 고정 순서가 아니다. 끝날 때마다 가장 낮은 쌍을 골라 한 단 위로 옮기고, 모든 쌍이 꼭대기면 SETTLE로 간다.
# 예) 3단 계단: 앞 → 중간 → 뒤 → 앞 → ... 총 9번. 지금 몇 단째로 가는지는 phase가 아니라 target이 기억한다.
PHASE_APPROACH = 0     # 앞발을 첫 수직면 바로 앞까지 전진
PHASE_MOVE_FRONT = 1   # 앞발 쌍을 target 단 위에 딛기
PHASE_MOVE_MIDDLE = 2  # 중간발 쌍을 target 단 위에 딛기
PHASE_MOVE_REAR = 3    # 뒷발 쌍을 target 단 위에 딛기
PHASE_SETTLE = 4       # 꼭대기에서 기본 자세로 서서 멈추기
PHASE_DONE = 5         # 성공. env_cfg의 success 종료 조건이 에피소드를 끝낸다
MOVE_PHASES = {PHASE_MOVE_FRONT: FRONT, PHASE_MOVE_MIDDLE: MIDDLE, PHASE_MOVE_REAR: REAR}
OBS_PHASES = 12  # obs one-hot 크기 고정 → phase를 늘려도 관측 차원 유지


@configclass
class SettleCfg:
    """SETTLE phase 설정: 꼭대기에서 기본 자세(init_state의 관절 각도)로 서서 멈추기."""

    reward: float = 4.0      # 정지 점수(0~1)에 곱하는 매 step 보상
    hold_time: float = 1.5   # 아래 허용 오차를 연속으로 지켜야 하는 시간 [s]. 채우면 성공

    # 허용 오차: 모든 발이 꼭대기에 있고 셋 다 이 안에 들면 "멈춘 상태"
    pose_tol: float = 0.2    # 기본 자세와의 관절 각도 RMS 오차 [rad]
    tilt_tol: float = 0.1    # 몸통 기울기 (중력 벡터의 수평 성분, 0.1 ≈ 6°)
    speed_tol: float = 0.1   # 몸통 속도 크기 (선속도 + 각속도)

    # 점수 폭: 오차가 이 값이면 해당 점수가 약 0.37. 허용 오차보다 넓어서 멀리 있어도 다가갈 방향이 보인다
    pose_std: float = 0.4
    tilt_std: float = 0.2
    speed_std: float = 0.3


def feet_tip_pos(env, asset_cfg, tip_offset):
    """발끝 위치 (env 원점 기준) (N, B, 3) = 무릎 위치 + 회전 × offset."""
    # env.scene[asset_cfg.name] : env.scene["robot"]
    data = env.scene[asset_cfg.name].data
    # quat (N, K, 4) : 환경 수, 링크 수, 쿼터니언
    quat = data.body_quat_w.torch[:, asset_cfg.body_ids]
    # 
    offset_w = quat_apply(quat, tip_offset.expand(quat.shape[0], -1, -1))
    return data.body_pos_w.torch[:, asset_cfg.body_ids] + offset_w - env.scene.env_origins.unsqueeze(1)


def pair_levels(stairs, terrain):
    """쌍별 단 번호 (N, 3): 두 발 중 낮은 발 아래의 단 (바닥 = 0)."""
    lvl = (terrain / stairs.step_height).round()
    return torch.stack([lvl[:, p].min(1).values for p in PAIRS], 1)


def get_phase(env):
    """env별 현재 phase (PHASE_* 참고). obs/보상/리셋/종료가 같이 씀."""
    if not hasattr(env, "stair_phase"):
        env.stair_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env.stair_phase


def stair_phase_obs(env, num_phases: int = OBS_PHASES):
    return F.one_hot(get_phase(env).clamp(max=num_phases - 1), num_phases).float()


def stair_phase_reached(env, phase: int):
    return get_phase(env) >= phase


def save_phase_states(env, env_ids, phases, size: int = 1000):
    """
    env_ids의 현재 상태를 phase별 버퍼에 추가 (위치는 env 원점 기준, 꽉 차면 오래된 것부터 버림).

    S -> size (버퍼 크기)
    stair_states : (S, 7 + 6 + J + J)
    """
    if not hasattr(env, "stair_states"):
        # env에 stair_states가 없으면 생성
        env.stair_states = {}

    # d : ArticulationData
    # d.root_pose_w : (N, 7) : xyz, 쿼터니언(회전){나중에 회전 관련해서 다시 공부}
    # d.root_vel_w : (N, 6) : 선속도, 각속도
    # d.joint_pos : (N, J) : 관절 각도
    # d.joint_vel : (N, J) : 관절 속도
    
    d = env.scene["robot"].data

    # torch는 꺼내기 위한 문법적 요소
    # clone 복사 , 원본에 영향을 미치지 않게 하기 위해서
    pose = d.root_pose_w.torch[env_ids].clone()

    # env.scene : InteractiveScene
    # env.scene의 속성(AssetBase, SensorBase)의 요소
        # cfg, device, num_instances, data

    # scene.env_origins[env_ids], reset하는 환경 원점 tensor 선택
    # pose[:, :3] reset하는 환경의 xyz tensor를 원점에서 빼기 -> (0, 0) 원점 기준 자세 구하기
    pose[:, :3] -= env.scene.env_origins[env_ids]

    # 합치기
    # J = 18 (관절 개수)
    # s : (len(env_ids), 7 + 6 + J + J)
    s = torch.cat([pose, d.root_vel_w.torch[env_ids], d.joint_pos.torch[env_ids], d.joint_vel.torch[env_ids]], 1)

    # phases.unique().tolist() : phases 중복 제거 리스트
    # s[phases == k] : phases가 k인 states stair_states[k]에 붙이기, 만약 없으면 빈 s에 s 붙이기
    # [-size:] : size만큼 자르기
    for k in phases.unique().tolist():
        env.stair_states[k] = torch.cat([env.stair_states.get(k, s[:0]), s[phases == k]])[-size:]
    

def reset_stair_phase(env, env_ids, x_ranges, buffer_ratio: float = 0.5):
    """시작 phase를 뽑아 그 시작 상태로 옮김 (reset_scene_to_default 다음에 실행).
    x_ranges의 phase(APPROACH, MOVE_FRONT): 기본 자세 + x_ranges 위치.
    buffer_ratio 비율로 그 뒤 phase(MOVE_MIDDLE ~ SETTLE): 저장된 상태 (비어 있으면 APPROACH)."""
    # env: ManagerBasedRLEnv
    # env_ids: tensor | slice(None)

    if isinstance(env_ids, slice): 
        # 첫 env.reset()은 slice(None)으로 호출됨
        # 궁극적으로 slice(None)은 인덱스 tensor와 똑같지만, 길이와 형태에 대한 데이터가 없다.
        env_ids = torch.arange(env.num_envs, device=env.device)

    # n: reset 해당 env 인덱스 갯수
    # S: buffer size
    # buf: env의 stair.states (S, 7 + 6 + J + J)
    n, buf = len(env_ids), getattr(env, "stair_states", {})

    # x_ranges : n개의 env에 대해 
    p = torch.randint(len(x_ranges), (n,), device=env.device)
    use_buf = torch.rand(n, device=env.device) < buffer_ratio
    p = torch.where(use_buf, torch.randint(len(x_ranges), PHASE_DONE, (n,), device=env.device), p)
    for k in range(len(x_ranges), PHASE_DONE):
        if k not in buf:
            p[p == k] = PHASE_APPROACH
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
    """계단 오르기 보상. 매 step 다음 순서로 계산한다.
    1. 측정: 발끝 위치, 쌍별 단 번호, 올라갈 단(target)
    2. 최고기록 갱신량: 전진/발 높이의 신기록만 보상 (제자리 왕복으로는 못 받음)
    3. phase별 진행 보상 + 완료 판정
       APPROACH: 앞발 x 전진 → 앞발이 첫 수직면 바로 앞에 도달하면 완료
       MOVE_*: 옮기는 쌍의 발 높이 + 몸통 x (중간발은 중간발 x 추가, 뒷발은 뒷발 x + 몸통 x × rear_body_fwd) → 두 발 다 target 단을 디디면 완료
       SETTLE: 기본 자세로 정지한 정도 → 멈춘 상태를 hold_time 동안 유지하면 완료
    4. 다음 phase 고르기 (완료 시 bonus 1회)
    5. 매 step 보상: 단 위 유지(hold), 몸통 전진(body_fwd), 뒷발 지지(support), 옮기는 발의 높이(lift_hold)"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        names = env.scene[asset_cfg.name].body_names
        self.offset = tip_offsets([names[i] for i in asset_cfg.body_ids], env.device)
        self.best_x = torch.zeros(env.num_envs, device=env.device)
        self.best_bx = torch.zeros(env.num_envs, device=env.device)
        self.best_mx = torch.zeros(env.num_envs, device=env.device)
        self.best_rx = torch.zeros(env.num_envs, device=env.device)
        self.best_z = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
        self.target = torch.ones(env.num_envs, device=env.device)  # 지금 옮기는 쌍이 올라갈 단 번호
        self.steady_time = torch.zeros(env.num_envs, device=env.device)  # SETTLE에서 멈춘 상태를 유지한 시간 [s]
        self.need_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_phase = torch.zeros(env.num_envs, device=env.device)
        self.last_level = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.need_init[env_ids] = True
        self.steady_time[env_ids] = 0.0
        ended_phase = self.last_phase[env_ids]  # 끝난 에피소드의 phase
        self._env.extras["log"]["Metrics/stair_phase"] = ended_phase.mean()
        self._env.extras["log"]["Metrics/stair_level"] = self.last_level[env_ids].mean()  # 끝난 에피소드의 쌍별 평균 단
        self._env.extras["log"]["Metrics/stair_reached_top"] = (ended_phase >= PHASE_SETTLE).float().mean()  # 꼭대기 도달 비율
        self._env.extras["log"]["Metrics/stair_success"] = (ended_phase == PHASE_DONE).float().mean()  # 정지까지 성공한 비율

    def _start_target(self, phase, levels, stairs):
        """리셋 직후의 target. 저장된 상태에서 시작하면 그 phase에 맞는 단을 지금 자세에서 계산한다."""
        moving_pair = (phase - PHASE_MOVE_FRONT).clamp(0, len(PAIRS) - 1)
        target = levels.gather(1, moving_pair.unsqueeze(1)).squeeze(1) + 1  # MOVE_*: 그 쌍의 지금 단 + 1
        target = target.masked_fill(phase == PHASE_APPROACH, 1.0)
        return target.masked_fill(phase == PHASE_SETTLE, float(stairs.num_steps))

    def _next_phase(self, phase, levels, stairs):
        """지금 phase를 끝냈을 때 넘어갈 phase. SETTLE → DONE, 나머지 → 가장 낮은 쌍의 MOVE (전부 꼭대기면 SETTLE)."""
        next_phase = PHASE_MOVE_FRONT + levels.argmin(1)
        next_phase = next_phase.masked_fill(levels.min(1).values >= stairs.num_steps, PHASE_SETTLE)
        return next_phase.masked_fill(phase == PHASE_SETTLE, PHASE_DONE)

    def _settle_state(self, env, asset_cfg, settle, feet_on_top):
        """SETTLE 판정. 반환: score (N) 0~1 정지 점수, steady (N) 멈춘 상태 여부."""
        data = env.scene[asset_cfg.name].data
        pose_error = (data.joint_pos.torch - data.default_joint_pos.torch).pow(2).mean(1).sqrt()
        tilt = data.projected_gravity_b.torch[:, :2].norm(dim=1)
        speed = data.root_vel_w.torch.norm(dim=1)
        score = (feet_on_top.float().mean(1)
                 * torch.exp(-(pose_error / settle.pose_std) ** 2)
                 * torch.exp(-(tilt / settle.tilt_std) ** 2)
                 * torch.exp(-(speed / settle.speed_std) ** 2))
        steady = (feet_on_top.all(1) & (pose_error < settle.pose_tol)
                  & (tilt < settle.tilt_tol) & (speed < settle.speed_tol))
        return score, steady

    def __call__(self, env, asset_cfg, stairs, swing_clearance: float = 0.08, contact_tol: float = 0.03,
                 bonus: float = 1.0, hold: float = 1.0, rear_body_fwd: float = 0.5, support: float = 0.5,
                 support_margin: float = 0.05, body_fwd: float = 1.0, fwd_ahead: float = 0.18,
                 fwd_ahead_before_middle: float = 0.05, lift_hold: float = 1.0, settle: SettleCfg = SettleCfg()):
        # ----- 1. 측정 -----
        tips = feet_tip_pos(env, asset_cfg, self.offset)
        terrain = stairs_terrain_height(stairs, tips[..., 0], tips[..., 1])
        levels = pair_levels(stairs, terrain)
        phase = get_phase(env)
        self.target = torch.where(self.need_init, self._start_target(phase, levels, stairs), self.target)

        top = (self.target * stairs.step_height).unsqueeze(1)  # 올라갈 단 높이 (N, 1)
        riser = stairs.x0 + (self.target - 1) * stairs.tread   # 올라갈 단 수직면 x
        approach_x = stairs.x0 - 0.05
        x = tips[:, FRONT, 0].min(dim=1).values.clamp(max=approach_x)  # 두 앞발 중 뒤처진 발
        z = torch.minimum(tips[..., 2], top + swing_clearance)  # 올라갈 단 + swing_clearance 위로는 보상 없음
        bx = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
        # 몸통 전진 보상의 상한. 중간발이 올라가기 전에 몸통이 너무 앞서면 중간다리가 단 모서리에 끼므로 상한을 낮춘다
        # (몸통이 수직면 + 0.05에 있으면 중간발의 기본 위치가 이미 단 위라서, 들었다 내리기만 해도 딛을 수 있다)
        before_middle = (phase == PHASE_MOVE_FRONT) | (phase == PHASE_MOVE_MIDDLE)
        body_limit = riser + torch.where(before_middle, fwd_ahead_before_middle, fwd_ahead)
        capped_bx = torch.minimum(bx, body_limit)
        mx = torch.minimum(tips[:, MIDDLE, 0].min(dim=1).values, riser + 0.05)  # 두 중간발 중 뒤처진 발 (단 위 5cm까지)
        rx = torch.minimum(tips[:, REAR, 0].min(dim=1).values, riser + 0.05)  # 두 뒷발 중 뒤처진 발 (단 위 5cm까지)
        # 옮기는 발이 지금 단에서 올라갈 단까지 얼마나 올라왔는지 (0~1). 단 높이에서 잘라서 단에 내려놓아도 점수가 줄지 않는다
        lift_score = ((tips[..., 2] - (top - stairs.step_height)) / stairs.step_height).clamp(0.0, 1.0)
        # 올라갈 단 이상을 딛고 있음: 발 아래 지형 ≥ 그 단 높이, 지형과의 높이 차 contact_tol 이내 (발이 단보다 낮으면 안 침)
        contact = (tips[..., 2] - terrain).abs() < contact_tol
        on = (terrain >= top - 1e-3) & contact

        # ----- 2. 최고기록 갱신량 -----
        self.best_x = torch.where(self.need_init, x, self.best_x)
        self.best_bx = torch.where(self.need_init, capped_bx, self.best_bx)
        self.best_mx = torch.where(self.need_init, mx, self.best_mx)
        self.best_rx = torch.where(self.need_init, rx, self.best_rx)
        self.best_z = torch.where(self.need_init.unsqueeze(1), z, self.best_z)
        self.need_init[:] = False
        gx = (x - self.best_x).clamp(min=0)
        gbx = (capped_bx - self.best_bx).clamp(min=0)
        gmx = (mx - self.best_mx).clamp(min=0)
        grx = (rx - self.best_rx).clamp(min=0)
        gz = (z - self.best_z).clamp(min=0)
        self.best_x = torch.maximum(self.best_x, x)
        self.best_bx = torch.maximum(self.best_bx, capped_bx)
        self.best_mx = torch.maximum(self.best_mx, mx)
        self.best_rx = torch.maximum(self.best_rx, rx)
        self.best_z = torch.maximum(self.best_z, z)

        # ----- 3. phase별 진행 보상 + 완료 판정 -----
        gain = torch.zeros(env.num_envs, device=env.device)
        lift_reward = torch.zeros(env.num_envs, device=env.device)
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        now = phase == PHASE_APPROACH
        gain += gx * now
        done |= now & (self.best_x >= approach_x)

        for k, feet in MOVE_PHASES.items():
            now = phase == k
            gain += gz[:, feet].sum(1) * now
            lift_reward += lift_score[:, feet].mean(1) * now
            done |= now & on[:, feet].all(1)
        gain += gbx * (phase == PHASE_MOVE_FRONT)
        gain += (gmx + gbx) * (phase == PHASE_MOVE_MIDDLE)
        # 뒷발은 몸을 밀지 않고 다리를 접어 단 안쪽으로
        gain += (grx + rear_body_fwd * gbx) * (phase == PHASE_MOVE_REAR)

        is_settle = phase == PHASE_SETTLE
        # SETTLE에 들어가도 손해가 없도록 lift_hold 최댓값을 그대로 준다. 줄어들면 일부러 뒷발을 든 채로 머무른다
        lift_reward += is_settle
        settle_score, steady = self._settle_state(env, asset_cfg, settle, on)
        settle_score = settle_score * is_settle
        # 멈춘 상태가 끊기면 시간을 처음부터 다시 잰다
        self.steady_time = torch.where(steady & is_settle, self.steady_time + env.step_dt, torch.zeros_like(self.steady_time))
        done |= is_settle & (self.steady_time >= settle.hold_time)

        # ----- 4. 다음 phase -----
        # 새 phase는 지금부터 다시 기록
        self.best_z[done] = z[done]
        self.best_bx[done] = bx[done]
        self.best_mx[done] = mx[done]
        self.best_rx[done] = rx[done]
        next_target = (levels.min(1).values + 1).clamp(max=stairs.num_steps)  # SETTLE의 target = 꼭대기 단
        self.target = torch.where(done, next_target, self.target)
        phase[:] = torch.where(done, self._next_phase(phase, levels, stairs), phase)
        self.last_phase = phase.float()
        self.last_level = levels.mean(1)
        # MOVE_FRONT 다음 phase에 막 들어선 상태 저장 → 리셋 때 시작 상태로 사용 (APPROACH, MOVE_FRONT는 x_ranges로 시작)
        save = done & (phase > PHASE_MOVE_FRONT) & (phase < PHASE_DONE)
        if save.any():
            save_phase_states(env, save.nonzero().squeeze(-1), phase[save])

        # ----- 5. 공통 보상 -----
        # 유지 보상: 1단 이상을 딛고 있는 다리 쌍마다 매 step
        on_step = (terrain >= stairs.step_height - 1e-3) & contact
        held = sum(on_step[:, p].all(1).float() for p in PAIRS)
        # 뒷발 지지: 몸통(무게중심)이 중간발보다 support_margin 앞 → 뒷발을 들어도 안 넘어짐 (매 step, 상한 1)
        margin = (bx - tips[:, MIDDLE, 0].mean(1)) / support_margin
        stable = margin.clamp(0.0, 1.0) * (phase == PHASE_MOVE_REAR)
        # SETTLE에 들어가도 손해가 없도록 support 최댓값을 그대로 준다. 줄어들면 일부러 뒷발 phase에 머무른다
        stable = stable + (phase == PHASE_SETTLE)
        # 몸통 전진: 앞에 있을수록 매 step (hold와 같은 크기로 경쟁), body_limit에서 포화 → 앞발이 다음 수직면에 안 닿게
        fwd = capped_bx.clamp(min=0.0)
        return ((gain + bonus * done) / env.step_dt + hold * held + support * stable + body_fwd * fwd
                + lift_hold * lift_reward + settle.reward * settle_score)
