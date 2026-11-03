"""계단 오르기 teacher의 단계 진행과 보상.

단마다 PHASES 표의 단계를 순서대로 밟는다. 전부 오르면 마지막 전진 → 정지 → 성공.

    앞발:   전진 → 올리기 → 전진 → 내려놓기
    중간발:        올리기 → 전진 → 내려놓기
    뒷발:   전진 → 올리기 → 전진 → 내려놓기

    전진: 몸통 x          올리기: 옮기는 발의 z          내려놓기: 옮기는 발이 올라갈 단에 닿았는가

발은 femur를 올려서 든다. tibia를 접으면 발이 몸 안쪽으로 들어오므로, 올리기와 든 채 전진에서는
tibia를 접거나 femur를 내린 발의 보상과 완료를 인정하지 않는다.

보상은 신기록 갱신량과 단계 완료 보너스뿐이라, 가만히 있으면 점수가 들어오지 않는다.
단계 번호 move는 앞으로만 증가한다. 한 step의 계산은 StairTracker.update()에 모여 있다.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from .observations import stairs_terrain_height, tip_offsets

# ===== 다리 =====
# 발 순서: tibia(L1), tibia_2 ~ tibia_6
FOOT_BODIES = ["tibia", "tibia_2", "tibia_3", "tibia_4", "tibia_5", "tibia_6"]
FEMUR_JOINTS = [f"L{i}_femur" for i in range(1, 7)]
TIBIA_JOINTS = [f"L{i}_tibia" for i in range(1, 7)]
FRONT, MIDDLE, REAR = [0, 5], [1, 4], [2, 3]

# ===== 단계 =====
ADVANCE, LIFT, CARRY, PLACE = 0, 1, 2, 3  # 전진, 올리기, 발을 든 채 전진, 내려놓기


class Phase(NamedTuple):
    kind: int
    feet: tuple = ()         # 옮기는 쌍
    body_x: float = 0.0      # 전진 단계에서 몸통 x의 목표이자 보상 상한 (이번에 오를 단의 수직면 기준 [m])


# 몸통 x는 URDF 계산에서 나온 값. 너무 앞서면 다리가 단 모서리에 끼므로 여기서 전진 보상을 멈춘다
PHASES = [
    Phase(ADVANCE, body_x=-0.13),
    Phase(LIFT, FRONT),
    Phase(CARRY, FRONT, -0.03),
    Phase(PLACE, FRONT),
    Phase(LIFT, MIDDLE),
    Phase(CARRY, MIDDLE, 0.03),
    Phase(PLACE, MIDDLE),
    # 뒷발을 들기 전에 무게중심을 중간발보다 앞으로 옮긴다
    Phase(ADVANCE, body_x=0.08),
    Phase(LIFT, REAR),
    Phase(CARRY, REAR, 0.14),
    Phase(PLACE, REAR),
]
NUM_PHASES = len(PHASES)
PHASE_OBS_SIZE = NUM_PHASES + 1  # 표의 단계 + 정지


@configclass
class StairGaitCfg:
    """판정 기준. 보상 크기는 env_cfg의 weight에서 정한다."""

    # 전진
    advance_tol: float = 0.01    # 전진 완료: 목표 x까지 남은 거리
    carry_foot_gain: float = 1.0 # 발을 든 채 전진할 때 옮기는 발 x 신기록의 비중 (뒷발은 몸통만 밀어서는 단을 넘지 못한다)
    over_margin: float = 0.02    # 발이 수직면을 이만큼 넘으면 "단 위로 넘어감"

    # 올리기
    raise_height: float = 0.03   # 단 윗면 위로 이만큼 올라오면 올리기 완료
    raise_reward_cap: float = 0.05  # 단 윗면 위로 이 높이까지만 보상
    tibia_fold_tol: float = 0.17 # 올리기 시작할 때보다 tibia를 이만큼 넘게 접으면 인정 안 함 [rad, 약 10°]
    femur_drop_tol: float = 0.09 # 올리기 시작할 때보다 femur를 이만큼 넘게 내리면 인정 안 함 [rad, 약 5°]

    # 접지
    tip_height: float = 0.013    # 딛었을 때 발끝 추정점이 지면보다 높은 만큼 (메시 최저점과의 차이)
    contact_tol: float = 0.03    # 발끝과 지면의 높이 차 허용치

    # 정지
    hold_time: float = 1.0       # 아래 속도 이하를 연속으로 지켜야 하는 시간 [s]
    speed_tol: float = 0.1       # 몸통 속도 크기
    speed_std: float = 0.3

    # 리셋 버퍼
    buffer_size: int = 1000
    save_speed_max: float = 0.5  # 이 속도 이하로 안정적일 때만 시작 상태로 저장


class StairTracker:
    """env별 진행 상태를 보관하고, step마다 한 번 진행 상태와 점수를 계산한다."""

    def __init__(self, env):
        self.env = env
        self.stairs = env.cfg.stairs
        self.gait = env.cfg.stair_gait
        self.asset = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.foot_ids, _ = self.asset.find_bodies(FOOT_BODIES, preserve_order=True)
        self.sensor_ids, _ = self.sensor.find_bodies(FOOT_BODIES, preserve_order=True)
        self.femur_ids, _ = self.asset.find_joints(FEMUR_JOINTS, preserve_order=True)
        self.tibia_ids, _ = self.asset.find_joints(TIBIA_JOINTS, preserve_order=True)
        self.tip_offset = tip_offsets(FOOT_BODIES, env.device)

        # 단계 번호: 0 ~ final_advance - 1 = 계단, final_advance = 마지막 전진, settle_move = 정지, done_move = 성공
        self.final_advance = NUM_PHASES * self.stairs.num_steps
        self.settle_move = self.final_advance + 1
        self.done_move = self.settle_move + 1

        # PHASES 표를 텐서로
        device, n, num_feet = env.device, env.num_envs, len(FOOT_BODIES)
        self.table_kind = torch.tensor([p.kind for p in PHASES], device=device)
        self.table_body_x = torch.tensor([p.body_x for p in PHASES], device=device)
        self.table_moving = torch.zeros(NUM_PHASES, num_feet, dtype=torch.bool, device=device)
        self.table_placed = torch.zeros(NUM_PHASES, num_feet, device=device)  # 이번 단에서 이미 올려놓은 발 (1 = 올려놓음)
        placed = []
        for k, phase in enumerate(PHASES):
            self.table_moving[k, list(phase.feet)] = True
            self.table_placed[k, placed] = 1.0
            if phase.kind == PLACE:
                placed = placed + list(phase.feet)
        self.table_feet = torch.tensor([p.feet if p.feet else (0, 0) for p in PHASES], device=device)

        # env별 상태
        self.move = torch.zeros(n, dtype=torch.long, device=device)
        self.need_init = torch.ones(n, dtype=torch.bool, device=device)  # 새 단계 시작 → 신기록을 지금 값으로
        self.best_body_x = torch.zeros(n, device=device)
        self.best_foot_x = torch.zeros(n, device=device)
        self.best_z = torch.zeros(n, num_feet, device=device)
        self.need_leg_ref = torch.ones(n, dtype=torch.bool, device=device)  # 올리기 시작 → 기준 각도를 지금 값으로
        self.femur_ref = torch.zeros(n, num_feet, device=device)
        self.tibia_ref = torch.zeros(n, num_feet, device=device)
        self.steady_time = torch.zeros(n, device=device)  # 멈춘 상태를 유지한 시간 [s]
        self.ended_move = torch.zeros(n, dtype=torch.long, device=device)  # 끝난 에피소드의 단계 번호 (로그용)
        self._updated_at = -1

        # 이번 step의 결과 (보상 함수들이 읽는다)
        self.just_done = torch.zeros(n, dtype=torch.bool, device=device)
        self.progress = torch.zeros(n, device=device)
        self.settle_score = torch.zeros(n, device=device)

    # ----- 지금 단계 -----
    def phase_and_level(self):
        """표의 몇 번째 단계인지, 몇 번째 단(바닥 = 0)을 오르는 중인지.
        마지막 전진과 정지는 "꼭대기 다음 단을 시작하는 전진"과 같은 값으로 계산한다."""
        climbing = self.move.clamp(max=self.final_advance)
        return climbing % NUM_PHASES, climbing // NUM_PHASES

    def riser_and_top(self, level):
        """이번에 오를 단의 수직면 x와 윗면 높이."""
        return self.stairs.x0 + level * self.stairs.tread, (level + 1) * self.stairs.step_height

    def moving_feet(self, phase):
        """지금 옮기는 발 (N, 6). 전진과 정지 중에는 없음."""
        return self.table_moving[phase] & (self.move < self.final_advance).unsqueeze(1)

    def tips(self):
        """발끝 추정 위치 (env 원점 기준) (N, 6, 3) = 무릎 위치 + 회전 × offset."""
        d = self.asset.data
        quat = d.body_quat_w.torch[:, self.foot_ids]
        offset_w = quat_apply(quat, self.tip_offset.expand(quat.shape[0], -1, -1))
        return d.body_pos_w.torch[:, self.foot_ids] + offset_w - self.env.scene.env_origins.unsqueeze(1)

    def body_state(self):
        """몸통 위치 (env 원점 기준) (N, 3)과 yaw 오차 (앞 방향 벡터의 y 성분) (N,)."""
        pose = self.asset.data.root_pose_w.torch
        forward = torch.zeros_like(pose[:, :3])
        forward[:, 0] = 1.0
        return pose[:, :3] - self.env.scene.env_origins, quat_apply(pose[:, 3:7], forward)[:, 1]

    # ----- step마다 한 번 -----
    def update(self):
        counter = self.env.common_step_counter
        if counter == self._updated_at:
            return
        self._updated_at = counter

        gait = self.gait
        phase, level = self.phase_and_level()
        riser, top = self.riser_and_top(level)
        moving = self.moving_feet(phase)
        kind = self.table_kind[phase]
        in_stairs = self.move < self.final_advance
        advancing = (kind == ADVANCE) & (self.move <= self.final_advance)
        lifting = (kind == LIFT) & in_stairs
        carrying = (kind == CARRY) & in_stairs
        placing = (kind == PLACE) & in_stairs
        settling = self.move == self.settle_move

        tips = self.tips()
        terrain = stairs_terrain_height(self.stairs, tips[..., 0], tips[..., 1])
        air_time = self.sensor.data.current_air_time.torch[:, self.sensor_ids]
        near_ground = (tips[..., 2] - terrain - gait.tip_height).abs() < gait.contact_tol
        body_x = self.body_state()[0][:, 0]

        # femur로 들고 있는 발: 올리기 시작할 때보다 tibia를 접거나 femur를 내리지 않음
        femur = self.asset.data.joint_pos.torch[:, self.femur_ids]
        tibia = self.asset.data.joint_pos.torch[:, self.tibia_ids]
        self.femur_ref = torch.where(self.need_leg_ref.unsqueeze(1), femur, self.femur_ref)
        self.tibia_ref = torch.where(self.need_leg_ref.unsqueeze(1), tibia, self.tibia_ref)
        self.need_leg_ref[:] = False
        by_femur = (tibia >= self.tibia_ref - gait.tibia_fold_tol) & (femur >= self.femur_ref - gait.femur_drop_tol)
        moving_by_femur = (by_femur | ~moving).all(1)

        # 현재 값 (보상 상한으로 자른 값)
        limit_x = riser + self.table_body_x[phase]
        capped_body_x = torch.minimum(body_x, limit_x)
        lagging_foot_x = tips[..., 0].masked_fill(~moving, float("inf")).min(1).values  # 옮기는 두 발 중 뒤처진 발
        capped_foot_x = torch.minimum(lagging_foot_x, riser + gait.over_margin)
        capped_z = torch.minimum(tips[..., 2], (top + gait.raise_reward_cap + gait.tip_height).unsqueeze(1))

        # 새 단계의 첫 step: 신기록을 지금 값으로
        self.best_body_x = torch.where(self.need_init, capped_body_x, self.best_body_x)
        self.best_foot_x = torch.where(self.need_init, capped_foot_x, self.best_foot_x)
        self.best_z = torch.where(self.need_init.unsqueeze(1), capped_z, self.best_z)
        self.need_init[:] = False

        # 발을 든 채 전진: 옮기는 두 발이 단 높이 아래로 떨어지거나 tibia를 접으면 보상도 신기록도 멈춘다
        feet_up = ((tips[..., 2] >= top.unsqueeze(1)) | ~moving).all(1) & moving_by_femur
        body_counts = advancing | (carrying & feet_up)
        foot_counts = carrying & feet_up

        # 보상: 신기록 갱신량 (단위 m)
        gain_body = (capped_body_x - self.best_body_x).clamp(min=0.0) * body_counts
        gain_foot = (capped_foot_x - self.best_foot_x).clamp(min=0.0) * foot_counts
        z_counts = lifting.unsqueeze(1) & moving & by_femur  # 올리기: femur로 들고 있는 발만
        gain_z = ((capped_z - self.best_z).clamp(min=0.0) * z_counts).sum(1)
        self.progress = gain_body + gait.carry_foot_gain * gain_foot + gain_z
        self.best_body_x = torch.where(body_counts, torch.maximum(self.best_body_x, capped_body_x), self.best_body_x)
        self.best_foot_x = torch.where(foot_counts, torch.maximum(self.best_foot_x, capped_foot_x), self.best_foot_x)
        self.best_z = torch.where(z_counts, torch.maximum(self.best_z, capped_z), self.best_z)

        # 단계 완료 조건
        advance_done = advancing & (self.best_body_x >= limit_x - gait.advance_tol)
        lift_done = lifting & moving_by_femur & ((tips[..., 2] >= (top + gait.raise_height).unsqueeze(1)) | ~moving).all(1)
        carry_done = carrying & feet_up & (lagging_foot_x >= riser + gait.over_margin)
        on_target = (terrain >= top.unsqueeze(1) - 1e-3) & near_ground & (air_time == 0)
        place_done = placing & (on_target | ~moving).all(1)

        # 옮기지 않는 발은 지금까지 올린 단에 있어야 단계를 끝낼 수 있다
        placed_height = (level.unsqueeze(1) + self.table_placed[phase]) * self.stairs.step_height
        feet_kept = ((terrain >= placed_height - 1e-3) & near_ground) | moving

        # 정지: 여섯 발이 꼭대기에 있고 몸통이 멈춤
        speed = self.asset.data.root_vel_w.torch.norm(dim=1)
        all_kept = feet_kept.all(1)
        steady = settling & all_kept & (air_time == 0).all(1) & (speed < gait.speed_tol)
        # 멈춘 상태가 끊기면 시간을 처음부터 다시 잰다
        self.steady_time = torch.where(steady, self.steady_time + self.env.step_dt, torch.zeros_like(self.steady_time))
        settle_done = settling & (self.steady_time >= gait.hold_time)
        self.settle_score = torch.exp(-(speed / gait.speed_std) ** 2) * settling * all_kept

        # 다음 단계로
        self.just_done = ((advance_done | lift_done | carry_done | place_done) & all_kept) | settle_done
        self.move = self.move + self.just_done.long()
        self.need_init |= self.just_done
        self.need_leg_ref |= self.just_done & (self.table_kind[self.phase_and_level()[0]] == LIFT)

        # 새 단계에 막 들어선 안정적인 상태 저장 → 리셋 때 시작 상태로 사용
        save = self.just_done & (self.move < self.done_move) & (speed < gait.save_speed_max)
        if save.any():
            save_stair_states(self.env, save.nonzero().squeeze(-1), self.move[save], gait.buffer_size)

    def reset(self, env_ids, moves):
        self.ended_move[env_ids] = self.move[env_ids]
        self.move[env_ids] = moves
        self.need_init[env_ids] = True
        self.need_leg_ref[env_ids] = True
        self.steady_time[env_ids] = 0.0
        self.just_done[env_ids] = False
        self.progress[env_ids] = 0.0
        self.settle_score[env_ids] = 0.0


def get_tracker(env) -> StairTracker:
    if not hasattr(env, "stair_tracker"):
        env.stair_tracker = StairTracker(env)
    return env.stair_tracker


# ===== 리셋 =====
def save_stair_states(env, env_ids, moves, size: int = 1000):
    """
    env_ids의 현재 상태를 단계 번호별 버퍼에 추가 (위치는 env 원점 기준, 꽉 차면 오래된 것부터 버림).

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

    # moves.unique().tolist() : moves 중복 제거 리스트
    # s[moves == k] : 단계 번호가 k인 상태를 stair_states[k]에 붙이기, 만약 없으면 빈 s에 s 붙이기
    # [-size:] : size만큼 자르기
    for k in moves.unique().tolist():
        env.stair_states[k] = torch.cat([env.stair_states.get(k, s[:0]), s[moves == k]])[-size:]


def reset_stair_state(env, env_ids, x_noise: float = 0.03, buffer_ratio: float = 0.5):
    """시작 상태를 뽑아 옮김 (reset_scene_to_default 다음에 실행).
    기본: 바닥에서 시작 자세 + x 오차. buffer_ratio 비율로 뒤쪽 단계의 저장된 상태 (비어 있으면 기본)."""
    # env: ManagerBasedRLEnv
    # env_ids: tensor | slice(None)

    if isinstance(env_ids, slice):
        # 첫 env.reset()은 slice(None)으로 호출됨
        # 궁극적으로 slice(None)은 인덱스 tensor와 똑같지만, 길이와 형태에 대한 데이터가 없다.
        env_ids = torch.arange(env.num_envs, device=env.device)

    tracker = get_tracker(env)
    # n: reset 해당 env 인덱스 갯수
    # buf: 단계 번호별 저장 상태 (S, 7 + 6 + J + J)
    n, buf = len(env_ids), getattr(env, "stair_states", {})

    # 단계 번호 뽑기: 1 ~ 정지 중 무작위, 저장된 상태가 없는 번호는 처음(0)부터
    moves = torch.randint(1, tracker.done_move, (n,), device=env.device)
    moves[torch.rand(n, device=env.device) >= buffer_ratio] = 0
    for k in range(1, tracker.done_move):
        if k not in buf:
            moves[moves == k] = 0
    tracker.reset(env_ids, moves)

    asset = env.scene["robot"]

    ids = env_ids[moves == 0]
    if len(ids):
        pose = asset.data.default_root_pose.torch[ids].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        pose[:, 0] += x_noise * (2.0 * torch.rand(len(ids), device=env.device) - 1.0)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)

    J = asset.num_joints
    for k, states in buf.items():
        ids = env_ids[moves == k]
        if len(ids) == 0:
            continue
        s = states[torch.randint(len(states), (len(ids),), device=env.device)]
        s[:, :3] += env.scene.env_origins[ids]
        asset.write_root_pose_to_sim_index(root_pose=s[:, :7], env_ids=ids)
        asset.write_root_velocity_to_sim_index(root_velocity=s[:, 7:13], env_ids=ids)
        asset.write_joint_position_to_sim_index(position=s[:, 13:13 + J], env_ids=ids)
        asset.write_joint_velocity_to_sim_index(velocity=s[:, 13 + J:], env_ids=ids)


# ===== 보상 =====
class stair_phase_bonus(ManagerTermBase):
    """단계를 마칠 때마다 1번 (weight = 단계 하나의 점수). 끝난 에피소드의 진행도도 로그로 남긴다."""

    def reset(self, env_ids=None):
        tracker = get_tracker(self._env)
        ended = tracker.ended_move[slice(None) if env_ids is None else env_ids]
        log = self._env.extras["log"]
        log["Metrics/stair_move"] = ended.float().mean()
        log["Metrics/stair_level"] = (ended // NUM_PHASES).float().mean()
        log["Metrics/stair_reached_top"] = (ended >= tracker.final_advance).float().mean()  # 꼭대기 도달 비율
        log["Metrics/stair_success"] = (ended == tracker.done_move).float().mean()  # 정지까지 성공한 비율

    def __call__(self, env):
        tracker = get_tracker(env)
        tracker.update()
        return tracker.just_done.float() / env.step_dt


def stair_progress(env) -> torch.Tensor:
    """지금 단계의 신기록 갱신량. 전진: 몸통 x, 올리기: 발 z, 발을 든 채 전진: 몸통 x + 발 x (weight = 1m당 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.progress / env.step_dt


def stair_settle_still(env) -> torch.Tensor:
    """정지: 꼭대기에서 몸통이 멈출수록 (weight = 초당 최대 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.settle_score


# ===== 종료 =====
def stair_success(env) -> torch.Tensor:
    tracker = get_tracker(env)
    return tracker.move >= tracker.done_move


# ===== 관측 =====
def stair_phase_obs(env) -> torch.Tensor:
    """지금 단계 one-hot (N, 12). 마지막 칸 = 정지."""
    tracker = get_tracker(env)
    phase, _ = tracker.phase_and_level()
    return F.one_hot(torch.where(tracker.move >= tracker.settle_move, NUM_PHASES, phase), PHASE_OBS_SIZE).float()


def stair_feet_target_obs(env) -> torch.Tensor:
    """옮기는 두 발의 위치: 수직면 기준 x, 단 윗면 기준 z, 기준 각도 대비 tibia 변화 (N, 6). 옮기는 발이 없으면 0."""
    tracker = get_tracker(env)
    phase, level = tracker.phase_and_level()
    riser, top = tracker.riser_and_top(level)
    feet = tracker.table_feet[phase]  # (N, 2)
    rows = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    tips = tracker.tips()[rows, feet]  # (N, 2, 3)
    tibia = tracker.asset.data.joint_pos.torch[:, tracker.tibia_ids]
    tibia_change = (tibia - tracker.tibia_ref)[rows, feet]  # (N, 2)
    obs = torch.cat([tips[..., 0] - riser.unsqueeze(1), tips[..., 2] - top.unsqueeze(1), tibia_change], dim=1)
    return obs * tracker.moving_feet(phase).any(1, keepdim=True)


def stair_body_target_obs(env) -> torch.Tensor:
    """전진 목표까지 남은 x, 옆 위치, yaw 오차 (N, 3)."""
    tracker = get_tracker(env)
    phase, level = tracker.phase_and_level()
    riser, _ = tracker.riser_and_top(level)
    body_pos, yaw_error = tracker.body_state()
    target_x = riser + tracker.table_body_x[phase]
    return torch.stack([target_x - body_pos[:, 0], body_pos[:, 1], yaw_error], dim=1)
