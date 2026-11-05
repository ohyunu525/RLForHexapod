"""계단 오르기 teacher의 단계 진행과 보상.

단마다 PHASES 표의 단계를 순서대로 밟는다. 전부 오르면 마지막 전진 → 정지 → 성공.

    전진 → 앞발 옮기기 → 중간발 옮기기 → 전진 → 뒷발 옮기기

    전진:   몸통 x가 목표에 도달하면 완료
    옮기기: 두 발을 들어 앞으로 보내고, 올라갈 단 위에 내려놓으면 완료

보상
    신기록 갱신량
        전진:   몸통 x
        옮기기: 옮기는 발끝 높이 (tibia가 45° 이내이고 안쪽으로 접히지 않았을 때만)
                몸통 x (옮기는 두 발이 단 높이 위에 있을 때만)
        뒷발 옮기기: 앞다리를 수직면과 평행하게 돌린 정도
    단계 완료 보너스
    발마다 단계당 1번: 단 끝에 처음 닿을 때, 단 위에 처음 놓을 때
    tibia 수직도: 항상, 여섯 다리 평균
    벌점: 옮기기 중 나머지 발이 제자리에서 떨어져 있을 때, 항상 몸통이 후진할 때

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
FRONT, MIDDLE, REAR = [0, 5], [1, 4], [2, 3]
FRONT_COXA_JOINTS = ["L1_coxa", "L6_coxa"]  # FRONT와 같은 순서

# ===== 단계 =====
ADVANCE, MOVE = 0, 1  # 전진, 옮기기


class Phase(NamedTuple):
    kind: int
    feet: tuple = ()         # 옮기는 쌍
    body_x: float = 0.0      # 몸통 x의 보상 상한이자 전진 단계의 목표 (이번에 오를 단의 수직면 기준 [m])


# 몸통 x: 기본 자세에서 발끝은 몸통 기준 앞 +15cm, 중간 0, 뒤 -15cm. 그 발이 수직면을 넘을 만큼만 전진하고 보상을 멈춘다
PHASES = [
    Phase(ADVANCE, body_x=-0.20),
    Phase(MOVE, FRONT, -0.10),
    Phase(MOVE, MIDDLE, 0.05),
    # 뒷발을 들기 전에 무게중심을 중간발보다 앞으로 옮긴다
    Phase(ADVANCE, body_x=0.08),
    Phase(MOVE, REAR, 0.20),
]
NUM_PHASES = len(PHASES)
PHASE_OBS_SIZE = NUM_PHASES + 1  # 표의 단계 + 정지


@configclass
class StairGaitCfg:
    """판정 기준. 보상 크기는 env_cfg의 weight에서 정한다."""

    # 전진
    advance_tol: float = 0.01    # 전진 완료: 목표 x까지 남은 거리
    backward_speed_tol: float = 0.05  # 이 속도 이하의 후진은 흔들림으로 보고 벌점 없음 [m/s]

    # 옮기기
    raise_reward_cap: float = 0.05  # 발끝 높이 보상의 상한: 단 위에 디뎠을 때보다 이만큼 높은 곳까지
    min_vertical: float = 0.71   # 높이 보상에 필요한 tibia 수직도 (1 = 수직, 0.71 = 45°). 단 높이에서는 완전한 수직이 불가능하다
    inward_tol: float = 0.03     # 발끝이 무릎보다 몸 안쪽으로 이만큼 넘게 들어오면 높이 보상 없음
    over_margin: float = 0.02    # 옮기기 완료: 발이 수직면을 이만큼 넘어 단 위에 닿음
    edge_reach: float = 0.015    # 단 끝 닿기: 발끝 추정점이 수직면보다 이만큼 앞이어도 인정 (실제 발 앞 끝은 추정점보다 1.1~1.6cm 앞)

    # 뒷발 옮기기: 앞다리를 수직면과 평행하게 돌리면 앞발이 몸통 기준 +15cm에서 +9cm로 들어와 뒷발 자리가 생긴다
    front_parallel_coxa: float = 0.96  # 앞다리가 수직면과 평행해지는 coxa 각도 크기 [rad] (55°, L1은 +, L6은 -)

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
        self.front_coxa_ids, _ = self.asset.find_joints(FRONT_COXA_JOINTS, preserve_order=True)
        self.tip_offset = tip_offsets(FOOT_BODIES, env.device)

        # 단계 번호: 0 ~ final_advance - 1 = 계단, final_advance = 마지막 전진, settle_move = 정지, done_move = 성공
        self.final_advance = NUM_PHASES * self.stairs.num_steps
        self.settle_move = self.final_advance + 1
        self.done_move = self.settle_move + 1

        # PHASES 표를 텐서로
        device, n, num_feet = env.device, env.num_envs, len(FOOT_BODIES)
        self.table_kind = torch.tensor([p.kind for p in PHASES], device=device)
        self.table_body_x = torch.tensor([p.body_x for p in PHASES], device=device)
        self.table_rear_move = torch.tensor([p.feet == REAR for p in PHASES], device=device)
        self.table_moving = torch.zeros(NUM_PHASES, num_feet, dtype=torch.bool, device=device)
        self.table_placed = torch.zeros(NUM_PHASES, num_feet, device=device)  # 이번 단에서 이미 올려놓은 발 (1 = 올려놓음)
        placed = []
        for k, phase in enumerate(PHASES):
            self.table_moving[k, list(phase.feet)] = True
            self.table_placed[k, placed] = 1.0
            if phase.kind == MOVE:
                placed = placed + list(phase.feet)
        self.table_feet = torch.tensor([p.feet if p.feet else (0, 0) for p in PHASES], device=device)

        # env별 상태
        self.move = torch.zeros(n, dtype=torch.long, device=device)
        self.need_init = torch.ones(n, dtype=torch.bool, device=device)  # 새 단계 시작 → 신기록을 지금 값으로
        self.best_body_x = torch.zeros(n, device=device)
        self.best_z = torch.zeros(n, num_feet, device=device)
        self.best_fold = torch.zeros(n, len(FRONT_COXA_JOINTS), device=device)
        self.front_parallel = torch.tensor([self.gait.front_parallel_coxa, -self.gait.front_parallel_coxa], device=device)
        self.foot_touched = torch.zeros(n, num_feet, dtype=torch.bool, device=device)  # 이번 단계에 단 끝에 닿은 발
        self.foot_placed = torch.zeros(n, num_feet, dtype=torch.bool, device=device)  # 이번 단계에 단 위에 놓인 발
        self.steady_time = torch.zeros(n, device=device)  # 멈춘 상태를 유지한 시간 [s]
        self.ended_move = torch.zeros(n, dtype=torch.long, device=device)  # 끝난 에피소드의 단계 번호 (로그용)
        self._updated_at = -1

        # 이번 step의 결과 (보상 함수들이 읽는다)
        self.just_done = torch.zeros(n, dtype=torch.bool, device=device)
        self.progress = torch.zeros(n, device=device)
        self.fold_gain = torch.zeros(n, device=device)
        self.new_touches = torch.zeros(n, device=device)
        self.new_placed = torch.zeros(n, device=device)
        self.vertical_score = torch.zeros(n, device=device)
        self.standing_feet_off = torch.zeros(n, device=device)
        self.backward_speed = torch.zeros(n, device=device)
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

    def knees_and_tips(self):
        """무릎과 발끝 추정 위치 (env 원점 기준). 각각 (N, 6, 3). 발끝 = 무릎 위치 + 회전 × offset."""
        d = self.asset.data
        quat = d.body_quat_w.torch[:, self.foot_ids]
        offset_w = quat_apply(quat, self.tip_offset.expand(quat.shape[0], -1, -1))
        knees = d.body_pos_w.torch[:, self.foot_ids] - self.env.scene.env_origins.unsqueeze(1)
        return knees, knees + offset_w

    def tips(self):
        return self.knees_and_tips()[1]

    def tibia_vertical(self, knees, tips):
        """tibia가 지면에 수직인 정도 (N, 6). 1 = 무릎 바로 아래 발끝, 0 = 수평."""
        tibia = tips - knees
        return (-tibia[..., 2] / tibia.norm(dim=-1).clamp(min=1e-6)).clamp(0.0, 1.0)

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
        moving_pair = (kind == MOVE) & in_stairs
        settling = self.move == self.settle_move

        knees, tips = self.knees_and_tips()
        terrain = stairs_terrain_height(self.stairs, tips[..., 0], tips[..., 1])
        air_time = self.sensor.data.current_air_time.torch[:, self.sensor_ids]
        near_ground = (tips[..., 2] - terrain - gait.tip_height).abs() < gait.contact_tol
        body_x = self.body_state()[0][:, 0]

        # 현재 값 (보상 상한으로 자른 값)
        limit_x = riser + self.table_body_x[phase]
        capped_body_x = torch.minimum(body_x, limit_x)
        capped_z = torch.minimum(tips[..., 2], (top + gait.tip_height + gait.raise_reward_cap).unsqueeze(1))

        # 앞다리를 수직면과 평행한 각도까지 돌린 비율 (0 = 기본 자세, 1 = 평행)
        fold = (self.asset.data.joint_pos.torch[:, self.front_coxa_ids] / self.front_parallel).clamp(max=1.0)

        # 새 단계의 첫 step: 신기록을 지금 값으로
        self.best_body_x = torch.where(self.need_init, capped_body_x, self.best_body_x)
        self.best_z = torch.where(self.need_init.unsqueeze(1), capped_z, self.best_z)
        self.best_fold = torch.where(self.need_init.unsqueeze(1), fold, self.best_fold)
        self.foot_touched &= ~self.need_init.unsqueeze(1)
        self.foot_placed &= ~self.need_init.unsqueeze(1)
        self.need_init[:] = False

        # tibia 수직도: 모든 단계에서 여섯 다리의 평균
        vertical = self.tibia_vertical(knees, tips)
        self.vertical_score = vertical.mean(1)

        # 발끝 높이는 tibia를 안쪽으로 접지 않은 자세로만 인정한다. 몸통 중심에서 수평 거리로 안쪽/바깥쪽을 판단
        center = self.body_state()[0][:, :2].unsqueeze(1)
        tip_radius = (tips[..., :2] - center).norm(dim=-1)
        knee_radius = (knees[..., :2] - center).norm(dim=-1)
        good_lift = (vertical >= gait.min_vertical) & (tip_radius >= knee_radius - gait.inward_tol)
        z_counts = moving & good_lift

        # 옮기는 중 몸통 x는 두 발이 단 높이 위에 있을 때만 인정한다 (들기 전에 전진하면 발이 수직면에 막힌다)
        feet_up = ((tips[..., 2] >= top.unsqueeze(1)) | ~moving).all(1)
        body_counts = advancing | (moving_pair & feet_up)

        # 보상: 신기록 갱신량 (단위 m)
        gain_body = (capped_body_x - self.best_body_x).clamp(min=0.0) * body_counts
        gain_z = ((capped_z - self.best_z).clamp(min=0.0) * z_counts).sum(1)
        self.progress = 2.0*gain_body + gain_z
        self.best_body_x = torch.where(body_counts, torch.maximum(self.best_body_x, capped_body_x), self.best_body_x)
        self.best_z = torch.where(z_counts, torch.maximum(self.best_z, capped_z), self.best_z)

        # 보상: 뒷발 옮기기 중 앞다리를 평행하게 돌린 비율의 신기록 (두 다리의 합)
        fold_counts = (moving_pair & self.table_rear_move[phase]).unsqueeze(1)
        self.fold_gain = ((fold - self.best_fold).clamp(min=0.0) * fold_counts).sum(1)
        self.best_fold = torch.where(fold_counts, torch.maximum(self.best_fold, fold), self.best_fold)

        # 단계 완료 조건
        advance_done = advancing & (self.best_body_x >= limit_x - gait.advance_tol)
        over_riser = tips[..., 0] >= (riser + gait.over_margin).unsqueeze(1)
        on_target = over_riser & (terrain >= top.unsqueeze(1) - 1e-3) & near_ground & (air_time == 0)
        move_done = moving_pair & (on_target | ~moving).all(1)

        # 발마다 단계당 1번: 단 끝에 처음 닿을 때, 단 위에 처음 놓을 때
        tip_z = tips[..., 2]
        at_edge = (
            (tips[..., 0] >= (riser - gait.edge_reach).unsqueeze(1))
            & (tip_z >= top.unsqueeze(1))
            & (tip_z < (top + gait.tip_height + gait.contact_tol).unsqueeze(1))
            & (air_time == 0)
        )
        first_touch = at_edge & moving & ~self.foot_touched
        first_placed = on_target & moving & ~self.foot_placed
        self.new_touches = first_touch.sum(1).float()
        self.new_placed = first_placed.sum(1).float()
        self.foot_touched |= first_touch
        self.foot_placed |= first_placed

        # 옮기지 않는 발은 지금까지 올린 단에 있어야 단계를 끝낼 수 있다
        placed_height = (level.unsqueeze(1) + self.table_placed[phase]) * self.stairs.step_height
        feet_kept = ((terrain >= placed_height - 1e-3) & near_ground) | moving

        # 벌점: 옮기는 동안 제자리에서 떨어진 나머지 발의 수 (전진 중에는 걸어야 하므로 없음)
        self.standing_feet_off = (~feet_kept).sum(1).float() * moving_pair

        # 벌점: 몸통의 후진 속도 (전진 보상은 신기록만 주므로 후진 비용이 따로 없다)
        backward = (-self.asset.data.root_vel_w.torch[:, 0]).clamp(min=0.0)
        self.backward_speed = backward * (backward > gait.backward_speed_tol)

        # 정지: 여섯 발이 꼭대기에 있고 몸통이 멈춤
        speed = self.asset.data.root_vel_w.torch.norm(dim=1)
        all_kept = feet_kept.all(1)
        steady = settling & all_kept & (air_time == 0).all(1) & (speed < gait.speed_tol)
        # 멈춘 상태가 끊기면 시간을 처음부터 다시 잰다
        self.steady_time = torch.where(steady, self.steady_time + self.env.step_dt, torch.zeros_like(self.steady_time))
        settle_done = settling & (self.steady_time >= gait.hold_time)
        self.settle_score = torch.exp(-(speed / gait.speed_std) ** 2) * settling * all_kept

        # 다음 단계로
        self.just_done = ((advance_done | move_done) & all_kept) | settle_done
        self.move = self.move + self.just_done.long()
        self.need_init |= self.just_done

        # 새 단계에 막 들어선 안정적인 상태 저장 → 리셋 때 시작 상태로 사용
        save = self.just_done & (self.move < self.done_move) & (speed < gait.save_speed_max)
        if save.any():
            save_stair_states(self.env, save.nonzero().squeeze(-1), self.move[save], gait.buffer_size)

    def reset(self, env_ids, moves):
        self.ended_move[env_ids] = self.move[env_ids]
        self.move[env_ids] = moves
        self.need_init[env_ids] = True
        self.steady_time[env_ids] = 0.0
        self.just_done[env_ids] = False
        self.progress[env_ids] = 0.0
        self.fold_gain[env_ids] = 0.0
        self.new_touches[env_ids] = 0.0
        self.new_placed[env_ids] = 0.0
        self.vertical_score[env_ids] = 0.0
        self.standing_feet_off[env_ids] = 0.0
        self.backward_speed[env_ids] = 0.0
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


def reset_stair_state(env, env_ids, x_ranges, buffer_ratio: float = 0.5):
    """시작 단계를 뽑아 그 시작 상태로 옮김 (reset_scene_to_default 다음에 실행).
    x_ranges의 단계(순서 = 단계 번호): 기본 자세 + x_ranges의 몸통 x 위치.
    buffer_ratio 비율로 그 뒤 단계: 저장된 상태 (비어 있으면 0번 단계)."""
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

    # 단계 번호 뽑기: x_ranges 단계 중 무작위, buffer_ratio 비율로 그 뒤 단계. 저장된 상태가 없는 번호는 0번부터
    moves = torch.randint(len(x_ranges), (n,), device=env.device)
    use_buf = torch.rand(n, device=env.device) < buffer_ratio
    moves = torch.where(use_buf, torch.randint(len(x_ranges), tracker.done_move, (n,), device=env.device), moves)
    for k in range(len(x_ranges), tracker.done_move):
        if k not in buf:
            moves[moves == k] = 0
    tracker.reset(env_ids, moves)

    asset = env.scene["robot"]

    by_hand = moves < len(x_ranges)
    ids = env_ids[by_hand]
    if len(ids):
        lo, hi = torch.tensor(x_ranges, device=env.device)[moves[by_hand]].unbind(-1)
        pose = asset.data.default_root_pose.torch[ids].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        pose[:, 0] += lo + (hi - lo) * torch.rand(len(ids), device=env.device)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)

    J = asset.num_joints
    for k, states in buf.items():
        if k < len(x_ranges):
            continue
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
    """지금 단계의 신기록 갱신량. 전진: 몸통 x, 옮기기: 발끝 높이 + 몸통 x (weight = 1m당 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.progress / env.step_dt


def stair_front_fold(env) -> torch.Tensor:
    """뒷발 옮기기 중 앞다리를 수직면과 평행하게 돌린 비율의 신기록, 두 다리의 합 (weight = 다리 하나를 끝까지 돌린 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.fold_gain / env.step_dt


def stair_edge_touch(env) -> torch.Tensor:
    """옮기는 발이 단 끝에 처음 닿은 수. 발마다 단계당 1번 (weight = 발 하나의 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.new_touches / env.step_dt


def stair_foot_placed(env) -> torch.Tensor:
    """옮기는 발이 단 위에 처음 놓인 수. 발마다 단계당 1번 (weight = 발 하나의 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.new_placed / env.step_dt


def stair_tibia_vertical(env) -> torch.Tensor:
    """모든 단계에서 여섯 다리의 tibia가 지면에 수직일수록 (weight = 초당 최대 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.vertical_score


def stair_standing_feet_off(env) -> torch.Tensor:
    """옮기기 중 제자리에서 떨어진 나머지 발의 수 (weight = 발 하나당 초당 벌점)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.standing_feet_off


def stair_backward(env) -> torch.Tensor:
    """모든 단계에서 몸통의 후진 속도 [m/s] (weight = 후진 1m당 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.backward_speed


def stair_settle_still(env) -> torch.Tensor:
    """정지: 꼭대기에서 몸통이 멈출수록 (weight = 초당 최대 점수)."""
    tracker = get_tracker(env)
    tracker.update()
    return tracker.settle_score


# ===== 종료 =====
def stair_success(env) -> torch.Tensor:
    tracker = get_tracker(env)
    return tracker.move >= tracker.done_move


def stair_reached_top(env) -> torch.Tensor:
    """마지막 전진까지 끝나 정지 단계에 들어섰는지."""
    tracker = get_tracker(env)
    return tracker.move >= tracker.settle_move


# ===== 관측 =====
def stair_phase_obs(env) -> torch.Tensor:
    """지금 단계 one-hot (N, 6). 마지막 칸 = 정지."""
    tracker = get_tracker(env)
    phase, _ = tracker.phase_and_level()
    return F.one_hot(torch.where(tracker.move >= tracker.settle_move, NUM_PHASES, phase), PHASE_OBS_SIZE).float()


def stair_feet_target_obs(env) -> torch.Tensor:
    """옮기는 두 발의 위치: 수직면 기준 x, 단 윗면 기준 z, tibia 수직도 (N, 6). 옮기는 발이 없으면 0."""
    tracker = get_tracker(env)
    phase, level = tracker.phase_and_level()
    riser, top = tracker.riser_and_top(level)
    feet = tracker.table_feet[phase]  # (N, 2)
    rows = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    knees, tips = tracker.knees_and_tips()
    vertical = tracker.tibia_vertical(knees, tips)[rows, feet]  # (N, 2)
    tips = tips[rows, feet]  # (N, 2, 3)
    obs = torch.cat([tips[..., 0] - riser.unsqueeze(1), tips[..., 2] - top.unsqueeze(1), vertical], dim=1)
    return obs * tracker.moving_feet(phase).any(1, keepdim=True)


def stair_body_target_obs(env) -> torch.Tensor:
    """전진 목표까지 남은 x, 옆 위치, yaw 오차 (N, 3)."""
    tracker = get_tracker(env)
    phase, level = tracker.phase_and_level()
    riser, _ = tracker.riser_and_top(level)
    body_pos, yaw_error = tracker.body_state()
    target_x = riser + tracker.table_body_x[phase]
    return torch.stack([target_x - body_pos[:, 0], body_pos[:, 1], yaw_error], dim=1)
