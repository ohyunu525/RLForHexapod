"""계단 오르기 teacher의 진행 상태와 보상.

목표: 수평을 유지하고 네 발 안에 무게중심을 둔 채, 한 쌍씩 다음 단으로 옮긴다.

이동 순서 (단마다 4번, STAGES 참고): 앞발 → 중간발(수직면 바로 위) → 뒷발 → 중간발(앞으로)
4번을 마치면 모든 발이 한 단 위에 있다. 전체 단을 오르면 SETTLE(정지) → DONE(성공).
이동 번호 move는 앞으로만 증가하고, 이동마다 시간제한이 있다.

한 step의 계산은 StairTracker.update()에 모여 있고, 보상 함수들은 그 결과를 읽기만 한다.
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


class Stage(NamedTuple):
    """한 단을 오르는 이동 하나. x는 이번에 오를 단의 수직면 기준, 몸통 z는 아래쪽 단 윗면 기준 [m]."""

    feet: list          # 옮기는 쌍
    foot_x: float       # 착지 목표 x
    via_x: float        # 들어 올릴 때의 경유점 x
    body_lift: tuple    # 발을 들 때 몸통 목표 (x, z)
    body_land: tuple    # 발이 단 위로 올라간 뒤 몸통 목표 (x, z)


# 값의 근거: URDF 계산. 몸통 수평, 다리가 닿고, 지지 발 안에 무게중심이 있고(여유 3cm 이상), 다리끼리 안 부딪히는 자세
STAGES = [
    Stage(FRONT, 0.25, -0.02, (-0.14, 0.26), (-0.13, 0.20)),
    # 뒷발을 들 때 무게중심이 중간발 앞에 있도록, 중간발을 수직면 바로 위에 먼저 놓는다
    Stage(MIDDLE, 0.03, -0.02, (0.02, 0.24), (0.02, 0.24)),
    Stage(REAR, 0.10, -0.02, (0.08, 0.22), (0.14, 0.26)),
    # 앞발을 들 때 무게중심이 중간발 뒤에 있도록, 중간발을 다시 앞으로 옮긴다. 끝나면 몸통 위치가 다음 단의 시작 위치가 된다
    Stage(MIDDLE, 0.20, 0.115, (0.17, 0.38), (0.17, 0.38)),
]
NUM_STAGES = len(STAGES)
STAGE_OBS_SIZE = NUM_STAGES + 1  # 이동 단계 + 정지


@configclass
class StairGaitCfg:
    """판정 기준과 추적 폭. 보상 크기는 env_cfg의 weight에서 정한다."""

    # 옮기는 발
    clearance: float = 0.04      # 경유점: 단 윗면 위 높이
    clear_height: float = 0.02   # 단 윗면 위로 이만큼 올라오면 경유 완료, 착지 목표로 전환
    land_x_tol: float = 0.04     # 착지 판정: 목표 x와의 오차
    track_std: float = 0.10      # 추적 점수 폭: 거리가 이 값이면 점수 약 0.37

    # 접지
    tip_height: float = 0.013    # 딛었을 때 발끝 추정점이 지면보다 높은 만큼 (메시 최저점과의 차이)
    contact_tol: float = 0.03    # 발끝과 지면의 높이 차 허용치

    # 몸통
    body_x_std: float = 0.10
    body_z_std: float = 0.10
    body_y_std: float = 0.10
    body_yaw_std: float = 0.30
    com_margin: float = 0.03     # 무게중심 여유가 이 값 이상이면 만점

    # 시간제한 [s]: 넘기면 실패 종료 → 시도하지 않고 버티는 동작을 막는다
    move_time_limit: float = 3.0
    settle_time_limit: float = 6.0

    # 정지
    hold_time: float = 1.5       # 아래 허용 오차를 연속으로 지켜야 하는 시간 [s]
    pose_tol: float = 0.2        # 시작 자세와의 관절 각도 RMS 오차 [rad]
    tilt_tol: float = 0.1        # 몸통 기울기 (중력 벡터의 수평 성분, 0.1 ≈ 6°)
    speed_tol: float = 0.1       # 몸통 속도 크기
    pose_std: float = 0.4
    tilt_std: float = 0.2
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
        self.tip_offset = tip_offsets(FOOT_BODIES, env.device)

        # 이동 번호: 0 ~ settle_move - 1 = 이동, settle_move = 정지, done_move = 성공
        self.settle_move = NUM_STAGES * self.stairs.num_steps
        self.done_move = self.settle_move + 1

        # STAGES 표를 텐서로
        device, n = env.device, env.num_envs
        self.table_moving = torch.zeros(NUM_STAGES, len(FOOT_BODIES), dtype=torch.bool, device=device)
        for k, stage in enumerate(STAGES):
            self.table_moving[k, stage.feet] = True
        self.table_feet = torch.tensor([s.feet for s in STAGES], device=device)
        self.table_foot_x = torch.tensor([s.foot_x for s in STAGES], device=device)
        self.table_via_x = torch.tensor([s.via_x for s in STAGES], device=device)
        self.table_body_lift = torch.tensor([s.body_lift for s in STAGES], device=device)
        self.table_body_land = torch.tensor([s.body_land for s in STAGES], device=device)

        # env별 상태
        self.move = torch.zeros(n, dtype=torch.long, device=device)
        self.cleared = torch.zeros(n, len(FOOT_BODIES), dtype=torch.bool, device=device)  # 경유 완료한 발
        self.move_time = torch.zeros(n, device=device)    # 이번 이동에 쓴 시간 [s]
        self.steady_time = torch.zeros(n, device=device)  # 멈춘 상태를 유지한 시간 [s]
        self.ended_move = torch.zeros(n, dtype=torch.long, device=device)  # 끝난 에피소드의 이동 번호 (로그용)
        self._updated_at = -1

        # 이번 step의 결과 (보상 함수들이 읽는다)
        self.just_done = torch.zeros(n, dtype=torch.bool, device=device)
        self.feet_score = torch.zeros(n, device=device)
        self.stance_score = torch.zeros(n, device=device)
        self.com_score = torch.zeros(n, device=device)
        self.body_score = torch.zeros(n, device=device)
        self.settle_score = torch.zeros(n, device=device)

    # ----- 지금 이동의 목표 -----
    def stage_and_level(self):
        """표의 몇 번째 이동인지, 몇 번째 단(바닥 = 0)을 오르는 중인지. 정지 중에는 마지막 이동의 값."""
        climbing = self.move.clamp(max=self.settle_move - 1)
        return climbing % NUM_STAGES, climbing // NUM_STAGES

    def riser_and_top(self, level):
        """이번에 오를 단의 수직면 x와 윗면 높이."""
        return self.stairs.x0 + level * self.stairs.tread, (level + 1) * self.stairs.step_height

    def moving_feet(self, stage):
        """지금 옮기는 발 (N, 6). 정지 중에는 없음."""
        return self.table_moving[stage] & (self.move < self.settle_move).unsqueeze(1)

    def foot_targets(self, stage, riser, top):
        """(경유점 x, z), (착지점 x, z). 각각 (N,)."""
        tip, clearance = self.gait.tip_height, self.gait.clearance
        return (riser + self.table_via_x[stage], top + clearance + tip), (riser + self.table_foot_x[stage], top + tip)

    def body_target(self, stage, level, moving, riser):
        """몸통 목표 x, z (N,). 옮기는 발이 모두 경유를 마치면 착지용 목표로 바뀐다."""
        all_cleared = (self.cleared | ~moving).all(1, keepdim=True)
        offset = torch.where(all_cleared, self.table_body_land[stage], self.table_body_lift[stage])
        return riser + offset[:, 0], level * self.stairs.step_height + offset[:, 1]

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

        gait, dt = self.gait, self.env.step_dt
        stage, level = self.stage_and_level()
        riser, top = self.riser_and_top(level)
        moving = self.moving_feet(stage)
        planted = ~moving
        settling = self.move == self.settle_move

        tips = self.tips()
        terrain = stairs_terrain_height(self.stairs, tips[..., 0], tips[..., 1])
        air_time = self.sensor.data.current_air_time.torch[:, self.sensor_ids]
        # 실제로 디딘 발: 접촉 센서가 닿았다고 하고, 발끝도 지면 높이 (정강이만 수직면에 닿은 경우를 거른다)
        near_ground = (tips[..., 2] - terrain - gait.tip_height).abs() < gait.contact_tol
        grounded = (air_time == 0) & near_ground
        on_top = (terrain >= top.unsqueeze(1) - 1e-3) & near_ground

        # 옮기는 발: 단 위로 올라오면 경유 완료 (되돌리지 않음)
        self.cleared |= moving & (tips[..., 2] >= (top + gait.clear_height).unsqueeze(1))

        # 목표 추적 점수: 경유 전 0 ~ 0.5, 경유 후 0.5 ~ 1. 경유를 마쳐도 점수가 줄지 않는다
        (via_x, via_z), (land_x, land_z) = self.foot_targets(stage, riser, top)
        dist_via = torch.hypot(tips[..., 0] - via_x.unsqueeze(1), tips[..., 2] - via_z.unsqueeze(1))
        dist_land = torch.hypot(tips[..., 0] - land_x.unsqueeze(1), tips[..., 2] - land_z.unsqueeze(1))
        foot_score = torch.where(self.cleared,
                                 0.5 + 0.5 * torch.exp(-dist_land / gait.track_std),
                                 0.5 * torch.exp(-dist_via / gait.track_std))
        feet_score = (foot_score * moving).sum(1) / moving.sum(1).clamp(min=1)
        # 정지 중에는 옮길 발이 없으므로 만점. 점수가 줄면 일부러 뒷단계에 머무른다
        self.feet_score = torch.where(self.move >= self.settle_move, torch.ones_like(feet_score), feet_score)

        # 접지 점수: 디뎌야 할 발 중 실제로 디딘 비율
        self.stance_score = (planted & grounded).sum(1) / planted.sum(1).clamp(min=1)

        # 무게중심 점수: 디딘 발의 x 범위 안쪽으로 얼마나 들어왔는지
        body_pos, yaw_error = self.body_state()
        planted_x_min = tips[..., 0].masked_fill(~planted, float("inf")).min(1).values
        planted_x_max = tips[..., 0].masked_fill(~planted, float("-inf")).max(1).values
        com_margin = torch.minimum(body_pos[:, 0] - planted_x_min, planted_x_max - body_pos[:, 0])
        self.com_score = (com_margin / gait.com_margin).clamp(0.0, 1.0)

        # 몸통 점수: 표의 위치, 옆 위치 0, 정면
        target_x, target_z = self.body_target(stage, level, moving, riser)
        self.body_score = torch.exp(-(body_pos[:, 0] - target_x).abs() / gait.body_x_std
                                    - (body_pos[:, 2] - target_z).abs() / gait.body_z_std
                                    - body_pos[:, 1].abs() / gait.body_y_std
                                    - yaw_error.abs() / gait.body_yaw_std)

        # 정지 점수와 유지 시간
        d = self.asset.data
        pose_error = (d.joint_pos.torch - d.default_joint_pos.torch).pow(2).mean(1).sqrt()
        tilt = d.projected_gravity_b.torch[:, :2].norm(dim=1)
        speed = d.root_vel_w.torch.norm(dim=1)
        score = (on_top.float().mean(1)
                 * torch.exp(-(pose_error / gait.pose_std) ** 2)
                 * torch.exp(-(tilt / gait.tilt_std) ** 2)
                 * torch.exp(-(speed / gait.speed_std) ** 2))
        self.settle_score = score * settling
        steady = (on_top.all(1) & grounded.all(1) & (pose_error < gait.pose_tol)
                  & (tilt < gait.tilt_tol) & (speed < gait.speed_tol))
        # 멈춘 상태가 끊기면 시간을 처음부터 다시 잰다
        self.steady_time = torch.where(steady & settling, self.steady_time + dt, torch.zeros_like(self.steady_time))

        # 완료 판정: 옮기는 두 발이 모두 목표 x 근처에서 단을 디딤
        landed = on_top & (air_time == 0) & ((tips[..., 0] - land_x.unsqueeze(1)).abs() < gait.land_x_tol)
        move_done = (landed | ~moving).all(1) & (self.move < self.settle_move)
        settle_done = settling & (self.steady_time >= gait.hold_time)
        self.just_done = move_done | settle_done

        # 다음 이동으로
        self.move_time = torch.where(self.just_done, torch.zeros_like(self.move_time), self.move_time + dt)
        self.move = self.move + self.just_done.long()
        self.cleared[self.just_done] = False

        # 새 이동에 막 들어선 안정적인 상태 저장 → 리셋 때 시작 상태로 사용
        save = self.just_done & (self.move < self.done_move) & (speed < gait.save_speed_max)
        if save.any():
            save_stair_states(self.env, save.nonzero().squeeze(-1), self.move[save], gait.buffer_size)

    def reset(self, env_ids, moves):
        self.ended_move[env_ids] = self.move[env_ids]
        self.move[env_ids] = moves
        self.cleared[env_ids] = False
        self.move_time[env_ids] = 0.0
        self.steady_time[env_ids] = 0.0
        self.just_done[env_ids] = False


def get_tracker(env) -> StairTracker:
    if not hasattr(env, "stair_tracker"):
        env.stair_tracker = StairTracker(env)
    return env.stair_tracker


# ===== 리셋 =====
def save_stair_states(env, env_ids, moves, size: int = 1000):
    """
    env_ids의 현재 상태를 이동 번호별 버퍼에 추가 (위치는 env 원점 기준, 꽉 차면 오래된 것부터 버림).

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
    # s[moves == k] : 이동 번호가 k인 상태를 stair_states[k]에 붙이기, 만약 없으면 빈 s에 s 붙이기
    # [-size:] : size만큼 자르기
    for k in moves.unique().tolist():
        env.stair_states[k] = torch.cat([env.stair_states.get(k, s[:0]), s[moves == k]])[-size:]


def reset_stair_state(env, env_ids, x_noise: float = 0.01, buffer_ratio: float = 0.5):
    """시작 상태를 뽑아 옮김 (reset_scene_to_default 다음에 실행).
    기본: 바닥에서 시작 자세 + x 오차. buffer_ratio 비율로 뒤쪽 이동의 저장된 상태 (비어 있으면 기본)."""
    # env: ManagerBasedRLEnv
    # env_ids: tensor | slice(None)

    if isinstance(env_ids, slice):
        # 첫 env.reset()은 slice(None)으로 호출됨
        # 궁극적으로 slice(None)은 인덱스 tensor와 똑같지만, 길이와 형태에 대한 데이터가 없다.
        env_ids = torch.arange(env.num_envs, device=env.device)

    tracker = get_tracker(env)
    # n: reset 해당 env 인덱스 갯수
    # buf: 이동 번호별 저장 상태 (S, 7 + 6 + J + J)
    n, buf = len(env_ids), getattr(env, "stair_states", {})

    # 이동 번호 뽑기: 1 ~ 정지 중 무작위, 저장된 상태가 없는 번호는 처음(0)부터
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
class stair_move_bonus(ManagerTermBase):
    """목표: 이동을 마칠 때마다 1번 (weight = 이동 한 번의 점수). 끝난 에피소드의 진행도도 로그로 남긴다."""

    def reset(self, env_ids=None):
        tracker = get_tracker(self._env)
        ended = tracker.ended_move[slice(None) if env_ids is None else env_ids]
        log = self._env.extras["log"]
        log["Metrics/stair_move"] = ended.float().mean()
        log["Metrics/stair_level"] = (ended // NUM_STAGES).float().mean()
        log["Metrics/stair_reached_top"] = (ended >= tracker.settle_move).float().mean()  # 꼭대기 도달 비율
        log["Metrics/stair_success"] = (ended == tracker.done_move).float().mean()  # 정지까지 성공한 비율

    def __call__(self, env):
        tracker = get_tracker(env)
        tracker.update()
        return tracker.just_done.float() / env.step_dt


def _score(env, name: str) -> torch.Tensor:
    tracker = get_tracker(env)
    tracker.update()
    return getattr(tracker, name)


def stair_feet_tracking(env) -> torch.Tensor:
    """과정: 옮기는 발이 경유점을 거쳐 착지점으로."""
    return _score(env, "feet_score")


def stair_body_pose(env) -> torch.Tensor:
    """과정: 몸통이 표의 위치로."""
    return _score(env, "body_score")


def stair_com_margin(env) -> torch.Tensor:
    """공통 조건: 디딘 발 안쪽에 무게중심."""
    return _score(env, "com_score")


def stair_stance_contact(env) -> torch.Tensor:
    """공통 조건: 옮기는 쌍을 뺀 발은 접지."""
    return _score(env, "stance_score")


def stair_settle_still(env) -> torch.Tensor:
    """정지: 꼭대기에서 시작 자세로 멈춤."""
    return _score(env, "settle_score")


# ===== 종료 =====
def stair_success(env) -> torch.Tensor:
    tracker = get_tracker(env)
    return tracker.move >= tracker.done_move


def stair_move_timed_out(env) -> torch.Tensor:
    tracker = get_tracker(env)
    limit = torch.where(tracker.move == tracker.settle_move,
                        tracker.gait.settle_time_limit, tracker.gait.move_time_limit)
    return tracker.move_time >= limit


# ===== 관측 =====
def stair_stage_obs(env) -> torch.Tensor:
    """지금 이동 단계 one-hot (N, 5). 마지막 칸 = 정지."""
    tracker = get_tracker(env)
    stage, _ = tracker.stage_and_level()
    return F.one_hot(torch.where(tracker.move >= tracker.settle_move, NUM_STAGES, stage), STAGE_OBS_SIZE).float()


def stair_feet_target_obs(env) -> torch.Tensor:
    """옮기는 두 발 각각의 현재 목표까지 (x, z) 오차와 경유 완료 여부 (N, 6). 정지 중에는 0."""
    tracker = get_tracker(env)
    stage, level = tracker.stage_and_level()
    riser, top = tracker.riser_and_top(level)
    (via_x, via_z), (land_x, land_z) = tracker.foot_targets(stage, riser, top)

    feet = tracker.table_feet[stage]  # (N, 2)
    rows = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    tips = tracker.tips()[rows, feet]  # (N, 2, 3)
    cleared = tracker.cleared[rows, feet]  # (N, 2)
    target_x = torch.where(cleared, land_x.unsqueeze(1), via_x.unsqueeze(1))
    target_z = torch.where(cleared, land_z.unsqueeze(1), via_z.unsqueeze(1))
    obs = torch.cat([target_x - tips[..., 0], target_z - tips[..., 2], cleared.float()], dim=1)
    return obs * (tracker.move < tracker.settle_move).unsqueeze(1)


def stair_body_target_obs(env) -> torch.Tensor:
    """몸통 목표까지 오차 (x, z), 옆 위치, yaw 오차 (N, 4)."""
    tracker = get_tracker(env)
    stage, level = tracker.stage_and_level()
    riser, _ = tracker.riser_and_top(level)
    target_x, target_z = tracker.body_target(stage, level, tracker.moving_feet(stage), riser)
    body_pos, yaw_error = tracker.body_state()
    return torch.stack([target_x - body_pos[:, 0], target_z - body_pos[:, 2], body_pos[:, 1], yaw_error], dim=1)


def stair_time_left_obs(env) -> torch.Tensor:
    """이번 이동의 시간제한 대비 쓴 시간 (N, 1). 정지 중에는 유지 시간 비율."""
    tracker = get_tracker(env)
    limit = torch.where(tracker.move == tracker.settle_move,
                        tracker.gait.settle_time_limit, tracker.gait.move_time_limit)
    return (tracker.move_time / limit).unsqueeze(1)
