import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from isaaclab.envs import (
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg, 
)
import RLforHexapod.tasks.walk.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab.sensors import ContactSensorCfg

from RLforHexapod.assets import ROBOT_USDA, OBSTACLES_DIR

JOINTS = [
    "L1_tibia", "L1_femur", "L1_coxa", 
    "L2_tibia", "L2_femur", "L2_coxa", 
    "L3_tibia", "L3_femur", "L3_coxa",
    "L4_tibia", "L4_femur", "L4_coxa", 
    "L5_tibia", "L5_femur", "L5_coxa", 
    "L6_tibia", "L6_femur", "L6_coxa"
]

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USDA)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*_coxa": 0.0,
            ".*_femur": math.radians(20.0),  # 0=아래로 수직, 90=수평 (한계 0~180)
            ".*_tibia": math.radians(-110.0),  # -90=femur와 일직선, 값이 작을수록 아래로 굽힘 (한계 -250~-70: 아래 160, 위 20)
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # MJWarp는 관절 속도 한계를 무시하므로 토크-속도 곡선으로 속도를 제한한다.
        # coxa=MG996R 6V 데이터시트, femur/tibia=RC920DMG 임시값(사양 불명, 7V 실측으로 교체)
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],   # 18개 관절 전부
            # None이면 Newton 액추에이터의 kp/kd가 0이 된다.
            stiffness=100.0,
            damping=1.0,
            # 감속기 반사 관성(MG996R 급 ~0.005-0.01). 0이면 가벼운 tibia가 불안정
            armature=0.005,
            saturation_effort={".*_coxa": 1.08, ".*_(femur|tibia)": 1.86},  # 스톨 토크 [N·m]
            actuator_velocity_limit={".*_coxa": 7.48, ".*_(femur|tibia)": 6.5},  # 무부하 속도 [rad/s]
            actuator_effort_limit={".*_coxa": 0.75, ".*_(femur|tibia)": 1.3},  # 지속 토크, 스톨의 70% [N·m]
        ),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000, color=(0.75, 0.75, 0.75)))
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = HEXAPOD_CFG
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True,
        scale={".*_coxa":0.5, ".*_femur": 1.0, ".*_tibia": 1.0}, 
    )


@configclass
class ObservationsCfg:
    # ObservationsCfg 안에는 ObsGroup만이 들어갈 수 있다고 한다.
    # 같은 task 내에서 ObsGroup을 바꿔 껴서 쉽게 할 수 있도록 했다.
    # 사용 예시로 증류가 있다.
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) 
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # 센서로 얻을 수 있는 값.
    

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True,
        scale={".*_coxa":0.5, ".*_femur": 1.0, ".*_tibia": 1.0},    
        # scale 증가 필요
    )


@configclass
class EventsCfg:
    # 실제 서보 회전 범위. robot.usda(LFS)의 0~270 override를 시작 시 덮어씀, coxa는 USD의 ±90 그대로
    joint_limits = EventTerm(func=mdp.set_joint_pos_limits, mode="startup", params={
        "limits_deg": {".*_femur": (0.0, 180.0), ".*_tibia": (-250.0, -70.0)}})
    # 리셋 때 로봇을 init_state(초기 자세)로 되돌림
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.01)
    track_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, params={"command_name": "base_velocity", "std": 0.5}, weight=1)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, params={"command_name": "base_velocity", "std": 0.5}, weight=0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-7)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})
    bad_ori  = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})  # 약 57°
    sim_nan = DoneTerm(func=mdp.sim_nan)


@configclass
class CommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.2),   # 수정해야함.
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(0, 0),
        ),
        debug_vis=True,
    )


@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """Step and reset the scene before adding teacher rewards."""

    # 7대장 (7개의 메니저가 있고, 그 메니저들의 설정이다.)
    scene: InteractiveSceneCfg = SceneCfg(num_envs=4, env_spacing=1.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandCfg = CommandCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0  # 정책 50Hz (decimation 4)
        self.episode_length_s = 20.0
        self.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=True)
        # Isaac Sim 대비 환경 설정이 간편함, 병렬 학습 속도는 비슷 비슷, 오픈소스라 마개조 가능, 다만 신버전이라 지원이 부족할 수 있음.
    

# Stairs-cfg  =====================================================

# 계단 형상: 씬 배치(StairsSceneCfg)와 특권 관측이 같은 값을 쓰도록 한 곳에서 정의
STAIRS = mdp.StairsGeometryCfg(x0=0.5, step_height=0.16)

@configclass
class StairsSceneCfg(SceneCfg):
    stairs = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stairs",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBSTACLES_DIR / f"stairs_{round(STAIRS.step_height * 1000)}mm.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(STAIRS.x0, 0.0, 0.0)),  # 로봇 앞 0.5m
    )


@configclass
class StairsTerminationsCfg(TerminationsCfg):
    # 조금 기울어지는 건 괜찮다. 약 29° 넘으면 실패
    bad_ori = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5})
    # 몸통 중심이 계단 폭(1.0m, stairs_*.usda의 y scale) 밖으로 나가면 실패
    off_stairs = DoneTerm(func=mdp.out_of_stairs_width, params={"half_width": STAIRS.width/2})
    sim_blowup = DoneTerm(func=mdp.joint_vel_out_of_manual_limit, params={"max_velocity": 50.0})
    # 꼭대기에 올라 멈추면 성공
    success = DoneTerm(func=mdp.stair_success, time_out=True)


@configclass
class StairsEventsCfg(EventsCfg):
    # x_ranges 순서 = 단계 번호 (몸통 x 범위). 0번 전진: 수직면까지 멀리서, 1번 앞발 옮기기: 앞발이 수직면 3~5cm 앞
    stair_reset = EventTerm(func=mdp.reset_stair_state, mode="reset",
                            params={"x_ranges": ((-0.1, 0.1), (0.30, 0.32)), "buffer_ratio": 0.5})


@configclass
class StairsRewardsCfg:
    # 단계를 마칠 때마다 1번 (weight = 점수)
    phase_bonus = RewTerm(func=mdp.stair_phase_bonus, weight=10.0)
    # 뒷발 옮기기 중 앞다리를 수직면과 평행하게 돌린 비율의 신기록 (weight = 다리 하나를 끝까지 돌린 점수)
    front_fold = RewTerm(func=mdp.stair_front_fold, weight=5.0)
    # 옮기는 발이 단 위에 처음 놓일 때, 발마다 단계당 1번 (weight = 발 하나의 점수)
    foot_placed = RewTerm(func=mdp.stair_foot_placed, weight=5.0)
    # 옮기는 발이 단 끝에라도 처음 닿을 때, 발마다 단계당 1번 (weight = 발 하나의 점수)
    edge_touch = RewTerm(func=mdp.stair_edge_touch, weight=2.0)
    # 지금 단계의 신기록 갱신량. 전진: 몸통 x, 옮기기: 발끝 높이 + 몸통 x (weight = 1m당 점수)
    progress = RewTerm(func=mdp.stair_progress, weight=30.0)
    # 모든 단계에서 여섯 다리의 tibia가 지면에 수직일수록 (weight = 초당 최대 점수)
    tibia_vertical = RewTerm(func=mdp.stair_tibia_vertical, weight=0.5)
    # 옮기기 중 제자리에서 떨어진 나머지 발의 수 (weight = 발 하나당 초당 벌점)
    standing_feet_off = RewTerm(func=mdp.stair_standing_feet_off, weight=-1.0)
    # 모든 단계에서 몸통의 후진 속도 (weight = 후진 1m당 점수)
    backward = RewTerm(func=mdp.stair_backward, weight=-30.0)
    # 꼭대기에서 멈출수록 (weight = 초당 최대 점수)
    settle_still = RewTerm(func=mdp.stair_settle_still, weight=4.0)
    # 실패: weight × step_dt(0.02)가 실제 점수 → -10점
    termination_penalty = RewTerm(func=mdp.is_terminated_term, weight=-500.0, params={
        "term_keys": ["base_low", "bad_ori", "off_stairs", "sim_blowup", "sim_nan"]})  # 성공 제외

    # 동작 규제
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # 동작 규제
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", "femur.*", "coxa.*"]),
        "threshold": 1.0})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-7)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)


@configclass
class StairsObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # 속도 명령을 따르지 않으므로 뺀다
        command = None
        # 계단 teacher 특권 관측
        # 1. 발끝이 지형에서 떠 있는 높이 (6) + 발끝 아래 지형의 몸통 대비 깊이 (6)
        feet_height = ObsTerm(func=mdp.feet_height_above_terrain, params={
            "asset_cfg": SceneEntityCfg("robot", body_names="tibia.*"), "stairs": STAIRS})
        terrain_under_feet = ObsTerm(func=mdp.terrain_height_under_feet, params={
            "asset_cfg": SceneEntityCfg("robot", body_names="tibia.*"), "stairs": STAIRS})
        # 2. 단 높이 (1)
        step_height = ObsTerm(func=mdp.stairs_step_height, params={"stairs": STAIRS})
        # 3. 지금 할 일: 단계 (6), 옮기는 발 위치와 tibia 수직도 (6), 전진 목표까지 거리와 방향 (3)
        phase = ObsTerm(func=mdp.stair_phase_obs)
        feet_target = ObsTerm(func=mdp.stair_feet_target_obs)
        body_target = ObsTerm(func=mdp.stair_body_target_obs)

    policy: PolicyCfg = PolicyCfg()


@configclass
class HexapodStairEnvCfg(HexapodEnvCfg):
    # ManagerBaseRLEnvCfg를 불러온 HexapodEnvCfg를 불러온 HexapodStairEvnCfg, 값만 바꾼거라 사실상 껍데기.
    scene: SceneCfg = StairsSceneCfg(num_envs=4, env_spacing=3.0)
    observations: ObservationsCfg = StairsObservationsCfg()
    events: EventsCfg = StairsEventsCfg()
    rewards: RewardsCfg = StairsRewardsCfg()
    terminations: TerminationsCfg = StairsTerminationsCfg()
    # StairTracker가 읽는 값
    stairs: mdp.StairsGeometryCfg = STAIRS
    stair_gait: mdp.StairGaitCfg = mdp.StairGaitCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0  # 단마다 5단계 × 3단 + 마지막 전진, 정지


