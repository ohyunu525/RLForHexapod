import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from isaaclab.envs import (
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg,  # noqa: F401 -- used by the lesson 04 replacement
)
import RLforHexapod.tasks.walk.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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
            ".*_femur": math.radians(20.0),  # 0=아래로 수직, 90=수평 (한계 0~270)
            ".*_tibia": math.radians(-110.0),  # -180=femur 수평일 때 수직 (한계 -270~0)
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],   # 18개 관절 전부
            stiffness=None,
            damping=None,
            # 감속기 반사 관성(MG996R 급 ~0.005-0.01). 0이면 가벼운 tibia가 불안정
            armature=0.005,
        ),
    },
)

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True,
        scale={".*_coxa":0.5, ".*_femur": 1.0, ".*_tibia": 1.0}, 
    )


@configclass
class ObservationsCfg:
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


@configclass
class EventsCfg:
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
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="tibia.*"),
            "command_name": "base_velocity",
            "threshold": 0.2,
        }
    )


# AI ===============================================
@configclass
class StairsRewardsCfg:
    track_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0,
                            params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5,
                            params={"command_name": "base_velocity", "std": 0.5})
    height_progress = RewTerm(func=mdp.base_height_progress, weight=1.5)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", "femur.*", "coxa.*"]),
        "threshold": 1.0})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-0.5, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="tibia.*")})
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=0.5, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="tibia.*"),
        "command_name": "base_velocity", "threshold": 0.25})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-7)
# AI ===============================================


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})
    bad_ori  = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})  # 약 57°


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
class StairsSceneCfg(SceneCfg):
    stairs = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stairs",          # env마다 복제됨
        spawn=sim_utils.UsdFileCfg(usd_path=str(OBSTACLES_DIR / "stairs_160mm.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),  # 로봇 앞 0.5m
    )


@configclass
class CommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(0, 0),
        ),
        debug_vis=True,
    )

@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """Step and reset the scene before adding teacher rewards."""

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
    

@configclass
class HexapodStairEnvCfg(HexapodEnvCfg):
    scene: SceneCfg = StairsSceneCfg(num_envs=4, env_spacing=3.0)
    rewards: RewardsCfg = StairsRewardsCfg()


if __name__ == "__main__":
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    )
