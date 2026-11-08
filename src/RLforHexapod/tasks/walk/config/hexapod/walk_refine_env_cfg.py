"""Refine the trained six-leg gait's yaw oscillation and stationary marching.

This task preserves WSP's support, hopping and velocity/height tracking rewards,
67 policy observations, action interface, command ranges and reset behavior.
It changes rewards only; no policy weight surgery or runtime gait controller.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

import RLforHexapod.tasks.walk.mdp as mdp
from .wsp_env_cfg import HexapodWspEnvCfg, WspRewardsCfg


@configclass
class RefineRewardsCfg(WspRewardsCfg):
    # model_1999 measured ~0.048 yaw error^2 at straight 0.15m/s. This cost
    # stays modest beside unchanged linear tracking (1) and yaw tracking (.5).
    yaw_error_l2 = RewTerm(func=mdp.refine_yaw_tracking_l2, weight=-1.0,
                          params={"command_name": "base_velocity",
                                  "linear_command_threshold": 0.025,
                                  "yaw_command_threshold": 0.1})
    # Its stationary qdot RMS was ~3rad/s: this initially costs ~0.20/s.
    # Translation or turning commands disable this term; no preferred pose.
    stand_joint_velocity = RewTerm(
        func=mdp.refine_stand_joint_velocity_l2, weight=-0.2,
        params={"command_name": "base_velocity", "velocity_scale": 3.0,
                "linear_command_threshold": 0.025, "yaw_command_threshold": 0.1},
    )


@configclass
class HexapodRefineEnvCfg(HexapodWspEnvCfg):
    rewards: RefineRewardsCfg = RefineRewardsCfg()
