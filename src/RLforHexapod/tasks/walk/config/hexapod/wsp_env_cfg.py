"""Reward-only fine-tuning task for the experimental flat WSP checkpoint.

The policy still receives the original 67 observations and outputs 18 actions.
Assets, motor limits, command ranges, observations and reset events are inherited.
"""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass

import RLforHexapod.tasks.walk.mdp as mdp
from .env_cfg import HexapodEnvCfg, RewardsCfg


WSP_FOOT_BODIES = ["tibia", "tibia_2", "tibia_3", "tibia_4", "tibia_5", "tibia_6"]


def _support_parameters():
    # Resolve sensor/asset independently: their body indexing need not be equal.
    return {
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WSP_FOOT_BODIES, preserve_order=True),
        "asset_cfg": SceneEntityCfg("robot", body_names=WSP_FOOT_BODIES, preserve_order=True),
        "force_threshold": 1.0,
        "min_tip_height": -0.01,
        "max_tip_height": 0.03,
    }


@configclass
class WspRewardsCfg(RewardsCfg):
    # These bounded costs do not pay a constant bonus for standing on six feet.
    support_participation = RewTerm(
        func=mdp.wsp_support_participation, weight=-0.6,
        params={**_support_parameters(), "command_name": "base_velocity",
                "time_constant_s": 0.6, "minimum_duty": 0.2, "maximum_duty": 0.85,
                "linear_command_threshold": 0.025, "yaw_command_threshold": 0.1},
    )
    all_feet_airborne = RewTerm(func=mdp.wsp_all_feet_airborne, weight=-0.8,
                                params=_support_parameters())
    sparse_support = RewTerm(func=mdp.wsp_support_deficit, weight=-0.25,
                             params={**_support_parameters(), "minimum_support": 3})
    # Existing speed/heading/height rewards retain their original weights.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.0)


@configclass
class HexapodWspEnvCfg(HexapodEnvCfg):
    rewards: WspRewardsCfg = WspRewardsCfg()
