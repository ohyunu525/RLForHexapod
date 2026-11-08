"""Per-episode collision geometry for the mixed stair lane."""

import torch

from .observations import StairsGeometryCfg, env_stair_step_height, stairs_approach_height

MAX_RANDOM_STAIR_HEIGHT = 0.16
_GROUND_CLEARANCE = 0.02


def mixed_stair_blocks_cfg(stairs: StairsGeometryCfg):
    """Build kinematic blocks with the same footprints as the fixed stair USD."""
    from isaaclab import sim
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

    thickness = _block_thickness(stairs)
    approach_length = stairs.x0 - stairs.approach_start
    footprints = [(stairs.approach_start + approach_length / 2, approach_length)]
    for index in range(stairs.num_steps):
        length = stairs.top_length if index == stairs.num_steps - 1 else stairs.tread
        footprints.append((stairs.x0 + index * stairs.tread + length / 2, length))

    blocks = {}
    for index, (center, length) in enumerate(footprints):
        top = index * stairs.step_height if index else -_GROUND_CLEARANCE
        name = f"StairBlock_{index}"
        blocks[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim.CuboidCfg(
                size=(length, stairs.width, thickness),
                rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim.UsdPhysicsCollisionCfg(),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(center, 0.0, top - thickness / 2)),
        )
    return RigidObjectCollectionCfg(rigid_objects=blocks)


def _block_thickness(stairs: StairsGeometryCfg) -> float:
    return stairs.num_steps * MAX_RANDOM_STAIR_HEIGHT + _GROUND_CLEARANCE


def set_mixed_stair_heights(env, env_ids: torch.Tensor, heights: torch.Tensor) -> None:
    """Move the selected blocks before spawning their robots, then publish the new geometry."""
    stairs = env.cfg.stairs
    blocks = env.scene["stair_blocks"]
    poses = blocks.data.default_body_pose.torch[env_ids].clone()
    levels = torch.arange(stairs.num_steps + 1, device=heights.device)
    tops = stairs_approach_height(stairs, heights)[:, None] + heights[:, None] * levels
    # Use the existing floor collider wherever the surface height is zero.
    tops = torch.where(tops > 0.0, tops, -_GROUND_CLEARANCE)
    poses[..., 2] = tops - _block_thickness(stairs) / 2
    poses[..., :3] += env.scene.env_origins[env_ids, None, :]
    blocks.write_body_pose_to_sim_index(body_poses=poses, env_ids=env_ids)

    if not hasattr(env, "mixed_stair_heights"):
        env.mixed_stair_heights = env_stair_step_height(env)
    env.mixed_stair_heights[env_ids] = heights
