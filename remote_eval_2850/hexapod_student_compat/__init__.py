"""Explicit registration of the pinned Hexapod student playback task.

Importing this module does not register tasks, start a simulator, or change the
installed RLforHexapod package. Call ``register()`` to add the private task ID.
"""

from __future__ import annotations

SOURCE_COMMIT = "ed53031ea774e167982bc9563a28435daefc619b"
TASK_ID = "WSP-Hexapod-Stair-Student-A"
PROVENANCE_FILENAME = "provenance.json"


def inspect_asset_provenance(robot_usda, obstacles_dir) -> dict:
    """Report installed asset hashes against this snapshot without modifying files."""
    from .asset_provenance import inspect_asset_provenance as inspect

    return inspect(robot_usda, obstacles_dir)


def register() -> str:
    """Register the private task once, refusing to replace an unrelated task."""
    import gymnasium as gym

    entry_point = f"{__name__}.hexapod_env:HexapodEnv"
    task_kwargs = {
        "env_cfg_entry_point": f"{__name__}.env_cfg:HexapodStairStudentAEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_distillation_cfg:HexapodStudentRunnerCfg"
        ),
    }
    existing = gym.registry.get(TASK_ID)
    if existing is not None:
        if existing.entry_point != entry_point or existing.kwargs != task_kwargs:
            raise RuntimeError(f"Refusing to overwrite a different task registered as {TASK_ID!r}")
        return TASK_ID

    gym.register(
        id=TASK_ID,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs=task_kwargs,
    )
    return TASK_ID
