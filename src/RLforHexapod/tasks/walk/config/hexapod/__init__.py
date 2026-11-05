import gymnasium as gym

_PACKAGE = __name__

gym.register(
    id="Hexapod-Walk",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodStatePPORunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodStatePPORunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair-Student-A",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairStudentAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:HexapodStudentRunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair-Student-B",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairStudentBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:HexapodStudentBRunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair-Student-A-Phase",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairStudentAPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:HexapodStudentAPhaseRunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair-Student-A-TopEnd",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairStudentATopEndEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:HexapodStudentATopEndRunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair-Student-A-Mem128",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairStudentATopEndEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:HexapodStudentMem128RunnerCfg"
    },
)

gym.register(
    id="Hexapod-Stair-Student-A-Mem256",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairStudentATopEndEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:HexapodStudentMem256RunnerCfg"
    },
)
