import gymnasium as gym

_PACKAGE = __name__

gym.register(
    id="Hexapod-Mixed-Flat-Stair",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.mixed_env_cfg:HexapodMixedEnvCfg",
        "rsl_rl_cfg_entry_point":f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodMixedPPORunnerCfg",
    },
)

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
    id="Hexapod-Walk-WSP",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.wsp_env_cfg:HexapodWspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodWspPPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Walk-Refine",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.walk_refine_env_cfg:HexapodRefineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodRefinePPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Walk-WspStop",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.walk_wsp_stop_env_cfg:HexapodWspStopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodWspStopPPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Walk-Commands",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        # The dedicated training launcher applies the selected command ranges
        # to this fresh instance before creating the environment.
        "env_cfg_entry_point": f"{_PACKAGE}.walk_wsp_stop_env_cfg:HexapodWalkCommandsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodCommandsPPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Stair",
    entry_point=f"{_PACKAGE}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodStairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodStairPPORunnerCfg"
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
