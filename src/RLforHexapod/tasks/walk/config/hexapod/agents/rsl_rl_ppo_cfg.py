"""RSL-RL PPO 설정: 걷기 teacher와 계단 teacher."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class HexapodStatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """걷기 teacher: actor와 critic이 같은 관측을 본다."""

    seed = 42
    num_steps_per_env = 64
    max_iterations = 800
    save_interval = 50
    experiment_name = "hexapod_walk"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        # Clamp the std: with clipped actions and an entropy bonus, PPO otherwise grows it without bound.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HexapodStairPPORunnerCfg(HexapodStatePPORunnerCfg):
    """계단 teacher: 설정은 걷기와 같고 로그 폴더만 나눈다. 2026-11-07 이전 계단 로그는 hexapod_walk 폴더에 있다."""

    experiment_name = "hexapod_stair"


@configclass
class HexapodWspPPORunnerCfg(HexapodStatePPORunnerCfg):
    """Conservative starting settings for WSP actor warm-start, not tuned guarantees."""

    experiment_name = "hexapod_walk_wsp"
    max_iterations = 3000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.1, std_type="log", std_range=(0.05, 0.2)),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.15,
        entropy_coef=0.002, num_learning_epochs=5, num_mini_batches=8,
        learning_rate=1.0e-4, schedule="fixed", gamma=0.995, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class HexapodRefinePPORunnerCfg(HexapodWspPPORunnerCfg):
    """Small PPO updates from model_1999; fresh optimizer/value model on warm-start."""

    experiment_name = "hexapod_walk_refine"
    max_iterations = 1000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.08, std_type="log", std_range=(0.03, 0.15)),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=0.001, num_learning_epochs=5, num_mini_batches=8,
        learning_rate=5.0e-5, schedule="fixed", gamma=0.995, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class HexapodWspStopPPORunnerCfg(HexapodRefinePPORunnerCfg):
    """Fresh PPO state from a verified WSP seed; lower exploration/update size."""

    experiment_name = "hexapod_walk_wsp_stop"
    max_iterations = 2000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.05, std_type="log", std_range=(0.02, 0.10)),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=0.0005, num_learning_epochs=5, num_mini_batches=8,
        learning_rate=3.0e-5, schedule="fixed", gamma=0.995, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )
