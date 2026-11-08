"""Start fresh PPO training from only a WSP actor's mean policy and normalization.

The source critic, optimizer, exploration distribution, iteration and logger
state are deliberately not loaded. This entrypoint targets the installed
Isaac Lab 3.0 / RSL-RL 5.5.1 API; ordinary --no-resume_optimizer_state still
restores the critic and iteration and therefore is not an actor-only warmstart.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import traceback
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
TASK = "Hexapod-Walk-WSP"
EXPERIMENT = "hexapod_walk_wsp"
CHECKPOINT = ROOT / "outputs/flat_wsp/physical_20261108/tripod_probes/model_13250_wsp_tripod_compact_v2.pt"
STD_KEY = "distribution.log_std_param"
NORMALIZATION_EPSILON = 0.01
INITIAL_STD = 0.10


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_only_warmstart(runner, checkpoint, expected_std=INITIAL_STD):
    """Copy mean actor+moments into a newly constructed runner, with exact checks.

    No checkpoint loader is called: this makes critic/optimizer/iteration
    restoration impossible. Validate all incoming tensors before any mutation.
    The returned facts describe initialization before the first PPO update.
    """
    import torch

    if runner.current_learning_iteration != 0:
        raise ValueError("Warmstart requires a new runner at iteration zero")
    if runner.alg.optimizer.state:
        raise ValueError("Warmstart requires an empty fresh optimizer state")
    actor = runner.alg.get_policy()
    if getattr(actor.obs_normalizer, "eps", None) != NORMALIZATION_EPSILON:
        raise ValueError("Actor normalization epsilon must remain 0.01")
    target = actor.state_dict()
    source = checkpoint.get("actor_state_dict")
    if not isinstance(source, dict) or set(source) != set(target) or STD_KEY not in target:
        raise ValueError("Source and fresh actor state keys must match exactly, including Gaussian log std")
    if any(not (key.startswith("mlp.") or key.startswith("obs_normalizer.") or key == STD_KEY)
           for key in source):
        raise ValueError("Unexpected actor state outside MLP, observation normalizer and Gaussian std")
    for key, value in source.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[key].shape or value.dtype != target[key].dtype:
            raise ValueError(f"Incompatible actor tensor: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite source actor tensor: {key}")
    if (source["obs_normalizer._std"] < 0).any() or (source["obs_normalizer._var"] < 0).any():
        raise ValueError("Negative normalization variance or standard deviation")
    if source["obs_normalizer.count"].item() < 0:
        raise ValueError("Negative normalizer sample count")
    fresh_std = target[STD_KEY].detach().clone()
    if not torch.allclose(fresh_std.exp(), torch.full_like(fresh_std, expected_std), rtol=0., atol=1e-6):
        raise ValueError(f"Fresh exploration std must be configured as {expected_std}")
    critic_before = {key: value.detach().clone() for key, value in runner.alg.critic.state_dict().items()}
    incoming = {key: value for key, value in source.items() if key != STD_KEY}
    result = actor.load_state_dict(incoming, strict=False)
    if set(result.missing_keys) != {STD_KEY} or result.unexpected_keys:
        raise RuntimeError(f"Unexpected actor-only load result: {result}")
    after = actor.state_dict()
    exact = all(torch.equal(after[key].detach().cpu(), value.detach().cpu()) for key, value in incoming.items())
    critic_unchanged = all(torch.equal(value, runner.alg.critic.state_dict()[key])
                           for key, value in critic_before.items())
    if not exact or not critic_unchanged or not torch.equal(after[STD_KEY], fresh_std):
        raise RuntimeError("Actor-only warmstart changed an unexpected tensor")
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise RuntimeError("Warmstart unexpectedly changed iteration or optimizer state")
    return {
        "actor_mean_and_normalizer_exact": exact,
        "loaded_actor_keys": sorted(incoming),
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "actor_normalizer_count": int(after["obs_normalizer.count"].item()),
        "exploration_std": after[STD_KEY].exp().detach().cpu().tolist(),
        "exploration_std_reinitialized": True,
        "critic_equals_fresh_initialization": critic_unchanged,
        "critic_normalizer_count": int(runner.alg.critic.state_dict()["obs_normalizer.count"].item()),
        "optimizer_state_entries": len(runner.alg.optimizer.state),
        "initial_learning_iteration": runner.current_learning_iteration,
        "source_iteration_not_loaded": int(checkpoint.get("iter", -1)),
        "ignored_source_states": ["critic_state_dict", "optimizer_state_dict", STD_KEY, "iter", "infos"],
    }


def parse_args():
    import warp as wp
    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="flat_wsp")
    parser.add_argument("--logger", choices=["tensorboard"], default="tensorboard")
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=[])
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_iterations < 1 or args.seed < 0:
        parser.error("Positive environments/iterations and nonnegative seed required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("run_name must contain only letters, digits, underscore, dot or hyphen")
    if not args.checkpoint.is_file():
        parser.error(f"Missing WSP checkpoint: {args.checkpoint}")
    return args


def run(args):
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, dump_train_configs, write_run_manifest
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config
    import RLforHexapod.tasks  # noqa: F401

    version = importlib.metadata.version("rsl-rl-lib")
    if version != "5.5.1":
        raise RuntimeError(f"Validated runner API is RSL-RL 5.5.1; installed {version}")
    args.task = TASK
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=[])
    apply_env_overrides(args, env_cfg)
    env_cfg.seed = args.seed
    args.device = env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = args.run_name
    agent_cfg.logger = args.logger
    agent_cfg.resume = False
    # Loading is exclusively performed by actor_only_warmstart below.
    agent_cfg.load_checkpoint = ""
    if agent_cfg.experiment_name != EXPERIMENT or agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("Expected the dedicated hexapod_walk_wsp PPO runner configuration")
    if list(agent_cfg.actor.hidden_dims) != [256, 256, 128] or agent_cfg.actor.activation != "elu":
        raise ValueError("WSP warmstart requires the original 67 -> 256 -> 256 -> 128 -> 18 ELU actor")
    agent_cfg.actor.distribution_cfg.init_std = INITIAL_STD
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version)
    torch.manual_seed(args.seed)
    source = args.checkpoint.resolve()
    source_hash = sha256(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ROOT / "logs/rsl_rl" / EXPERIMENT / f"{stamp}_{args.run_name}_{uuid4().hex[:8]}"
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)
    status = {"status": "preparing", "task": TASK, "log_dir": str(log_dir), "source_checkpoint": str(source),
              "source_sha256": source_hash, "max_iterations": args.max_iterations,
              "started_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "num_envs": args.num_envs}
    status_path = log_dir / "training_status.json"
    print(f"WSP training log directory: {log_dir}", flush=True)
    write_run_manifest(str(log_dir), library="rsl_rl", task=TASK,
                       metadata={"agent": "rsl_rl_cfg_entry_point", "warmstart": "actor_mean_and_normalizer_only"})
    runner = None
    try:
        with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(TASK, cfg=env_cfg)
            cleanup.callback(close_env, env)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(wrapped, agent_cfg, log_dir=str(log_dir))
            proof = actor_only_warmstart(runner, checkpoint)
            proof.update({
                "task": TASK, "source_checkpoint": str(source), "source_sha256": source_hash,
                "created_at": datetime.now(timezone.utc).isoformat(), "rsl_rl_version": version,
                "seed": args.seed, "num_envs": args.num_envs, "max_iterations": args.max_iterations,
                "device": str(args.device), "logger": args.logger,
                "algorithm": agent_cfg.algorithm.to_dict(),
                "actor_configuration": agent_cfg.actor.to_dict(),
                "num_steps_per_env": agent_cfg.num_steps_per_env,
                "save_interval": agent_cfg.save_interval,
                "scope": "Initialization proof before any PPO update; no gait improvement claim.",
            })
            with (log_dir / "initialization.json").open("x", encoding="utf-8") as stream:
                json.dump(proof, stream, indent=2, allow_nan=False)
                stream.write("\n")
            dump_train_configs(str(log_dir), env_cfg, agent_cfg)
            runner.add_git_repo_to_log(__file__)
            status["status"] = "training"
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print("Loaded WSP actor mean + normalizer; fresh critic, optimizer, iteration 0 and std 0.10.", flush=True)
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=agent_cfg.init_at_random_ep_len)
            status["status"] = "completed"
            status["final_learning_iteration"] = runner.current_learning_iteration
            status["saved_checkpoints"] = [str(path) for path in sorted(log_dir.glob("model_*.pt"))]
            if not status["saved_checkpoints"]:
                raise RuntimeError("Training returned without saving a checkpoint")
    except (Exception, KeyboardInterrupt) as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        # Keep the originating stack even if the console/pipe itself has failed.
        status["error_traceback"] = traceback.format_exc()
        status["last_learning_iteration"] = None if runner is None else runner.current_learning_iteration
        status["saved_checkpoints"] = [str(path) for path in sorted(log_dir.glob("model_*.pt"))]
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        status["source_sha256_unchanged"] = sha256(source) == source_hash
        if not status["source_sha256_unchanged"]:
            status["status"] = "failed"
            status["source_integrity_error"] = "Source checkpoint changed during training"
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not status["source_sha256_unchanged"]:
        raise RuntimeError("Source checkpoint changed during training")
    print(f"WSP training completed: {log_dir}", flush=True)
    return log_dir


def main():
    try:
        run(parse_args())
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
