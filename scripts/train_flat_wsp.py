"""Start fresh PPO training from only an actor's mean policy and normalization.

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
import math
from pathlib import Path
import re
import sys
import traceback
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
TASK = "Hexapod-Walk-WSP"
EXPERIMENT = "hexapod_walk_wsp"
CHECKPOINT = ROOT / "outputs/flat_wsp/physical_20261108/tripod_probes/model_13250_wsp_tripod_compact_v2.pt"
REFINE_CHECKPOINT = ROOT / "logs/rsl_rl/hexapod_walk_wsp/2026-11-08_05-09-04_flat_wsp_20261108T050840347Z_abbf5ff6_63b7c699/model_1999.pt"
REFINE_SHA256 = "e8a7da57b3fbcf7a9a2bc4569b1aa93653c0691525cb2f5f1c4cf1cad98b8734"
WSP_STOP_PARENT = ROOT / "logs/rsl_rl/hexapod_walk_refine/2026-11-08_08-40-19_flat_refine_20261108T083955889Z_0af00a16_5f77c9f4/model_999.pt"
WSP_STOP_PARENT_SHA256 = "c8828d64d8f1257f9a140a3990cf1250c91c111d8afcb80f1aa4f2f66d648601"
STD_KEY = "distribution.log_std_param"
NORMALIZATION_EPSILON = 0.01
INITIAL_STD = 0.10
TASK_SETTINGS = {
    TASK: {"experiment": EXPERIMENT, "checkpoint": CHECKPOINT, "checkpoint_sha256": None,
           "num_envs": 1024, "max_iterations": 3000, "run_name": "flat_wsp", "initial_std": INITIAL_STD},
    "Hexapod-Walk-Refine": {"experiment": "hexapod_walk_refine", "checkpoint": REFINE_CHECKPOINT,
                            "checkpoint_sha256": REFINE_SHA256, "num_envs": 5120,
                            "max_iterations": 1000, "run_name": "flat_refine", "initial_std": 0.08},
    "Hexapod-Walk-WspStop": {"experiment": "hexapod_walk_wsp_stop", "checkpoint": None,
                             "checkpoint_sha256": None, "num_envs": 5120,
                             "max_iterations": 2000, "run_name": "flat_wsp_stop", "initial_std": 0.05},
}


def apply_task_defaults(args):
    """Resolve explicit task defaults without silently selecting another source."""
    if args.task not in TASK_SETTINGS:
        raise ValueError(f"Unsupported actor-only training task: {args.task}")
    settings = TASK_SETTINGS[args.task]
    using_default_checkpoint = args.checkpoint is None
    for key in ("checkpoint", "num_envs", "max_iterations", "run_name", "initial_std"):
        if getattr(args, key) is None:
            setattr(args, key, settings[key])
    if args.checkpoint_sha256 is None and using_default_checkpoint:
        args.checkpoint_sha256 = settings["checkpoint_sha256"]
    if args.checkpoint_sha256 is not None:
        args.checkpoint_sha256 = args.checkpoint_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a 64-character SHA-256")
    if not math.isfinite(args.initial_std) or not 0 < args.initial_std <= 1:
        raise ValueError("initial_std must be finite and in (0, 1]")
    if args.task == "Hexapod-Walk-WspStop" and (args.checkpoint is None or args.checkpoint_sha256 is None
                                               or getattr(args, "wsp_provenance", None) is None):
        raise ValueError("WspStop requires explicit checkpoint, checkpoint_sha256 and wsp_provenance")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wsp_lineage(path, candidate_path, candidate_hash, candidate_checkpoint):
    """Prove the candidate's declared, bounded actor edit against model_999."""
    import torch

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("source_sha256") != WSP_STOP_PARENT_SHA256 or report.get("candidate_sha256") != candidate_hash:
        raise ValueError("WSP provenance source/candidate SHA-256 mismatch")
    for key in ("source_sha256_unchanged", "normalizer_preserved", "upstream_preserved"):
        if report.get(key) is not True:
            raise ValueError(f"WSP provenance does not prove {key}")
    if report.get("runtime_inputs_added") is not False or report.get("normalization_epsilon") != NORMALIZATION_EPSILON:
        raise ValueError("WSP provenance changed the input/normalization contract")
    if not report.get("source_checkpoint") or not report.get("candidate"):
        raise ValueError("WSP provenance must identify the original and candidate paths")
    edit_scope = report.get("edit_scope", "head_only")
    allowed = {"mlp.6.weight", "mlp.6.bias"}
    preserved_prefix = [0, 2, 4]
    scope_description = "final-layer-only"
    if edit_scope == "last_two_layers":
        if (report.get("method") != "nonlinear" or report.get("prefix_preserved") is not True
                or report.get("preserved_prefix_layers") != [0, 2]):
            raise ValueError("Last-two-layer WSP requires nonlinear method and exact preserved prefix [0, 2] claim")
        allowed |= {"mlp.4.weight", "mlp.4.bias"}
        preserved_prefix = [0, 2]
        scope_description = "last-two-layers-only"
    elif edit_scope != "head_only":
        raise ValueError(f"Unsupported WSP edit_scope: {edit_scope}")
    # Artifact paths may have been copied between the coding and training PCs.
    # The original hash is authoritative; the known repository path is the fallback.
    reported_parent = Path(report["source_checkpoint"])
    parent = reported_parent if reported_parent.is_file() else WSP_STOP_PARENT
    if not parent.is_file() or sha256(parent) != WSP_STOP_PARENT_SHA256:
        raise ValueError("Original model_999 is missing or changed")
    original = torch.load(parent, map_location="cpu", weights_only=True)
    if original.get("iter") != 999 or candidate_checkpoint.get("iter") != 999:
        raise ValueError("WSP source/candidate must retain source iteration999; PPO starts fresh")
    old, new = original["actor_state_dict"], candidate_checkpoint["actor_state_dict"]
    if set(old) != set(new):
        raise ValueError("WSP changed actor state keys")
    for key in old:
        if (not isinstance(new[key], torch.Tensor) or new[key].shape != old[key].shape
                or new[key].dtype != old[key].dtype):
            raise ValueError(f"WSP changed actor tensor shape/dtype: {key}")
    changed = sorted(key for key in old if not torch.equal(old[key], new[key]))
    if not changed or not set(changed) <= allowed or set(report.get("changed_actor_tensors", [])) != set(changed):
        raise ValueError(f"WSP candidate/report must agree on nonempty {scope_description} actor changes")
    if sha256(parent) != WSP_STOP_PARENT_SHA256:
        raise ValueError("Original model_999 changed during lineage verification")
    return {"report_path": str(Path(path).resolve()), "report_sha256": sha256(path),
            "parent_checkpoint_verified": str(parent.resolve()), "parent_sha256": WSP_STOP_PARENT_SHA256,
            "candidate_checkpoint": str(candidate_path), "candidate_sha256": candidate_hash,
            "actual_changed_actor_tensors": changed, "normalizer_exact_to_parent": True,
            "edit_scope": edit_scope, "preserved_prefix_layers": preserved_prefix,
            "preserved_actor_tensors": sorted(set(old) - allowed),
            "upstream_actor_exact_to_parent": True, "prefix_actor_exact_to_parent": True,
            "upstream_definition": f"Unedited actor MLP prefix layers {preserved_prefix}; all actor state outside the explicitly allowed edit scope is exact.",
            "provenance": report}


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
    parser.add_argument("--task", choices=list(TASK_SETTINGS), default=TASK)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_sha256", help="Require this exact source SHA-256 before simulation starts")
    parser.add_argument("--wsp_provenance", type=Path, help="Required model_999-to-WSP lineage JSON for WspStop")
    parser.add_argument("--initial_std", type=float, help="Fresh exploration std; task default unless explicitly set")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name")
    parser.add_argument("--logger", choices=["tensorboard"], default="tensorboard")
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=[])
    args = parser.parse_args()
    try:
        apply_task_defaults(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.num_envs < 1 or args.max_iterations < 1 or args.seed < 0:
        parser.error("Positive environments/iterations and nonnegative seed required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("run_name must contain only letters, digits, underscore, dot or hyphen")
    if not args.checkpoint.is_file():
        parser.error(f"Missing source checkpoint: {args.checkpoint}")
    if args.wsp_provenance is not None and not args.wsp_provenance.is_file():
        parser.error(f"Missing WSP provenance: {args.wsp_provenance}")
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
    settings = TASK_SETTINGS[args.task]
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=[])
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
    if agent_cfg.experiment_name != settings["experiment"] or agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Expected the dedicated {settings['experiment']} PPO runner configuration")
    if list(agent_cfg.actor.hidden_dims) != [256, 256, 128] or agent_cfg.actor.activation != "elu":
        raise ValueError("Warmstart requires the original 67 -> 256 -> 256 -> 128 -> 18 ELU actor")
    distribution = agent_cfg.actor.distribution_cfg
    configured_std = distribution.init_std
    if not math.isclose(configured_std, settings["initial_std"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Task configuration exploration std differs from the audited task default")
    if not distribution.std_range[0] <= args.initial_std <= distribution.std_range[1]:
        raise ValueError(f"initial_std must fit the configured std range {distribution.std_range}")
    distribution.init_std = args.initial_std
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version)
    torch.manual_seed(args.seed)
    source = args.checkpoint.resolve()
    source_hash = sha256(source)
    if args.checkpoint_sha256 is not None and source_hash != args.checkpoint_sha256:
        raise ValueError(f"Source checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, got {source_hash}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    lineage = (verify_wsp_lineage(args.wsp_provenance, source, source_hash, checkpoint)
               if args.task == "Hexapod-Walk-WspStop" else None)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ROOT / "logs/rsl_rl" / settings["experiment"] / f"{stamp}_{args.run_name}_{uuid4().hex[:8]}"
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)
    status = {"status": "preparing", "task": args.task, "log_dir": str(log_dir), "source_checkpoint": str(source),
              "source_sha256": source_hash, "max_iterations": args.max_iterations,
              "expected_source_sha256": args.checkpoint_sha256, "initial_exploration_std": args.initial_std,
              "started_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "num_envs": args.num_envs}
    status_path = log_dir / "training_status.json"
    print(f"{args.task} training log directory: {log_dir}", flush=True)
    write_run_manifest(str(log_dir), library="rsl_rl", task=args.task,
                       metadata={"agent": "rsl_rl_cfg_entry_point", "warmstart": "actor_mean_and_normalizer_only"})
    runner = None
    try:
        with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(args.task, cfg=env_cfg)
            cleanup.callback(close_env, env)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(wrapped, agent_cfg, log_dir=str(log_dir))
            proof = actor_only_warmstart(runner, checkpoint, expected_std=args.initial_std)
            proof.update({
                "task": args.task, "source_checkpoint": str(source), "source_sha256": source_hash,
                "expected_source_sha256": args.checkpoint_sha256,
                "task_default_exploration_std": configured_std, "initial_exploration_std": args.initial_std,
                "initialization_method": "actor_mean_and_normalizer_only_no_weight_edit",
                "source_wsp_lineage": lineage,
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
            print(f"Loaded exact actor mean + normalizer; fresh critic, optimizer, iteration 0 and std {args.initial_std:.3f}.", flush=True)
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
    print(f"{args.task} training completed: {log_dir}", flush=True)
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
