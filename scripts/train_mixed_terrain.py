"""Train a shared flat/stair policy from a flat or mixed PPO checkpoint.

Warmstarting preserves the first 67 actor inputs and zeros the 28 new columns.
Resuming restores the full 95-input PPO state. --verify-only constructs the
environment and runner and writes initialization proofs without performing PPO.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
# An isolated runtime must win over any older editable install or user branch.
sys.path.insert(0, str(SOURCE_ROOT))
os.environ["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
from mixed_actor_adapter import (JOINT_NAMES, MIXED_OBSERVATIONS, PREFIX_TERMS, SOURCE_SHA256,
                                 initialize_mixed_runner)
from train_flat_wsp import sha256


TASK = "Hexapod-Mixed-Flat-Stair"
EXPERIMENT = "hexapod_mixed_flat_stair"
INITIAL_STD = .05
CURRICULUM_INFO_KEY = "stair_curriculum"


def validate_initial_episode_ages(agent, episode_lengths):
    """Both cohorts start at zero despite their different 20/45 s horizons."""
    if agent.init_at_random_ep_len is not False:
        raise ValueError("Mixed PPO must disable global random episode ages")
    if episode_lengths.count_nonzero().item() != 0:
        raise ValueError("Fresh mixed PPO rollout must start with zero episode ages")
    return {"init_at_random_ep_len": False, "initial_episode_ages_zero": True,
            "reason": "Flat episodes have a 20 s horizon; stair episodes have a 45 s horizon."}


def checkpoint_curriculum_state(checkpoint):
    infos = checkpoint.get("infos")
    if infos is None:
        return None
    if not isinstance(infos, dict):
        raise ValueError("Checkpoint infos must be a dictionary")
    state = infos.get(CURRICULUM_INFO_KEY)
    if state is not None and not isinstance(state, dict):
        raise ValueError("Checkpoint stair curriculum state must be a dictionary")
    return state


def save_curriculum_with_checkpoints(runner, get_curriculum_state):
    save = runner.save

    def save_with_curriculum(path, infos=None):
        state = get_curriculum_state()
        if state is not None:
            infos = {**(infos or {}), CURRICULUM_INFO_KEY: state}
        save(path, infos=infos)

    runner.save = save_with_curriculum


def validate_adam_checkpoint(optimizer, saved):
    """Adam checks group sizes but can silently accept incompatible moments."""
    import torch

    if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
        raise ValueError("Mixed resume requires native Adam optimizer state")
    groups = saved.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("Checkpoint Adam parameter groups differ from the runner")
    state = saved["state"]
    ids = set()
    for group, target in zip(groups, optimizer.param_groups, strict=True):
        indices = group.get("params") if isinstance(group, dict) else None
        if not isinstance(indices, list) or len(indices) != len(target["params"]):
            raise ValueError("Checkpoint Adam parameter group size differs from the runner")
        learning_rate = group.get("lr")
        if type(learning_rate) not in (float, int) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Checkpoint Adam learning rate must be positive and finite")
        for index, parameter in zip(indices, target["params"], strict=True):
            if type(index) is not int or index in ids:
                raise ValueError("Invalid checkpoint Adam parameter indices")
            ids.add(index)
            moments = state.get(index)
            keys = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad"):
                keys.add("max_exp_avg_sq")
            if not isinstance(moments, dict) or set(moments) != keys:
                raise ValueError(f"Missing or incompatible Adam state for parameter {index}")
            for name, value in moments.items():
                shape = () if name == "step" else parameter.shape
                if (not isinstance(value, torch.Tensor) or value.shape != shape
                        or (name != "step" and value.dtype != parameter.dtype) or not torch.isfinite(value).all()):
                    raise ValueError(f"Incompatible Adam state for parameter {index}: {name}")
    if ids != set(state):
        raise ValueError("Checkpoint Adam state contains unexpected parameters")


def resume_mixed_runner(runner, checkpoint):
    """Restore native RSL-RL PPO state and start after its last completed update."""
    import torch

    iteration = checkpoint.get("iter")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Mixed checkpoint requires a nonnegative completed iteration")
    targets = runner.alg.save()
    for name in ("actor_state_dict", "critic_state_dict"):
        saved = checkpoint.get(name)
        target = targets[name]
        if not isinstance(saved, dict) or set(saved) != set(target):
            raise ValueError(f"Mixed checkpoint keys differ for {name}")
        for key, value in saved.items():
            if (not isinstance(value, torch.Tensor) or value.shape != target[key].shape
                    or value.dtype != target[key].dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Incompatible mixed checkpoint tensor: {name}.{key}")
        if any((saved[key] < 0).any() for key in ("obs_normalizer._var", "obs_normalizer._std",
                                                   "obs_normalizer.count")):
            raise ValueError(f"Invalid normalization state in {name}")
    validate_adam_checkpoint(runner.alg.optimizer, checkpoint.get("optimizer_state_dict"))
    # Use the native algorithm loader with the checkpoint already read in weights-only mode.
    if not runner.alg.load(checkpoint, load_cfg=None, strict=True):
        raise RuntimeError("The RSL-RL algorithm did not request iteration restoration")
    # RSL-RL 5.5.1 saves after updating the policy but starts learn() at this index.
    runner.current_learning_iteration = iteration + 1
    return {"source_learning_iteration": iteration, "initial_learning_iteration": runner.current_learning_iteration,
            "actor_and_critic_restored": True, "optimizer_state_entries": len(runner.alg.optimizer.state),
            "learning_rate": runner.alg.learning_rate}


def code_provenance():
    paths = list(SOURCE_ROOT.rglob("*.py"))
    paths += [Path(__file__), ROOT / "scripts/mixed_actor_adapter.py", ROOT / "scripts/train_flat_wsp.py"]
    files = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted(set(paths))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"runtime_root": str(ROOT), "files": files, "manifest_sha256": digest}


def check_prefix_configuration(cfg):
    """Compare effective prefix definitions against the original flat config."""
    from RLforHexapod.tasks.walk.config.hexapod.env_cfg import ActionsCfg, ObservationsCfg
    from RLforHexapod.tasks.walk.config.hexapod.walk_wsp_stop_env_cfg import HexapodWspStopEnvCfg

    original = ObservationsCfg().policy
    for name in PREFIX_TERMS:
        if getattr(cfg.observations.policy, name).to_dict() != getattr(original, name).to_dict():
            raise ValueError(f"Mixed task changed a raw source observation term: {name}")
    for name in ("concatenate_terms", "concatenate_dim", "history_length", "flatten_history_dim", "history_order",
                 "enable_corruption"):
        if getattr(cfg.observations.policy, name) != getattr(original, name):
            raise ValueError(f"Mixed task changed the source observation group setting: {name}")
    if cfg.actions.joint_pos.to_dict() != ActionsCfg().joint_pos.to_dict():
        raise ValueError("Mixed task changed the source joint-position action configuration")
    original_robot = HexapodWspStopEnvCfg().scene.robot
    if cfg.scene.robot.init_state.joint_pos != original_robot.init_state.joint_pos:
        raise ValueError("Mixed task changed source default joint positions")
    return {"raw_prefix_terms_exact": PREFIX_TERMS, "prefix_configuration_exact": True,
            "action_configuration_exact": True, "default_joint_positions_exact": True}


def check_runtime_contract(raw, cfg, expected_terms):
    import torch

    names = list(raw.observation_manager.active_terms["policy"])
    if names != list(expected_terms) or names[:8] != PREFIX_TERMS:
        raise ValueError(f"Runtime policy term order differs: {names}")
    dimensions = [int(math.prod(shape)) for shape in raw.observation_manager._group_obs_term_dim["policy"]]
    if dimensions[:8] != [18, 18, 3, 3, 3, 3, 1, 18] or sum(dimensions) != MIXED_OBSERVATIONS:
        raise ValueError(f"Runtime policy dimensions differ: {dimensions}")
    obs = raw.observation_manager.compute(update_history=False)["policy"]
    if tuple(obs.shape) != (raw.num_envs, 95) or not torch.isfinite(obs).all().item():
        raise ValueError("Mixed actor must receive finite num_envs x 95 observations")
    robot = raw.scene["robot"]
    action = raw.action_manager.get_term("joint_pos")
    if list(robot.joint_names) != JOINT_NAMES or list(action._joint_names) != JOINT_NAMES:
        raise ValueError("Runtime action/joint ordering differs from original flat actor")
    scale = torch.as_tensor(action._scale)
    expected_scale = scale.new_tensor([.5, 1., 1.] * 6)
    if scale.shape[-1] != 18 or not torch.equal(scale, expected_scale.expand_as(scale)):
        raise ValueError("Runtime action scales differ from original flat actor")
    if not torch.equal(action._offset, robot.data.default_joint_pos.torch):
        raise ValueError("Runtime action offset is not the source default joint position")
    proof = {"policy_terms": names, "policy_term_dimensions": dimensions, "policy_observation_size": 95,
             "source_prefix_size": 67, "new_feature_count": 28, "joint_names": JOINT_NAMES,
             "action_scales": [.5, 1., 1.] * 6, "action_default_offset_exact": True,
             "initial_observations_finite": True, "robot_usd": str(robot.cfg.spawn.usd_path),
             "robot_usd_sha256": sha256(Path(robot.cfg.spawn.usd_path)),
             "flat_fraction": cfg.flat_fraction, "stair_height": cfg.stair_height,
             "stair_reward_scale": cfg.stair_reward_scale}
    indices = torch.linspace(0, raw.num_envs - 1, min(raw.num_envs, 32), device=obs.device).round().long()
    proof["physical_prefix_environment_indices"] = indices.tolist()
    return obs[indices, :67].detach().cpu(), proof


def parse_args():
    import warp as wp
    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Warmstart from the verified final flat model_1999")
    source.add_argument("--resume-checkpoint", type=Path, help="Continue from a native mixed PPO model_*.pt")
    parser.add_argument("--checkpoint_sha256", help="Expected SHA-256 (required hash is built in for warmstart)")
    parser.add_argument("--num_envs", type=int, default=5120)
    parser.add_argument("--max_iterations", type=int, default=2000, help="Number of additional PPO updates")
    parser.add_argument("--flat_fraction", type=float, default=.5)
    parser.add_argument("--stair_height", type=float, default=.04)
    parser.add_argument("--stair_reward_scale", type=float, default=.1)
    parser.add_argument("--height-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random-stair-height", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="mixed_flat_stair")
    parser.add_argument("--verify-only", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=[])
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_iterations < 1 or args.seed < 0:
        parser.error("Positive environment/iteration counts and nonnegative seed required")
    if not 0 <= args.flat_fraction <= 1 or not .005 <= args.stair_height <= .16 or not 0 < args.stair_reward_scale <= 1:
        parser.error("flat_fraction [0,1], fixed stair_height [.005,.16] and stair_reward_scale (0,1] required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("run_name must use letters, digits, underscore, dot or hyphen")
    if args.checkpoint_sha256 is not None:
        args.checkpoint_sha256 = args.checkpoint_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
            parser.error("checkpoint_sha256 must be a 64-character SHA-256")
    if args.checkpoint is not None:
        if args.checkpoint_sha256 not in (None, SOURCE_SHA256):
            parser.error("Warmstart requires the verified final model_1999 SHA-256")
        args.checkpoint_sha256 = SOURCE_SHA256
    if not (args.checkpoint or args.resume_checkpoint).is_file():
        parser.error("The checkpoint must be an existing file")
    args.task = TASK
    return args


def run(args):
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, dump_train_configs, write_run_manifest
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config
    import RLforHexapod
    import RLforHexapod.tasks.walk.config.hexapod  # noqa: F401
    from RLforHexapod.tasks.walk.config.hexapod.mixed_env_cfg import MIXED_POLICY_TERMS, MIXED_OBSERVATION_SIZE
    from RLforHexapod.tasks.walk.mdp.mixed_terrain import mixed_curriculum_state

    if not Path(RLforHexapod.__file__).resolve().is_relative_to(SOURCE_ROOT.resolve()):
        raise ValueError("Training imported RLforHexapod outside the selected isolated runtime")
    if MIXED_OBSERVATION_SIZE != 95:
        raise ValueError("Mixed task declared an unexpected observation dimension")
    version = importlib.metadata.version("rsl-rl-lib")
    if version != "5.5.1":
        raise RuntimeError(f"Validated RSL-RL version is 5.5.1, installed {version}")
    resuming = args.resume_checkpoint is not None
    source = (args.resume_checkpoint or args.checkpoint).resolve()
    source_hash = sha256(source)
    expected_hash = args.checkpoint_sha256 if resuming else SOURCE_SHA256
    if expected_hash is not None and source_hash != expected_hash:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a native PPO training checkpoint dictionary")
    if not resuming and checkpoint.get("iter") != 1999:
        raise ValueError("Pinned source must retain its original iteration 1999")
    cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=[])
    apply_env_overrides(args, cfg)
    cfg.seed = args.seed
    cfg.configure_mixed_terrain(flat_fraction=args.flat_fraction, stair_height=args.stair_height,
                                stair_reward_scale=args.stair_reward_scale, height_curriculum=args.height_curriculum,
                                random_stair_height=args.random_stair_height)
    if resuming and args.height_curriculum:
        cfg.stair_curriculum_state = checkpoint_curriculum_state(checkpoint)
    prefix_proof = check_prefix_configuration(cfg)
    if agent.experiment_name != EXPERIMENT or agent.class_name != "OnPolicyRunner":
        raise ValueError("Wrong mixed PPO runner configuration")
    if (list(agent.actor.hidden_dims) != [256, 256, 128] or agent.actor.activation != "elu"
            or agent.actor.distribution_cfg.init_std != INITIAL_STD or agent.clip_actions != 1.):
        raise ValueError("Mixed actor architecture/std/action clipping contract differs")
    args.device = cfg.sim.device
    agent.device, agent.seed = args.device, args.seed
    agent.max_iterations, agent.run_name = args.max_iterations, args.run_name
    agent.resume, agent.load_checkpoint, agent.logger = resuming, str(source) if resuming else "", "tensorboard"
    agent = handle_deprecated_rsl_rl_cfg(agent, version)
    if agent.init_at_random_ep_len is not False:
        raise ValueError("Mixed PPO must disable global random episode ages")
    torch.manual_seed(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ROOT / "logs/rsl_rl" / EXPERIMENT / f"{stamp}_{args.run_name}_{uuid4().hex[:8]}"
    log_dir.mkdir(parents=True, exist_ok=False)
    cfg.log_dir = str(log_dir)
    code = code_provenance()
    status = {"status": "preparing", "task": TASK, "source_checkpoint": str(source), "source_sha256": source_hash,
              "resume": resuming,
              "num_envs": args.num_envs, "max_iterations": args.max_iterations, "seed": args.seed,
              "flat_fraction": args.flat_fraction, "stair_height": args.stair_height,
              "stair_reward_scale": args.stair_reward_scale, "automatic_height_curriculum": args.height_curriculum,
              "random_stair_height": args.random_stair_height,
              "init_at_random_ep_len": False,
              "verify_only": args.verify_only, "log_dir": str(log_dir), "code_manifest_sha256": code["manifest_sha256"],
              "started_at": datetime.now(timezone.utc).isoformat()}
    status_path = log_dir / "training_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    metadata = ({"resume": "native_mixed_ppo"} if resuming else
                {"warmstart": "67_prefix_exact_28_zero_input_columns_actor_only"})
    write_run_manifest(str(log_dir), library="rsl_rl", task=TASK,
                       metadata={**metadata, "runtime_root": str(ROOT)})
    print(f"Mixed training log directory: {log_dir}", flush=True)
    runner = None
    try:
        with launch_simulation(cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(TASK, cfg=cfg)
            cleanup.callback(close_env, env)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = create_rsl_rl_runner(wrapped, agent, log_dir=str(log_dir))
            episode_proof = validate_initial_episode_ages(agent, env.unwrapped.episode_length_buf)
            physical_prefix, runtime_proof = check_runtime_contract(env.unwrapped, cfg, MIXED_POLICY_TERMS)
            if resuming:
                initialization = resume_mixed_runner(runner, checkpoint)
            else:
                initialization = initialize_mixed_runner(runner, checkpoint, physical_prefix, expected_std=INITIAL_STD)
            if args.height_curriculum:
                state = mixed_curriculum_state(env.unwrapped)
                initialization["stair_curriculum"] = state
                save_curriculum_with_checkpoints(runner, lambda: mixed_curriculum_state(env.unwrapped))
            proof = {**status, "initialization": initialization, "configuration_contract": prefix_proof,
                     "runtime_contract": runtime_proof, "code_provenance": code, "rsl_rl_version": version,
                     "initial_episode_state": episode_proof,
                     "algorithm": agent.algorithm.to_dict(), "actor_configuration": agent.actor.to_dict(),
                     "num_steps_per_env": agent.num_steps_per_env, "save_interval": agent.save_interval,
                     "scope": "Mixed training can change flat behavior. Evaluate both groups separately."}
            with (log_dir / "initialization.json").open("x", encoding="utf-8") as stream:
                json.dump(proof, stream, indent=2, allow_nan=False)
                stream.write("\n")
            dump_train_configs(str(log_dir), cfg, agent)
            if (ROOT / ".git").exists():
                runner.add_git_repo_to_log(__file__)
            if args.verify_only:
                status["status"] = "verified_initialization_without_ppo"
                status["ppo_updates_performed"] = 0
            else:
                status["status"] = "training"
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=agent.init_at_random_ep_len)
                status["status"] = "completed"
                status["final_learning_iteration"] = runner.current_learning_iteration
                status["saved_checkpoints"] = [str(path) for path in sorted(log_dir.glob("model_*.pt"))]
                if not status["saved_checkpoints"]:
                    raise RuntimeError("PPO returned without saving any checkpoint")
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}", error_traceback=traceback.format_exc(),
                      last_learning_iteration=None if runner is None else runner.current_learning_iteration)
        raise
    finally:
        status["source_sha256_unchanged"] = source.is_file() and sha256(source) == source_hash
        status["code_manifest_unchanged"] = code_provenance()["manifest_sha256"] == code["manifest_sha256"]
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        if not status["source_sha256_unchanged"] or not status["code_manifest_unchanged"]:
            status["status"] = "failed"
            status["integrity_error"] = "Source checkpoint or selected runtime code changed during the run"
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if status["status"] == "failed":
        raise RuntimeError(status["integrity_error"])
    print(f"Mixed run {status['status']}: {log_dir}", flush=True)
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
