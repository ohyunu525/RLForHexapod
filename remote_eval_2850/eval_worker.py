"""First-episode evaluation of one verified 2850 model; no training or export."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import torch

MODELS = {
    "original": ("model_2850.pt", "218aa9b0bd3a708cd9a0ce8b5447f9b5b623c068c213420cda2cdecc7118e9e5"),
    "wsp": ("model_2850_wsp.pt", "d3fc4f998759c81c8d97e83deae3ae091337b77319fbeac6c8d810db38bc577e"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_for_evaluation(runner, checkpoint, device):
    """Remap serialized CUDA tensors to the actual evaluation device."""
    return runner.load(str(checkpoint), map_location=str(device))


def collect(env, policy, obs, max_steps, progress=None):
    """Keep exactly the first episode of each env, including all failures."""
    base = env.unwrapped
    tracker, tm = base.stair_tracker, base.termination_manager
    start = tracker.move.clone()
    if not (start == 0).all():
        raise ValueError("Evaluation must begin in phase 0 for every environment")
    finished = torch.zeros_like(start, dtype=torch.bool)
    top_step = torch.full_like(start, -1)
    final_move = torch.full_like(start, -1)
    length = torch.zeros_like(start)
    terms = list(tm.active_terms)
    reasons = torch.zeros((len(start), len(terms)), dtype=torch.bool, device=start.device)
    initial = {
        "root_state_w": base.scene["robot"].data.root_state_w.detach().cpu().tolist(),
        "joint_pos": base.scene["robot"].data.joint_pos.detach().cpu().tolist(),
        "joint_vel": base.scene["robot"].data.joint_vel.detach().cpu().tolist(),
    }
    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            action = policy(obs)
            if not torch.isfinite(action).all():
                raise ValueError("Non-finite inference action")
            obs, _, done, _ = env.step(action)
            done = done.bool()
            policy.reset(done)
            # The env has auto-reset by this point; ended_move preserves pre-reset phase.
            move = torch.where(done, tracker.ended_move, tracker.move)
            newly_top = (~finished) & (top_step < 0) & (move >= tracker.final_advance)
            top_step[newly_top] = step
            newly = done & ~finished
            final_move[newly] = tracker.ended_move[newly]
            length[newly] = step
            if terms:
                reasons[newly] = torch.stack([tm.get_term(name) for name in terms], dim=1)[newly]
            finished |= newly
            if progress is not None and (step % 250 == 0 or finished.all()):
                progress(step, int(finished.sum().item()), len(start))
            if finished.all():
                break
    if not finished.all():
        raise RuntimeError(f"{int((~finished).sum())} first episodes unfinished after {max_steps} steps")
    return {"start_phase": start.cpu().tolist(), "final_phase": final_move.cpu().tolist(),
            "length_steps": length.cpu().tolist(), "top_step": top_step.cpu().tolist(),
            "termination_names": terms, "termination_flags": reasons.cpu().tolist(),
            "initial_state": initial}


def summarize(data, dt, horizon_s, done_move):
    count = len(data["top_step"])
    if count == 0:
        raise ValueError("No episodes collected")
    reached = [step >= 0 for step in data["top_step"]]
    observed = [step * dt for step in data["top_step"] if step >= 0]
    capped = [min(step * dt, horizon_s) if step >= 0 else horizon_s for step in data["top_step"]]
    reason_counts = {name: sum(bool(row[j]) for row in data["termination_flags"])
                     for j, name in enumerate(data["termination_names"])}
    fall_indices = [i for i, n in enumerate(data["termination_names"]) if n in ("bad_ori", "base_contact")]
    fall_count = sum(any(row[j] for j in fall_indices) for row in data["termination_flags"])
    return {"episodes": count, "reached_top_count": sum(reached), "reached_top_rate": sum(reached) / count,
            "settled_success_count": sum(p >= done_move for p in data["final_phase"]),
            "fall_terms": [data["termination_names"][j] for j in fall_indices],
            "fall_count": fall_count, "fall_rate": fall_count / count,
            "termination_counts": reason_counts,
            "successful_top_time_mean_s": sum(observed) / len(observed) if observed else None,
            "all_episode_capped_top_time_mean_s": sum(capped) / count,
            "capped_top_time_per_episode_s": capped,
            "capped_time_definition": "First recorded phase>=final_advance time; every noncompletion receives the full horizon, including early falls.",
            "horizon_s": horizon_s, "step_dt_s": dt}


def main():
    import gymnasium as gym
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.cli import load_external_tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
    from hexapod_student_compat import register, inspect_asset_provenance, SOURCE_COMMIT
    from hexapod_student_compat.verify_snapshot import verify_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=MODELS, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    if args.result.exists() or args.num_envs < 1:
        raise ValueError("Result must be new and num_envs positive")
    bundle = Path(__file__).resolve().parent
    filename, expected = MODELS[args.role]
    checkpoint = bundle / filename
    if sha(checkpoint) != expected:
        raise ValueError("Checkpoint SHA256 mismatch")
    snapshot = verify_snapshot()
    load_external_tasks()
    from RLforHexapod.assets import ROBOT_USDA, OBSTACLES_DIR
    assets = inspect_asset_provenance(ROBOT_USDA, OBSTACLES_DIR)
    if not assets["matches_pinned"]:
        write_json(args.result.with_suffix(".asset_failure.json"), assets)
        raise ValueError("Installed robot/stair assets differ from pinned source")
    task = register()
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.events.stair_reset.params["buffer_ratio"] = 0.0
    env_cfg.events.stair_reset.params["x_ranges"] = ((-0.1, 0.1),)
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    started = time.monotonic()
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env = gym.make(task, cfg=env_cfg)
        cleanup.callback(env.close)
        base = env.unwrapped
        dims = base.observation_manager.group_obs_dim
        if tuple(dims["student"]) != (271,) or tuple(dims["policy"]) != (91,) or base.action_manager.total_action_dim != 18:
            raise ValueError("Student-A observation/action contract mismatch")
        if agent_cfg.class_name != "DistillationRunner":
            raise ValueError("Expected DistillationRunner")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        agent_cfg.device = str(base.device)
        runner = create_rsl_rl_runner(env, agent_cfg)
        load_for_evaluation(runner, checkpoint, base.device)
        policy = runner.get_inference_policy(device=base.device)
        progress = lambda step, done, total: print(f"[{args.role} seed {args.seed}] step {step}; first episodes {done}/{total}", flush=True)
        data = collect(env, policy, env.get_observations(), base.max_episode_length + 1, progress)
        dt = float(base.step_dt)
        horizon = float(base.max_episode_length * dt)
        result = {"schema_version": 1, "status": "complete", "role": args.role, "seed": args.seed,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": expected, "num_envs": args.num_envs,
                  "task": task, "source_commit": SOURCE_COMMIT, "training_task_proven": False,
                  "snapshot": snapshot, "assets": assets,
                  "config_changes": {"buffer_ratio": 0.0, "x_ranges": [[-0.1, 0.1]]},
                  "height_map_noise_enabled": bool(base.cfg.observations.student.enable_corruption),
                  "normalizer": "Checkpoint runner inference policy; no external renormalization",
                  "policy": "Mean deterministic action, GRU reset on every done; no training/export",
                  "final_advance": int(base.stair_tracker.final_advance), "done_move": int(base.stair_tracker.done_move),
                  "summary": summarize(data, dt, horizon, int(base.stair_tracker.done_move)), "episodes": data,
                  "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                              "isaaclab": importlib.metadata.version("isaaclab"), "rsl_rl": importlib.metadata.version("rsl-rl-lib"),
                              "device": str(base.device), "wall_seconds": time.monotonic() - started},
                  "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        if sha(checkpoint) != expected:
            raise ValueError("Checkpoint changed during evaluation")
        write_json(args.result, result)
        print(json.dumps(result["summary"], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
