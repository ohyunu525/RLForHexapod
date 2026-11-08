"""Bounded mixed-task integration check, not a climbing-success evaluation."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TASK = "Hexapod-Mixed-Flat-Stair"


def main():
    import warp as wp
    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2300)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--flat_fraction", type=float, default=.5)
    parser.add_argument("--stair_height", type=float, default=.04)
    parser.add_argument("--stair_reward_scale", type=float, default=.1)
    parser.add_argument("--checkpoint", type=Path, help="Optional original67-input actor; otherwise neutral actions")
    parser.add_argument("--report", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=[])
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 1 or not 0 < args.flat_fraction < 1:
        parser.error("Smoke requires at least2envs, positive steps and both cohorts")
    if args.report.exists():
        parser.error("Refusing to overwrite an existing report")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "preparing", "task": TASK, "steps_requested": args.steps,
              "num_envs": args.num_envs, "scope": "Integration/finite-state checks, not a gait-success claim",
              "action_mode": "source67_actor" if args.checkpoint else "neutral"}
    try:
        import gymnasium as gym
        import torch
        from isaaclab.managers import TerminationTermCfg
        from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import resolve_task_config
        import RLforHexapod.tasks  # noqa: F401
        from RLforHexapod.tasks.walk.mdp.mixed_terrain import mixed_masks

        cfg, _ = resolve_task_config(TASK, "", overrides=[])
        cfg.configure_mixed_terrain(flat_fraction=args.flat_fraction, stair_height=args.stair_height,
                                    stair_reward_scale=args.stair_reward_scale)
        apply_env_overrides(args, cfg)
        cfg.seed = args.seed
        args.device = cfg.sim.device
        torch.manual_seed(args.seed)
        counts = {"flat": {}, "stair": {}}
        raw_checks = 0
        step = -1
        maximum_stair_phase = 0

        def capture_before_reset(env):
            nonlocal raw_checks, maximum_stair_phase
            flat, stair = mixed_masks(env)
            data = env.scene["robot"].data
            for name in ("root_pose_w", "root_vel_w", "joint_pos", "joint_vel"):
                value = getattr(data, name).torch
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite raw {name} before reset at step{step}")
            for name in env.termination_manager.active_terms:
                if name == "mixed_smoke_probe":
                    continue
                value = env.termination_manager.get_term(name)
                for cohort, mask in (("flat", flat), ("stair", stair)):
                    counts[cohort][name] = counts[cohort].get(name, 0) + int(value[mask].sum().item())
            raw_checks += 1
            tracker = getattr(env, "stair_tracker", None)
            if tracker is not None:
                maximum_stair_phase = max(maximum_stair_phase, int(tracker.move[stair].max()))
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        cfg.terminations.mixed_smoke_probe = TerminationTermCfg(func=capture_before_reset)
        actor = None
        if args.checkpoint:
            from flat_wsp import FlatActor
            original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            actor = FlatActor(original["actor_state_dict"]).to(args.device).eval()
            report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        with launch_simulation(cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(TASK, cfg=cfg)
            cleanup.callback(close_env, env)
            raw = env.unwrapped
            obs, _ = env.reset(seed=args.seed)
            flat, stair = mixed_masks(raw)
            assert bool(flat.any()) and bool(stair.any())
            assert not bool((flat & stair).any()) and bool((flat | stair).all())
            original_flat, original_stair = flat.clone(), stair.clone()
            local_pose = raw.scene["robot"].data.root_pose_w.torch[:, :3] - raw.scene.env_origins
            if not torch.allclose(local_pose[flat, 1], torch.full_like(local_pose[flat, 1], 4.), atol=1e-4):
                raise RuntimeError("Flat cohort did not reset into its physically separated floor lane")
            if not bool((local_pose[stair, 1].abs() < .5).all()):
                raise RuntimeError("Stair cohort did not reset into the stair corridor")
            report.update(flat_count=int(flat.sum()), stair_count=int(stair.sum()),
                          step_dt_s=float(raw.step_dt), flat_lane_y_m=4.,
                          stair_height_m=float(cfg.stairs.step_height),
                          initial_local_positions=local_pose[:min(args.num_envs, 16)].cpu().tolist(),
                          asset_usd_path=str(raw.scene["robot"].cfg.spawn.usd_path))
            if raw.termination_manager.active_terms[-1] != "mixed_smoke_probe":
                raise RuntimeError("Finite-state probe must run before automatic reset and after other terminations")
            reward_sum = {"flat": 0., "stair": 0.}
            reward_abs_sum = {"flat": 0., "stair": 0.}
            maximum_abs_reward = {"flat": 0., "stair": 0.}
            max_context = 0.
            for step in range(args.steps):
                with torch.inference_mode():
                    flat, stair = mixed_masks(raw)
                    if not torch.equal(flat, original_flat) or not torch.equal(stair, original_stair):
                        raise RuntimeError("Fixed mixed cohort assignment changed")
                    value = obs["policy"]
                    if tuple(value.shape) != (args.num_envs, 95) or not torch.isfinite(value).all():
                        raise RuntimeError("Mixed policy observation must be finite95-dimensional")
                    extra = value[flat, 67:].abs().max().item()
                    max_context = max(max_context, extra)
                    if extra > 1e-7:
                        raise RuntimeError("Flat cohort received nonzero privileged staircase context")
                    if not torch.allclose(value[stair, 79], torch.full_like(value[stair, 79], args.stair_height),
                                          atol=1e-6, rtol=0):
                        raise RuntimeError("Physical staircase height and policy context differ")
                    commands = raw.command_manager.get_command("base_velocity")
                    if not torch.allclose(value[:, 45:48], commands, atol=1e-6, rtol=0):
                        raise RuntimeError("Policy input and live velocity command differ")
                    height = raw.command_manager.get_command("base_height")
                    if not torch.allclose(value[:, 48:49], height, atol=1e-6, rtol=0):
                        raise RuntimeError("Policy input and live height command differ")
                    if not bool((commands[stair, 1:].abs() < 1e-6).all()):
                        raise RuntimeError("Stair commands randomly turned or strafed")
                    if not bool(((commands[stair, 0]-.1).abs() < 1e-6).logical_or(commands[stair, 0].abs() < 1e-6).all()):
                        raise RuntimeError("Stair forward/settle command mismatch")
                    action = (actor(value[:, :67]).clamp(-1, 1) if actor is not None else
                              torch.zeros(args.num_envs, 18, device=raw.device))
                    obs, reward, terminated, truncated, extras = env.step(action)
                    if not torch.isfinite(reward).all():
                        raise RuntimeError("Mixed reward is nonfinite")
                    for cohort, mask in (("flat", flat), ("stair", stair)):
                        reward_sum[cohort] += float(reward[mask].mean())
                        reward_abs_sum[cohort] += float(reward[mask].abs().mean())
                        maximum_abs_reward[cohort] = max(maximum_abs_reward[cohort], float(reward[mask].abs().max()))
                report["steps_completed"] = step + 1
            tracker = getattr(raw, "stair_tracker", None)
            if tracker is None:
                raise RuntimeError("Stair tracker was not exercised")
            if bool((tracker.move[flat] != 0).any()):
                raise RuntimeError("Flat environments advanced the staircase tracker")
            report.update(status="passed", raw_state_checks=raw_checks, termination_counts=counts,
                          flat_extra_context_max_abs=max_context,
                          cohort_mean_reward_per_step={k:v/args.steps for k,v in reward_sum.items()},
                          cohort_mean_absolute_reward_per_step={k:v/args.steps for k,v in reward_abs_sum.items()},
                          cohort_maximum_absolute_reward_per_step=maximum_abs_reward,
                          maximum_observed_stair_phase=maximum_stair_phase,
                          final_stair_phase_min_max=[int(tracker.move[stair].min()), int(tracker.move[stair].max())],
                          final_flat_phase_max=int(tracker.move[flat].max()),
                          stairs_buffer_keys=sorted(int(k) for k in getattr(raw, "stair_states", {})))
            if torch.cuda.is_available():
                report["torch_cuda_peak_allocated_MiB"] = torch.cuda.max_memory_allocated()/1024**2
                report["torch_cuda_peak_reserved_MiB"] = torch.cuda.max_memory_reserved()/1024**2
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({k: report.get(k) for k in ("status", "num_envs", "steps_completed", "error")}), flush=True)


if __name__ == "__main__":
    main()
