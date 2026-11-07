"""Check the loaded walking robot with neutral actions; this is not a policy evaluation.

    uv run --locked python scripts/smoke_walk.py --device cuda:0 --viz none

Uses Isaac Lab 3.0's kitless launcher. A report path must not already exist.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from uuid import uuid4


class SmokeFailure(RuntimeError):
    """A state or observation violates the smoke test's contract."""


class StateProbe:
    """Inspect control-step states before the environment can automatically reset them."""

    def __init__(self, report, torch):
        self.report = report
        self.torch = torch
        self.step = 0
        self.checks = 0
        self.height_min = None
        self.height_max = None

    def fail(self, message, **details):
        self.report.setdefault("first_failure", {"step": self.step, "reason": message, **details})
        raise SmokeFailure(message)

    def check(self, env):
        torch = self.torch
        data = env.scene["robot"].data
        states = {
            "root_pose_w": data.root_pose_w.torch,
            "root_vel_w": data.root_vel_w.torch,
            "joint_pos": data.joint_pos.torch,
            "joint_vel": data.joint_vel.torch,
        }
        widths = {"root_pose_w": 7, "root_vel_w": 6, "joint_pos": 18, "joint_vel": 18}
        for name, value in states.items():
            if tuple(value.shape) != (env.num_envs, widths[name]):
                self.fail(f"Unexpected raw {name} shape: {tuple(value.shape)}")
        finite = {name: torch.isfinite(value).all(dim=1) for name, value in states.items()}
        if not torch.stack(list(finite.values())).all().item():
            invalid = {
                name: (~valid).nonzero(as_tuple=False).flatten().cpu().tolist()
                for name, valid in finite.items() if not valid.all().item()
            }
            self.fail("Non-finite raw simulator state before reset", invalid_envs_by_field=invalid)
        self.checks += 1
        height = states["root_pose_w"][:, 2] - env.scene.env_origins[:, 2]
        low, high = height.min(), height.max()
        self.height_min = low if self.height_min is None else torch.minimum(self.height_min, low)
        self.height_max = high if self.height_max is None else torch.maximum(self.height_max, high)

    def observations(self, obs, num_envs):
        if "policy" not in obs:
            self.fail("Missing policy observation group")
        policy = obs["policy"]
        if tuple(policy.shape) != (num_envs, 67):
            self.fail(f"Expected policy observations ({num_envs}, 67), got {tuple(policy.shape)}")
        if not self.torch.isfinite(policy).all().item():
            self.fail("Non-finite policy observations")

    def summarize(self):
        self.report["raw_state_checks"] = self.checks
        if self.height_min is not None:
            self.report["root_height_range_m"] = [self.height_min.item(), self.height_max.item()]


def parse_args():
    # Match the official entrypoints: disable unused Warp adjoint code before simulator imports.
    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="Hexapod-Walk")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("outputs/full_robot_smoke.json"))
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=[])
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num_envs and --steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.report.exists():
        parser.error(f"Report already exists; choose a new --report path: {args.report}")
    return args


def run(args, report):
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils.math import quat_apply
    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    import RLforHexapod.tasks  # noqa: F401

    torch.manual_seed(args.seed)
    # argparse consumed this script's flags; they are not Hydra configuration overrides.
    env_cfg, _ = resolve_task_config(args.task, "", overrides=[])
    apply_env_overrides(args, env_cfg)
    env_cfg.seed = args.seed
    args.device = env_cfg.sim.device
    probe = StateProbe(report, torch)

    def inspect_before_reset(env):
        # Terminations run after physics, before rewards and automatic resets. Checking only
        # env.step()'s return would miss a NaN that was replaced by a fresh reset state.
        probe.check(env)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    probe_term = "smoke_raw_state_finite"
    if hasattr(env_cfg.terminations, probe_term):
        raise SmokeFailure(f"Reserved termination name already exists: {probe_term}")
    setattr(env_cfg.terminations, probe_term, TerminationTermCfg(func=inspect_before_reset))

    try:
        with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(args.task, cfg=env_cfg)
            cleanup.callback(close_env, env)
            raw_env = env.unwrapped
            robot = raw_env.scene["robot"]
            report.update({
                "device": str(raw_env.device),
                "num_envs": raw_env.num_envs,
                "step_dt_s": raw_env.step_dt,
                "episode_length_s": env_cfg.episode_length_s,
                "asset_usd_path": str(robot.cfg.spawn.usd_path),
                "asset_prim_path": robot.cfg.prim_path,
                "joint_names": list(robot.joint_names),
                "body_names": list(robot.body_names),
                "sampling": "Initial reset and each control-step state before automatic resets; not each physics substep.",
            })
            print(f"Loaded asset: {report['asset_usd_path']}", flush=True)
            print(f"Joint order: {report['joint_names']}", flush=True)
            print(f"Body order: {report['body_names']}", flush=True)
            obs, _ = env.reset(seed=args.seed)
            probe.check(raw_env)
            probe.observations(obs, raw_env.num_envs)
            actions = torch.zeros_like(raw_env.action_manager.action)
            if tuple(actions.shape) != (raw_env.num_envs, 18):
                probe.fail(f"Expected actions ({raw_env.num_envs}, 18), got {tuple(actions.shape)}")
            report["policy_observation_shape"] = list(obs["policy"].shape)
            report["action_shape"] = list(actions.shape)
            pose = robot.data.root_pose_w.torch
            forward = torch.zeros(raw_env.num_envs, 3, device=raw_env.device)
            forward[:, 0] = 1.0
            forward = quat_apply(pose[:, 3:7], forward)
            report["initial_root_pose_env0"] = pose[0].cpu().tolist()
            report["initial_body_x_world_first_envs"] = forward[:8].cpu().tolist()
            print(f"Initial body +x in world (env 0): {report['initial_body_x_world_first_envs'][0]}", flush=True)
            expected_forward = torch.zeros_like(forward)
            expected_forward[:, 0] = 1.0
            facing_forward = torch.isclose(forward, expected_forward, atol=1e-5, rtol=0.0).all(dim=1)
            if not facing_forward.all().item():
                probe.fail(
                    "Initial body +x does not face world +x; check the asset frame and root quaternion",
                    env_ids=(~facing_forward).nonzero(as_tuple=False).flatten().cpu().tolist(),
                )

            manager = raw_env.termination_manager
            if probe_term not in manager.active_terms:
                probe.fail("Pre-reset raw-state probe was not registered")
            names = [name for name in manager.active_terms if name != probe_term]
            failure_names = [name for name in names if not manager.get_term_cfg(name).time_out]
            report["termination_counts"] = {name: 0 for name in names}
            report["termination_count_note"] = "Counts episode endings per term across all environments; terms can overlap."
            report["episodes_ended"] = 0
            report["failed_episodes"] = 0
            report["failure_terms"] = failure_names

            for step in range(1, args.steps + 1):
                probe.step = step
                checks_before = probe.checks
                with torch.inference_mode():
                    obs, reward, terminated, truncated, _ = env.step(actions)
                report["steps_completed"] = step
                if probe.checks != checks_before + 1:
                    probe.fail("Pre-reset raw-state probe did not run exactly once for this control step")
                # Raw states were already checked by the probe, before any automatic reset.
                probe.observations(obs, raw_env.num_envs)
                if not torch.isfinite(reward).all().item():
                    probe.fail("Non-finite returned rewards")
                reasons = {name: manager.get_term(name) for name in names}
                for name, flags in reasons.items():
                    report["termination_counts"][name] += int(flags.sum().item())
                report["episodes_ended"] += int((terminated | truncated).sum().item())
                failed = torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)
                for name in failure_names:
                    failed |= reasons[name]
                report["failed_episodes"] += int(failed.sum().item())
                if failed.any().item() and "first_failure" not in report:
                    report["first_failure"] = {
                        "step": step,
                        "reason": "Neutral-pose stability termination (not a trained-policy result)",
                        "env_ids": failed.nonzero(as_tuple=False).flatten().cpu().tolist(),
                        "terms": [name for name in failure_names if reasons[name].any().item()],
                    }
            report["simulated_time_s"] = args.steps * raw_env.step_dt
            report["timeout_reset_exercised"] = report["termination_counts"].get("time_out", 0) > 0
            report["status"] = "failed" if report["failed_episodes"] else "passed"
    finally:
        probe.summarize()


def main():
    args = parse_args()
    report = {
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": "Neutral-action asset/runtime and default-pose stability check; not walking or stair policy performance.",
        "task": args.task,
        "seed": args.seed,
        "steps_requested": args.steps,
        "steps_completed": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation reserves this run's path, including against concurrent invocations.
    try:
        stream = args.report.open("x", encoding="utf-8")
    except FileExistsError:
        print(f"Report already exists; choose a new --report path: {args.report}", file=sys.stderr)
        return 2
    with stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        try:
            run(args, report)
        except KeyboardInterrupt:
            report["status"] = "interrupted"
            report["error"] = "Interrupted before the requested step budget completed"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            stream.seek(0)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.truncate()
    print(f"Neutral-action smoke: {report['status']}; steps={report['steps_completed']}/{args.steps}")
    print(f"Root height range [m]: {report.get('root_height_range_m')}")
    print(f"Termination counts: {report.get('termination_counts', {})}")
    if "first_failure" in report:
        print(f"First failure: {report['first_failure']}")
    print(f"Report: {args.report.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
