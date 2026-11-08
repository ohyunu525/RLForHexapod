"""Bounded deterministic Hexapod-Walk rollout with pre-reset gait diagnostics.

Run separately for baseline and candidate with identical arguments. A completed
evaluation is not a claim that a candidate improves walking. No rewards, motor
settings, robot assets, or policy observation definitions are changed here.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


JOINT_NAMES = [f"L{leg}_{joint}" for leg in range(1, 7) for joint in ("coxa", "femur", "tibia")]
TIBIA_BODIES = ["tibia", *[f"tibia_{leg}" for leg in range(2, 7)]]
PROBE_TERM = "flat_wsp_capture_before_reset"


def episode_segments(episode_id, env):
    """Yield contiguous episode spans without joining repeated IDs."""
    changes = np.flatnonzero(np.diff(episode_id[:, env]) != 0) + 1
    edges = np.r_[0, changes, len(episode_id)]
    return zip(edges[:-1], edges[1:])


def episode_weighted_excursion(values, episode_id):
    """Time-weight each episode's p95-p05; reset pose changes are excluded."""
    if values.shape[:2] != episode_id.shape:
        raise ValueError("Episode IDs must match time and environment dimensions")
    total = np.zeros(values.shape[2:], dtype=np.float64)
    count = 0
    for env in range(values.shape[1]):
        for start, end in episode_segments(episode_id, env):
            span = values[start:end, env]
            total += (end - start) * (np.quantile(span, .95, axis=0) - np.quantile(span, .05, axis=0))
            count += end - start
    return total / count


def joint_motion(q, episode_id, dt, speed_deadband=.05, reversal_hysteresis=.03):
    """Within-episode travel and angle-hysteresis reversals, per env/joint.

    Travel ignores per-step speeds below 0.05 rad/s. A direction reversal needs
    at least 0.03 rad away from the preceding extremum, preventing tiny jitter
    from being interpreted as repeated flexion. Neither a reset jump nor a
    direction change across a reset contributes. These are motion descriptors,
    not proof of alternating foot support or an efficient gait.
    """
    if q.ndim != 3 or q.shape[:2] != episode_id.shape or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Joint motion requires time × env × joint samples, episode IDs and positive dt")
    positive = np.zeros(q.shape[1:], dtype=np.float64)
    negative = np.zeros_like(positive)
    bidirectional = np.zeros_like(positive)
    reversals = np.zeros(q.shape[1:], dtype=np.int64)
    measured_time = np.zeros(q.shape[1], dtype=np.float64)
    for env in range(q.shape[1]):
        for start, end in episode_segments(episode_id, env):
            span = q[start:end, env]
            delta = np.diff(span, axis=0)
            delta = np.where(np.abs(delta) >= speed_deadband * dt, delta, 0.0)
            segment_positive = np.maximum(delta, 0).sum(0)
            segment_negative = np.maximum(-delta, 0).sum(0)
            positive[env] += segment_positive
            negative[env] += segment_negative
            bidirectional[env] += np.minimum(segment_positive, segment_negative)
            measured_time[env] += max(end - start - 1, 0) * dt
            direction = np.zeros(q.shape[2], dtype=np.int8)
            peak, trough = span[0].copy(), span[0].copy()
            for sample in span[1:]:
                peak = np.maximum(peak, sample)
                trough = np.minimum(trough, sample)
                rise = (direction <= 0) & (sample - trough >= reversal_hysteresis)
                fall = (direction >= 0) & (peak - sample >= reversal_hysteresis)
                reversals[env] += (rise & (direction == -1)) | (fall & (direction == 1))
                direction = np.where(rise, 1, np.where(fall, -1, direction))
                peak = np.where(rise, sample, peak)
                trough = np.where(fall, sample, trough)
    return {"positive_travel_rad": positive, "negative_travel_rad": negative,
            "bidirectional_travel_rad": bidirectional, "direction_reversals": reversals,
            "measured_time_s": measured_time}


def summarize_trace(trace, dt, vx, target_height, vy=0.0, yaw_rate=0.0):
    """Summarize actual samples, including terminated episodes and final frames.

    Progress uses each step's pre/post state, so automatic reset displacement is
    never counted as locomotion. Contact here means the tibia link touched an
    object; it does not establish that the distal foot alone supports the robot.
    """
    if not trace or len(trace["root_pose"]) == 0:
        raise ValueError("No captured rollout frames")
    for key, array in trace.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Non-finite captured {key}; rollout cannot be scored")
    pose = trace["root_pose"]
    velocity = trace["root_vel"]
    q = trace["joint_pos"]
    raw = trace["actions_raw"]
    if q.ndim != 3 or q.shape[-1] != 18 or raw.shape != q.shape:
        raise ValueError("Expected time × environments × 18 joint/action arrays")
    episode_id = trace["episode_id"]
    motion = joint_motion(q, episode_id, dt)
    joint_excursion = episode_weighted_excursion(q, episode_id)
    # World-space body forward direction from an XYZW quaternion.
    quat = pose[..., 3:7]
    x, y, z, w = np.moveaxis(quat, -1, 0)
    forward = np.stack((1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w)), axis=-1)
    speed_bx = (velocity[..., :3] * forward).sum(-1)
    side = np.stack((2 * (x*y - z*w), 1 - 2 * (x*x + z*z), 2 * (y*z + x*w)), axis=-1)
    up = np.stack((2 * (x*z + y*w), 2 * (y*z - x*w), 1 - 2 * (x*x + y*y)), axis=-1)
    speed_by = (velocity[..., :3] * side).sum(-1)
    yaw_speed = (velocity[..., 3:6] * up).sum(-1)
    tip_relative = trace["foot_tip_pos_w"] - pose[..., None, :3]
    tip_forward = (tip_relative * forward[..., None, :]).sum(-1)
    tip_forward_excursion = episode_weighted_excursion(tip_forward, episode_id)
    tip_height_excursion = episode_weighted_excursion(trace["foot_tip_pos_w"][..., 2], episode_id)
    delta_x = pose[..., 0] - trace["pre_root_pose"][..., 0]
    gravity_z = 1 - 2 * (x*x + y*y)
    tilt = np.arccos(np.clip(gravity_z, -1.0, 1.0))
    support = np.linalg.norm(trace["tibia_force_w"], axis=-1) > 1.0
    steps, envs = pose.shape[:2]
    elapsed = steps * dt
    per_leg = {}
    for leg in range(6):
        indices = slice(3 * leg, 3 * leg + 3)
        per_leg[f"L{leg+1}"] = {
            "joint_excursion_p95_p05_rad": joint_excursion[indices].tolist(),
            "joint_positive_travel_rad_per_env": motion["positive_travel_rad"][:, indices].mean(0).tolist(),
            "joint_negative_travel_rad_per_env": motion["negative_travel_rad"][:, indices].mean(0).tolist(),
            "joint_bidirectional_travel_rad_per_env": motion["bidirectional_travel_rad"][:, indices].mean(0).tolist(),
            "joint_direction_reversals_per_env": motion["direction_reversals"][:, indices].mean(0).tolist(),
            "joint_mean_rad": q[:, :, indices].mean((0, 1)).tolist(),
            "action_clip_fraction": (np.abs(raw[:, :, indices]) >= 1.0).mean((0, 1)).tolist(),
            "tibia_contact_fraction": float(support[:, :, leg].mean()),
            "maximum_air_time_s": float(trace["tibia_air_time"][:, :, leg].max()),
            "tip_height_p95_p05_m": float(tip_height_excursion[leg]),
            "tip_forward_excursion_p95_p05_m": float(tip_forward_excursion[leg]),
        }
    return {
        "captured_steps": steps,
        "environments": envs,
        "simulated_time_s_per_env": elapsed,
        "progress_world_x_m_per_env": delta_x.sum(0).tolist(),
        "mean_progress_speed_world_x_m_s": float(delta_x.sum() / (elapsed * envs)),
        "mean_forward_speed_body_m_s": float(speed_bx.mean()),
        "forward_tracking_rmse_m_s": float(np.sqrt(np.mean((speed_bx - vx)**2))),
        "xy_tracking_rmse_m_s": float(np.sqrt(np.mean((speed_bx - vx)**2 + (speed_by - vy)**2))),
        "yaw_tracking_rmse_rad_s": float(np.sqrt(np.mean((yaw_speed - yaw_rate)**2))),
        "vertical_velocity_rms_m_s": float(np.sqrt(np.mean(velocity[..., 2]**2))),
        "roll_pitch_rate_world_rms_rad_s": float(np.sqrt(np.mean(np.sum(velocity[..., 3:5]**2, axis=-1)))),
        "tilt_rms_rad": float(np.sqrt(np.mean(tilt**2))),
        "height_tracking_rmse_m": float(np.sqrt(np.mean((pose[..., 2] - target_height)**2))),
        "failed_episodes": int(trace["terminated"].sum()),
        "timeout_episodes": int(trace["timed_out"].sum()),
        "environments_with_failure": int(trace["terminated"].any(0).sum()),
        "all_tibia_air_fraction": float((trace["tibia_air_time"] > 0).all(-1).mean()),
        "fewer_than_three_tibia_contacts_fraction": float((support.sum(-1) < 3).mean()),
        "per_leg": per_leg,
        "joint_component_order": ["coxa", "femur", "tibia"],
        "joint_motion_measured_time_s_per_env": motion["measured_time_s"].tolist(),
        "joint_motion_definition": "Within-episode q differences; speed deadband 0.05 rad/s, reversal hysteresis 0.03 rad. Reset transitions excluded. Positive/negative travel are direction magnitudes; reversals are not gait cycles.",
        "excursion_definition": "Episode p95-p05 weighted by captured time across environments. Pose range alone is not evidence of repeated movement. Tip forward uses the current body frame.",
        "support_definition": "Tibia-link net force norm > 1 N; not distal-foot contact identification.",
        "scope": "All captured steps and failed episodes included; no settling frames dropped.",
    }


class RolloutProbe:
    """Copy state in the last termination callback, before automatic reset."""

    def __init__(self, torch, quat_apply):
        self.torch = torch
        self.quat_apply = quat_apply
        self.frames = []
        self.pending = None
        self.episode_id = None
        self.body_ids = None
        self.sensor_ids = None
        self.termination_counts = {}

    def begin(self, env, obs, actions):
        torch = self.torch
        if tuple(obs.shape) != (env.num_envs, 67) or tuple(actions.shape) != (env.num_envs, 18):
            raise ValueError("Policy must consume 67 observations and produce 18 actions")
        if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
            raise ValueError("Non-finite input or policy output before physics step")
        if self.pending is not None:
            raise RuntimeError("Previous step has not reached the pre-reset probe")
        if self.episode_id is None:
            self.episode_id = np.zeros(env.num_envs, dtype=np.int64)
        self.pending = {
            "obs": obs.detach().cpu().numpy().copy(),
            "actions_raw": actions.detach().cpu().numpy().copy(),
            "actions_clipped": actions.clamp(-1.0, 1.0).detach().cpu().numpy().copy(),
            "pre_root_pose": env.scene["robot"].data.root_pose_w.torch.detach().cpu().numpy().copy(),
            "episode_id": self.episode_id.copy(),
        }

    def capture(self, env):
        torch = self.torch
        if self.pending is None:
            raise RuntimeError("Pre-reset callback without a matching policy step")
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        data = robot.data
        manager = env.termination_manager
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        timed_out = torch.zeros_like(terminated)
        for name in manager.active_terms:
            if name == PROBE_TERM:
                continue
            flags = manager.get_term(name)
            self.termination_counts[name] = self.termination_counts.get(name, 0) + int(flags.sum().item())
            if manager.get_term_cfg(name).time_out:
                timed_out |= flags
            else:
                terminated |= flags
        quat = data.body_quat_w.torch[:, self.body_ids]
        offset = torch.tensor([[-.004, -.147, -.021], *[[.004, .147, -.021]] * 5],
                              dtype=quat.dtype, device=env.device)
        tips = data.body_pos_w.torch[:, self.body_ids] + self.quat_apply(quat, offset.expand(env.num_envs, -1, -1))
        values = {
            "root_pose": data.root_pose_w.torch,
            "root_vel": data.root_vel_w.torch,
            "joint_pos": data.joint_pos.torch,
            "joint_vel": data.joint_vel.torch,
            "tibia_air_time": sensor.data.current_air_time.torch[:, self.sensor_ids],
            "tibia_force_w": sensor.data.net_forces_w.torch[:, self.sensor_ids],
            "foot_tip_pos_w": tips,
            "terminated": terminated,
            "timed_out": timed_out,
        }
        frame = self.pending
        frame.update({key: value.detach().cpu().numpy().copy() for key, value in values.items()})
        self.frames.append(frame)
        self.pending = None
        self.episode_id += (frame["terminated"] | frame["timed_out"]).astype(np.int64)
        # Preserve the failing frame before stopping on corrupt physics.
        for key, value in values.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite pre-reset {key}; captured failing frame")
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def arrays(self):
        if not self.frames:
            return {}
        return {key: np.stack([frame[key] for frame in self.frames]) for key in self.frames[0]}


def parse_args():
    import warp as wp
    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--vx", type=float, default=.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=.205)
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=[])
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.seed < 0:
        parser.error("Positive steps/environments and nonnegative seed required")
    if not (-.1 <= args.vx <= .2 and -.1 <= args.vy <= .1 and -.5 <= args.yaw_rate <= .5
            and .17 <= args.height <= .24):
        parser.error("Use training-range vx [-.1,.2], vy [-.1,.1], yaw-rate [-.5,.5], height [.17,.24]")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    for destination in (args.report, args.trace):
        if destination is not None and destination.exists():
            parser.error(f"Refusing to overwrite existing output: {destination}")
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
    from flat_wsp import FlatActor, sha256

    torch.manual_seed(args.seed)
    args.task = "Hexapod-Walk"
    cfg, _ = resolve_task_config(args.task, "", overrides=[])
    apply_env_overrides(args, cfg)
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = False
    velocity = cfg.commands.base_velocity
    velocity.ranges.lin_vel_x = (args.vx, args.vx)
    velocity.ranges.lin_vel_y = (args.vy, args.vy)
    velocity.ranges.ang_vel_z = (args.yaw_rate, args.yaw_rate)
    velocity.heading_command = False
    velocity.rel_standing_envs = 1.0 if args.vx == args.vy == args.yaw_rate == 0.0 else 0.0
    velocity.debug_vis = False
    cfg.commands.base_height.height_range = (args.height, args.height)
    args.device = cfg.sim.device
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = FlatActor(checkpoint["actor_state_dict"]).to(args.device).eval()
    probe = RolloutProbe(torch, quat_apply)

    def capture_before_reset(env):
        # A closure retains the same probe even if a config container is copied.
        return probe.capture(env)

    if hasattr(cfg.terminations, PROBE_TERM):
        raise ValueError("Reserved probe termination already present")
    setattr(cfg.terminations, PROBE_TERM, TerminationTermCfg(func=capture_before_reset))
    report["checkpoint_sha256"] = sha256(args.checkpoint)
    report["source_sha256"] = report["checkpoint_sha256"]
    try:
        with launch_simulation(cfg, args), contextlib.ExitStack() as cleanup:
            env = gym.make(args.task, cfg=cfg)
            cleanup.callback(close_env, env)
            raw = env.unwrapped
            robot = raw.scene["robot"]
            sensor = raw.scene.sensors["contact_forces"]
            action = raw.action_manager.get_term("joint_pos")
            if list(robot.joint_names) != JOINT_NAMES or list(action._joint_names) != JOINT_NAMES:
                raise ValueError("Runtime joint/action order differs from the checkpoint's verified contract")
            if raw.termination_manager.active_terms[-1] != PROBE_TERM:
                raise ValueError("Capture must be the last termination callback")
            probe.body_ids = [list(robot.body_names).index(name) for name in TIBIA_BODIES]
            probe.sensor_ids = [list(sensor.body_names).index(name) for name in TIBIA_BODIES]
            obs, _ = env.reset(seed=args.seed)
            initial_state = {
                "root_pose": robot.data.root_pose_w.torch.cpu().tolist(),
                "root_vel": robot.data.root_vel_w.torch.cpu().tolist(),
                "joint_pos": robot.data.joint_pos.torch.cpu().tolist(),
                "joint_vel": robot.data.joint_vel.torch.cpu().tolist(),
                "obs": obs["policy"].cpu().tolist(),
            }
            report.update({
                "step_dt_s": raw.step_dt,
                "joint_names": JOINT_NAMES,
                "tibia_bodies": TIBIA_BODIES,
                "initial_root_pose": robot.data.root_pose_w.torch.cpu().tolist(),
                "initial_joint_pos": robot.data.joint_pos.torch.cpu().tolist(),
                "asset_usd_path": str(robot.cfg.spawn.usd_path),
                "asset_usd_sha256": sha256(Path(robot.cfg.spawn.usd_path)),
                "asset_manifest_sha256": sha256(Path(__file__).resolve().parents[1]
                    / "src/RLforHexapod/assets/robot/full/conversion_manifest.json"),
                "quaternion_order": "xyzw",
                "normalization_epsilon": 0.01,
                "initial_state": initial_state,
                "initial_state_sha256": hashlib.sha256(json.dumps(initial_state, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                "environment_source_sha256": sha256(Path(__file__).resolve().parents[1]
                    / "src/RLforHexapod/tasks/walk/config/hexapod/env_cfg.py"),
                "sampling": "Policy input before step; raw result captured before same-step automatic reset.",
                "independence_limit": "Fixed reset states and cloned environments; more environments or seeds do not establish independent trials.",
            })
            for step in range(args.steps):
                with torch.inference_mode():
                    policy_obs = obs["policy"]
                    actions = actor(policy_obs)
                    probe.begin(raw, policy_obs, actions)
                    before = len(probe.frames)
                    obs, _, _, _, _ = env.step(actions.clamp(-1., 1.))
                    if len(probe.frames) != before + 1 or probe.pending is not None:
                        raise RuntimeError("Expected exactly one pre-reset capture per control step")
                report["steps_completed"] = step + 1
            report["status"] = "completed"
    finally:
        trace = probe.arrays()
        report["termination_counts"] = probe.termination_counts
        report["captured_frames"] = len(probe.frames)
        if trace:
            if args.trace is not None:
                args.trace.parent.mkdir(parents=True, exist_ok=True)
                with args.trace.open("xb") as stream:
                    np.savez_compressed(stream, **trace, joint_names=np.array(JOINT_NAMES),
                                        checkpoint_sha256=np.array(report["checkpoint_sha256"]),
                                        source_sha256=np.array(report["checkpoint_sha256"]),
                                        normalization_epsilon=np.array(0.01))
                report["trace"] = str(args.trace.resolve())
                report["trace_sha256"] = sha256(args.trace)
            report["metrics"] = summarize_trace(trace, report["step_dt_s"], args.vx, args.height,
                                                args.vy, args.yaw_rate)


def main():
    args = parse_args()
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "task": "Hexapod-Walk", "checkpoint": str(args.checkpoint.resolve()),
              "seed": args.seed, "num_envs": args.num_envs, "steps": args.steps, "steps_requested": args.steps,
              "steps_completed": 0,
              "command": {"vx": args.vx, "vy": args.vy, "yaw_rate": args.yaw_rate, "height": args.height},
              "comparison_note": "Compare against baseline with identical asset, commands, seed, initial states and budget."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        try:
            run(args, report)
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.truncate()
    print(f"Flat evaluation {report['status']}: {args.report.resolve()}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
