"""Capture flat walking, fit a WSP candidate, and compare it in the same simulator.

    uv run --locked python scripts/run_flat_wsp.py --dry-run
    uv run --locked python scripts/run_flat_wsp.py --checkpoint path/to/model.pt

No PPO updates or default checkpoint replacement. Each invocation creates a new run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / (
    "logs/rsl_rl/hexapod_walk/2026-11-08_10-24-48_full_robot_resume_20261108_102439/model_13250.pt"
)
# Fixed, distinct control conditions. Cloned environments are not independent trials.
CASES = (
    ("forward10", 0.10, 0.0, 0.0, 0.205),
    ("forward15", 0.15, 0.0, 0.0, 0.205),
    ("forward20", 0.20, 0.0, 0.0, 0.205),
    ("stand", 0.0, 0.0, 0.0, 0.205),
    ("turn_left", 0.0, 0.0, 0.25, 0.205),
    ("turn_right", 0.0, 0.0, -0.25, 0.205),
    ("forward_low", 0.15, 0.0, 0.0, 0.17),
    ("forward_high", 0.15, 0.0, 0.0, 0.24),
)


def evaluation_command(args, checkpoint, folder, case):
    name, vx, vy, yaw, height = case
    return [
        sys.executable, str(ROOT / "scripts/eval_flat_wsp.py"),
        "--checkpoint", str(checkpoint), "--report", str(folder / f"{name}.json"),
        "--trace", str(folder / f"{name}.npz"),
        "--num_envs", str(args.num_envs), "--steps", str(args.steps), "--seed", str(args.seed),
        "--vx", str(vx), "--vy", str(vy), "--yaw-rate", str(yaw), "--height", str(height),
        "--device", args.device, "--viz", "none",
    ]


def invoke(command, log_path):
    print(subprocess.list2cmdline(command), flush=True)
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def merge_traces(paths, output, expected_hash):
    """Combine observations only; never join separate episodes into fake trajectories."""
    import numpy as np

    observations = []
    joints = None
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            if str(trace["source_sha256"].item()) != expected_hash:
                raise ValueError(f"Trace source mismatch: {path}")
            names = trace["joint_names"].tolist()
            if joints is not None and joints != names:
                raise ValueError(f"Joint order mismatch: {path}")
            joints = names
            if abs(float(trace["normalization_epsilon"].item()) - 0.01) > 1e-9:
                raise ValueError(f"Normalizer contract mismatch: {path}")
            obs = trace["obs"]
            if obs.ndim != 3 or obs.shape[-1] != 67 or not np.isfinite(obs).all():
                raise ValueError(f"Invalid observations: {path}")
            observations.append(obs.reshape(-1, 67))
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, obs=np.concatenate(observations), source_sha256=expected_hash,
            joint_names=np.asarray(joints), normalization_epsilon=0.01,
            sampling="Actual pre-action observations; unordered calibration rows from separate cases",
            source_traces=np.asarray([str(path) for path in paths]),
        )


def compare_pair(before, after):
    """Reject incomparable runs and expose regressions without adopting a policy."""
    if before.get("status") != "completed" or after.get("status") != "completed":
        raise ValueError("Both rollouts must complete their full step budget")
    for key in ("task", "seed", "num_envs", "steps_requested", "steps_completed", "command", "step_dt_s",
                "joint_names", "asset_usd_sha256", "asset_manifest_sha256", "environment_source_sha256",
                "initial_state_sha256", "normalization_epsilon"):
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError(f"Paired evaluation contract mismatch: {key}")
    if before["steps_completed"] != before["steps_requested"]:
        raise ValueError("Incomplete step budget")
    first, second = before["metrics"], after["metrics"]
    scalar_keys = (
        "mean_forward_speed_body_m_s", "xy_tracking_rmse_m_s", "yaw_tracking_rmse_rad_s",
        "vertical_velocity_rms_m_s", "tilt_rms_rad", "height_tracking_rmse_m", "failed_episodes",
        "all_tibia_air_fraction", "fewer_than_three_tibia_contacts_fraction",
    )
    changes = {key: second[key] - first[key] for key in scalar_keys}
    legs = {}
    for leg in (f"L{index}" for index in range(1, 7)):
        a, b = first["per_leg"][leg], second["per_leg"][leg]
        legs[leg] = {
            "tibia_excursion_before_rad": a["joint_excursion_p95_p05_rad"][2],
            "tibia_excursion_after_rad": b["joint_excursion_p95_p05_rad"][2],
            "tibia_mean_change_rad": b["joint_mean_rad"][2] - a["joint_mean_rad"][2],
            "tibia_clip_fraction_change": b["action_clip_fraction"][2] - a["action_clip_fraction"][2],
            "tibia_contact_fraction_change": b["tibia_contact_fraction"] - a["tibia_contact_fraction"],
        }
    flags = []
    if changes["failed_episodes"] > 0:
        flags.append("more_failed_episodes")
    if before["command"]["vx"] > 0 and second["mean_forward_speed_body_m_s"] < max(
        0.0, first["mean_forward_speed_body_m_s"] * .95
    ):
        flags.append("forward_speed_lost_more_than_five_percent")
    if changes["xy_tracking_rmse_m_s"] > .01:
        flags.append("xy_tracking_rmse_increased_over_001_m_s")
    if changes["height_tracking_rmse_m"] > .01:
        flags.append("height_rmse_increased_over_001_m")
    if changes["all_tibia_air_fraction"] > .01:
        flags.append("all_air_fraction_increased_over_001")
    return {"delta_candidate_minus_baseline": changes, "per_leg": legs, "regression_flags": flags,
            "interpretation": "Screening flags only; absence of flags does not establish normal gait or improvement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/flat_wsp")
    parser.add_argument("--candidate", type=Path, help="Compare an existing candidate; do not fit another one")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--joint-scope", choices=("tibia", "all"), default="tibia")
    parser.add_argument("--protected-cap", type=float, default=.1)
    parser.add_argument("--max-applied-change", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1 or args.seed < 0:
        parser.error("steps/num_envs must be positive and seed nonnegative")
    if not 0 < args.strength <= 0.4:
        parser.error("strength must be in (0, 0.4]")
    if not 0 <= args.protected_cap <= 1 or (args.max_applied_change is not None
                                          and not 0 < args.max_applied_change <= 2):
        parser.error("protected-cap must be in [0,1] and max-applied-change in (0,2]")
    source = args.checkpoint.resolve()
    folder = args.output_root.resolve() / (
        datetime.now(timezone.utc).strftime("compare_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    )
    candidate = args.candidate.resolve() if args.candidate else folder / "model_flat_wsp.pt"
    baseline_dir, candidate_dir = folder / "baseline", folder / "candidate"
    fit_command = [
        sys.executable, str(ROOT / "scripts/flat_wsp.py"), "--checkpoint", str(source),
        "--output", str(candidate), "--trace", str(folder / "calibration.npz"),
        "--strength", str(args.strength),
        "--joint-scope", args.joint_scope, "--protected-cap", str(args.protected_cap),
    ]
    if args.max_applied_change is not None:
        fit_command += ["--max-applied-change", str(args.max_applied_change)]
    if args.dry_run:
        print(f"Run directory: {folder}")
        for case in CASES:
            print(subprocess.list2cmdline(evaluation_command(args, source, baseline_dir, case)))
        if not args.candidate:
            print("Merge captured baseline observations and verify hashes, then:")
            print(subprocess.list2cmdline(fit_command))
        for case in CASES:
            print(subprocess.list2cmdline(evaluation_command(args, candidate, candidate_dir, case)))
        return 0

    from flat_wsp import SOURCE_SHA256, sha256

    if not source.is_file():
        parser.error(f"Missing baseline checkpoint: {source}")
    if not args.candidate and sha256(source) != SOURCE_SHA256:
        parser.error("WSP fitting is pinned to reviewed model_13250.pt; source hash differs")
    if args.candidate and (not candidate.is_file() or candidate == source):
        parser.error("--candidate must be a different existing checkpoint")
    baseline_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir.mkdir()
    manifest = {
        "status": "running", "source": str(source), "source_sha256": sha256(source),
        "candidate": str(candidate), "seed": args.seed, "steps": args.steps,
        "num_envs": args.num_envs, "cases": CASES, "adopted": False,
        "scope": "Fixed command comparison, including failures; no statistical population claim or automatic promotion.",
    }
    try:
        for case in CASES:
            invoke(evaluation_command(args, source, baseline_dir, case), baseline_dir / f"{case[0]}.log")
        if not args.candidate:
            merge_traces([baseline_dir / f"{case[0]}.npz" for case in CASES],
                         folder / "calibration.npz", manifest["source_sha256"])
            invoke(fit_command, folder / "fit.log")
        for case in CASES:
            invoke(evaluation_command(args, candidate, candidate_dir, case), candidate_dir / f"{case[0]}.log")
        pairs = []
        for case in CASES:
            before = json.loads((baseline_dir / f"{case[0]}.json").read_text(encoding="utf-8"))
            after = json.loads((candidate_dir / f"{case[0]}.json").read_text(encoding="utf-8"))
            if before["source_sha256"] != manifest["source_sha256"] or after["source_sha256"] != sha256(candidate):
                raise ValueError("Evaluated checkpoint hash changed")
            pairs.append({"case": case[0], "comparison": compare_pair(before, after),
                          "baseline_report": str(baseline_dir / f"{case[0]}.json"),
                          "candidate_report": str(candidate_dir / f"{case[0]}.json")})
        manifest.update(status="comparison_recorded_requires_review", candidate_sha256=sha256(candidate), pairs=pairs)
        if sha256(source) != manifest["source_sha256"]:
            raise RuntimeError("Baseline changed during comparison")
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (folder / "comparison.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                                encoding="utf-8")
        print(f"Comparison report: {folder / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
