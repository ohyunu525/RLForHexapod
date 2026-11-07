"""Run six isolated original/WSP evaluations with the remote repository Python."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def verify(bundle):
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["sha256"].items():
        path = (bundle / rel).resolve()
        if bundle not in path.parents or sha(path) != expected:
            raise ValueError(f"Bundle integrity mismatch: {rel}")
    return sha(bundle / "bundle_manifest.json")


def aggregate(results):
    seeds = sorted({r["seed"] for r in results})
    pairs = []
    for seed in seeds:
        roles = {r["role"]: r for r in results if r["seed"] == seed}
        if set(roles) != {"original", "wsp"}:
            raise ValueError(f"Incomplete seed pair: {seed}")
        a, b = roles["original"], roles["wsp"]
        if a["episodes"]["initial_state"] != b["episodes"]["initial_state"]:
            raise ValueError(f"Initial physical states differ for seed {seed}")
        for key in ("num_envs", "task", "source_commit", "config_changes", "height_map_noise_enabled", "final_advance", "done_move"):
            if a[key] != b[key]:
                raise ValueError(f"Pair contract mismatch: {key}")
        sa, sb = a["summary"], b["summary"]
        pairs.append({"seed": seed, "initial_states_exact": True, "original": sa, "wsp": sb,
                      "wsp_minus_original_top_rate": sb["reached_top_rate"] - sa["reached_top_rate"],
                      "wsp_minus_original_fall_rate": sb["fall_rate"] - sa["fall_rate"],
                      "wsp_minus_original_capped_seconds": sb["all_episode_capped_top_time_mean_s"] - sa["all_episode_capped_top_time_mean_s"]})
    totals = {}
    for role in ("original", "wsp"):
        rows = [r["summary"] for r in results if r["role"] == role]
        n = sum(r["episodes"] for r in rows)
        totals[role] = {"episodes": n, "top_rate": sum(r["reached_top_count"] for r in rows) / n,
                        "fall_rate": sum(r["fall_count"] for r in rows) / n,
                        "settled_success_rate": sum(r["settled_success_count"] for r in rows) / n,
                        "capped_top_time_mean_s": sum(sum(r["capped_top_time_per_episode_s"]) for r in rows) / n}
    return {"pairs": pairs, "totals": totals,
            "limits": ["Same physical initial states and seeds; subsequent stochastic observations may diverge after different resets.",
                       "Pinned Student-A benchmark; checkpoint does not prove original training variant or asset.",
                       "No model is automatically promoted or installed."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="cpu")
    p.add_argument("--viz", default="newton_gl")
    p.add_argument("--python", type=Path, help="Override remote .venv Python")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    bundle = Path(__file__).resolve().parent
    manifest_sha = verify(bundle)
    if args.verify_only:
        print(json.dumps({"bundle_verified": True, "manifest_sha256": manifest_sha}))
        return
    if args.num_envs < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive num-envs and unique seeds required")
    repository = bundle.parent
    python = args.python or repository / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"Use remote RLForHexapod .venv: {python}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = bundle / f"ab2850_{stamp}_{uuid.uuid4().hex[:8]}"
    run.mkdir()
    state = {"status": "running", "settings": {"num_envs": args.num_envs, "seeds": args.seeds, "device": args.device, "viz": args.viz},
             "bundle_manifest_sha256": manifest_sha, "runs": []}
    write(run / "run.json", state)
    results = []
    try:
        for seed in args.seeds:
            for role in ("original", "wsp"):
                result = run / f"{role}_seed{seed}.json"
                log = run / f"{role}_seed{seed}.log"
                cmd = [str(python), "-u", str(bundle / "eval_worker.py"), "--role", role, "--result", str(result),
                       "--num_envs", str(args.num_envs), "--seed", str(seed), "--device", args.device, "--viz", args.viz]
                entry = {"role": role, "seed": seed, "command": cmd, "log": str(log), "result": str(result), "status": "running"}
                state["runs"].append(entry)
                write(run / "run.json", state)
                print(f"Starting {role}, seed {seed}; log: {log}", flush=True)
                with log.open("x", encoding="utf-8") as stream:
                    process = subprocess.run(cmd, cwd=repository, stdout=stream, stderr=subprocess.STDOUT)
                entry["returncode"] = process.returncode
                if process.returncode or not result.is_file():
                    entry["status"] = "failed"
                    raise RuntimeError(f"Evaluation failed; inspect {log}")
                if verify(bundle) != manifest_sha:
                    raise ValueError("Bundle changed while running")
                value = json.loads(result.read_text(encoding="utf-8"))
                if value["status"] != "complete" or value["role"] != role or value["seed"] != seed:
                    raise ValueError("Result identity mismatch")
                results.append(value)
                entry.update(status="complete", result_sha256=sha(result))
                write(run / "run.json", state)
        summary = aggregate(results)
        write(run / "summary.json", summary)
        state.update(status="complete", summary_sha256=sha(run / "summary.json"))
        print(json.dumps(summary["totals"], indent=2), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(run / "run.json", state)
        print(f"Results: {run}", flush=True)


if __name__ == "__main__":
    main()
