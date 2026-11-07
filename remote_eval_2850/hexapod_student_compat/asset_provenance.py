"""Check installed USD assets against the pinned repository's Git LFS hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def inspect_asset_provenance(robot_usda, obstacles_dir) -> dict:
    """Return a JSON-ready asset report; never download, replace, or parse USD assets.

    Callers must reject ``matches_pinned == False`` before treating the private
    task as compatible with the pinned training source. Paths come from the
    installed ``RLforHexapod.assets`` module rather than an assumed checkout.
    """
    manifest_path = Path(__file__).resolve().parent / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    robot_usda = Path(robot_usda).expanduser().resolve()
    obstacles_dir = Path(obstacles_dir).expanduser().resolve()
    roots = {"robot_directory": robot_usda.parent, "obstacles_directory": obstacles_dir}
    checks = []
    for expected in manifest["files"]:
        base = roots[expected["root"]]
        if expected["root"] == "robot_directory" and expected["relative_path"] == manifest["robot_root_filename"]:
            path = robot_usda
        else:
            path = (base / expected["relative_path"]).resolve()
        if base not in path.parents:
            raise ValueError(f"Asset manifest path escapes its root: {expected['relative_path']}")

        record = {
            "root": expected["root"],
            "relative_path": expected["relative_path"],
            "path": str(path),
            "expected_sha256": expected["expected_sha256"],
            "expected_size": expected["expected_size"],
            "upstream_git_blob_sha1": expected["upstream_git_blob_sha1"],
            "actual_sha256": None,
            "actual_size": None,
        }
        try:
            digest = hashlib.sha256()
            size = 0
            prefix = b""
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    if not prefix:
                        prefix = chunk[:100]
                    digest.update(chunk)
                    size += len(chunk)
            record["actual_sha256"] = digest.hexdigest()
            record["actual_size"] = size
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                record["status"] = "unresolved_git_lfs_pointer"
            elif size == expected["expected_size"] and digest.hexdigest() == expected["expected_sha256"]:
                record["status"] = "match"
            else:
                record["status"] = "hash_mismatch"
        except FileNotFoundError:
            record["status"] = "missing"
        except OSError as exc:
            record["status"] = "unreadable"
            record["error"] = str(exc)
        checks.append(record)

    return {
        "source_commit": manifest["upstream_commit"],
        "scope": manifest["scope"],
        "robot_usda": str(robot_usda),
        "obstacles_dir": str(obstacles_dir),
        "matches_pinned": all(check["status"] == "match" for check in checks),
        "files_checked": len(checks),
        "mismatched_files": [check["relative_path"] for check in checks if check["status"] != "match"],
        "files": checks,
    }
