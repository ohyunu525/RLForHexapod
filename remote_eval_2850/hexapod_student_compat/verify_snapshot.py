"""Verify the private task snapshot without importing Isaac Lab or launching it."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def verify_snapshot(root: Path | None = None) -> dict:
    root = Path(__file__).resolve().parent if root is None else root.resolve()
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    edited_files = []
    for record in manifest["files"]:
        target = (root / record["local_path"]).resolve()
        if root not in target.parents:
            raise ValueError(f"Snapshot entry escapes package: {record['local_path']}")
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["vendored_sha256"]:
            raise ValueError(f"Vendored file hash mismatch: {record['local_path']}")

        original = data
        for transform in reversed(record["transformations"]):
            before = transform["before"].encode("utf-8")
            after = transform["after"].encode("utf-8")
            if original.count(after) != transform["occurrences"]:
                raise ValueError(f"Import transform count mismatch: {record['local_path']}")
            original = original.replace(after, before)
        if hashlib.sha256(original).hexdigest() != record["upstream_sha256"]:
            raise ValueError(f"Reconstructed upstream hash mismatch: {record['local_path']}")
        git_blob = b"blob " + str(len(original)).encode("ascii") + b"\0" + original
        if hashlib.sha1(git_blob).hexdigest() != record["upstream_git_blob_sha1"]:
            raise ValueError(f"Reconstructed upstream Git blob mismatch: {record['local_path']}")
        if record["transformations"]:
            edited_files.append(record["local_path"])

    for relative_path, expected_hash in manifest.get("authored_sha256", {}).items():
        target = (root / relative_path).resolve()
        if root not in target.parents:
            raise ValueError(f"Authored entry escapes package: {relative_path}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Authored file hash mismatch: {relative_path}")

    python_files = sorted(root.rglob("*.py"))
    for path in python_files:
        source = path.read_bytes()
        compile(source, str(path), "exec")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                if name.startswith("RLforHexapod") and name != "RLforHexapod.assets":
                    raise ValueError(f"Unexpected installed-task import in {path.name}: {name}")

    return {
        "source_commit": manifest["upstream_commit"],
        "task_id": manifest["task_id"],
        "verified_upstream_files": len(manifest["files"]),
        "compiled_python_files": len(python_files),
        "relocated_import_files": edited_files,
        "simulation_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_snapshot(), indent=2))
