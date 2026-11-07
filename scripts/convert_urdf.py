"""Convert and validate the complete robot without starting Isaac Sim or using a GPU.

Use a separate Python 3.12 environment with requirements-asset-conversion.txt.
The old USD is retained; the generated asset lives in assets/robot/full/robot/.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "RLforHexapod" / "assets"


def source_hashes(urdf):
    """Hash the URDF and its meshes so changing a mesh cannot reuse a stale USD."""
    text = urdf.read_text(encoding="utf-8")
    meshes = {}
    for mesh in ET.fromstring(text).iter("mesh"):
        name = mesh.attrib["filename"]
        path = (urdf.parent / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF mesh is missing: {path}")
        data = path.read_bytes()
        if data[:64].startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"URDF mesh is a Git LFS pointer; download its contents: {path}")
        meshes[name] = hashlib.sha256(data).hexdigest()
    return {
        "urdf_sha256_utf8_lf": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "mesh_sha256": dict(sorted(meshes.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSETS / "urdf" / "robot.urdf")
    parser.add_argument("--output-dir", type=Path, default=ASSETS / "robot" / "full")
    parser.add_argument("--force", action="store_true", help="Regenerate an existing output asset")
    args = parser.parse_args()
    urdf, output = args.urdf.resolve(), args.output_dir.resolve()
    expected = output / urdf.stem / f"{urdf.stem}.usda"
    if expected.parent.exists() and not args.force:
        parser.error(f"Output exists: {expected}; pass --force to regenerate it")
    inputs = source_hashes(urdf)

    # Same standalone importer used by Isaac Lab 3.0 in kitless mode.
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    options = {
        "merge_fixed_joints": True,
        "fix_base": False,
        "collision_from_visuals": False,
        "collision_type": "Convex Hull",
        "allow_self_collision": False,
        "joint_drive_type": "force",
        "joint_target_type": "position",
        "override_joint_stiffness": 100.0,
        "override_joint_damping": 1.0,
        "run_asset_transformer": True,
        "run_multi_physics_conversion": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    report = output / "validation.json"
    manifest_path = output / "conversion_manifest.json"
    versions = {name: importlib.metadata.version(name) for name in (
        "isaacsim-asset-isolated", "urdf-usd-converter", "usd-exchange", "newton-usd-schemas",
    )}
    manifest = {
        **inputs,
        "urdf": urdf.relative_to(ROOT).as_posix() if urdf.is_relative_to(ROOT) else str(urdf),
        "usd": expected.relative_to(output).as_posix(),
        "converter_versions": versions,
        "converter_options": options,
        "physics_variant": "physics",
    }

    def validate(usd, report_path, require_manifest=False):
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "validate_robot_asset.py"),
            "--urdf", str(urdf), "--usd", str(usd), "--report", str(report_path),
            *(["--require-manifest"] if require_manifest else []),
        ], check=True)

    # An empty destination prevents the importer from silently choosing robot_1.
    # Validate before replacing the previous asset, and verify relative payloads
    # again at the final location. All temporary files stay inside output.
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=output) as directory:
        work = Path(directory)
        staged_manifest = work / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        scratch = work / "scratch"
        scratch.mkdir()
        previous_tempdir = tempfile.tempdir
        try:
            tempfile.tempdir = str(scratch)
            generated = URDFImporter(URDFImporterConfig(
                urdf_path=str(urdf), usd_path=str(work / "asset"), **options,
            )).import_urdf()
        finally:
            tempfile.tempdir = previous_tempdir
        if not generated:
            raise RuntimeError("URDF importer did not return an output asset")
        usd = Path(generated).resolve()
        if not usd.is_file() or not usd.is_relative_to(work):
            raise RuntimeError(f"Unexpected converter output: {usd}")

        # Multi-physics export does not select a variant by default. Isaac Lab
        # needs the common USD Physics bodies and joints to be composed.
        from pxr import Usd

        stage = Usd.Stage.Open(str(usd))
        if not stage or not stage.GetDefaultPrim():
            raise RuntimeError(f"Converter output has no default prim: {usd}")
        physics = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics")
        if "physics" not in physics.GetVariantNames():
            raise RuntimeError("Converter output has no common USD Physics variant")
        physics.SetVariantSelection("physics")
        # The transformer embeds its temporary Windows path as documentation.
        # Keep the portable asset independent of the converter workstation.
        for layer in stage.GetUsedLayers():
            if layer.realPath and Path(layer.realPath).is_relative_to(work) and layer.documentation:
                layer.documentation = "Generated from the project's complete robot URDF."
                layer.Save()
        stage.GetRootLayer().Save()
        del layer, physics, stage
        validate(usd, work / "validation.json")
        backups, published = {}, []
        try:
            for index, path in enumerate((expected.parent, manifest_path, report)):
                if path.exists():
                    backup = work / f"previous-{index}"
                    path.rename(backup)
                    backups[path] = backup
            usd.parent.rename(expected.parent)
            published.append(expected.parent)
            staged_manifest.rename(manifest_path)
            published.extend((manifest_path, report))
            validate(expected, report, require_manifest=True)
        except BaseException:
            for index, path in enumerate(published):
                if path.exists():
                    path.rename(work / f"failed-{index}")
            for path, backup in backups.items():
                backup.rename(path)
            raise
    print(f"[OK] Validated USD: {expected}")
    print(f"[OK] Validation report: {report}")


if __name__ == "__main__":
    main()
