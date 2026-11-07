"""Validate the project's URDF and composed USD before training; no simulator required.

URDF checks use the standard library. USD checks require the ``pxr`` package;
an unavailable USD runtime is a failed validation, never a successful partial check.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
import xml.etree.ElementTree as ET


ASSETS = Path(__file__).resolve().parents[1] / "src" / "RLforHexapod" / "assets"
EXPECTED_LINKS = 19
EXPECTED_JOINTS = 18
EXPECTED_TOTAL_MASS = 2.28614
EXPECTED_BODY_MASS = 1.28096
MASS_TOLERANCE = 1e-5
COM_TOLERANCE = 1e-6
ANGLE_TOLERANCE_DEG = 1e-3
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def check(checks, name, passed, **details):
    checks.append({"name": name, "passed": bool(passed), **details})


def file_problem(path):
    """Return a concrete missing/pointer/read error, or None for a materialized file."""
    if not path.is_file():
        return "missing file"
    try:
        with path.open("rb") as stream:
            prefix = stream.read(128)
    except OSError as exc:
        return str(exc)
    if prefix.startswith(LFS_HEADER):
        return "Git LFS pointer; fetch the actual asset"
    if not prefix:
        return "empty file"
    return None


def mesh_path(urdf, filename):
    if filename.startswith("file://"):
        parsed = urlparse(filename)
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("remote file URI is unsupported")
        return Path(url2pathname(unquote(parsed.path))).resolve()
    if "://" in filename:
        raise ValueError("unresolved mesh URI; use a local relative or absolute path")
    return (urdf.parent / filename).resolve()


def inspect_urdf(path, checks):
    problem = file_problem(path)
    check(checks, "urdf_file", problem is None, path=str(path), error=problem)
    if problem:
        return None
    try:
        root = ET.parse(path).getroot()
        if root.tag != "robot":
            raise ValueError("root element is not <robot>")
    except (ET.ParseError, OSError, ValueError) as exc:
        check(checks, "urdf_xml", False, error=str(exc))
        return None

    mesh_files, mesh_errors = set(), []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        try:
            if not filename:
                raise ValueError("mesh has no filename")
            resolved = mesh_path(path, filename)
            mesh_files.add(str(resolved))
            problem = file_problem(resolved)
            if problem:
                mesh_errors.append({"mesh": filename, "error": problem})
        except (OSError, ValueError) as exc:
            mesh_errors.append({"mesh": filename, "error": str(exc)})
    check(checks, "urdf_meshes", bool(mesh_files) and not mesh_errors,
          files=sorted(mesh_files), errors=mesh_errors)

    links, invalid_links = {}, []
    link_names = [link.get("name", "") for link in root.findall("link")]
    for link in root.findall("link"):
        name = link.get("name", "")
        try:
            inertial = link.find("inertial")
            mass = float(inertial.find("mass").attrib["value"])
            origin = inertial.find("origin")
            com = [float(value) for value in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError("mass must be finite and positive")
            if len(com) != 3 or not all(math.isfinite(value) for value in com):
                raise ValueError("center of mass must contain three finite coordinates")
            links[name] = {"mass": mass, "com": com}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            invalid_links.append({"link": name, "error": str(exc)})
    check(checks, "urdf_links", len(link_names) == EXPECTED_LINKS and len(set(link_names)) == len(link_names)
          and all(link_names) and not invalid_links, count=len(link_names), errors=invalid_links)
    total_mass = sum(link["mass"] for link in links.values())
    check(checks, "urdf_total_mass", math.isclose(total_mass, EXPECTED_TOTAL_MASS, abs_tol=MASS_TOLERANCE),
          actual=total_mass, expected=EXPECTED_TOTAL_MASS)
    body_mass = links.get("body", {}).get("mass")
    check(checks, "urdf_body_mass", body_mass is not None and math.isclose(
        body_mass, EXPECTED_BODY_MASS, abs_tol=MASS_TOLERANCE), actual=body_mass, expected=EXPECTED_BODY_MASS)

    joints, invalid_joints, children = {}, [], []
    all_joints = root.findall("joint")
    joint_names = [joint.get("name", "") for joint in all_joints]
    for joint in all_joints:
        name = joint.get("name", "")
        try:
            if joint.get("type") != "revolute":
                raise ValueError("the training source must contain only revolute joints")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            children.append(child)
            if parent not in links or child not in links or parent == child:
                raise ValueError("joint references invalid parent/child links")
            limit = joint.find("limit")
            bounds = [float(limit.attrib[key]) for key in ("lower", "upper")]
            if not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
                raise ValueError("joint limits must be finite and ordered")
            joints[name] = {"limits_deg": [math.degrees(value) for value in bounds], "parent": parent, "child": child}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            invalid_joints.append({"joint": name, "error": str(exc)})
    check(checks, "urdf_revolute_joints", len(joint_names) == EXPECTED_JOINTS
          and len(set(joint_names)) == len(joint_names) and all(joint_names) and not invalid_joints,
          count=len(joint_names), errors=invalid_joints)
    roots = sorted(set(links) - set(children))
    connected = {"body"}
    for _ in range(len(links)):
        connected.update(joint["child"] for joint in joints.values() if joint["parent"] in connected)
    check(checks, "urdf_tree", roots == ["body"] and len(children) == len(set(children))
          and connected == set(links), roots=roots, reachable_links=len(connected))
    return {"links": links, "joints": joints, "total_mass": total_mass, "root_link": "body"}


def inspect_usd(path, checks):
    problem = file_problem(path)
    check(checks, "usd_file", problem is None, path=str(path), error=problem)
    if problem:
        return None
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    except (ImportError, OSError) as exc:
        check(checks, "usd_runtime", False, error=f"pxr is required for USD validation: {exc}")
        return None
    check(checks, "usd_runtime", True, version=list(Usd.GetVersion()))
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
        dependency_files = sorted({layer.realPath for layer in layers} | set(assets))
        dependency_errors = []
        for filename in dependency_files:
            problem = file_problem(Path(filename))
            if problem:
                dependency_errors.append({"path": filename, "error": problem})
        check(checks, "usd_dependencies", bool(layers) and not unresolved and not dependency_errors,
              files=dependency_files, unresolved=list(unresolved), errors=dependency_errors)
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
        if not stage:
            raise ValueError("USD stage could not be opened")
    except Exception as exc:
        check(checks, "usd_open", False, error=str(exc))
        return None

    check(checks, "usd_composition", not stage.GetCompositionErrors(),
          errors=[str(error) for error in stage.GetCompositionErrors()])
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    kilograms = UsdPhysics.GetStageKilogramsPerUnit(stage)
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    check(checks, "usd_units", meters == 1.0 and kilograms == 1.0 and up_axis == "Z",
          meters_per_unit=meters, kilograms_per_unit=kilograms, up_axis=up_axis)
    check(checks, "usd_default_prim", bool(stage.GetDefaultPrim()), path=str(stage.GetDefaultPrim().GetPath()))

    # Imported meshes can be instance proxies; ordinary traversal omits their collision schemas.
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    counts = Counter(prim.GetName() for prim in bodies)
    links, invalid_links = {}, []
    for prim in bodies:
        name = prim.GetName()
        try:
            api = UsdPhysics.MassAPI(prim)
            mass = api.GetMassAttr().Get()
            com = api.GetCenterOfMassAttr().Get()
            if not prim.HasAPI(UsdPhysics.MassAPI) or mass is None or com is None:
                raise ValueError("explicit MassAPI mass and center of mass are required")
            mass, com = float(mass), [float(value) for value in com]
            if not math.isfinite(mass) or mass <= 0 or not all(math.isfinite(value) for value in com):
                raise ValueError("invalid mass or center of mass")
            rigid = UsdPhysics.RigidBodyAPI(prim)
            if not rigid.GetRigidBodyEnabledAttr().Get() or rigid.GetKinematicEnabledAttr().Get():
                raise ValueError("link must be a dynamic enabled rigid body")
            links[name] = {"mass": mass, "com": com, "path": str(prim.GetPath())}
        except (TypeError, ValueError) as exc:
            invalid_links.append({"link": name, "error": str(exc)})
    check(checks, "usd_links", len(bodies) == EXPECTED_LINKS and all(n == 1 for n in counts.values())
          and not invalid_links, count=len(bodies), errors=invalid_links,
          duplicate_names=[name for name, count in counts.items() if count > 1])

    collisions_by_link = {str(prim.GetPath()): [] for prim in bodies}
    invalid_colliders, active_colliders = [], 0
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        active_colliders += 1
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        try:
            if not owner or str(owner.GetPath()) not in collisions_by_link:
                raise ValueError("enabled collider has no owning robot rigid body")
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                faces = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                if points is None or faces is None or indices is None or len(points) < 3 or not len(faces):
                    raise ValueError("collision mesh has no usable points/faces")
                if not all(math.isfinite(value) for point in points for value in point):
                    raise ValueError("collision mesh contains non-finite points")
                if min(faces) < 3 or sum(faces) != len(indices) or min(indices) < 0 or max(indices) >= len(points):
                    raise ValueError("collision mesh has invalid face topology")
            else:
                dimensions = {
                    "Cube": ("size",), "Sphere": ("radius",), "Capsule": ("radius", "height"),
                    "Cylinder": ("radius", "height"), "Cone": ("radius", "height"),
                }.get(prim.GetTypeName())
                if dimensions is None:
                    raise ValueError(f"unsupported collision geometry: {prim.GetTypeName()}")
                values = [prim.GetAttribute(name).Get() for name in dimensions]
                if not all(value is not None and math.isfinite(value) and value > 0 for value in values):
                    raise ValueError("collision primitive dimensions must be finite and positive")
            collisions_by_link[str(owner.GetPath())].append(str(prim.GetPath()))
        except (TypeError, ValueError) as exc:
            invalid_colliders.append({"path": str(prim.GetPath()), "error": str(exc)})
    missing_colliders = [path for path, colliders in collisions_by_link.items() if not colliders]
    check(checks, "usd_link_collisions", active_colliders > 0 and not invalid_colliders and not missing_colliders,
          active_count=active_colliders, colliders_by_link=collisions_by_link,
          missing_links=missing_colliders, errors=invalid_colliders)

    joints, invalid_joints, world_anchors = {}, [], []
    all_joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    joint_counts = Counter(prim.GetName() for prim in all_joints)
    body_paths = {str(prim.GetPath()): prim.GetName() for prim in bodies}
    for prim in all_joints:
        joint = UsdPhysics.Joint(prim)
        targets = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        # A missing body target means the joint attaches to the world; an external target is also unsafe here.
        internal = all(len(target) == 1 and str(target[0]) in body_paths for target in targets)
        if joint.GetJointEnabledAttr().Get() and not internal:
            world_anchors.append(str(prim.GetPath()))
        try:
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError("unexpected non-revolute joint")
            if not joint.GetJointEnabledAttr().Get() or joint.GetExcludeFromArticulationAttr().Get():
                raise ValueError("joint must be enabled and included in articulation")
            if not internal:
                raise ValueError("joint must connect two robot rigid bodies")
            revolute = UsdPhysics.RevoluteJoint(prim)
            limits = [float(revolute.GetLowerLimitAttr().Get()), float(revolute.GetUpperLimitAttr().Get())]
            if not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]:
                raise ValueError("joint limits must be finite and ordered")
            joints[prim.GetName()] = {
                "limits_deg": limits,
                "parent": body_paths[str(targets[0][0])],
                "child": body_paths[str(targets[1][0])],
            }
        except (TypeError, ValueError) as exc:
            invalid_joints.append({"joint": prim.GetName(), "error": str(exc)})
    check(checks, "usd_revolute_joints", len(all_joints) == EXPECTED_JOINTS
          and all(n == 1 for n in joint_counts.values()) and not invalid_joints,
          count=len(all_joints), errors=invalid_joints)

    articulation_paths = [prim.GetPath() for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    root_body = links.get("body")
    root_is_articulated = root_body is not None and any(
        Sdf.Path(root_body["path"]).HasPrefix(root) for root in articulation_paths
    )
    check(checks, "usd_free_articulated_base", root_is_articulated and len(articulation_paths) == 1
          and not world_anchors, articulation_roots=[str(root) for root in articulation_paths],
          world_anchors=world_anchors)
    return {"links": links, "joints": joints, "total_mass": sum(link["mass"] for link in links.values())}


def compare_models(source, converted, checks):
    """Compare data extracted from the two formats; independent of the USD runtime."""
    check(checks, "link_names_match", source["links"].keys() == converted["links"].keys(),
          missing=sorted(source["links"].keys() - converted["links"].keys()),
          extra=sorted(converted["links"].keys() - source["links"].keys()))
    for name in sorted(source["links"].keys() & converted["links"].keys()):
        expected, actual = source["links"][name], converted["links"][name]
        check(checks, f"link_mass:{name}", math.isclose(expected["mass"], actual["mass"], abs_tol=MASS_TOLERANCE),
              expected=expected["mass"], actual=actual["mass"])
        check(checks, f"link_com:{name}", all(math.isclose(a, b, abs_tol=COM_TOLERANCE)
              for a, b in zip(expected["com"], actual["com"], strict=True)),
              expected=expected["com"], actual=actual["com"])
    check(checks, "usd_total_mass", math.isclose(converted["total_mass"], EXPECTED_TOTAL_MASS,
          abs_tol=MASS_TOLERANCE), actual=converted["total_mass"], expected=EXPECTED_TOTAL_MASS)
    check(checks, "joint_names_match", source["joints"].keys() == converted["joints"].keys(),
          missing=sorted(source["joints"].keys() - converted["joints"].keys()),
          extra=sorted(converted["joints"].keys() - source["joints"].keys()))
    for name in sorted(source["joints"].keys() & converted["joints"].keys()):
        expected, actual = source["joints"][name], converted["joints"][name]
        check(checks, f"joint_limits:{name}", all(math.isclose(a, b, abs_tol=ANGLE_TOLERANCE_DEG)
              for a, b in zip(expected["limits_deg"], actual["limits_deg"], strict=True)),
              expected_deg=expected["limits_deg"], actual_deg=actual["limits_deg"])
        check(checks, f"joint_links:{name}", all(expected[key] == actual[key] for key in ("parent", "child")),
              expected=[expected["parent"], expected["child"]], actual=[actual["parent"], actual["child"]])


def inspect_manifest(urdf, usd, checks):
    """Require conversion provenance to match current source bytes (URDF line endings normalized)."""
    candidates = [usd.parent / "conversion_manifest.json", usd.parent.parent / "conversion_manifest.json"]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must contain a JSON object")
        usd_entry = manifest.get("usd")
        if not isinstance(usd_entry, str) or (path.parent / usd_entry.replace("\\", "/")).resolve() != usd:
            raise ValueError("manifest USD path does not identify the requested asset")
        text = urdf.read_text(encoding="utf-8")  # Universal newlines match convert_urdf.source_hashes.
        urdf_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mesh_hashes = {}
        for mesh in ET.fromstring(text).iter("mesh"):
            filename = mesh.attrib["filename"]
            mesh_hashes[filename] = hashlib.sha256(mesh_path(urdf, filename).read_bytes()).hexdigest()
        expected_meshes = manifest.get("mesh_sha256")
        if not isinstance(expected_meshes, dict):
            raise ValueError("manifest mesh_sha256 must be an object")
        mismatched = sorted(name for name in set(mesh_hashes) | set(expected_meshes)
                            if mesh_hashes.get(name) != expected_meshes.get(name))
        check(checks, "conversion_manifest", urdf_hash == manifest.get("urdf_sha256_utf8_lf") and not mismatched,
              path=str(path), urdf_hash_matches=urdf_hash == manifest.get("urdf_sha256_utf8_lf"),
              mesh_count=len(mesh_hashes), mismatched_meshes=mismatched)
    except (OSError, ValueError, KeyError, ET.ParseError) as exc:
        check(checks, "conversion_manifest", False, path=str(path), error=str(exc))


def validate_asset(urdf, usd, require_manifest=False):
    urdf, usd = Path(urdf).resolve(), Path(usd).resolve()
    checks = []
    source = inspect_urdf(urdf, checks)
    converted = inspect_usd(usd, checks)
    if source is not None and converted is not None:
        compare_models(source, converted, checks)
    if require_manifest:
        inspect_manifest(urdf, usd, checks)
    return {
        "schema_version": 1,
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "urdf": str(urdf), "usd": str(usd), "manifest_required": require_manifest,
        "tolerances": {"mass_kg": MASS_TOLERANCE, "com_m": COM_TOLERANCE, "joint_angle_deg": ANGLE_TOLERANCE_DEG},
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSETS / "urdf" / "robot.urdf")
    parser.add_argument("--usd", type=Path, default=ASSETS / "robot" / "full" / "robot" / "robot.usda")
    parser.add_argument("--report", type=Path, help="Write the full JSON report to this path")
    parser.add_argument("--require-manifest", action="store_true", help="Require current URDF/mesh hashes in conversion_manifest.json")
    args = parser.parse_args(argv)
    report = validate_asset(args.urdf, args.usd, require_manifest=args.require_manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
