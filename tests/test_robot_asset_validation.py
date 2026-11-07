"""Asset contracts without a simulator; optional USD tests use real composed stages."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


SPEC = importlib.util.spec_from_file_location(
    "validate_robot_asset", Path(__file__).resolve().parents[1] / "scripts" / "validate_robot_asset.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except (ImportError, OSError):
    Usd = None


def make_urdf(directory):
    root = ET.Element("robot", name="test_robot")
    extra_mass = (validator.EXPECTED_TOTAL_MASS - validator.EXPECTED_BODY_MASS) / 18
    names = ["body"] + [f"leg_{index}" for index in range(18)]
    for name in names:
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value=str(validator.EXPECTED_BODY_MASS if name == "body" else extra_mass))
        ET.SubElement(inertial, "origin", xyz="-0.01 0 0.02", rpy="0 0 0")
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", filename="meshes/part.stl")
    for index, name in enumerate(names[1:]):
        joint = ET.SubElement(root, "joint", name=f"joint_{index}", type="revolute")
        ET.SubElement(joint, "parent", link="body")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "limit", lower="-1", upper="1", effort="2", velocity="3")
    path = directory / "robot.urdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    mesh = directory / "meshes" / "part.stl"
    mesh.parent.mkdir()
    mesh.write_text("solid part\nendsolid part\n", encoding="ascii")
    return path, mesh


def make_usd(path, model):
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    for name, link in model["links"].items():
        prim = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(link["mass"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*link["com"]))
        collider = UsdGeom.Cube.Define(stage, f"/Robot/{name}/collision")
        collider.CreateSizeAttr(0.02)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        if name == "body":
            UsdPhysics.ArticulationRootAPI.Apply(prim)
    for name, joint in model["joints"].items():
        prim = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}")
        prim.CreateBody0Rel().SetTargets([f"/Robot/{joint['parent']}"])
        prim.CreateBody1Rel().SetTargets([f"/Robot/{joint['child']}"])
        prim.CreateLowerLimitAttr(joint["limits_deg"][0])
        prim.CreateUpperLimitAttr(joint["limits_deg"][1])
    stage.GetRootLayer().Save()
    return stage


class AssetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.urdf, self.mesh = make_urdf(self.directory)
        self.checks = []
        self.source = validator.inspect_urdf(self.urdf, self.checks)

    def test_source_contract_accepts_negative_com_and_converts_radians(self):
        self.assertTrue(all(check["passed"] for check in self.checks), self.checks)
        self.assertEqual(len(self.source["links"]), 19)
        self.assertEqual(self.source["links"]["body"]["com"], [-0.01, 0.0, 0.02])
        self.assertAlmostEqual(self.source["joints"]["joint_0"]["limits_deg"][1], 180 / math.pi)

    def test_missing_mesh_and_lfs_pointer_fail(self):
        for content in (None, validator.LFS_HEADER + b"\noid sha256:test\nsize 100\n"):
            with self.subTest(content=content):
                if content is None:
                    self.mesh.unlink()
                else:
                    self.mesh.write_bytes(content)
                checks = []
                validator.inspect_urdf(self.urdf, checks)
                result = next(check for check in checks if check["name"] == "urdf_meshes")
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"])

    def test_stale_usd_mass_com_and_degree_limits_are_detected(self):
        converted = copy.deepcopy(self.source)
        converted["links"]["body"]["mass"] = 0.79963
        converted["links"]["body"]["com"][2] = -0.0186076
        converted["total_mass"] = sum(link["mass"] for link in converted["links"].values())
        converted["joints"]["joint_0"]["limits_deg"] = [-1.0, 1.0]  # Incorrectly retained radians.
        checks = []
        validator.compare_models(self.source, converted, checks)
        failed = {check["name"] for check in checks if not check["passed"]}
        self.assertEqual(failed, {"link_mass:body", "link_com:body", "usd_total_mass", "joint_limits:joint_0"})

    def test_nonfinite_source_mass_is_rejected(self):
        document = ET.parse(self.urdf)
        document.find("link/inertial/mass").set("value", "nan")
        document.write(self.urdf)
        checks = []
        validator.inspect_urdf(self.urdf, checks)
        self.assertFalse(next(check for check in checks if check["name"] == "urdf_links")["passed"])

    def test_manifest_requires_current_mesh_bytes_but_normalizes_urdf_newlines(self):
        usd_path = self.directory / "robot" / "robot.usda"
        original = self.urdf.read_text(encoding="utf-8")
        manifest = {
            "usd": "robot\\robot.usda",
            "urdf_sha256_utf8_lf": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "mesh_sha256": {"meshes/part.stl": hashlib.sha256(self.mesh.read_bytes()).hexdigest()},
        }
        (self.directory / "conversion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # LF/CRLF differences from Git checkout must not invalidate equivalent URDF text.
        self.urdf.write_bytes(original.replace("\n", "\r\n").encode("utf-8"))
        checks = []
        validator.inspect_manifest(self.urdf, usd_path, checks)
        self.assertTrue(checks[0]["passed"], checks)
        self.mesh.write_text("solid updated_geometry\nendsolid updated_geometry\n", encoding="ascii")
        checks = []
        validator.inspect_manifest(self.urdf, usd_path, checks)
        self.assertFalse(checks[0]["passed"])
        self.assertEqual(checks[0]["mismatched_meshes"], ["meshes/part.stl"])
        # A source-only change must also be detected even if the XML remains valid.
        self.urdf.write_text(original.replace('xyz="-0.01 0 0.02"', 'xyz="-0.02 0 0.02"'), encoding="utf-8")
        checks = []
        validator.inspect_manifest(self.urdf, usd_path, checks)
        self.assertFalse(checks[0]["urdf_hash_matches"])

    def test_missing_required_manifest_fails(self):
        checks = []
        validator.inspect_manifest(self.urdf, self.directory / "robot" / "robot.usda", checks)
        self.assertFalse(checks[0]["passed"])

    @unittest.skipIf(Usd is None, "pxr is unavailable; pure validation tests still run")
    def test_real_usd_passes_then_stale_mass_and_limits_fail(self):
        usd_path = self.directory / "robot.usda"
        stage = make_usd(usd_path, self.source)
        report = validator.validate_asset(self.urdf, usd_path)
        self.assertTrue(report["passed"], [check for check in report["checks"] if not check["passed"]])
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/Robot/body")).GetMassAttr().Set(0.79963)
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Robot/joint_0")).GetUpperLimitAttr().Set(270.0)
        stage.GetRootLayer().Save()
        report = validator.validate_asset(self.urdf, usd_path)
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertFalse(report["passed"])
        self.assertIn("link_mass:body", failed)
        self.assertIn("joint_limits:joint_0", failed)

    @unittest.skipIf(Usd is None, "pxr is unavailable; pure validation tests still run")
    def test_missing_usd_dependency_and_fixed_base_are_rejected(self):
        usd_path = self.directory / "robot.usda"
        stage = make_usd(usd_path, self.source)
        stage.GetPrimAtPath("/Robot").GetReferences().AddReference("missing_geometry.usda")
        anchor = UsdPhysics.FixedJoint.Define(stage, "/Robot/world_anchor")
        anchor.CreateBody1Rel().SetTargets(["/Robot/body"])
        stage.GetRootLayer().Save()
        report = validator.validate_asset(self.urdf, usd_path)
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("usd_dependencies", failed)
        self.assertIn("usd_free_articulated_base", failed)
        self.assertFalse(report["passed"])

    @unittest.skipIf(Usd is None, "pxr is unavailable; pure validation tests still run")
    def test_missing_and_disabled_link_colliders_fail(self):
        usd_path = self.directory / "robot.usda"
        stage = make_usd(usd_path, self.source)
        collider = stage.GetPrimAtPath("/Robot/body/collision")
        UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
        for remove in (False, True):
            with self.subTest(remove=remove):
                if remove:
                    stage.RemovePrim("/Robot/body/collision")
                stage.GetRootLayer().Save()
                report = validator.validate_asset(self.urdf, usd_path)
                result = next(check for check in report["checks"] if check["name"] == "usd_link_collisions")
                self.assertFalse(result["passed"])
                self.assertEqual(result["missing_links"], ["/Robot/body"])

    @unittest.skipIf(Usd is None, "pxr is unavailable; pure validation tests still run")
    def test_instanced_link_collision_geometry_is_traversed(self):
        shape_path = self.directory / "shape.usda"
        shape_stage = Usd.Stage.CreateNew(str(shape_path))
        shape = UsdGeom.Xform.Define(shape_stage, "/Shape").GetPrim()
        shape_stage.SetDefaultPrim(shape)
        cube = UsdGeom.Cube.Define(shape_stage, "/Shape/cube")
        cube.CreateSizeAttr(0.02)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        shape_stage.GetRootLayer().Save()
        usd_path = self.directory / "robot.usda"
        stage = make_usd(usd_path, self.source)
        stage.RemovePrim("/Robot/body/collision")
        instance = UsdGeom.Xform.Define(stage, "/Robot/body/collision").GetPrim()
        instance.GetReferences().AddReference("shape.usda")
        instance.SetInstanceable(True)
        stage.GetRootLayer().Save()
        report = validator.validate_asset(self.urdf, usd_path)
        self.assertTrue(report["passed"], [check for check in report["checks"] if not check["passed"]])
        result = next(check for check in report["checks"] if check["name"] == "usd_link_collisions")
        self.assertEqual(result["colliders_by_link"]["/Robot/body"], ["/Robot/body/collision/cube"])


if __name__ == "__main__":
    unittest.main()
