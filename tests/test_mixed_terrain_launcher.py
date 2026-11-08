"""Bounded Windows native stand-ins; never register tasks or start Isaac Lab."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.test_train_flat_wsp_launcher import FAKE_UV, POWERSHELL


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "725c9696e82ef25e4b13e20c648562d17be6c79a6c1bc6ebd81936fbf2590527"


@unittest.skipUnless(os.name == "nt" and POWERSHELL.is_file(), "Requires Windows PowerShell")
class MixedLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import base64
        cls.temporary = tempfile.TemporaryDirectory(prefix="mixed launcher ", dir=ROOT / "outputs")
        cls.folder = Path(cls.temporary.name)
        cls.python = cls.folder / "python standin.exe"
        code = cls.folder / "fake.cs"
        code.write_text(FAKE_UV, encoding="utf-8")
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = f"Add-Type -Path {quote(code)} -OutputAssembly {quote(cls.python)} -OutputType ConsoleApplication"
        encoded = base64.b64encode(command.encode("utf-16-le")).decode()
        subprocess.run([str(POWERSHELL), "-NoProfile", "-EncodedCommand", encoded], capture_output=True, check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def prepare(self, name):
        repo = self.folder / name
        (repo / "scripts").mkdir(parents=True)
        (repo / "src").mkdir()
        (repo / "src/marker.py").write_text("# isolated runtime\n")
        checkpoint = repo / "source weights.pt"
        checkpoint.write_bytes(b"bounded native-argument test; no actual checkpoint or simulation")
        fake_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        # Only the temporary fixture's pinned hash is replaced. Production files
        # keep the real model_1999 hash and are never changed by this test.
        for script in ("submit_mixed_terrain.ps1", "run_mixed_terrain.ps1", "train_mixed_terrain.py",
                       "mixed_actor_adapter.py", "train_flat_wsp.py", "validate_robot_asset.py"):
            content = (ROOT / "scripts" / script).read_text(encoding="utf-8")
            (repo / "scripts" / script).write_text(content.replace(SOURCE_HASH, fake_hash), encoding="utf-8")
        capture = repo / "captured_native.txt"
        env = os.environ.copy()
        env["WSP_FAKE_CAPTURE"], env["WSP_FAKE_FAIL"] = str(capture), ""
        return repo, checkpoint, capture, env

    def test_prepare_only_records_isolated_runtime_original_python_and_no_task(self):
        repo, checkpoint, capture, env = self.prepare("prepare only")
        command = [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(repo / "scripts/submit_mixed_terrain.ps1"), "-PythonPath", str(self.python),
                   "-Checkpoint", str(checkpoint), "-PrepareOnly"]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        receipts = list((repo / "outputs/mixed_terrain_submit").glob("*/submission.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        self.assertEqual((receipt["num_envs"], receipt["max_iterations"], receipt["flat_fraction"],
                          receipt["stair_height"], receipt["stair_reward_scale"]), (5120, 2000, .5, .04, .1))
        self.assertEqual(receipt["python_executable"], str(self.python))
        self.assertEqual(receipt["runtime_root"], str(repo))
        self.assertFalse(receipt["registered"] or receipt["launch_accepted"] or receipt["monitoring_enabled"])
        self.assertTrue(receipt["code_provenance"])
        self.assertFalse(capture.exists())

    def test_native_run_preserves_arguments_and_verify_only_without_console_pipe(self):
        repo, checkpoint, capture, env = self.prepare("native bounded")
        command = [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(repo / "scripts/run_mixed_terrain.ps1"), "-PythonPath", str(self.python),
                   "-Checkpoint", str(checkpoint), "-NumEnvs", "16", "-MaxIterations", "2", "-VerifyOnly"]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        files = list((repo / "outputs/mixed_terrain_launch").glob("*/launcher_status.json"))
        state = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "verified_initialization_without_ppo")
        self.assertEqual(len(state["steps"]), 2)
        arguments = state["steps"][1]["arguments"]
        self.assertIn("--verify-only", arguments)
        self.assertEqual(arguments[arguments.index("--num_envs") + 1], "16")
        self.assertEqual(arguments[arguments.index("--stair_reward_scale") + 1], "0.1")
        for step in state["steps"]:
            self.assertEqual(step["exit_code"], 0)
            self.assertIn("FAKE_COMPLETED", Path(step["stdout"]).read_text(encoding="utf-8-sig"))
            self.assertIn("FAKE_STDERR", Path(step["stderr"]).read_text(encoding="utf-8-sig"))
        self.assertTrue(capture.exists())

    def test_resume_launch_passes_checkpoint_and_terrain_modes(self):
        repo, checkpoint, _, env = self.prepare("resume launch")
        checkpoint.write_bytes(b"different from the pinned flat checkpoint")
        command = [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(repo / "scripts/run_mixed_terrain.ps1"), "-PythonPath", str(self.python),
                   "-ResumeCheckpoint", str(checkpoint), "-HeightCurriculum", "-RandomStairHeight", "-VerifyOnly"]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        status_path = next((repo / "outputs/mixed_terrain_launch").glob("*/launcher_status.json"))
        state = json.loads(status_path.read_text(encoding="utf-8"))
        arguments = state["steps"][1]["arguments"]
        self.assertEqual(arguments[arguments.index("--resume-checkpoint") + 1], str(checkpoint))
        self.assertEqual(state["source_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertIn("--height-curriculum", arguments)
        self.assertIn("--random-stair-height", arguments)

    def test_resume_submission_pins_checkpoint_digest(self):
        repo, checkpoint, capture, env = self.prepare("resume submission")
        checkpoint.write_bytes(b"different from the pinned flat checkpoint")
        command = [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(repo / "scripts/submit_mixed_terrain.ps1"), "-PythonPath", str(self.python),
                   "-ResumeCheckpoint", str(checkpoint), "-HeightCurriculum", "-RandomStairHeight", "-PrepareOnly"]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        receipt_path = next((repo / "outputs/mixed_terrain_submit").glob("*/submission.json"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        arguments = receipt["arguments"]
        self.assertEqual(arguments[arguments.index("-ResumeCheckpoint") + 1], str(checkpoint))
        self.assertEqual(arguments[arguments.index("-CheckpointSha256") + 1], receipt["source_sha256"])
        self.assertEqual(receipt["source_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertIn("-HeightCurriculum", arguments)
        self.assertIn("-RandomStairHeight", arguments)
        self.assertFalse(capture.exists())


if __name__ == "__main__":
    unittest.main()
