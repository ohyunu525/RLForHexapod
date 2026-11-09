"""Exercise the Windows launcher without starting Isaac Lab or training."""

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
FAKE_PYTHON = r'''
using System;
using System.IO;
using System.Text;
public class FakePython {
    public static int Main(string[] args) {
        Console.OutputEncoding = new UTF8Encoding(false);
        string capture = Environment.GetEnvironmentVariable("FLAT_COMMAND_CAPTURE");
        using (var writer = new StreamWriter(capture, false, Encoding.UTF8)) {
            foreach (string arg in args) writer.WriteLine(Convert.ToBase64String(Encoding.UTF8.GetBytes(arg)));
        }
        Console.Out.WriteLine("FAKE_COMPLETED");
        Console.Error.WriteLine("FAKE_DIAGNOSTIC");
        return Environment.GetEnvironmentVariable("FLAT_COMMAND_FAIL") == "1" ? 7 : 0;
    }
}
'''


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == "nt" and POWERSHELL.is_file(), "Requires Windows PowerShell 5.1")
class FlatCommandLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "outputs").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="flat command launcher tests ", dir=ROOT / "outputs")
        cls.work = Path(cls.temporary.name)
        cls.fake_python = cls.work / "python with spaces.exe"
        code = cls.work / "fake.cs"
        code.write_text(FAKE_PYTHON, encoding="utf-8")
        command = f"Add-Type -Path {quote(code)} -OutputAssembly {quote(cls.fake_python)} -OutputType ConsoleApplication"
        cls.invoke(command, os.environ.copy(), check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def invoke(command, env, check=False):
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        return subprocess.run([str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                              env=env, capture_output=True, check=check, timeout=30)

    def prepare(self, name):
        repo = self.work / name
        (repo / "scripts").mkdir(parents=True)
        launcher = repo / "scripts/run_flat_commands.ps1"
        shutil.copy2(ROOT / "scripts/run_flat_commands.ps1", launcher)
        capture = repo / "captured.txt"
        env = os.environ.copy()
        env["FLAT_COMMAND_CAPTURE"] = str(capture)
        command = ("[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
                   f"& {quote(launcher)} -PythonPath {quote(self.fake_python)} ")
        return repo, capture, env, command

    def read_state(self, repo):
        paths = list((repo / "outputs/flat_commands_launch").glob("*/launcher_status.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_dry_run_uses_default_pinned_source_without_launching(self):
        repo, capture, env, command = self.prepare("default dry run")
        result = self.invoke(command + "-DryRun -VerifyOnly", env)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        plan = json.loads(result.stdout.decode("utf-8-sig"))
        args = plan["arguments"]
        self.assertEqual(args[args.index("--task") + 1], "Hexapod-Walk-Commands")
        self.assertEqual(args[args.index("--command-difficulty") + 1], "hard")
        self.assertEqual(args[args.index("--num_envs") + 1], "5120")
        self.assertEqual(args[args.index("--max_iterations") + 1], "2000")
        self.assertIn("--verify-only", args)
        self.assertNotIn("--checkpoint", args)
        self.assertNotIn("--checkpoint_sha256", args)
        self.assertEqual(plan["source_selection"], "pinned_pre_stair_default")
        self.assertFalse(plan["monitoring_enabled"])
        self.assertFalse(capture.exists())
        self.assertFalse((repo / "outputs").exists())

    def test_native_arguments_preserve_ranges_culture_quotes_and_checkpoint(self):
        repo, capture, env, command = self.prepare("native custom settings")
        checkpoint = repo / "weights with spaces.pt"
        checkpoint.write_bytes(b"fake checkpoint")
        expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        device = 'probe "quoted" trailing\\'
        command = ("[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::GetCultureInfo('fr-FR'); "
                   + command + f"-Checkpoint {quote(checkpoint)} -CheckpointSha256 {expected_hash} "
                   f"-Device {quote(device)} -Difficulty expert -LinVelX @(-0.2, 0.4) -LinVelY @(-0.2, 0.2) "
                   "-YawRate @(-1.0, 1.0) -CommandInterval @(1.5, 3.0) -StandingFraction 0.25 -VerifyOnly")
        result = self.invoke(command, env)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        args = [base64.b64decode(line).decode("utf-8") for line in capture.read_text(encoding="utf-8-sig").splitlines()]
        for option, expected in (("--checkpoint", str(checkpoint)), ("--checkpoint_sha256", expected_hash),
                                 ("--device", device), ("--command-difficulty", "expert"), ("--standing-fraction", "0.25")):
            self.assertEqual(args[args.index(option) + 1], expected)
        for option, expected in (("--lin-vel-x", ["-0.2", "0.4"]), ("--lin-vel-y", ["-0.2", "0.2"]),
                                 ("--yaw-rate", ["-1", "1"]), ("--command-interval", ["1.5", "3"])):
            index = args.index(option)
            self.assertEqual(args[index + 1:index + 3], expected)
        self.assertIn("--verify-only", args)
        state = self.read_state(repo)
        self.assertEqual(state["status"], "verified_initialization_without_ppo")
        self.assertEqual(state["arguments"], args)
        self.assertEqual(state["exit_code"], 0)
        self.assertIsNone(state["active_process_id"])
        self.assertIn("FAKE_COMPLETED", Path(state["stdout"]).read_text(encoding="utf-8-sig"))
        self.assertIn("FAKE_DIAGNOSTIC", Path(state["stderr"]).read_text(encoding="utf-8-sig"))

    def test_invalid_ranges_and_hashes_never_launch_worker(self):
        invalid = (
            "-LinVelX @(0.3, -0.1)", "-LinVelY @(0.1)", "-YawRate @([double]::NaN, 1)",
            "-CommandInterval @(0, 2)", "-StandingFraction ([double]::PositiveInfinity)",
            "-StandingFraction 1.1", "-CheckpointSha256 " + "0" * 64,
        )
        for i, options in enumerate(invalid):
            with self.subTest(options=options):
                repo, capture, env, command = self.prepare(f"invalid {i}")
                result = self.invoke(command + options, env)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(capture.exists())
                self.assertFalse((repo / "outputs").exists())
        repo, capture, env, command = self.prepare("hash mismatch")
        checkpoint = repo / "weights.pt"
        checkpoint.write_bytes(b"wrong hash")
        result = self.invoke(command + f"-Checkpoint {quote(checkpoint)} -CheckpointSha256 " + "0" * 64, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(capture.exists())

    def test_nonzero_native_exit_is_saved_and_reported(self):
        repo, capture, env, command = self.prepare("native failure")
        env["FLAT_COMMAND_FAIL"] = "1"
        result = self.invoke(command, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(capture.exists())
        state = self.read_state(repo)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["exit_code"], 7)
        self.assertIn("code 7", state["error"])
        self.assertIsNone(state["active_process_id"])


if __name__ == "__main__":
    unittest.main()
