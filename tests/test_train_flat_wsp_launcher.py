"""Real Windows PowerShell 5.1 launcher tests using a bounded native uv stand-in."""

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
FAKE_UV = r'''
using System;
using System.IO;
using System.Text;
using System.Threading;
public class FakeUv {
    public static int Main(string[] args) {
        Console.OutputEncoding = new UTF8Encoding(false);
        string capture = Environment.GetEnvironmentVariable("WSP_FAKE_CAPTURE");
        using (var writer = new StreamWriter(capture, true, Encoding.UTF8)) {
            foreach (string arg in args) writer.WriteLine(Convert.ToBase64String(Encoding.UTF8.GetBytes(arg)));
            writer.WriteLine("END");
        }
        Console.Out.WriteLine("FAKE_STDOUT: " + String.Join("|", args));
        Console.Error.WriteLine("FAKE_STDERR: persistent diagnostic");
        Thread.Sleep(700);
        string fail = Environment.GetEnvironmentVariable("WSP_FAKE_FAIL");
        foreach (string arg in args) if (!String.IsNullOrEmpty(fail) && arg == fail) return 7;
        Console.Out.WriteLine("FAKE_COMPLETED");
        return 0;
    }
}
'''


@unittest.skipUnless(os.name == "nt" and POWERSHELL.is_file(), "Requires Windows PowerShell 5.1")
class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "outputs").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="wsp launcher tests ", dir=ROOT / "outputs")
        cls.work = Path(cls.temporary.name)
        cls.bin = cls.work / "native tools"
        cls.bin.mkdir()
        code = cls.bin / "fake_uv.cs"
        code.write_text(FAKE_UV, encoding="utf-8")
        target = cls.bin / "uv.exe"
        quote = lambda path: "'" + str(path).replace("'", "''") + "'"
        command = f"Add-Type -Path {quote(code)} -OutputAssembly {quote(target)} -OutputType ConsoleApplication"
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        subprocess.run([str(POWERSHELL), "-NoProfile", "-EncodedCommand", encoded],
                       capture_output=True, check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def prepare(self, name, fail=""):
        repo = self.work / name
        (repo / "scripts").mkdir(parents=True)
        for script in ("train_flat_wsp.ps1", "train_flat_refine.ps1", "submit_flat_wsp_stop.ps1"):
            shutil.copy2(ROOT / "scripts" / script, repo / "scripts" / script)
        checkpoint = repo / "weights with spaces.pt"
        checkpoint.write_bytes(b"fake checkpoint; no simulator will execute")
        capture = repo / "native_arguments.txt"
        env = os.environ.copy()
        env["PATH"] = str(self.bin) + os.pathsep + env["PATH"]
        env["WSP_FAKE_CAPTURE"] = str(capture)
        env["WSP_FAKE_FAIL"] = fail
        device = 'probe "quoted" trailing\\'
        command = [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(repo / "scripts/train_flat_wsp.ps1"), "-Checkpoint", str(checkpoint),
                   "-MaxIterations", "2", "-NumEnvs", "4", "-Device", device]
        return repo, capture, env, command, device

    def state(self, repo, prefix="flat_wsp"):
        files = list((repo / "outputs" / (prefix + "_launch")).glob("*/launcher_status.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding="utf-8"))

    def test_refine_wrapper_passes_pinned_source_and_5120_by_1000_defaults(self):
        repo, capture, env, command, _ = self.prepare("refine default dimensions repo")
        command[5] = str(repo / "scripts/train_flat_refine.ps1")
        # Retain the fake source/device, but exercise the real wrapper defaults.
        command = command[:8] + command[12:]
        source_hash = hashlib.sha256((repo / "weights with spaces.pt").read_bytes()).hexdigest()
        command += ["-CheckpointSha256", source_hash]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        state = self.state(repo, "flat_refine")
        self.assertEqual(state["status"], "completed")
        self.assertEqual((state["num_envs"], state["max_iterations"], state["initial_std"]), (5120, 1000, .08))
        self.assertEqual(state["task"], "Hexapod-Walk-Refine")
        self.assertEqual(state["source_sha256"], source_hash)
        invocations = capture.read_text(encoding="utf-8-sig").strip().split("\nEND")
        decoded = [[base64.b64decode(line).decode("utf-8") for line in invocation.strip().splitlines()]
                   for invocation in invocations if invocation.strip()]
        training_args = decoded[2]
        for option, expected in (("--task", "Hexapod-Walk-Refine"), ("--initial_std", "0.08"),
                                 ("--checkpoint_sha256", source_hash), ("--num_envs", "5120"),
                                 ("--max_iterations", "1000")):
            self.assertEqual(training_args[training_args.index(option) + 1], expected)

    def test_refine_source_hash_mismatch_stops_before_any_native_worker(self):
        repo, capture, env, command, _ = self.prepare("refine hash mismatch repo")
        command[5] = str(repo / "scripts/train_flat_refine.ps1")
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        state = self.state(repo, "flat_refine")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(len(state["steps"]), 0)
        self.assertIn("SHA-256 mismatch", state["error"])
        self.assertFalse(capture.exists())

    def test_wsp_stop_native_worker_waits_without_console_tail_and_preserves_arguments(self):
        repo, capture, env, command, _ = self.prepare("wsp stop quiet repo")
        checkpoint_hash = hashlib.sha256((repo / "weights with spaces.pt").read_bytes()).hexdigest()
        provenance = repo / "lineage with spaces.json"
        provenance.write_text('{}', encoding="utf-8")
        command += ["-Task", "Hexapod-Walk-WspStop", "-CheckpointSha256", checkpoint_hash,
                    "-WspProvenance", str(provenance), "-NoConsoleTail", "-RunName", "quiet_test"]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        state = self.state(repo, "flat_wsp_stop")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["run_name"], "quiet_test")
        self.assertFalse(state["console_tail_enabled"])
        self.assertEqual(state["initial_std"], .05)
        smoke_arguments = state["steps"][1]["arguments"]
        self.assertEqual(smoke_arguments[smoke_arguments.index("--task") + 1], "Hexapod-Walk-WspStop")
        arguments = state["steps"][-1]["arguments"]
        self.assertEqual(arguments[arguments.index("--wsp_provenance") + 1], str(provenance))
        self.assertEqual(arguments[arguments.index("--initial_std") + 1], "0.05")
        self.assertTrue(capture.exists())

    def test_detached_prepare_validates_lineage_and_creates_no_worker_or_scheduled_task(self):
        repo, capture, env, _, _ = self.prepare("detached prepare repo")
        checkpoint = repo / "weights with spaces.pt"
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        provenance = repo / "lineage with spaces.json"
        provenance.write_text(json.dumps({
            "source_sha256": "c8828d64d8f1257f9a140a3990cf1250c91c111d8afcb80f1aa4f2f66d648601",
            "candidate_sha256": checkpoint_hash, "source_sha256_unchanged": True,
            "normalizer_preserved": True, "upstream_preserved": True,
            "runtime_inputs_added": False, "normalization_epsilon": .01}), encoding="utf-8")
        command = [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(repo / "scripts/submit_flat_wsp_stop.ps1"), "-Checkpoint", str(checkpoint),
                   "-CheckpointSha256", checkpoint_hash, "-WspProvenance", str(provenance), "-PrepareOnly"]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        receipts = list((repo / "outputs/flat_wsp_stop_submit").glob("*/submission.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "prepared")
        self.assertFalse(receipt["registered"])
        self.assertFalse(receipt["launch_accepted"])
        self.assertFalse(receipt["monitoring_enabled"])
        self.assertEqual((receipt["num_envs"], receipt["max_iterations"], receipt["initial_std"]), (5120, 2000, .05))
        self.assertIn("-NoConsoleTail", receipt["arguments"])
        self.assertIn(str(provenance), receipt["arguments"])
        self.assertFalse(capture.exists())
        self.assertFalse((repo / "outputs/flat_wsp_stop_launch").exists())
        nonlinear = json.loads(provenance.read_text(encoding="utf-8"))
        nonlinear.update(edit_scope="last_two_layers", method="nonlinear", prefix_preserved=True,
                         preserved_prefix_layers=[0, 2], changed_actor_tensors=["mlp.4.weight", "mlp.6.bias"])
        provenance.write_text(json.dumps(nonlinear), encoding="utf-8")
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        receipts = [json.loads(path.read_text(encoding="utf-8"))
                    for path in (repo / "outputs/flat_wsp_stop_submit").glob("*/submission.json")]
        self.assertEqual(len(receipts), 2)
        self.assertEqual({record["wsp_edit_scope"] for record in receipts}, {"head_only", "last_two_layers"})
        self.assertTrue(all(not record["registered"] and not record["launch_accepted"] for record in receipts))
        nonlinear["prefix_preserved"] = False
        provenance.write_text(json.dumps(nonlinear), encoding="utf-8")
        rejected = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(len(list((repo / "outputs/flat_wsp_stop_submit").glob("*/submission.json"))), 2)
        self.assertFalse(capture.exists())

    def test_stdout_stderr_and_quoted_arguments_survive_native_process_launch(self):
        repo, capture, env, command, device = self.prepare("success repo")
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        state = self.state(repo)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["steps"]), 3)
        invocations = capture.read_text(encoding="utf-8-sig").strip().split("\nEND")
        decoded = [[base64.b64decode(line).decode("utf-8") for line in invocation.strip().splitlines()]
                   for invocation in invocations if invocation.strip()]
        self.assertEqual(len(decoded), 3)
        self.assertIn(device, decoded[1])
        self.assertIn(str(repo / "weights with spaces.pt"), decoded[2])
        self.assertIn("-u", decoded[2])
        for step in state["steps"]:
            self.assertEqual(step["exit_code"], 0)
            self.assertIn("FAKE_STDOUT", Path(step["stdout"]).read_text(encoding="utf-8-sig"))
            self.assertIn("FAKE_STDERR", Path(step["stderr"]).read_text(encoding="utf-8-sig"))

    def test_native_failure_stops_later_steps_and_keeps_diagnostics(self):
        repo, _, env, command, _ = self.prepare("failure repo", "scripts/validate_robot_asset.py")
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        state = self.state(repo)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(len(state["steps"]), 1)
        self.assertEqual(state["steps"][0]["exit_code"], 7)
        self.assertIn("FAKE_STDERR", Path(state["steps"][0]["stderr"]).read_text(encoding="utf-8-sig"))

    def test_closed_parent_output_pipe_does_not_fail_native_worker(self):
        repo, capture, env, command, _ = self.prepare("disconnected output repo")
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 15
            while not capture.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(capture.exists(), "Native worker did not start")
            process.stdout.close()
            process.stderr.close()
            process.wait(timeout=30)
            state = self.state(repo)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(len(state["steps"]), 3)
            for step in state["steps"]:
                self.assertEqual(step["exit_code"], 0)
                self.assertIn("FAKE_COMPLETED", Path(step["stdout"]).read_text(encoding="utf-8-sig"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for stream in (process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
