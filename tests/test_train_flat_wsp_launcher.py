"""Real Windows PowerShell 5.1 launcher tests using a bounded native uv stand-in."""

import base64
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
        shutil.copy2(ROOT / "scripts/train_flat_wsp.ps1", repo / "scripts/train_flat_wsp.ps1")
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

    def state(self, repo):
        files = list((repo / "outputs/flat_wsp_launch").glob("*/launcher_status.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding="utf-8"))

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
