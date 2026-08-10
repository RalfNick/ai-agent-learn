import os
import sys
import tempfile
import unittest
from pathlib import Path

from agent_cli_lab.runner import run_process, safe_environment


class RunnerTests(unittest.TestCase):
    def test_success_and_nonzero_exit_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            ok = run_process([sys.executable, "-c", "print('ok')"], cwd=Path(temp))
            failed = run_process([sys.executable, "-c", "raise SystemExit(7)"], cwd=Path(temp))
        self.assertEqual(ok.exit_code, 0)
        self.assertEqual(ok.stdout.strip(), "ok")
        self.assertEqual(failed.exit_code, 7)

    def test_timeout_and_output_limit_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            timed_out = run_process(
                [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(5)"],
                cwd=Path(temp),
                timeout_seconds=0.1,
            )
            clipped = run_process(
                [sys.executable, "-c", "print('x' * 100)"], cwd=Path(temp), output_limit=10
            )
        self.assertEqual(timed_out.status, "timed_out")
        self.assertIn("start", timed_out.stdout)
        self.assertTrue(clipped.stdout_truncated)

    def test_environment_uses_allowlist(self):
        os.environ["AGENT_CLI_TEST_SECRET"] = "must-not-leak"
        env = safe_environment({"LAB_MODE": "test"})
        self.assertNotIn("AGENT_CLI_TEST_SECRET", env)
        self.assertEqual(env["LAB_MODE"], "test")

    def test_missing_working_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "working directory"):
            run_process([sys.executable, "-c", "pass"], cwd=Path("missing-directory"))


if __name__ == "__main__":
    unittest.main()

