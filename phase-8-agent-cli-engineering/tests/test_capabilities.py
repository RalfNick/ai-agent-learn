import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_cli_lab import cli
from agent_cli_lab.records import RunNotFoundError, export_report, get_run, list_runs


class CapabilityTests(unittest.TestCase):
    def test_list_get_and_export_share_stable_records(self):
        self.assertEqual(list_runs(2)["count"], 2)
        self.assertEqual(get_run("run-003")["status"], "needs_human")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            artifact = export_report("run-003", output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["id"], "run-003")
            self.assertEqual(artifact["bytes"], output.stat().st_size)

    def test_invalid_limit_and_unknown_id_are_explicit(self):
        with self.assertRaises(ValueError):
            list_runs(0)
        with self.assertRaises(RunNotFoundError):
            get_run("missing")

    def test_cli_json_and_error_exit_codes(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(["runs", "list", "--limit", "1", "--format", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["count"], 1)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli.main(["runs", "get", "missing", "--format", "json"])
        self.assertEqual(code, 2)
        self.assertIn("unknown run id", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

