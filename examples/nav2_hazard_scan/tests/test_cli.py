"""scan_demo.py's CLI: the bundled map end to end (self-check + report generation), a malformed
map's error path, and the honest-skip behavior for the junction rule -- the specific thing
Engineering Development asked for a dedicated test of (never silently fall back to the demo graph
for a real scan; say plainly in both console output and the report that it was skipped).

Run with: python3 -m unittest discover -s examples/nav2_hazard_scan/tests -v
(from the repo root, with numpy/pillow/pyyaml installed -- see the example README)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_DIR = os.path.dirname(_HERE)
SCAN_DEMO = os.path.join(_EXAMPLE_DIR, "scan_demo.py")
BUNDLED_MAP = os.path.join(_EXAMPLE_DIR, "maps", "warehouse.yaml")


def _run(*args):
    return subprocess.run(
        [sys.executable, SCAN_DEMO, *args], cwd=_EXAMPLE_DIR,
        capture_output=True, text=True, timeout=120,
    )


class BundledMapCliTest(unittest.TestCase):
    def test_default_invocation_self_checks_and_exits_zero(self):
        result = _run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_report_is_written_with_expected_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.md")
            result = _run("--report", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.isfile(out))
            with open(out) as f:
                text = f.read()
            for heading in ("# Nav2 Hazard Scan Report", "## Coverage", "## Findings", "## Method",
                             "## Parameters used", "What this report is"):
                self.assertIn(heading, text)

    def test_map_can_be_pointed_at_an_absolute_path(self):
        # --map defaults to the bundled map resolved relative to the script's own directory, not
        # the process's cwd -- this is the fix for the README's documented invocation not actually
        # working from the repo root (see the PR). Confirm an explicit absolute path also works,
        # run from a cwd that is NOT the example directory at all.
        result = subprocess.run(
            [sys.executable, SCAN_DEMO, "--map", BUNDLED_MAP], cwd=tempfile.gettempdir(),
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class MalformedMapCliTest(unittest.TestCase):
    def test_nonexistent_map_exits_nonzero_with_a_clear_message_not_a_traceback(self):
        result = _run("--map", "/definitely/does/not/exist.yaml")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("error:", result.stderr)


class HonestSkipTest(unittest.TestCase):
    """The behavior Engineering Development specifically asked be tested: the junction rule must
    never silently run against the toy demonstration graph for a real scan."""

    def test_no_route_graph_skips_the_junction_rule_honestly(self):
        result = _run("--map", BUNDLED_MAP)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Junction review: not evaluated, no route graph supplied", result.stdout)
        self.assertIn("intersection_flagged_for_review: skipped", result.stdout)
        # Never a FLAG/OK line for the junction rule when it was skipped.
        self.assertNotIn("intersection_flagged_for_review  @", result.stdout)

    def test_skip_is_recorded_in_the_report_coverage_table_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.md")
            result = _run("--map", BUNDLED_MAP, "--report", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out) as f:
                text = f.read()
            self.assertIn("Junction / crossing review | Skipped", text)
            self.assertIn("**Not evaluated: no route graph supplied", text)

    def test_real_route_graph_is_actually_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = os.path.join(tmp, "graph.json")
            with open(graph_path, "w") as f:
                json.dump({
                    "nodes": {"a": [0, 0], "b": [5, 0], "c": [5, 5], "j": [5, 2.5], "d": [10, 2.5]},
                    "edges": [["a", "j"], ["j", "b"], ["j", "c"], ["j", "d"]],
                }, f)
            result = _run("--map", BUNDLED_MAP, "--route_graph", graph_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("intersection_flagged_for_review  @", result.stdout)
            self.assertNotIn("not evaluated", result.stdout)

    def test_demo_graph_is_clearly_labeled_as_a_demonstration_not_a_real_finding(self):
        result = _run("--map", BUNDLED_MAP, "--demo_graph")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DEMONSTRATION graph", result.stdout)

    def test_demo_graph_and_route_graph_together_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = os.path.join(tmp, "graph.json")
            with open(graph_path, "w") as f:
                json.dump({"nodes": {"a": [0, 0]}, "edges": []}, f)
            result = _run("--demo_graph", "--route_graph", graph_path)
            self.assertNotEqual(result.returncode, 0)

    def test_malformed_route_graph_exits_nonzero_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = os.path.join(tmp, "bad.json")
            with open(graph_path, "w") as f:
                f.write("not json")
            result = _run("--map", BUNDLED_MAP, "--route_graph", graph_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
