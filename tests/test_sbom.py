"""SBOM generation (safety_harness/sbom.py). The point of this file isn't "the script runs" --
it's the dependency-drift guard in DependencyDriftTest: this list is a hand-checked mirror of
pyproject.toml's [project].dependencies as of this writing (exactly one runtime dependency,
pyyaml>=6.0 -- see sbom.py's module docstring for why that's the honest answer, not zero). If a
future change silently adds a runtime dependency to pyproject.toml without a matching, deliberate
edit here, this test fails instead of the drift going unnoticed.

Requires the package installed (`pip install -e .`, same prerequisite as everything else -- see
README "Install & test"), since the SBOM is generated from installed-distribution metadata, not by
re-parsing pyproject.toml.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_harness import sbom  # noqa: E402

# Mirrors pyproject.toml's [project].dependencies. Deliberately hand-maintained, not derived from
# the SBOM itself -- an assertion that reads its own expected value from the code under test would
# never fail no matter what changed.
EXPECTED_RUNTIME_DEPENDENCIES = ["pyyaml>=6.0"]


class GenerateSbomTest(unittest.TestCase):
    def test_generate_sbom_does_not_crash(self):
        sbom.generate_sbom()  # no exception

    def test_top_level_structure_is_valid_cyclonedx(self):
        bom = sbom.generate_sbom()
        self.assertEqual(bom["bomFormat"], "CycloneDX")
        self.assertEqual(bom["specVersion"], "1.5")
        self.assertIn("metadata", bom)
        self.assertIn("component", bom["metadata"])
        self.assertIn("components", bom)
        self.assertIn("dependencies", bom)

    def test_output_round_trips_through_json(self):
        bom = sbom.generate_sbom()
        # json.dumps/loads must not raise, and must reproduce the same structure -- i.e. everything
        # in the document is plain JSON-serializable data, not e.g. a stray Distribution object.
        round_tripped = json.loads(json.dumps(bom))
        self.assertEqual(bom, round_tripped)

    def test_root_component_is_this_package(self):
        bom = sbom.generate_sbom()
        root = bom["metadata"]["component"]
        self.assertEqual(root["name"], "safety-harness")
        self.assertEqual(root["type"], "application")
        self.assertTrue(root["purl"].startswith("pkg:pypi/safety-harness@"))

    def test_every_component_has_name_version_and_purl(self):
        bom = sbom.generate_sbom()
        for component in bom["components"]:
            self.assertIn("name", component)
            self.assertIn("version", component)
            self.assertTrue(component["purl"].startswith("pkg:pypi/"))
            self.assertIn(component["type"], ("library", "application", "framework"))

    def test_dependency_graph_is_internally_consistent(self):
        bom = sbom.generate_sbom()
        known_refs = {bom["metadata"]["component"]["purl"]} | {c["purl"] for c in bom["components"]}
        seen_refs = set()
        for entry in bom["dependencies"]:
            self.assertIn(entry["ref"], known_refs)
            seen_refs.add(entry["ref"])
            for dep in entry["dependsOn"]:
                self.assertIn(dep, known_refs)
        # every known component (including the root) appears exactly once as a "ref"
        self.assertEqual(seen_refs, known_refs)

    def test_reproducible_given_fixed_source_date_epoch(self):
        old = os.environ.get("SOURCE_DATE_EPOCH")
        os.environ["SOURCE_DATE_EPOCH"] = "1700000000"
        try:
            first = sbom.generate_sbom()
            second = sbom.generate_sbom()
        finally:
            if old is None:
                os.environ.pop("SOURCE_DATE_EPOCH", None)
            else:
                os.environ["SOURCE_DATE_EPOCH"] = old
        self.assertEqual(first, second)
        self.assertEqual(first["metadata"]["timestamp"], "2023-11-14T22:13:20Z")

    def test_no_serial_number_emitted(self):
        # Deliberate: a random serialNumber would break reproducibility even with a fixed
        # SOURCE_DATE_EPOCH. CycloneDX makes the field optional.
        bom = sbom.generate_sbom()
        self.assertNotIn("serialNumber", bom)


class DependencyDriftTest(unittest.TestCase):
    """The actual point of this file: pyproject.toml's declared runtime dependencies must match
    what this test expects, and what the generated SBOM reports. A future dependency added to
    pyproject.toml's [project].dependencies without updating EXPECTED_RUNTIME_DEPENDENCIES here
    (deliberately, having thought about whether it belongs) fails this test."""

    def test_installed_distribution_requires_matches_expected(self):
        requires = importlib.metadata.distribution("safety-harness").requires or []
        self.assertEqual(sorted(requires), sorted(EXPECTED_RUNTIME_DEPENDENCIES))

    def test_sbom_components_match_expected_runtime_dependencies(self):
        bom = sbom.generate_sbom()
        component_names = sorted(c["name"].lower() for c in bom["components"])
        expected_names = sorted(sbom._requirement_name(r).lower() for r in EXPECTED_RUNTIME_DEPENDENCIES)
        self.assertEqual(component_names, expected_names)

    def test_pyproject_toml_dependencies_line_matches_expected(self):
        # Belt-and-suspenders: read the actual source of truth too, not just the installed
        # metadata derived from it (in case the environment's install is stale relative to a
        # locally edited pyproject.toml that hasn't been reinstalled yet).
        pyproject_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")
        with open(pyproject_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        match = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("dependencies = ["):
                match = stripped
                break
        self.assertIsNotNone(match, "couldn't find a 'dependencies = [...]' line in pyproject.toml")
        # Extract the quoted requirement strings on that line.
        import re
        found = re.findall(r'"([^"]*)"', match)
        self.assertEqual(sorted(found), sorted(EXPECTED_RUNTIME_DEPENDENCIES))


class CliTest(unittest.TestCase):
    def test_main_prints_valid_json_to_stdout(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = sbom.main([])
        self.assertEqual(exit_code, 0)
        bom = json.loads(buf.getvalue())
        self.assertEqual(bom["bomFormat"], "CycloneDX")

    def test_main_rejects_arguments(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = sbom.main(["unexpected-arg"])
        self.assertEqual(exit_code, 2)


class CheckedInReferenceSbomTest(unittest.TestCase):
    """The reference SBOM committed under examples/sbom/results/ is real generated evidence, not
    hand-typed -- see examples/sbom/README.md. This pins that it's still valid CycloneDX and that
    its dependency list hasn't silently drifted from what generate_sbom() produces today (aside
    from the timestamp, which is expected to differ run to run)."""

    def test_checked_in_sbom_matches_current_generation_modulo_timestamp(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reference_path = os.path.join(repo_root, "examples", "sbom", "results", "safety-harness.cyclonedx.json")
        with open(reference_path, "r", encoding="utf-8") as fh:
            checked_in = json.load(fh)

        current = sbom.generate_sbom()
        checked_in["metadata"]["timestamp"] = current["metadata"]["timestamp"]
        self.assertEqual(checked_in, current)


if __name__ == "__main__":
    unittest.main()
