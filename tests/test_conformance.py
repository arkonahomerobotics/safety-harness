"""Tests for safety_harness.conformance itself -- the third-party-facing fixture, not the engine.

Every test here builds a deliberately broken (or deliberately under-wired) stand-in and asserts the
fixture's report actually flags it, the same "prove it catches the bug, not just that it runs"
standard the rest of tests/ holds itself to (see test_fuzz.py's own docstring).

Run with: python3 -m unittest tests.test_conformance -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness.action_schema import ActionSchemaRegistry  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, FallbackController, Logger, PerceptionAdapter  # noqa: E402
from safety_harness.conformance import ConformanceSuite  # noqa: E402
from safety_harness.conformance.contract import run_contract_fuzz  # noqa: E402
from safety_harness.conformance.mutation import run_mutation_battery  # noqa: E402
from safety_harness.conformance.structural import validate_trajectory, validate_world_state  # noqa: E402
from safety_harness.schema import Pose, PredictedTrajectory  # noqa: E402


class _FixedPerception(PerceptionAdapter):
    def __init__(self, state):
        self._state = state

    def get_world_state(self):
        return self._state


class _StraightLineDynamics(DynamicsAdapter):
    """Ignores state/action entirely -- good enough for schemas that don't check joint-level
    trajectory data, which is all the minimal schemas below wire."""

    def predict_trajectory(self, state, action, horizon_s):
        return fixtures.straight_line_trajectory(horizon_s=horizon_s)


class _RaisingDynamics(DynamicsAdapter):
    def predict_trajectory(self, state, action, horizon_s):
        raise ValueError("simulator connection dropped")


class _NoneFallback(FallbackController):
    def execute(self, state):
        return None


class _RaisingLogger(Logger):
    def record(self, decision, state):
        raise RuntimeError("telemetry sink unavailable")


# A schema wired well enough to actually enforce the fields the mutation battery perturbs.
WELL_WIRED_GRASP_SCHEMA = ActionSchemaRegistry.from_dict({
    "action_types": {
        "grasp": {
            "checks": [
                {"name": "object_hazard_confirmed", "kwargs": {"object_id_param": "object_id", "min_confidence": 0.6}},
                {"name": "object_pose_confirmed", "kwargs": {"object_id_param": "object_id", "min_confidence": 0.6}},
                {"name": "object_cleared_for_interaction", "kwargs": {"object_id_param": "object_id"}},
                {"name": "current_position_confirmed_stable", "kwargs": {"object_id_param": "object_id"}},
                {"name": "fall_consequence_acceptable", "kwargs": {"object_id_param": "object_id"}},
                {"name": "mass_within_force_budget", "kwargs": {"object_id_param": "object_id", "force_budget_kg": 3.0}},
                {"name": "visibility_above_threshold", "kwargs": {"min_visibility": 0.5}},
            ],
        },
    },
})

# Registers grasp, but with a check that never looks at the target object or the environment at
# all -- real adapters that wire too little end up exactly here, and the mutation battery exists to
# catch it.
UNDER_WIRED_GRASP_SCHEMA = ActionSchemaRegistry.from_dict({
    "action_types": {"grasp": {"checks": [{"name": "robot_state_confirmed", "kwargs": {}}]}},
})

NO_GRASP_SCHEMA = ActionSchemaRegistry.from_dict({"action_types": {"reach": {"checks": []}}})


def _good_state():
    return fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=())


class StructuralConformanceTests(unittest.TestCase):
    def test_well_formed_state_has_no_violations(self):
        self.assertEqual(validate_world_state(_good_state()), [])

    def test_well_formed_trajectory_has_no_violations(self):
        self.assertEqual(validate_trajectory(fixtures.straight_line_trajectory()), [])

    def test_wrong_type_in_a_loosely_typed_container_is_caught(self):
        # WorldState.objects is declared as a bare `tuple` in schema.py -- see structural.py's
        # docstring for why that needs an explicit element-type rule rather than pure reflection.
        state = replace(_good_state(), objects=(42,))
        violations = validate_world_state(state)
        self.assertTrue(any("objects[0]" in v.message and "TrackedObject" in v.message for v in violations), violations)

    def test_nan_position_is_caught(self):
        obj = replace(fixtures.confirmed_object(), pose=Pose(position=(math.nan, 0.0, 0.0)))
        state = replace(_good_state(), objects=(obj,))
        violations = validate_world_state(state)
        self.assertTrue(any("not a finite number" in v.message for v in violations), violations)

    def test_confidence_outside_unit_range_is_caught(self):
        obj = replace(fixtures.confirmed_object(), class_confidence=1.5)
        state = replace(_good_state(), objects=(obj,))
        violations = validate_world_state(state)
        self.assertTrue(any("outside the documented [0, 1]" in v.message for v in violations), violations)

    def test_wrong_top_level_type_is_caught(self):
        self.assertTrue(validate_world_state("not a world state at all"))
        self.assertTrue(validate_trajectory(PredictedTrajectory(points=("not a point",))))


class ContractFuzzTests(unittest.TestCase):
    def test_well_behaved_adapter_produces_no_violations(self):
        violations, histogram = run_contract_fuzz(
            _FixedPerception(_good_state()), _StraightLineDynamics(), WELL_WIRED_GRASP_SCHEMA, n_iterations=100, seed=7,
        )
        self.assertEqual(violations, (), violations)
        self.assertTrue(histogram)

    def test_exception_that_escapes_gate_is_flagged_not_swallowed(self):
        # A logger that raises escapes ActuatorGate.gate() uncaught (engine.py only wraps
        # perception/dynamics/check failures, not the logger or fallback calls) -- confirms the
        # fixture notices rather than silently treating a crash as "it ran."
        violations, _ = run_contract_fuzz(
            _FixedPerception(_good_state()), _StraightLineDynamics(), WELL_WIRED_GRASP_SCHEMA,
            logger=_RaisingLogger(), n_iterations=5, seed=1,
        )
        self.assertTrue(violations)
        self.assertTrue(all(v.category == "undocumented_exception" for v in violations))

    def test_fallback_returning_none_is_flagged(self):
        violations, _ = run_contract_fuzz(
            _FixedPerception(_good_state()), _StraightLineDynamics(), NO_GRASP_SCHEMA,
            fallback=_NoneFallback(), n_iterations=10, seed=2,
        )
        self.assertTrue(any(v.category == "block_without_fallback" for v in violations), violations)

    def test_dynamics_exception_is_contained_by_the_engine_not_by_this_fixture(self):
        # engine.gate() already catches a dynamics failure and turns it into a normal BLOCK -- so
        # this must NOT show up as an undocumented_exception; it's the engine's existing default-
        # deny safety net working, not something the fixture needs to separately detect.
        violations, _ = run_contract_fuzz(
            _FixedPerception(_good_state()), _RaisingDynamics(), WELL_WIRED_GRASP_SCHEMA, n_iterations=10, seed=3,
        )
        self.assertEqual([v for v in violations if v.category == "undocumented_exception"], [])


class MutationBatteryTests(unittest.TestCase):
    def _dynamics(self):
        return _StraightLineDynamics()

    def test_well_wired_schema_passes(self):
        result = run_mutation_battery(
            _FixedPerception(_good_state()), self._dynamics(), WELL_WIRED_GRASP_SCHEMA, fixtures.grasp_action(),
        )
        self.assertTrue(result["baseline_permitted"], result["skip_reason"])
        self.assertEqual(result["violations"], (), result["violations"])

    def test_under_wired_schema_is_caught(self):
        result = run_mutation_battery(
            _FixedPerception(_good_state()), self._dynamics(), UNDER_WIRED_GRASP_SCHEMA, fixtures.grasp_action(),
        )
        self.assertTrue(result["baseline_permitted"], result["skip_reason"])
        self.assertTrue(result["violations"])
        self.assertTrue(all(v.category == "mutation_not_blocked" for v in result["violations"]))
        # Every curated object mutation plus the environment mutation should have escaped.
        self.assertGreaterEqual(len(result["violations"]), 8)

    def test_baseline_that_never_permits_skips_rather_than_fails(self):
        result = run_mutation_battery(
            _FixedPerception(_good_state()), self._dynamics(), NO_GRASP_SCHEMA, fixtures.grasp_action(),
        )
        self.assertFalse(result["baseline_permitted"])
        self.assertEqual(result["violations"], ())
        self.assertIn("did not PERMIT", result["skip_reason"])


class ConformanceSuiteTests(unittest.TestCase):
    def test_end_to_end_report_for_a_conformant_adapter(self):
        suite = ConformanceSuite(
            perception=_FixedPerception(_good_state()), dynamics=_StraightLineDynamics(),
            action_schema=WELL_WIRED_GRASP_SCHEMA, baseline_action=fixtures.grasp_action(),
            n_fuzz_iterations=50, seed=11,
        )
        report = suite.run()
        self.assertTrue(report.ok, report.summary())
        self.assertEqual({o.name for o in report.outcomes}, {"structural_conformance", "contract_fuzz", "mutation_battery"})
        # Round-trips without raising -- a third party is expected to serialize this, not just print it.
        as_dict = report.to_dict()
        self.assertTrue(as_dict["ok"])
        self.assertIsInstance(report.summary(), str)

    def test_mutation_battery_is_skipped_without_a_baseline_action(self):
        suite = ConformanceSuite(
            perception=_FixedPerception(_good_state()), dynamics=_StraightLineDynamics(),
            action_schema=WELL_WIRED_GRASP_SCHEMA, n_fuzz_iterations=10,
        )
        report = suite.run()
        mutation_outcome = next(o for o in report.outcomes if o.name == "mutation_battery")
        self.assertFalse(mutation_outcome.ran)
        self.assertTrue(mutation_outcome.passed)
        self.assertTrue(report.ok)

    def test_report_surfaces_an_under_wired_schema_as_a_failure(self):
        suite = ConformanceSuite(
            perception=_FixedPerception(_good_state()), dynamics=_StraightLineDynamics(),
            action_schema=UNDER_WIRED_GRASP_SCHEMA, baseline_action=fixtures.grasp_action(),
            n_fuzz_iterations=10,
        )
        report = suite.run()
        self.assertFalse(report.ok)
        self.assertTrue(any(v.category == "mutation_not_blocked" for v in report.violations))


if __name__ == "__main__":
    unittest.main()
