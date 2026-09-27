"""Regression tests for the NaN-comparison false-PERMIT bugs found by adversarial stress testing on
2026-09-27 (design doc: "NaN-Sensor Stress Test") and fixed in preconditions.py via the
``_below``/``_exceeds``/``_at_or_within``/``_safe_max`` helpers.

Every numeric precondition enforces default-deny through a ``<``/``>`` comparison against a
sensor-derived float. IEEE-754 makes any such comparison against NaN evaluate False in *both*
directions, so a corrupted or never-actually-measured reading used to silently satisfy the very
check meant to catch its absence -- the same root cause the project had already found and partially
patched once for one field on one check (``object_pose_confirmed``'s NaN-*position* guard). This
suite is what found it didn't generalize, and now confirms the general fix holds.

Each test drives a single corrupted float through a real ``ActuatorGate.gate()`` call (not just the
precondition function in isolation) with every other signal clean and safe, so a BLOCK verdict can
only be explained by the fix catching that field -- and a control case confirms BLOCK for the
equivalent honest-but-bad value too, so this isn't just "anything blocks now."

Run with: python3 -m unittest discover -s tests -p "test_*.py" -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402
from test_engine import make_gate  # noqa: E402

from safety_harness import DecisionVerdict  # noqa: E402


def _failed_names(decision):
    return [r.name for r in decision.precondition_results if not r.satisfied]


class NanMassNoLongerBypassesForceBudget(unittest.TestCase):
    """mass_within_force_budget now runs its comparison through ``_exceeds``, which treats a
    non-finite mass as automatically exceeding the budget. It's the only guard on mass for `grasp`,
    so this alone determines whether an unmeasured mass can slip through."""

    def test_nan_mass_now_blocks_the_lift(self):
        obj = replace(fixtures.confirmed_object(), estimated_mass_kg=math.nan)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn("mass_within_force_budget", _failed_names(decision))

    def test_control_infinite_mass_still_blocks(self):
        obj = replace(fixtures.confirmed_object(), estimated_mass_kg=math.inf)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn("mass_within_force_budget", _failed_names(decision))

    def test_control_normal_heavy_mass_still_blocks(self):
        obj = replace(fixtures.confirmed_object(), estimated_mass_kg=5.0)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_control_mass_within_budget_still_permits(self):
        """Confirms the fix didn't overcorrect into blocking legitimate, confirmed-light objects."""
        obj = replace(fixtures.confirmed_object(), estimated_mass_kg=0.05)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)


class NanAgentPositionNoLongerDefeatsProximityChecks(unittest.TestCase):
    """swept_path_clear_of_agents and iso15066_separation_distance_maintained both now run their
    distance comparison through ``_below``, which treats a non-finite distance as automatically
    below the required clearance -- an unconfirmed position is the worst case (could be touching),
    not a free pass. Before the fix, a NaN coordinate made both individually report
    satisfied=True; the previous version of this test proved that and then found that the overall
    decision still happened to BLOCK for a standard-speed human, purely by coincidence in a third,
    unrelated check (iso15066_power_force_limiting). That check's own "too far, skip" gate is fixed
    too (see below), so the compound case is now blocked for the right reason, not by luck."""

    def _nan_agent(self, worst_case_speed_mps=1.5):
        base = replace(fixtures.close_agent(), worst_case_speed_mps=worst_case_speed_mps)
        return replace(base, pose=replace(base.pose, position=(math.nan, 0.0, 0.05)))

    def test_nan_position_now_fails_both_proximity_checks(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(self._nan_agent(),))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = set(_failed_names(decision))
        for name in ("swept_path_clear_of_agents", "iso15066_separation_distance_maintained"):
            self.assertIn(name, failed, f"{name} should now correctly report satisfied=False on a NaN position")

    def test_control_same_agent_without_nan_still_blocks_on_those_two(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.close_agent(),))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = set(_failed_names(decision))
        for name in ("swept_path_clear_of_agents", "iso15066_separation_distance_maintained"):
            self.assertIn(name, failed)


class NanAgentPositionNoLongerPermitsForASlowerTrackedAgent(unittest.TestCase):
    """This is the case that used to reach full PERMIT once the pedestrian-speed coincidence was
    removed: a tracked agent with worst_case_speed_mps=0.15 (a slow mobile-base teammate, not a
    pedestrian) kept iso15066_power_force_limiting's force estimate under its 150N limit even while
    the NaN position defeated the two proximity checks above, and its own "too far, skip" gate used
    to stop skipping on NaN only as an accidental side effect, not because it was fixed to. All three
    are now fixed on purpose: this must BLOCK, matching the exact same agent at the exact same real
    (non-NaN) position, which BLOCKs on the same two proximity checks."""

    def test_nan_position_now_blocks_grasp_next_to_a_slow_tracked_agent(self):
        agent = replace(fixtures.close_agent(), worst_case_speed_mps=0.15)
        nan_agent = replace(agent, pose=replace(agent.pose, position=(math.nan, 0.0, 0.05)))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(nan_agent,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = set(_failed_names(decision))
        self.assertIn("swept_path_clear_of_agents", failed)
        self.assertIn("iso15066_separation_distance_maintained", failed)

    def test_control_same_agent_and_position_without_nan_still_blocks(self):
        agent = replace(fixtures.close_agent(), worst_case_speed_mps=0.15)
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(agent,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = set(_failed_names(decision))
        self.assertIn("swept_path_clear_of_agents", failed)
        self.assertIn("iso15066_separation_distance_maintained", failed)


class NanConfidenceNoLongerBypassesConfidenceGates(unittest.TestCase):
    """object_pose_confirmed's pose_confidence and object_hazard_confirmed's class_confidence both
    now run through ``_below``, which treats a non-finite confidence score as automatically below
    the minimum -- an unconfirmed confidence is the worst case, same principle as everywhere else."""

    def test_nan_pose_confidence_now_blocks(self):
        obj = replace(fixtures.confirmed_object(), pose_confidence=math.nan)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn("object_pose_confirmed", _failed_names(decision))

    def test_nan_class_confidence_now_blocks(self):
        obj = replace(fixtures.confirmed_object(), class_confidence=math.nan)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn("object_hazard_confirmed", _failed_names(decision))

    def test_control_low_pose_confidence_still_blocks(self):
        obj = replace(fixtures.confirmed_object(), pose_confidence=0.1)
        state = fixtures.base_world_state(objects=(obj,))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_control_high_confidence_still_permits(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        decision = make_gate(state).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)


class NanBalanceMarginNoLongerDefeatsLeggedPlatformCheck(unittest.TestCase):
    """balance_margin_maintained isn't wired into the example grasp/place/reach schema (no legged
    platform in this reference config), so this is exercised directly against the function, as a
    latent-bug fix for whenever a humanoid/legged adapter registers it."""

    def test_nan_center_of_mass_component_now_fails_the_margin_check(self):
        from safety_harness.preconditions import balance_margin_maintained
        from safety_harness.schema import TrajectoryPoint

        robot = replace(
            fixtures.robot_state(),
            center_of_mass=(math.nan, 0.0, 0.5),
            support_polygon=((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)),
        )
        point = TrajectoryPoint(t=0.0, robot=robot, swept_volume_center=(0.0, 0.0, 0.5), swept_volume_radius_m=0.1)
        trajectory = replace(fixtures.straight_line_trajectory(), points=(point,))
        result = balance_margin_maintained(fixtures.base_world_state(), fixtures.grasp_action(), trajectory)
        self.assertFalse(result.satisfied)

    def test_control_finite_in_margin_center_of_mass_still_permits(self):
        from safety_harness.preconditions import balance_margin_maintained
        from safety_harness.schema import TrajectoryPoint

        robot = replace(
            fixtures.robot_state(),
            center_of_mass=(0.0, 0.0, 0.5),
            support_polygon=((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)),
        )
        point = TrajectoryPoint(t=0.0, robot=robot, swept_volume_center=(0.0, 0.0, 0.5), swept_volume_radius_m=0.1)
        trajectory = replace(fixtures.straight_line_trajectory(), points=(point,))
        result = balance_margin_maintained(fixtures.base_world_state(), fixtures.grasp_action(), trajectory)
        self.assertTrue(result.satisfied)


if __name__ == "__main__":
    unittest.main()
