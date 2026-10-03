"""Regression test for the ``_distance()`` length-mismatch gap flagged by an independent
third-party review, 2026-09-29: unlike every other ``zip()`` in preconditions.py (see
``joint_position_limits_respected``/``joint_velocity_within_limits``, fixed for the identical flaw
earlier), ``_distance`` had no length check before zipping its two coordinate tuples. That earlier
fix matters because a truncating zip() there silently *skips* a check; here it's worse -- ``zip()``
still returns a distance, just the wrong one, computed between coordinates that don't correspond to
each other, with nothing to flag it as wrong. Reachable only via a malformed ``Pose`` (the dataclass's
type hint isn't runtime-enforced), but the failure mode (confidently-wrong distance) is worse than a
crash would be.
"""

import math
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness import preconditions as pc  # noqa: E402
from safety_harness.schema import Action, Pose, TrackedAgent, WorldState  # noqa: E402


class DistanceLengthCheckTest(unittest.TestCase):
    def test_equal_length_points_unaffected(self):
        self.assertAlmostEqual(pc._distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), 5.0)

    def test_mismatched_length_raises_instead_of_silently_truncating(self):
        with self.assertRaises(ValueError):
            pc._distance((0.0, 0.0, 0.0), (1.0, 1.0))

    def test_mismatched_length_does_not_silently_report_a_wrong_short_distance(self):
        # Before the fix, zip() truncated to 2 coordinates and happily returned 0.0 -- a confirmed
        # close agent would have read as "no violation" rather than raising anything at all.
        far_but_truncated = (0.0, 0.0)  # missing a third coordinate
        with self.assertRaises(ValueError):
            pc._distance((0.0, 0.0, 5.0), far_but_truncated)


class SweptPathCheckFailsClosedOnMalformedPoseTest(unittest.TestCase):
    """One real caller of _distance(), exercised end to end: a malformed agent Pose must make the
    check fail (by raising, which ActuatorGate.gate() already converts into a safe BLOCK -- see
    test_fuzz.PathologicalInputTests.test_buggy_custom_check_that_raises_still_blocks_not_crashes
    for that engine-level guarantee), never silently report the agent as cleared."""

    def test_malformed_agent_position_length_raises_rather_than_silently_passing(self):
        trajectory = fixtures.straight_line_trajectory()
        malformed_agent = TrackedAgent(
            agent_id="person_1",
            pose=Pose(position=(0.5, 0.0)),  # missing z -- e.g. a buggy 2D-only tracker
            tracking_confidence=0.95,
        )
        state = WorldState(agents=(malformed_agent,))
        with self.assertRaises(ValueError):
            pc.swept_path_clear_of_agents(state, Action(action_type="grasp", params={}), trajectory)

    def test_well_formed_agent_position_still_works(self):
        trajectory = fixtures.straight_line_trajectory()
        state = WorldState(agents=(fixtures.close_agent(),))
        result = pc.swept_path_clear_of_agents(state, Action(action_type="grasp", params={}), trajectory)
        self.assertFalse(result.satisfied)  # close_agent() is deliberately right on the swept path


if __name__ == "__main__":
    unittest.main()
