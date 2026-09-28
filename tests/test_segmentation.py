"""GripActionSegmenter (safety_harness/segmentation.py) -- the decision-segmentation fix.

Covers: the exact deadlock it fixes (gating "grasp" on every closing step) does NOT reproduce
against it; edge-triggering fires exactly once per transition, not on every step past the edge;
multi-object sequencing (object_id changing between calls); the defensive is_holding guards on
both edges; reset() between episodes; and a reproduction of the validated Franka closed-loop
control sequence (open -> close -> hold/travel -> open -> travel), asserting the exact same
action-type sequence that measurement's own inline logic produced."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_harness.schema import Action  # noqa: E402
from safety_harness.segmentation import GripActionSegmenter  # noqa: E402

OPEN, CLOSE = 1.0, -1.0


class BasicEdgeTriggerTest(unittest.TestCase):
    def test_first_call_closing_fires_grasp(self):
        seg = GripActionSegmenter()
        a = seg.propose(CLOSE, "cube_2", (0.5, 0.0, 0.05), is_holding=False)
        self.assertEqual(a.action_type, "grasp")
        self.assertEqual(a.params["object_id"], "cube_2")
        self.assertEqual(a.params["target_position"], (0.5, 0.0, 0.05))

    def test_first_call_open_fires_reach_not_place(self):
        seg = GripActionSegmenter()
        a = seg.propose(OPEN, "cube_2", (0.5, 0.0, 0.05), is_holding=False)
        self.assertEqual(a.action_type, "reach")

    def test_repeated_closing_after_the_edge_does_not_refire_grasp(self):
        """The exact deadlock this class exists to fix: gating "grasp" on every step while the
        fingers are still closing. Only the FIRST closing step should ever propose "grasp"."""
        seg = GripActionSegmenter()
        first = seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)
        types = [seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False).action_type for _ in range(50)]
        self.assertEqual(first.action_type, "grasp")
        self.assertTrue(all(t == "reach" for t in types), types)

    def test_opening_edge_only_fires_place_while_holding(self):
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)  # grasp edge
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True)  # now genuinely holding
        a = seg.propose(OPEN, "cube_2", (0.6, -0.2, 0.05), is_holding=True)
        self.assertEqual(a.action_type, "place")
        self.assertEqual(a.params["target_position"], (0.6, -0.2, 0.05))

    def test_opening_without_ever_having_held_does_not_fire_place(self):
        # e.g. a failed grasp attempt that un-commands CLOSE without ever actually gripping
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)
        a = seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=False)
        self.assertEqual(a.action_type, "reach")

    def test_repeated_opening_after_the_edge_does_not_refire_place(self):
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True)
        first = seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=True)
        types = [seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=False).action_type for _ in range(20)]
        self.assertEqual(first.action_type, "place")
        self.assertTrue(all(t == "reach" for t in types), types)

    def test_steady_open_never_fires_grasp_or_place(self):
        seg = GripActionSegmenter()
        types = [seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=False).action_type for _ in range(10)]
        self.assertTrue(all(t == "reach" for t in types))


class DefensiveGuardTest(unittest.TestCase):
    """Guards beyond the minimum the validated Franka reference needed for its own well-behaved
    phase machine -- included because a general-purpose library shouldn't assume every policy is
    equally well-behaved. Neither guard can reintroduce the original deadlock: neither depends on
    any object-stability check, only on the caller's own is_holding signal."""

    def test_reclosing_while_already_holding_does_not_refire_grasp(self):
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)  # grasp
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True)
        seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=True)  # commanded open but grip oscillates
        a = seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True)  # re-closes while STILL genuinely holding
        self.assertEqual(a.action_type, "reach")

    def test_reclosing_after_a_genuine_release_fires_a_new_grasp(self):
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True)
        seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=True)  # place edge: genuinely released
        a = seg.propose(CLOSE, "cube_3", (1, 0, 0), is_holding=False)  # a new object, next task
        self.assertEqual(a.action_type, "grasp")
        self.assertEqual(a.params["object_id"], "cube_3")


class ParamsAndTypeOverrideTest(unittest.TestCase):
    def test_grasp_params_merged_only_into_grasp(self):
        seg = GripActionSegmenter()
        a = seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False, grasp_params={"grip_force_n": 5.0})
        self.assertEqual(a.params["grip_force_n"], 5.0)
        b = seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True, grasp_params={"grip_force_n": 99.0})
        self.assertNotIn("grip_force_n", b.params)  # this step is "reach", not "grasp" -- param not carried over

    def test_place_params_merged_only_into_place(self):
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=True)
        a = seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=True, place_params={"target_surface_id": "cube_1"})
        self.assertEqual(a.params["target_surface_id"], "cube_1")

    def test_custom_close_threshold_convention(self):
        # a 0..1 "closed fraction" convention instead of the default -1..1
        seg = GripActionSegmenter(is_closing=lambda g: g > 0.5)
        a = seg.propose(0.9, "cube_2", (0, 0, 0), is_holding=False)
        self.assertEqual(a.action_type, "grasp")
        b = seg.propose(0.1, "cube_2", (0, 0, 0), is_holding=False)  # never crossed 0.5 while holding
        self.assertEqual(b.action_type, "reach")

    def test_custom_reach_action_type(self):
        seg = GripActionSegmenter()
        a = seg.propose(OPEN, "cube_2", (0, 0, 0), is_holding=False, reach_action_type="approach")
        self.assertEqual(a.action_type, "approach")

    def test_propose_never_returns_none(self):
        seg = GripActionSegmenter()
        for g in (OPEN, CLOSE, OPEN, CLOSE, CLOSE, OPEN):
            self.assertIsInstance(seg.propose(g, "x", (0, 0, 0), is_holding=False), Action)


class ResetTest(unittest.TestCase):
    def test_reset_forgets_prior_state(self):
        seg = GripActionSegmenter()
        seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)  # grasp edge consumed
        seg.reset()
        a = seg.propose(CLOSE, "cube_2", (0, 0, 0), is_holding=False)  # a fresh episode's first step
        self.assertEqual(a.action_type, "grasp")  # not suppressed as a "repeat" of the pre-reset state


class FrankaClosedLoopReproductionTest(unittest.TestCase):
    """Reproduces the exact phase sequence the validated closed-loop measurement's own inline
    edge-detection produced (examples/closed_loop/closed_loop_gate.py), and asserts
    GripActionSegmenter yields the identical action-type sequence for it -- so the extraction is
    provably behavior-preserving relative to what was actually run on real hardware-in-the-loop
    simulation, not just independently "looks right" in isolation."""

    def test_grasp_lift_carry_place_retreat_sequence(self):
        seg = GripActionSegmenter()
        script = [
            # (grip_command, holding, expected_action_type) -- "above"/"descend": approaching, open
            (OPEN, False, "reach"), (OPEN, False, "reach"),
            # "grasp": fingers start closing
            (CLOSE, False, "grasp"),
            # still closing/settling, not yet holding by the caller's own contact signal
            (CLOSE, False, "reach"), (CLOSE, False, "reach"),
            # contact achieved -- now genuinely holding, still commanding closed while lifting/moving
            (CLOSE, True, "reach"), (CLOSE, True, "reach"), (CLOSE, True, "reach"),
            # "release": commanding open while still (for one last step) holding
            (OPEN, True, "place"),
            # "retreat": open, no longer holding
            (OPEN, False, "reach"), (OPEN, False, "reach"),
        ]
        got = [seg.propose(g, "cube_2", (0, 0, 0), is_holding=h).action_type for g, h, _ in script]
        expected = [t for _, _, t in script]
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
