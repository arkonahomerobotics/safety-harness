"""swept_path_observed with known solid geometry (WorldState.solid_regions, 0.3.1).

The motivating live finding: with realistic coverage (only the space above a tabletop observed),
swept_path_observed blocked ~98% of nominal near-table reaches in the G1 stacking test, because the
margin sphere around any near-table path dips under the tabletop -- space no camera sees, and no
person can be in. Declaring the table as known-solid fixes that without letting any genuinely
unobserved space through."""

import math
import os
import sys
import unittest

from safety_harness import preconditions as pc
from safety_harness.schema import (
    Action, KnownSolidRegion, ObservedRegion, PredictedTrajectory, TrajectoryPoint, WorldState,
)

sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402

TABLE_TOP = 0.70
ABOVE_TABLE = ObservedRegion(min_corner=(-1.0, -1.0, TABLE_TOP), max_corner=(1.0, 1.0, 2.0))
TABLE = KnownSolidRegion(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, TABLE_TOP))


def _traj(center, radius=0.08):
    return PredictedTrajectory(points=(TrajectoryPoint(t=0.0, robot=None, swept_volume_center=center,
                                                       swept_volume_radius_m=radius),), horizon_s=1.0)


def _check(observed, solids, center, margin=0.15):
    state = WorldState(observed_regions=observed, solid_regions=solids)
    return pc.swept_path_observed(state, Action("reach", {}), _traj(center), margin_m=margin)


class KnownSolidRegionTest(unittest.TestCase):
    NEAR_TABLE = (0.0, 0.3, TABLE_TOP + 0.04)  # a hand 4cm above the tabletop

    def test_near_table_path_blocks_without_solid_geometry(self):
        self.assertFalse(_check((ABOVE_TABLE,), None, self.NEAR_TABLE).satisfied)

    def test_near_table_path_permits_with_table_declared_solid(self):
        self.assertTrue(_check((ABOVE_TABLE,), (TABLE,), self.NEAR_TABLE).satisfied)

    def test_solid_does_not_cover_unobserved_space_beside_it(self):
        # the path's margin reaches past the table's edge into space that is neither observed nor solid
        edge = (0.95, 0.3, TABLE_TOP + 0.04)
        self.assertFalse(_check((ABOVE_TABLE,), (TABLE,), edge).satisfied)

    def test_solid_alone_is_not_observation(self):
        self.assertFalse(_check((), (TABLE,), self.NEAR_TABLE).satisfied)
        self.assertFalse(_check(None, (TABLE,), self.NEAR_TABLE).satisfied)

    def test_malformed_solid_regions_add_no_coverage(self):
        for bad in (KnownSolidRegion((math.nan, -1.0, 0.0), (1.0, 1.0, TABLE_TOP)),
                    KnownSolidRegion((1.0, 1.0, TABLE_TOP), (-1.0, -1.0, 0.0)),  # inverted
                    ObservedRegion((-1.0, -1.0, 0.0), (1.0, 1.0, TABLE_TOP)),  # wrong type: observation isn't solidity
                    "table"):
            self.assertFalse(_check((ABOVE_TABLE,), (bad,), self.NEAR_TABLE).satisfied, repr(bad))

    def test_nan_center_fails_closed_even_with_solids(self):
        self.assertFalse(_check((ABOVE_TABLE,), (TABLE,), (math.nan, 0.3, 0.74)).satisfied)

    def test_through_the_gate_near_table(self):
        state = fixtures.base_world_state(observed_regions=(ABOVE_TABLE,))
        from dataclasses import replace
        traj = _traj(self.NEAR_TABLE)
        without = pc.swept_path_observed(state, Action("reach", {}), traj)
        with_solid = pc.swept_path_observed(replace(state, solid_regions=(TABLE,)), Action("reach", {}), traj)
        self.assertFalse(without.satisfied)
        self.assertTrue(with_solid.satisfied)


class AabbCoveredTest(unittest.TestCase):
    """Direct tests of _aabb_covered, the exact-coverage algorithm behind every multi-region claim
    above (a shared face counts, a real gap doesn't). Everything in KnownSolidRegionTest exercises
    it only through a single observed+solid pair that happens to share a face by construction; none
    of it would catch a regression in the union logic itself (an L-shaped gap, two regions that
    overlap instead of merely touching, a query that straddles more than two boxes). This class
    tests the algorithm on its own, with no swept_path_observed or WorldState involved."""

    def test_two_boxes_sharing_a_face_cover_a_straddling_query(self):
        left = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        right = ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))  # touches `left` exactly at x=1, no overlap
        straddling = ((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        self.assertTrue(pc._aabb_covered(*straddling, [left, right]))

    def test_a_real_gap_between_two_boxes_is_not_covered(self):
        left = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        right = ((1.1, 0.0, 0.0), (2.1, 1.0, 1.0))  # 0.1 gap from `left`
        straddling = ((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        self.assertFalse(pc._aabb_covered(*straddling, [left, right]))

    def test_overlapping_boxes_cover_a_straddling_query(self):
        # regions that overlap (not just touch) must still union correctly, not double-count
        left = ((0.0, 0.0, 0.0), (1.2, 1.0, 1.0))
        right = ((0.8, 0.0, 0.0), (2.0, 1.0, 1.0))
        straddling = ((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))
        self.assertTrue(pc._aabb_covered(*straddling, [left, right]))

    def test_l_shaped_union_leaves_the_missing_corner_uncovered(self):
        # bottom bar + left bar form an L; the top-right cell of their bounding box is in neither
        bottom = ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        left = ((0.0, 0.0, 0.0), (1.0, 2.0, 1.0))
        corner_query = ((0.5, 0.5, 0.0), (1.5, 1.5, 1.0))  # reaches into the L's missing corner
        self.assertFalse(pc._aabb_covered(*corner_query, [bottom, left]))
        within_bottom_bar = ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        self.assertTrue(pc._aabb_covered(*within_bottom_bar, [bottom, left]))

    def test_no_boxes_never_covers(self):
        self.assertFalse(pc._aabb_covered((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), []))

    def test_degenerate_zero_size_query_still_checked_against_the_union(self):
        # a zero-radius sphere's bounding box collapses to a point -- must still fail outside coverage
        box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        inside_point = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        outside_point = ((5.0, 5.0, 5.0), (5.0, 5.0, 5.0))
        self.assertTrue(pc._aabb_covered(*inside_point, [box]))
        self.assertFalse(pc._aabb_covered(*outside_point, [box]))


class FiniteBoxTest(unittest.TestCase):
    """_finite_box: turns a region into a (lo, hi) float tuple, or None if it can't be trusted as
    coverage -- non-finite, wrong shape, or inverted (min > max, which would otherwise silently
    invert into a box covering everything OUTSIDE the intended region)."""

    def test_well_formed_region_becomes_float_tuple(self):
        self.assertEqual(pc._finite_box(TABLE), ((-1.0, -1.0, 0.0), (1.0, 1.0, TABLE_TOP)))

    def test_inverted_box_rejected(self):
        self.assertIsNone(pc._finite_box(KnownSolidRegion((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))))

    def test_nan_corner_rejected(self):
        self.assertIsNone(pc._finite_box(KnownSolidRegion((math.nan, 0.0, 0.0), (1.0, 1.0, 1.0))))

    def test_infinite_corner_rejected(self):
        self.assertIsNone(pc._finite_box(KnownSolidRegion((0.0, 0.0, 0.0), (math.inf, 1.0, 1.0))))

    def test_wrong_arity_rejected(self):
        self.assertIsNone(pc._finite_box(KnownSolidRegion((0.0, 0.0), (1.0, 1.0))))

    def test_a_box_exactly_at_its_own_bounds_is_well_formed(self):
        # lo == hi on every axis (a degenerate, zero-volume box) is not "inverted" -- a > b is
        # false when a == b, so this must be accepted, not rejected.
        self.assertEqual(pc._finite_box(KnownSolidRegion((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))),
                          ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
