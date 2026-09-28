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


if __name__ == "__main__":
    unittest.main()
