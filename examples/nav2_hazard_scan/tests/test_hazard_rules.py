"""hazard_rules.py against small hand-built grids and graphs -- fast, deterministic, no real map
needed (that's scan_demo.py's job, against the real navigation2 warehouse map).
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hazard_rules import blind_corner_absent, corridor_width_sufficient, intersection_flagged_for_review  # noqa: E402
from map_io import FREE, OCCUPIED, OccupancyGrid  # noqa: E402


class CorridorWidthTest(unittest.TestCase):
    def test_wide_open_area_is_satisfied(self):
        cells = np.full((20, 20), FREE, dtype=np.uint8)
        grid = OccupancyGrid(cells=cells, resolution_m=0.1, origin_xy=(0.0, 0.0))
        route = [(10, 10)]
        findings = corridor_width_sufficient(grid, route, min_clearance_m=0.5)
        self.assertTrue(findings[0].satisfied)

    def test_narrow_gap_is_flagged(self):
        cells = np.full((5, 20), OCCUPIED, dtype=np.uint8)
        cells[2, :] = FREE  # a single-cell-wide corridor down the middle
        grid = OccupancyGrid(cells=cells, resolution_m=0.1, origin_xy=(0.0, 0.0))
        route = [(2, 10)]
        findings = corridor_width_sufficient(grid, route, min_clearance_m=0.5)
        self.assertFalse(findings[0].satisfied)
        self.assertIn("0.10m", findings[0].reason)  # clearance to the nearest obstacle, one cell = 0.1m


class BlindCornerTest(unittest.TestCase):
    def test_long_open_corridor_is_satisfied(self):
        # Tall enough that a +/-50 degree cone ray doesn't trivially run off the map's top/bottom
        # edge before it runs the required 3m/0.1m=30 cells -- a real map is never this thin, and
        # running off the mapped area correctly counts as "blocked" (default-deny on the unknown),
        # so a too-thin test grid would fail for a reason that has nothing to do with the rule itself.
        cells = np.full((81, 100), FREE, dtype=np.uint8)
        grid = OccupancyGrid(cells=cells, resolution_m=0.1, origin_xy=(0.0, 0.0))
        route = [(40, i) for i in range(0, 50, 5)]
        findings = blind_corner_absent(grid, route, required_sightline_m=3.0, sample_every=5)
        self.assertTrue(all(f.satisfied for f in findings), [(f.satisfied, f.reason) for f in findings])

    def test_forward_direction_continues_at_the_routes_own_end(self):
        # The route heads east then stops; a wall sits east of the endpoint. With the heading fix,
        # the endpoint's forward direction is still "east" (continuing the travel direction), so it
        # correctly sees the close wall -- not "west" (back the way it came, which is open), which
        # would wrongly read as satisfied.
        cells = np.full((21, 20), FREE, dtype=np.uint8)
        cells[:, 12] = OCCUPIED
        grid = OccupancyGrid(cells=cells, resolution_m=0.1, origin_xy=(0.0, 0.0))
        route = [(10, i) for i in range(0, 11, 2)]  # ends at (10, 10), wall at column 12
        findings = blind_corner_absent(grid, route, required_sightline_m=3.0, cone_half_angle_deg=5,
                                        n_rays=1, sample_every=len(route) - 1)
        self.assertFalse(findings[-1].satisfied, findings[-1].reason)

    def test_wall_right_ahead_is_flagged(self):
        cells = np.full((3, 20), FREE, dtype=np.uint8)
        cells[:, 5] = OCCUPIED  # a wall 5 cells ahead of the route's start
        grid = OccupancyGrid(cells=cells, resolution_m=0.1, origin_xy=(0.0, 0.0))
        route = [(1, 0), (1, 1), (1, 2)]  # heading east, straight toward the wall
        findings = blind_corner_absent(grid, route, required_sightline_m=3.0, cone_half_angle_deg=5,
                                        n_rays=1, sample_every=1)
        self.assertFalse(findings[0].satisfied)


class IntersectionTest(unittest.TestCase):
    def test_pass_through_node_not_flagged(self):
        nodes = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}
        edges = [("a", "b"), ("b", "c")]
        findings = intersection_flagged_for_review(nodes, edges)
        by_id = {f.location_world_xy: f for f in findings}
        self.assertTrue(by_id[(1.0, 0.0)].satisfied)  # b: degree 2, under the default min_degree=3

    def test_real_junction_is_flagged(self):
        nodes = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (1.0, 1.0), "j": (1.0, -1.0), "d": (2.0, -1.0)}
        edges = [("a", "j"), ("j", "b"), ("j", "c"), ("j", "d")]
        findings = intersection_flagged_for_review(nodes, edges)
        by_id = {f.location_world_xy: f for f in findings}
        self.assertFalse(by_id[(1.0, -1.0)].satisfied)  # j: degree 4
        self.assertIn("4 routes", by_id[(1.0, -1.0)].reason)

    def test_min_degree_is_configurable(self):
        nodes = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (1.0, 1.0)}
        edges = [("a", "b"), ("b", "c")]  # b has degree 2
        findings = intersection_flagged_for_review(nodes, edges, min_degree=2)
        by_id = {f.location_world_xy: f for f in findings}
        self.assertFalse(by_id[(1.0, 0.0)].satisfied)  # flagged now that min_degree=2


if __name__ == "__main__":
    unittest.main()
