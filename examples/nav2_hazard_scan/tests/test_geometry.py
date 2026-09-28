"""Pure-Python/numpy tests for geometry.py, against small hand-built grids (fast; the real
1.6MB warehouse map is exercised by scan_demo.py itself, not here).

Run with: python3 -m unittest discover -s examples/nav2_hazard_scan/tests -v
(from the repo root, with numpy and pillow installed -- see the example README)
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import astar, bresenham_cells, clearance_transform, sightline_cells  # noqa: E402
from map_io import FREE, OCCUPIED, UNKNOWN  # noqa: E402


class ClearanceTransformTest(unittest.TestCase):
    def test_single_obstacle_radial_distances(self):
        g = np.full((5, 5), FREE, dtype=np.uint8)
        g[2, 2] = OCCUPIED
        d = clearance_transform(g)
        self.assertEqual(d[2, 2], 0.0)
        self.assertAlmostEqual(d[1, 2], 1.0)
        self.assertAlmostEqual(d[2, 1], 1.0)
        self.assertAlmostEqual(d[1, 1], math.sqrt(2))
        self.assertAlmostEqual(d[0, 0], 2 * math.sqrt(2))

    def test_unknown_counts_as_obstacle_by_default(self):
        g = np.full((3, 3), FREE, dtype=np.uint8)
        g[1, 1] = UNKNOWN
        d = clearance_transform(g)
        self.assertEqual(d[1, 1], 0.0)

    def test_unknown_excluded_when_asked(self):
        g = np.full((3, 3), FREE, dtype=np.uint8)
        g[1, 1] = UNKNOWN
        d = clearance_transform(g, occupied_as_obstacle=False)
        self.assertTrue(np.isinf(d[1, 1]))

    def test_long_corridor_propagates_along_the_row(self):
        # a 1x21 free corridor with obstacles only at both ends -- the center cell's true distance
        # to the nearest obstacle is 10 cells, which only the within-row sequential sweep can reach.
        g = np.full((1, 21), FREE, dtype=np.uint8)
        g[0, 0] = OCCUPIED
        g[0, 20] = OCCUPIED
        d = clearance_transform(g)
        self.assertEqual(d[0, 10], 10.0)


class BresenhamAndSightlineTest(unittest.TestCase):
    def test_horizontal_line(self):
        self.assertEqual(bresenham_cells(0, 0, 0, 5), [(0, i) for i in range(6)])

    def test_diagonal_line(self):
        self.assertEqual(bresenham_cells(0, 0, 3, 3), [(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_sightline_stops_at_obstacle(self):
        g = np.full((1, 10), FREE, dtype=np.uint8)
        g[0, 5] = OCCUPIED
        hit = sightline_cells(g, (0, 0), angle_rad=0.0, max_range_cells=9)
        self.assertAlmostEqual(hit, 5.0)

    def test_sightline_reaches_max_range_when_clear(self):
        g = np.full((1, 10), FREE, dtype=np.uint8)
        hit = sightline_cells(g, (0, 0), angle_rad=0.0, max_range_cells=9)
        self.assertAlmostEqual(hit, 9.0)

    def test_unknown_blocks_sightline_too(self):
        g = np.full((1, 10), FREE, dtype=np.uint8)
        g[0, 3] = UNKNOWN
        hit = sightline_cells(g, (0, 0), angle_rad=0.0, max_range_cells=9)
        self.assertAlmostEqual(hit, 3.0)


class AStarTest(unittest.TestCase):
    def test_straight_line_when_clear(self):
        g = np.full((5, 5), FREE, dtype=np.uint8)
        path = astar(g, (0, 0), (0, 4))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 4))
        self.assertEqual(len(path), 5)

    def test_routes_around_a_wall(self):
        g = np.full((5, 5), FREE, dtype=np.uint8)
        g[1:4, 2] = OCCUPIED  # a wall splitting the grid, gap at row 0 and row 4
        path = astar(g, (2, 0), (2, 4))
        self.assertIsNotNone(path)
        for r, c in path:
            self.assertNotEqual(g[r, c], OCCUPIED)

    def test_no_path_returns_none(self):
        g = np.full((3, 3), FREE, dtype=np.uint8)
        g[:, 1] = OCCUPIED  # a solid wall, no gap
        self.assertIsNone(astar(g, (0, 0), (0, 2)))


if __name__ == "__main__":
    unittest.main()
