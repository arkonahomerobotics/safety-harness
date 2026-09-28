"""live_costmap.py against a small hand-built message, shaped exactly like a real
nav_msgs/OccupancyGrid -- no rclpy/ROS install needed, same duck-typing convention as
tests/test_ros2_adapter.py uses for the ROS2 adapter.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_costmap import occupancy_grid_from_msg  # noqa: E402
from map_io import FREE, OCCUPIED, UNKNOWN  # noqa: E402


def _occupancy_grid_msg(width, height, data, resolution=0.1, origin_xy=(0.0, 0.0)):
    return SimpleNamespace(
        info=SimpleNamespace(
            resolution=resolution, width=width, height=height,
            origin=SimpleNamespace(position=SimpleNamespace(x=origin_xy[0], y=origin_xy[1])),
        ),
        data=data,
    )


class LiveCostmapTest(unittest.TestCase):
    def test_free_occupied_unknown_thresholds(self):
        # 1x4 grid: clearly free, ambiguous-but-below-occupied, clearly occupied, unknown
        msg = _occupancy_grid_msg(4, 1, [0, 50, 100, -1])
        grid = occupancy_grid_from_msg(msg)
        self.assertEqual(list(grid.cells[0]), [FREE, UNKNOWN, OCCUPIED, UNKNOWN])

    def test_resolution_and_origin_carried_through(self):
        msg = _occupancy_grid_msg(2, 2, [0, 0, 0, 0], resolution=0.05, origin_xy=(-1.5, 2.0))
        grid = occupancy_grid_from_msg(msg)
        self.assertEqual(grid.resolution_m, 0.05)
        self.assertEqual(grid.origin_xy, (-1.5, 2.0))

    def test_row_order_flipped_to_top_origin(self):
        # Message convention: row 0 = bottom. data[0]=bottom-left occupied, data[width]=top-left free.
        # After flipping to this tool's top-origin convention, cells[0] (top row) must be the FREE one.
        msg = _occupancy_grid_msg(1, 2, [100, 0])
        grid = occupancy_grid_from_msg(msg)
        self.assertEqual(grid.cells[0, 0], FREE)
        self.assertEqual(grid.cells[1, 0], OCCUPIED)

    def test_same_shape_as_static_loader_output(self):
        # The whole point: hazard_rules.py must not need to know or care which source built this.
        from map_io import OccupancyGrid
        msg = _occupancy_grid_msg(3, 3, [0] * 9)
        grid = occupancy_grid_from_msg(msg)
        self.assertIsInstance(grid, OccupancyGrid)


if __name__ == "__main__":
    unittest.main()
