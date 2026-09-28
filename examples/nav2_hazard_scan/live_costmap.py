"""Converts one real, live `nav_msgs/OccupancyGrid` message (a running Nav2 costmap topic) into the
same `map_io.OccupancyGrid` every hazard rule in this tool already operates on -- so a live costmap
and a downloaded `.yaml`/`.pgm` map are interchangeable inputs to the exact same rules.

DESIGN BOUNDARY (see the example README and hazard_rules.py's own docstring): this reads the topic
ONCE -- a single message, converted to a report -- the same "analyze once, human reviews" shape as
the static-file path. It is not a subscription that keeps re-evaluating; nothing here loops or feeds
a decision back into navigation. Building that would be a real classification-relevant design
change (EU Machinery Regulation Art. 3(3)), not a convenience feature to add casually -- don't.
"""

from __future__ import annotations

import numpy as np

from map_io import FREE, OCCUPIED, UNKNOWN, OccupancyGrid


def occupancy_grid_from_msg(msg, occupied_thresh: int = 65, free_thresh: int = 25) -> OccupancyGrid:
    """``msg``: a real `nav_msgs/OccupancyGrid` (or anything duck-typed the same way -- `.info.
    resolution`, `.info.width`, `.info.height`, `.info.origin.position.{x,y}`, `.data`). Values in
    ``.data`` are 0-100 (probability of occupancy) or -1 (unknown), per the standard ROS convention
    -- not a scale this tool invented. A cell in the ambiguous middle (neither clearly free nor
    clearly occupied by the given thresholds) reads as UNKNOWN, not interpolated to something in
    between -- same default-deny spirit as the static-map loader's own trinary interpretation.

    ``.data`` is row-major from the grid's origin (bottom-left, row 0 = bottom) per the message's
    own documented convention -- flipped here to row 0 = top, matching `map_io.load_occupancy_grid`
    and keeping `OccupancyGrid.cell_to_world`/`world_to_cell` correct for either source unmodified.
    """
    width, height = msg.info.width, msg.info.height
    data = np.array(msg.data, dtype=np.int16).reshape((height, width))
    data = np.flipud(data)  # bottom-left-origin -> top-left-origin, matching the PGM loader

    cells = np.full((height, width), UNKNOWN, dtype=np.uint8)
    cells[(data >= 0) & (data <= free_thresh)] = FREE
    cells[data >= occupied_thresh] = OCCUPIED

    origin = msg.info.origin.position
    return OccupancyGrid(cells=cells, resolution_m=float(msg.info.resolution), origin_xy=(float(origin.x), float(origin.y)))
