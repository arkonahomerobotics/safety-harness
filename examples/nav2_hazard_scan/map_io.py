"""Loads a real ROS/Nav2 map: the standard <name>.yaml + <name>.pgm pair the map_server / Nav2 map
saver produces (unchanged format for well over a decade -- see
https://navigation.ros.org/configuration/packages/costmap-plugins/static.html and the map_server
YAML spec it links). No ROS install needed to read it: it's a plain YAML file plus a plain PGM image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image

FREE, OCCUPIED, UNKNOWN = 0, 1, 2


@dataclass(frozen=True)
class OccupancyGrid:
    cells: np.ndarray  # (rows, cols) of FREE/OCCUPIED/UNKNOWN, row 0 = the image's top row
    resolution_m: float  # meters per cell
    origin_xy: tuple  # (x, y) world coordinates of the BOTTOM-LEFT cell (the map_server convention)

    def world_to_cell(self, x: float, y: float):
        """World (x, y) meters -> (row, col). Row grows downward in `cells` but the map_server
        convention has y grow upward from origin -- row 0 is the map's northernmost row, i.e. the
        LAST row of the image as saved (PGM stores top row first, which is the map's north edge for
        a right-handed world frame with no rotation, the map_server's stated assumption)."""
        col = int(round((x - self.origin_xy[0]) / self.resolution_m))
        row_from_bottom = int(round((y - self.origin_xy[1]) / self.resolution_m))
        row = self.cells.shape[0] - 1 - row_from_bottom
        return row, col

    def cell_to_world(self, row: int, col: int):
        x = self.origin_xy[0] + col * self.resolution_m
        row_from_bottom = self.cells.shape[0] - 1 - row
        y = self.origin_xy[1] + row_from_bottom * self.resolution_m
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.cells.shape[0] and 0 <= col < self.cells.shape[1]


def load_occupancy_grid(yaml_path: str) -> OccupancyGrid:
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    image_path = meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)
    img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)  # 0..255, one channel

    negate = bool(meta.get("negate", 0))
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    # Standard map_server convention: occupancy probability from pixel intensity. negate=0 (the
    # common case): white (255) = free, black (0) = occupied, so probability-of-occupied = 1 -
    # pixel/255. negate=1 flips that.
    occ_prob = (img / 255.0) if negate else (1.0 - img / 255.0)

    cells = np.full(img.shape, UNKNOWN, dtype=np.uint8)
    cells[occ_prob <= free_thresh] = FREE
    cells[occ_prob >= occ_thresh] = OCCUPIED

    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    return OccupancyGrid(cells=cells, resolution_m=resolution, origin_xy=(float(origin[0]), float(origin[1])))
