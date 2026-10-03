"""Loads a real ROS/Nav2 map: the standard <name>.yaml + <name>.pgm pair the map_server / Nav2 map
saver produces (unchanged format for well over a decade -- see
https://navigation.ros.org/configuration/packages/costmap-plugins/static.html and the map_server
YAML spec it links). No ROS install needed to read it: it's a plain YAML file plus a plain PGM image.

Supports both documented occupancy modes. ``trinary`` (the default, and what the overwhelming
majority of real map_server YAML files use) classifies every pixel as FREE, OCCUPIED, or UNKNOWN
purely by threshold. ``scale`` keeps a continuously-scaled occupancy probability for pixels that
fall between the two thresholds instead of flatly calling them UNKNOWN -- this tool's own internal
grid is strictly trinary (geometry.py/hazard_rules.py need FREE/OCCUPIED/UNKNOWN cells, not a
continuous probability), so that scaled value is binarized at 50% here, a deliberate, stated
simplification, not a silent one. ``raw`` mode (pixel value used directly as the published
occupancy data, no thresholding) is rare in practice and its exact pixel semantics are
consumer-defined rather than standardized -- rather than guess at behavior for a mode this tool has
no real test map for, loading a ``mode: raw`` YAML raises ``MapLoadError`` with a clear message
instead of silently producing a questionable classification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image

FREE, OCCUPIED, UNKNOWN = 0, 1, 2
_SUPPORTED_MODES = ("trinary", "scale")


class MapLoadError(Exception):
    """Anything wrong with a map file or its YAML metadata -- missing file, missing/invalid key,
    unreadable image, an unsupported mode. Always a clear, specific message; callers (the CLI) are
    expected to catch this and print it without a traceback, since a malformed map file is routine
    user error, not a programming bug."""


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
    if not os.path.isfile(yaml_path):
        raise MapLoadError(f"map YAML not found: {yaml_path!r}")
    try:
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise MapLoadError(f"{yaml_path!r} is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise MapLoadError(f"{yaml_path!r} does not contain a YAML mapping (map_server metadata) at its top level")

    for key in ("image", "resolution"):
        if key not in meta:
            raise MapLoadError(f"{yaml_path!r} is missing the required map_server key {key!r}")

    image_path = meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)
    if not os.path.isfile(image_path):
        raise MapLoadError(f"{yaml_path!r} points at image {image_path!r}, which does not exist")
    try:
        img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)  # 0..255, one channel
    except Exception as exc:  # noqa: BLE001 -- Pillow's own exception types vary by failure mode
        raise MapLoadError(f"could not read {image_path!r} as an image: {exc}") from exc
    if img.size == 0:
        raise MapLoadError(f"{image_path!r} decoded to an empty image")

    try:
        resolution = float(meta["resolution"])
    except (TypeError, ValueError) as exc:
        raise MapLoadError(f"{yaml_path!r}'s resolution {meta['resolution']!r} is not a number") from exc
    if not (resolution > 0):
        raise MapLoadError(f"{yaml_path!r}'s resolution must be positive, got {resolution!r}")

    origin = meta.get("origin", [0.0, 0.0, 0.0])
    if not (isinstance(origin, (list, tuple)) and len(origin) >= 2):
        raise MapLoadError(f"{yaml_path!r}'s origin must be a list of at least [x, y], got {origin!r}")
    try:
        origin_xy = (float(origin[0]), float(origin[1]))
    except (TypeError, ValueError) as exc:
        raise MapLoadError(f"{yaml_path!r}'s origin {origin!r} is not numeric") from exc

    mode = meta.get("mode", "trinary")
    if mode not in _SUPPORTED_MODES:
        raise MapLoadError(
            f"{yaml_path!r} specifies mode {mode!r}, which this tool does not support "
            f"(supported: {', '.join(_SUPPORTED_MODES)}) -- 'raw' mode's pixel semantics are "
            f"consumer-defined rather than standardized, so this tool refuses to guess at them "
            f"rather than silently produce a questionable classification"
        )

    negate = meta.get("negate", 0)
    try:
        negate = bool(int(negate))
    except (TypeError, ValueError) as exc:
        raise MapLoadError(f"{yaml_path!r}'s negate {negate!r} is not interpretable as 0/1") from exc

    try:
        occ_thresh = float(meta.get("occupied_thresh", 0.65))
        free_thresh = float(meta.get("free_thresh", 0.196))
    except (TypeError, ValueError) as exc:
        raise MapLoadError(f"{yaml_path!r}'s occupied_thresh/free_thresh are not numeric: {exc}") from exc
    if not (0.0 <= free_thresh < occ_thresh <= 1.0):
        raise MapLoadError(
            f"{yaml_path!r}'s thresholds must satisfy 0 <= free_thresh < occupied_thresh <= 1, "
            f"got free_thresh={free_thresh!r} occupied_thresh={occ_thresh!r}"
        )

    # Standard map_server convention: occupancy probability from pixel intensity. negate=0 (the
    # common case): white (255) = free, black (0) = occupied, so probability-of-occupied = 1 -
    # pixel/255. negate=1 flips that.
    occ_prob = (img / 255.0) if negate else (1.0 - img / 255.0)

    cells = np.full(img.shape, UNKNOWN, dtype=np.uint8)
    cells[occ_prob <= free_thresh] = FREE
    cells[occ_prob >= occ_thresh] = OCCUPIED
    if mode == "scale":
        # The documented map_server "scale" behavior: pixels strictly between the two thresholds
        # get a continuously-scaled 0-100 probability instead of a flat UNKNOWN. This tool's grid
        # is trinary, so that continuous value is binarized at the 50% mark -- see the module
        # docstring for why that's a deliberate simplification, not a silent one.
        mid = (occ_prob > free_thresh) & (occ_prob < occ_thresh)
        scaled = (occ_prob[mid] - free_thresh) / (occ_thresh - free_thresh) * 100.0
        cells_mid = cells[mid]
        cells_mid[scaled >= 50.0] = OCCUPIED
        cells_mid[scaled < 50.0] = FREE
        cells[mid] = cells_mid

    return OccupancyGrid(cells=cells, resolution_m=resolution, origin_xy=origin_xy)
