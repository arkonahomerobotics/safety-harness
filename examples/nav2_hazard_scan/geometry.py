"""Grid geometry primitives, all operating on a `map_io.OccupancyGrid`'s plain numpy `cells` array
-- no scipy, kept to numpy so this example needs only one extra dependency beyond the standard
library, on top of what map_io.py already needs to read a PGM.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from map_io import FREE, OCCUPIED, UNKNOWN

SQRT2 = math.sqrt(2.0)


def clearance_transform(cells: np.ndarray, occupied_as_obstacle: bool = True) -> np.ndarray:
    """Distance (in CELLS, not meters -- multiply by resolution_m for real distance) from every
    cell to the nearest OCCUPIED cell. UNKNOWN cells count as obstacles too when
    ``occupied_as_obstacle`` is True (the default, and the conservative choice: unmapped space
    earns no clearance credit, same default-deny spirit as the rest of this project) -- pass False
    only if a caller has independently confirmed unknown space is actually clear.

    Two-pass raster chamfer distance transform (real-valued 1 / sqrt(2) weights): the standard,
    well-known sequential algorithm -- each pass visits every cell in raster order and relaxes it
    against the neighbors already visited that pass. That makes the within-row step (a cell's
    left neighbor on the forward pass, right neighbor on the backward pass) a genuine sequential
    dependency -- a long free corridor's distance value has to propagate one cell at a time along
    it -- so only that inner step is a plain Python loop; the three neighbors from the row already
    fully computed (up to 3 of them per pass) are vectorized with numpy. Approximates true Euclidean
    distance to within a few percent; not claimed to be exact.
    """
    obstacle = (cells == OCCUPIED) | (occupied_as_obstacle & (cells == UNKNOWN))
    d = np.where(obstacle, 0.0, np.inf)
    rows, cols = d.shape

    def sweep_left_to_right(row):
        for c in range(1, cols):
            v = row[c - 1] + 1.0
            if v < row[c]:
                row[c] = v

    def sweep_right_to_left(row):
        for c in range(cols - 2, -1, -1):
            v = row[c + 1] + 1.0
            if v < row[c]:
                row[c] = v

    for r in range(rows):  # forward pass: top-left -> bottom-right
        row = d[r]
        if r > 0:
            prev = d[r - 1]
            np.minimum(row, prev + 1.0, out=row)
            row[1:] = np.minimum(row[1:], prev[:-1] + SQRT2)
            row[:-1] = np.minimum(row[:-1], prev[1:] + SQRT2)
        sweep_left_to_right(row)
    for r in range(rows - 1, -1, -1):  # backward pass: bottom-right -> top-left
        row = d[r]
        if r < rows - 1:
            nxt = d[r + 1]
            np.minimum(row, nxt + 1.0, out=row)
            row[1:] = np.minimum(row[1:], nxt[:-1] + SQRT2)
            row[:-1] = np.minimum(row[:-1], nxt[1:] + SQRT2)
        sweep_right_to_left(row)
    return d


def bresenham_cells(r0: int, c0: int, r1: int, c1: int):
    """Integer grid cells from (r0,c0) to (r1,c1) inclusive, Bresenham's line algorithm -- the
    standard way to sample a straight ray across a grid for a line-of-sight check."""
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def sightline_cells(cells: np.ndarray, origin, angle_rad: float, max_range_cells: float) -> float:
    """Casts one ray from ``origin`` (row, col) at ``angle_rad`` (0 = along +col/east, increasing
    counter-clockwise in (row,col) image space, i.e. toward -row = "up" the image) out to
    ``max_range_cells``, stopping at the first OCCUPIED or UNKNOWN cell (unmapped space blocks
    sight, same default-deny reasoning as clearance_transform). Returns the distance in cells to
    that stop, or max_range_cells if nothing blocks the ray that far."""
    r0, c0 = origin
    r1 = r0 - max_range_cells * math.sin(angle_rad)
    c1 = c0 + max_range_cells * math.cos(angle_rad)
    for r, c in bresenham_cells(r0, c0, int(round(r1)), int(round(c1))):
        if not (0 <= r < cells.shape[0] and 0 <= c < cells.shape[1]):
            return math.hypot(r - r0, c - c0)  # ran off the mapped area -- treat the edge as the block
        if cells[r, c] != FREE:
            return math.hypot(r - r0, c - c0)
    return max_range_cells


def astar(cells: np.ndarray, start, goal):
    """A real grid A* (8-connected, real per-step cost, not a straight-line stand-in) from
    ``start`` to ``goal`` (row, col), through FREE cells only. Returns the path as a list of
    (row, col), inclusive of both ends, or None if no path exists. Used only to get an honest,
    genuinely-computed route to run the hazard rules along, in place of a real nav2_route export
    this public map doesn't ship with -- see the example README."""
    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = [(h(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return list(reversed(path))
        for dr, dc, cost in neighbors:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < cells.shape[0] and 0 <= nb[1] < cells.shape[1]):
                continue
            if cells[nb] != FREE:
                continue
            ng = g + cost
            if ng < g_score.get(nb, math.inf):
                g_score[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb, goal), ng, nb))
    return None
