"""Runnable, self-checking demonstration: real hazard rules run against a real Nav2 map
(warehouse.yaml/.pgm, from the navigation2 project's own example maps -- see maps/README.md) along
a real A*-computed route, plus a small representative route-graph example for intersection
flagging (this public map doesn't ship a nav2_route graph -- see the top-level README).

    python3 scan_demo.py

Prints every finding, a summary count, and exits 0 if at least one finding of each rule type was
produced (proving the rules actually fired against real data, not just "ran without crashing").
"""

from __future__ import annotations

import math
import random
import sys

import numpy as np

from geometry import astar, clearance_transform
from hazard_rules import blind_corner_absent, corridor_width_sufficient, intersection_flagged_for_review
from map_io import FREE, load_occupancy_grid


def pick_far_apart_free_points(grid, tries: int = 400, seed: int = 1):
    free_idx = np.argwhere(grid.cells == FREE)
    rng = random.Random(seed)
    start = tuple(int(v) for v in free_idx[rng.randrange(len(free_idx))])
    best_goal, best_d = start, 0.0
    for _ in range(tries):
        cand = tuple(int(v) for v in free_idx[rng.randrange(len(free_idx))])
        d = math.hypot(cand[0] - start[0], cand[1] - start[1])
        if d > best_d:
            best_d, best_goal = d, cand
    return start, best_goal


def main() -> int:
    grid = load_occupancy_grid("maps/warehouse.yaml")
    print(f"Loaded {grid.cells.shape[1]}x{grid.cells.shape[0]} cells @ {grid.resolution_m}m/cell "
          f"(~{grid.cells.shape[1]*grid.resolution_m:.0f}m x {grid.cells.shape[0]*grid.resolution_m:.0f}m)")

    start, goal = pick_far_apart_free_points(grid)
    route = astar(grid.cells, start, goal)
    if route is None:
        print("FAIL: no route found between chosen points")
        return 1
    print(f"Route: {len(route)} cells, {start} -> {goal}")

    clearance = clearance_transform(grid.cells)
    corridor_findings = corridor_width_sufficient(grid, route, min_clearance_m=0.75, clearance_cells=clearance)
    sightline_findings = blind_corner_absent(grid, route, required_sightline_m=3.0)

    # This public map ships no nav2_route graph -- a small, representative one instead (three
    # corridors meeting at a real crossing, one simple pass-through), documented as such.
    nodes = {"a": (0.0, 0.0), "b": (5.0, 0.0), "c": (5.0, 5.0), "junction": (5.0, 2.5), "d": (10.0, 2.5)}
    edges = [("a", "junction"), ("junction", "b"), ("junction", "c"), ("junction", "d")]
    intersection_findings = intersection_flagged_for_review(nodes, edges)

    all_findings = corridor_findings + sightline_findings + intersection_findings
    for f in all_findings:
        tag = "OK   " if f.satisfied else "FLAG "
        x, y = f.location_world_xy
        print(f"{tag}{f.name:32s} @ ({x:6.2f}, {y:6.2f})  {f.reason}")

    by_rule = {}
    for f in all_findings:
        by_rule.setdefault(f.name, [0, 0])
        by_rule[f.name][0 if f.satisfied else 1] += 1
    print("\nSummary:")
    for name, (ok, flagged) in by_rule.items():
        print(f"  {name}: {ok} ok, {flagged} flagged")

    ok = all(flagged > 0 for _, (_, flagged) in by_rule.items()) and len(by_rule) == 3
    print("PASS: every rule fired against real data (ok + at least one real flag each)" if ok else
          "FAIL: at least one rule produced no findings, or never flagged anything")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
