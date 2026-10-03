"""Runnable, self-checking demonstration AND a client-usable CLI: real hazard rules run against
any real Nav2 map (<name>.yaml + <name>.pgm) along a real A*-computed route.

    python3 scan_demo.py                                    # bundled warehouse map, self-check
    python3 scan_demo.py --map path/to/map.yaml --report out.md
    python3 scan_demo.py --map path/to/map.yaml --route-graph route.json --report out.md

By default (no --map), uses the bundled real navigation2 warehouse map (maps/, see
maps/README.md) and --self_check, matching this tool's original purpose: proving the rules fire
against real data, not just "ran without crashing" (exits non-zero if any rule produced zero
findings or zero flags against that known map).

**The intersection/junction rule is never run against a stand-in graph by default.** This public
map ships no real `nav2_route` graph, and a client report must never silently include a finding
from this tool's own toy demonstration graph -- see --demo_graph below if you want that graph
specifically for demonstrating the rule, and the README's "Design boundary"/"Honest limitations"
for why this distinction matters. Pass --route_graph with a real graph to actually evaluate it;
otherwise the report and console output say plainly that it was skipped, not silently omit it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np

from geometry import astar, clearance_transform
from hazard_rules import blind_corner_absent, corridor_width_sufficient, intersection_flagged_for_review
from map_io import FREE, MapLoadError, load_occupancy_grid
from report import RuleCoverage, render_markdown_report

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP = os.path.join(_HERE, "maps", "warehouse.yaml")

# This tool's own demonstration graph for the junction rule -- three corridors meeting at a real
# crossing, one simple pass-through. Only ever used behind --demo_graph, explicitly opted into;
# never a silent default for a real --map scan. See the module docstring.
_DEMO_GRAPH_NODES = {"a": (0.0, 0.0), "b": (5.0, 0.0), "c": (5.0, 5.0), "junction": (5.0, 2.5), "d": (10.0, 2.5)}
_DEMO_GRAPH_EDGES = [("a", "junction"), ("junction", "b"), ("junction", "c"), ("junction", "d")]


class RouteGraphLoadError(Exception):
    """A malformed --route_graph JSON file -- same "clear message, no traceback" contract as
    map_io.MapLoadError."""


def load_route_graph(path: str):
    """JSON: {"nodes": {id: [x, y], ...}, "edges": [[from_id, to_id], ...]} -- the same shape
    nav2_route's own graph model uses (directional edges; see hazard_rules.py's docstring)."""
    if not os.path.isfile(path):
        raise RouteGraphLoadError(f"route graph not found: {path!r}")
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RouteGraphLoadError(f"{path!r} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        raise RouteGraphLoadError(f"{path!r} must be a JSON object with 'nodes' and 'edges' keys")
    try:
        nodes = {str(k): (float(v[0]), float(v[1])) for k, v in data["nodes"].items()}
        edges = [(str(a), str(b)) for a, b in data["edges"]]
    except (TypeError, ValueError, IndexError) as exc:
        raise RouteGraphLoadError(
            f"{path!r}'s nodes must be {{id: [x, y]}} and edges must be [[from_id, to_id], ...]: {exc}"
        ) from exc
    for a, b in edges:
        for n in (a, b):
            if n not in nodes:
                raise RouteGraphLoadError(f"{path!r}'s edge references node {n!r}, which isn't in 'nodes'")
    return nodes, edges


def pick_far_apart_free_points(grid, tries: int = 400, seed: int = 1):
    free_idx = np.argwhere(grid.cells == FREE)
    if len(free_idx) == 0:
        raise ValueError("map has no FREE cells at all -- nothing to route across")
    rng = random.Random(seed)
    start = tuple(int(v) for v in free_idx[rng.randrange(len(free_idx))])
    best_goal, best_d = start, 0.0
    for _ in range(tries):
        cand = tuple(int(v) for v in free_idx[rng.randrange(len(free_idx))])
        d = math.hypot(cand[0] - start[0], cand[1] - start[1])
        if d > best_d:
            best_d, best_goal = d, cand
    return start, best_goal


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default=DEFAULT_MAP, help="path to a Nav2 map .yaml (companion .pgm alongside it)")
    p.add_argument("--report", default=None, help="write a Markdown report to this path")
    p.add_argument("--route_graph", default=None,
                    help="JSON route graph ({nodes: {id: [x,y]}, edges: [[from,to]]}) to evaluate the "
                         "junction rule against. Omit to skip that rule honestly (see module docstring).")
    p.add_argument("--demo_graph", action="store_true",
                    help="use this tool's own small demonstration graph for the junction rule, instead of "
                         "--route_graph -- for demonstrating the rule only; never implied by default, and "
                         "refuses to combine with --route_graph.")
    p.add_argument("--min_clearance_m", type=float, default=0.75)
    p.add_argument("--required_sightline_m", type=float, default=3.0)
    p.add_argument("--junction_min_degree", type=int, default=3)
    p.add_argument("--seed", type=int, default=1, help="seed for picking the demo start/goal route points")
    p.add_argument("--self_check", action="store_true",
                    help="exit non-zero unless every EVALUATED rule produced both an ok and a flagged "
                         "finding -- this tool's own regression check against known data, not a meaningful "
                         "pass/fail criterion for an arbitrary client map (which may legitimately have zero "
                         "hazards of some kind). Implied when --map is left at its bundled default.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo_graph and args.route_graph:
        print("error: --demo_graph and --route_graph are mutually exclusive", file=sys.stderr)
        return 2
    self_check = args.self_check or (args.map == DEFAULT_MAP and args.route_graph is None and not args.demo_graph)

    try:
        grid = load_occupancy_grid(args.map)
    except MapLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Loaded {grid.cells.shape[1]}x{grid.cells.shape[0]} cells @ {grid.resolution_m}m/cell "
          f"(~{grid.cells.shape[1]*grid.resolution_m:.0f}m x {grid.cells.shape[0]*grid.resolution_m:.0f}m)")

    try:
        start, goal = pick_far_apart_free_points(grid, seed=args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    route = astar(grid.cells, start, goal)
    if route is None:
        print("FAIL: no route found between chosen points", file=sys.stderr)
        return 1
    print(f"Route: {len(route)} cells, {start} -> {goal}")

    params = {
        "min_clearance_m": args.min_clearance_m,
        "required_sightline_m": args.required_sightline_m,
        "junction_min_degree": args.junction_min_degree,
    }
    clearance = clearance_transform(grid.cells)
    corridor_findings = corridor_width_sufficient(grid, route, min_clearance_m=params["min_clearance_m"], clearance_cells=clearance)
    sightline_findings = blind_corner_absent(grid, route, required_sightline_m=params["required_sightline_m"])

    coverage = [
        RuleCoverage(name="corridor_width_sufficient", evaluated=True, findings=tuple(corridor_findings)),
        RuleCoverage(name="blind_corner_absent", evaluated=True, findings=tuple(sightline_findings)),
    ]

    if args.route_graph:
        try:
            nodes, edges = load_route_graph(args.route_graph)
        except RouteGraphLoadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        intersection_findings = intersection_flagged_for_review(nodes, edges, min_degree=params["junction_min_degree"])
        coverage.append(RuleCoverage(name="intersection_flagged_for_review", evaluated=True, findings=tuple(intersection_findings)))
    elif args.demo_graph:
        intersection_findings = intersection_flagged_for_review(_DEMO_GRAPH_NODES, _DEMO_GRAPH_EDGES, min_degree=params["junction_min_degree"])
        coverage.append(RuleCoverage(name="intersection_flagged_for_review", evaluated=True, findings=tuple(intersection_findings)))
        print("Junction review: evaluated against this tool's own DEMONSTRATION graph, not a real route graph.")
    else:
        skip_reason = "no route graph supplied (--route_graph)"
        coverage.append(RuleCoverage(name="intersection_flagged_for_review", evaluated=False, skip_reason=skip_reason))
        print(f"Junction review: not evaluated, {skip_reason}")

    for rc in coverage:
        if not rc.evaluated:
            continue
        for f in rc.findings:
            tag = "OK   " if f.satisfied else "FLAG "
            x, y = f.location_world_xy
            print(f"{tag}{f.name:32s} @ ({x:6.2f}, {y:6.2f})  {f.reason}")

    print("\nSummary:")
    for rc in coverage:
        if not rc.evaluated:
            print(f"  {rc.name}: skipped ({rc.skip_reason})")
            continue
        ok = sum(1 for f in rc.findings if f.satisfied)
        flagged = len(rc.findings) - ok
        print(f"  {rc.name}: {ok} ok, {flagged} flagged")

    if args.report:
        report_text = render_markdown_report(
            map_path=args.map, map_shape=grid.cells.shape, resolution_m=grid.resolution_m,
            origin_xy=grid.origin_xy, route_description=f"{len(route)} cells, {start} -> {goal} (A*, seed {args.seed})",
            coverage=coverage, params=params,
        )
        with open(args.report, "w") as f:
            f.write(report_text)
        print(f"\nWrote report: {args.report}")

    if self_check:
        evaluated = [rc for rc in coverage if rc.evaluated]
        ok = len(evaluated) > 0 and all(
            any(f.satisfied for f in rc.findings) and any(not f.satisfied for f in rc.findings) for rc in evaluated
        )
        print("PASS: every evaluated rule fired against real data (ok + at least one real flag each)" if ok else
              "FAIL: at least one evaluated rule produced no findings, or never flagged anything")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
