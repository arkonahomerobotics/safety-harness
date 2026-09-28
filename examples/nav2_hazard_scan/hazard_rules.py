"""ISO 3691-4-oriented hazard rules against a real Nav2 map -- the "someone spends 2-3 days
manually going through the Nav2 map flagging hazards" step from the ROS Discourse thread that
prompted this proof-of-concept (see the example README for the full story and honest scope).

Each rule returns a list of ``HazardFinding``, deliberately shaped like this project's own
``PreconditionResult`` (name, satisfied, reason) plus a map location -- the same "named check,
explicit pass/fail, plain-language reason" pattern as safety_harness/preconditions.py, applied once
at map/deployment time instead of continuously against live perception. This module does not import
safety_harness and is not wired through ActuatorGate: a static map scan is a genuinely different
data shape (no Action being proposed, no live WorldState) from what ActuatorGate.gate() checks, so
reusing that machinery directly would be a forced fit, not a real one -- this is a proof-of-concept
tool that borrows the pattern, not a new precondition check registered on the runtime gate.

Every threshold below (min_clearance_m, required_sightline_m, junction degree) is a stated,
adjustable parameter, not a hidden constant -- same "disclosed, not standards-certified" spirit as
the ISO/TS 15066 placeholder figures elsewhere in this project. ISO 3691-4 itself does not publish a
single universal clearance or sightline number; a real deployment sets these from its own risk
assessment (rated speed, stopping distance, aisle traffic), which this tool does not do for you.

DESIGN BOUNDARY (see the example README's own section on this): this module reads a map ONCE and
returns findings for a human to review before deployment. It must never be wired into a live control
or path-planning loop that re-evaluates continuously and acts on its own findings -- that crosses
from a pre-deployment analysis tool into the same "safety component" classification territory
safety_harness itself sits in (EU Machinery Regulation Art. 3(3)), which is a real, deliberate design
line, not an implementation detail to casually cross for convenience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geometry import astar, bresenham_cells, clearance_transform, sightline_cells  # noqa: F401
from map_io import FREE, OccupancyGrid


@dataclass(frozen=True)
class HazardFinding:
    name: str
    satisfied: bool  # False = flagged for review, same polarity as PreconditionResult
    reason: str
    location_world_xy: tuple


def _route_headings(route_cells, lookahead: int = 3):
    """Local direction of travel (radians, image (row,col) convention -- see
    geometry.sightline_cells's own docstring) at each route point, from a few cells ahead. Near the
    route's own end, where there's nothing ahead to look at, continues the direction already being
    traveled (an earlier point -> here) rather than reversing it -- a route's endpoint is often near
    a wall or dock precisely because that's the destination, so getting its forward direction
    backward would check sightline the wrong way at exactly the point most likely to matter."""
    n = len(route_cells)
    headings = []
    for i in range(n):
        if i + lookahead < n:
            j0, j1 = i, i + lookahead
        else:
            j0, j1 = max(i - lookahead, 0), i
        r0, c0 = route_cells[j0]
        r1, c1 = route_cells[j1]
        headings.append(math.atan2(-(r1 - r0), c1 - c0))  # -(dr) because "up the image" = +y = 0 rad's sin
    return headings


def corridor_width_sufficient(grid: OccupancyGrid, route_cells, min_clearance_m: float = 0.75,
                               sample_every: int = 10, clearance_cells: np.ndarray = None):
    """Flags any sampled route point whose distance to the nearest obstacle is under
    ``min_clearance_m``. This is clearance FROM the route point, not full corridor width -- a
    materially simpler, still-useful proxy that doesn't assume the route runs down the exact center
    of its corridor (see the module docstring on why an assumption like that isn't made silently).
    """
    if clearance_cells is None:
        clearance_cells = clearance_transform(grid.cells)
    findings = []
    for i in range(0, len(route_cells), sample_every):
        r, c = route_cells[i]
        clearance_m = float(clearance_cells[r, c]) * grid.resolution_m
        satisfied = clearance_m >= min_clearance_m
        findings.append(HazardFinding(
            name="corridor_width_sufficient",
            satisfied=satisfied,
            reason=(f"{clearance_m:.2f}m clearance to the nearest obstacle" if satisfied else
                    f"only {clearance_m:.2f}m clearance to the nearest obstacle, below the {min_clearance_m}m minimum"),
            location_world_xy=grid.cell_to_world(r, c),
        ))
    return findings


def blind_corner_absent(grid: OccupancyGrid, route_cells, required_sightline_m: float = 3.0,
                         cone_half_angle_deg: float = 50.0, n_rays: int = 7,
                         max_range_m: float = 15.0, sample_every: int = 10):
    """Flags any sampled route point where the clearest sightline within a forward cone (the
    direction of travel, +/- ``cone_half_angle_deg``) is shorter than ``required_sightline_m`` --
    real ray-casting against the real occupied/unknown cells, not a synthetic stand-in. A short
    sightline in the direction of travel is exactly what "blind corner" means operationally: by the
    time the vehicle (or a crossing pedestrian) is visible, there may not be room left to stop.
    """
    headings = _route_headings(route_cells)
    max_range_cells = max_range_m / grid.resolution_m
    findings = []
    for i in range(0, len(route_cells), sample_every):
        r, c = route_cells[i]
        heading = headings[i]
        hits_m = []
        for k in range(n_rays):
            frac = (k / (n_rays - 1)) - 0.5 if n_rays > 1 else 0.0
            angle = heading + frac * 2 * math.radians(cone_half_angle_deg)
            hit_cells = sightline_cells(grid.cells, (r, c), angle, max_range_cells)
            hits_m.append(hit_cells * grid.resolution_m)
        worst = min(hits_m)
        satisfied = worst >= required_sightline_m
        findings.append(HazardFinding(
            name="blind_corner_absent",
            satisfied=satisfied,
            reason=(f"clearest forward sightline {worst:.2f}m" if satisfied else
                    f"clearest forward sightline only {worst:.2f}m, below the {required_sightline_m}m required stopping sightline"),
            location_world_xy=grid.cell_to_world(r, c),
        ))
    return findings


def intersection_flagged_for_review(nodes: dict, edges: list, min_degree: int = 3):
    """``nodes``: {node_id: (x, y)}. ``edges``: [(from_id, to_id), ...], directional (mirroring
    nav2_route's own graph model -- see that package's docs). Flags any node whose total
    (in + out) degree is at least ``min_degree`` -- a real crossing point where two or more travel
    directions meet, exactly the kind of location ISO 3691-4's zone-classification exercise cares
    about (does it need a personnel-detection zone, a speed reduction, a mirror, right-of-way rules).
    Unlike the two geometric rules above, this is exact graph analysis, not a sampled approximation.
    """
    degree = {n: 0 for n in nodes}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    findings = []
    for node_id, deg in degree.items():
        satisfied = deg < min_degree
        findings.append(HazardFinding(
            name="intersection_flagged_for_review",
            satisfied=satisfied,
            reason=(f"{deg} route(s) meet here" if satisfied else
                    f"{deg} routes meet here -- a real junction, needs a personnel-detection/speed-reduction review"),
            location_world_xy=nodes[node_id],
        ))
    return findings
