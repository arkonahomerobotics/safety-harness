# Nav2 hazard scan: automating the "2-3 days per site" part of an ISO 3691-4 risk assessment

A proof-of-concept, not a pivot. Pure Python + numpy + Pillow, no ROS install needed to run it (it
reads a plain map file), no GPU, no physical robot.

## Why this exists

A [ROS Discourse thread](https://discourse.openrobotics.org/t/iso-risk-assessment/55420) (poster
"Sleem45", asking how others handle ISO 3691-4 risk assessments before AMR deployment): "Every site
to deploy to, someone spends 2-3 days manually going through the Nav2 map flagging hazards and
writing them up against the standard. Feels like it should be automatable." Separately, a real ROS
Discourse moderator (a real OSRA TGC member) had already pushed back on an earlier `safety-harness`
post as unvalidated and told us to build something the community has actually asked for — this is
that.

**Where this sits relative to `safety_harness`:** this is the same "named check, explicit pass/fail,
default-deny on missing evidence" pattern the core engine uses (`safety_harness/preconditions.py`,
`PreconditionResult`), applied once at map/deployment time instead of continuously against live
perception. It is **not** wired through `ActuatorGate` — a static map scan has no `Action` being
proposed and no live `WorldState`, so forcing it through the runtime-gating machinery would be a
contrived fit, not a real one. This module doesn't even import `safety_harness`. It borrows the
pattern; it isn't a new precondition check on the live gate.

**Competitive landscape, checked honestly, not assumed:** [`jherrodthomas/robotics-skills-suite`](https://github.com/jherrodthomas/robotics-skills-suite)
(245 stars, active) already has an ISO 3691-4 "risk assessment builder," but it generates the
*documentation* (audit-ready spreadsheets from JSON inputs) — not automated hazard detection from
map geometry. That's the "writing it up" half of the 2-3 days. This tool targets the other half:
"going through the map" itself.

## What's here

- **`map_io.py`**: loads a real ROS/Nav2 map — the standard `<name>.yaml` + `<name>.pgm` pair the
  map_server / Nav2 map saver produces (a format essentially unchanged for over a decade). No ROS
  install needed; it's a plain YAML file and a plain PGM image.
- **`geometry.py`**: real algorithms, not stand-ins — a two-pass chamfer distance transform (grid
  clearance-to-nearest-obstacle), Bresenham ray casting (line-of-sight), and a real 8-connected A*
  (used to get an honest, actually-computed route to scan, in place of a real `nav2_route` export
  this public map doesn't ship with).
- **`hazard_rules.py`**: three rules, each returning `HazardFinding` (name, satisfied, reason,
  location) — the same shape as `PreconditionResult`:
  - `corridor_width_sufficient`: flags a route point whose clearance to the nearest obstacle is
    under a configurable minimum.
  - `blind_corner_absent`: real ray-casting in a forward cone around the route's local direction of
    travel; flags where the clearest sightline is shorter than a configurable required stopping
    distance.
  - `intersection_flagged_for_review`: exact graph analysis (not sampled) — flags any route-graph
    node where 3+ directions meet, a real crossing point ISO 3691-4's zone classification cares
    about.
- **`scan_demo.py`**: loads the real warehouse map (`maps/`, from the `navigation2` project's own
  example maps — see `maps/README.md`), computes a real A* route across it, runs the two geometric
  rules along that route, and runs the graph rule against a small representative route-graph example
  (this public map doesn't ship a `nav2_route` graph — see "Honest limitations" below).
- **`tests/`**: fast unit tests against small hand-built grids/graphs (`python3 -m unittest discover
  -s examples/nav2_hazard_scan/tests -v`, from the repo root, with `numpy`/`pillow` installed).

## Run it

```bash
pip install numpy pillow pyyaml
python3 examples/nav2_hazard_scan/scan_demo.py
```

## Results (2026-09-28, against the real `navigation2` warehouse map)

```
corridor_width_sufficient: 83 ok, 52 flagged
blind_corner_absent: 41 ok, 94 flagged
intersection_flagged_for_review: 4 ok, 1 flagged
PASS: every rule fired against real data (ok + at least one real flag each)
```

All three rules produce real findings against real map geometry, not synthetic pass-throughs. The
`blind_corner_absent` flag rate looks high at first glance — worth explaining honestly rather than
leaving it looking alarming: the route scanned is a raw shortest-path A*, which (like any shortest
path) often hugs walls and corners tightly, exactly where sightlines are naturally short. A real
AMR's planned route usually carries its own clearance margin from the path planner; this demo
doesn't add one, so it's a legitimately more conservative (stricter) test than a deployed route would
see, not a sign the rule itself is miscalibrated. Found and fixed one real bug while building this:
the route's own endpoint was computing its forward direction backward (toward where it came from,
not where it was heading) — caught by testing against a wall placed deliberately at a route's end,
not caught by the happier all-clear cases. See `hazard_rules.py`'s `_route_headings` docstring.

## Honest limitations, stated plainly

- **One scenario against one real map, not a validated campaign.** This proves the approach is
  tractable and the three rules work against real data; it is not a validated tool for a real site
  assessment yet.
- **No real `nav2_route` graph for this map.** `intersection_flagged_for_review` is demonstrated
  against a small, clearly-labeled representative graph in `scan_demo.py`, not this map's own real
  route topology (it doesn't ship one). Fully tested against real graph shapes in `tests/`.
- **Ramps/inclines are out of scope.** A plain 2D occupancy grid carries no elevation data; detecting
  slope hazards needs a supplementary elevation source this tool doesn't have.
- **Semantic zones (doors, charging stations, load-transfer stations) aren't inferable from raw
  geometry alone.** A real deployment would need either manual zone-tagging or a richer map source —
  not attempted here.
- **Every threshold is a stated, adjustable parameter** (`min_clearance_m`, `required_sightline_m`,
  junction `min_degree`), not a certified number. ISO 3691-4 itself doesn't publish one universal
  figure for any of these — a real deployment sets them from its own risk assessment (rated speed,
  stopping distance, actual traffic), which this tool does not do for you. Same "disclosed, not
  standards-certified" spirit as the ISO/TS 15066 placeholder figures elsewhere in this project.
- **Not reviewed by a functional-safety or industrial-safety assessor.** Engineering proof-of-concept
  evidence, not a certified or audit-ready risk assessment.
