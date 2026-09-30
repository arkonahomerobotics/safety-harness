# Nav2 hazard scan: automating the "2-3 days per site" part of an ISO 3691-4 risk assessment

A proof-of-concept, not a pivot. Pure Python + numpy + Pillow, no ROS install needed to run it (it
reads a plain map file), no GPU, no physical robot.

## Design boundary — read before extending this

**This is a pre-deployment analysis tool: read a static map once, produce a report, a human reviews
it before deployment. It is not, and must not become, a live/runtime system that re-evaluates
continuously and feeds decisions back into a robot's path planning.** That boundary is deliberate,
not an accident of the current implementation, and it's the reason this stays a standalone tool
instead of a `safety_harness` precondition check (see "Where this sits" below): a tool that analyzes
a static map offline and hands a human a report sits outside "safety component" classification under
the EU Machinery Regulation (Art. 3(3)) — closer to a CAD safety-check plugin than a runtime safety
system. The moment something like this re-flags hazards live and feeds directly into path-planning
decisions, it crosses into the same classification territory `safety_harness` itself already sits
in, with everything that implies (see the design doc's
[Development Roadmap](../../docs/design.md#development-roadmap)).
Internal legal research (not outside counsel) flagged this as cheaper to design in now than to argue about later — so it's written down here,
not left implicit. A planned follow-up (running this against a *live* Nav2 costmap topic instead of
only a downloaded map file) stays inside this boundary as long as it's still a one-time read that
produces a report for a human — the moment it starts continuously re-evaluating and influencing
navigation decisions on its own, that's a real classification-relevant design change, not a
convenience feature, and needs to be flagged as such before it ships.

**Naming:** this is "ISO 3691-4-*mapped* hazard flagging," never "certified against ISO 3691-4" or
"ISO 3691-4 compliant" — same non-certification framing this project uses everywhere else (see
"Honest limitations" below and the ISO/TS 15066 caveats in the main README).

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
- **`live_costmap.py`** / **`live_scan_demo.py`** / **`Dockerfile.live`**: the same rules fed a real
  *live* Nav2 costmap instead of a downloaded map file — see "Live variant" below. This is the only
  part of this directory that needs ROS/rclpy at all; everything above needs none.

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

## Live variant: the same rules against a real running Nav2 costmap

`live_scan_demo.py` proves the same `hazard_rules.py` functions work unmodified against a real
*live* `/global_costmap/costmap` topic, not just a downloaded `.yaml`/`.pgm` file — a real headless
Gazebo, a real spawned TurtleBot3, real `nav2_bringup` producing a real costmap, read exactly once
(see "Design boundary" above for why exactly once, deliberately).

```bash
docker build --platform linux/amd64 -f examples/nav2_hazard_scan/Dockerfile.live -t nav2-live-scan-demo .
docker run --platform linux/amd64 --rm nav2-live-scan-demo
```

Real run, 2026-09-28, output pasted verbatim (trimmed to a representative slice of the 24 real
findings):

```
gzserver starting with world /opt/ros/humble/share/turtlebot3_gazebo/worlds/turtlebot3_world.world ...
Robot spawned
Launching nav2_bringup against .../maps/warehouse.yaml for a live costmap ...
Published initial pose (-2.0, -0.5) to bootstrap AMCL
Live costmap received: 1006x1674 cells @ 0.03m/cell, 1101018 free cells
FLAG corridor_width_sufficient    only 0.03m clearance to the nearest obstacle, below the 0.3m minimum
OK   corridor_width_sufficient    1.44m clearance to the nearest obstacle
...
FLAG blind_corner_absent          clearest forward sightline only 0.04m, below the 1.5m required stopping sightline
OK   blind_corner_absent          clearest forward sightline 4.89m
...
PASS: 24 findings against a REAL live costmap, same rules as the static-map demo
```

**A real bug found and fixed by live-diagnosing the actual running container, not guessed:**
`global_costmap` sat forever logging `Timed out waiting for transform from base_link to map ...
Invalid frame ID "map" ... frame does not exist`. Root cause, found by execing into a debug
container and inspecting the real Nav2 lifecycle logs: AMCL never publishes a `map`→`odom` transform
until it receives an initial pose — nothing in `nav2_bringup`'s default launch provides one
automatically (that's normally done by hand in RViz). Without that transform, `global_costmap` can
wait indefinitely for a `map` frame that will never appear, regardless of which map file was given —
confirmed by testing with the *matching* `turtlebot3_navigation2` map first and seeing the exact same
symptom. Fixed by publishing one `/initialpose` message at the real known spawn point
(`turtlebot3_world.launch.py`'s own default `x_pose`/`y_pose`) before waiting for the costmap.
Deliberately still uses this project's own checked-in `warehouse.yaml`, not a map of the actual
Gazebo world the robot is standing in — AMCL's own localization accuracy is irrelevant here, since
nothing downstream of "get one real, structurally valid costmap message" depends on true
localization.

## Honest limitations, stated plainly

- **One scenario against one real map, not a validated campaign.** This proves the approach is
  tractable and the three rules work against real data; it is not a validated tool for a real site
  assessment yet.
- **No real `nav2_route` graph for this map.** `intersection_flagged_for_review` is demonstrated
  against a small, clearly-labeled representative graph in `scan_demo.py`, not this map's own real
  route topology (it doesn't ship one). Fully tested against real graph shapes in `tests/`.
- **The live variant's map and the Gazebo world it's spawned in don't match** (see above) — fine for
  proving the live-data plumbing works, not a real site assessment of that world.
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
