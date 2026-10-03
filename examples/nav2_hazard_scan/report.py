"""Renders scan_demo.py's findings into a clean, self-contained Markdown report -- the "writing it
up" half of the 2-3-day manual process this tool exists to shorten (see the example README's "Why
this exists"). Deliberately generic/unbranded: this lives in the Apache-2.0 safety-harness repo, so
nothing here claims certification, and nothing here is a client-branded deliverable layer (company
name, logo, "prepared for" headers) -- that's a separate, intentionally proprietary concern, not
something to fold into the open-source engine. See the PR discussion for why.

No new claims about coverage or certification beyond what the example README and hazard_rules.py's
own module docstring already state -- every disclaimer paragraph below is reused from there, not
invented for this report.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

RULE_LABELS = {
    "corridor_width_sufficient": "Corridor clearance",
    "blind_corner_absent": "Sightline / blind corners",
    "intersection_flagged_for_review": "Junction / crossing review",
}

# One line per rule, reused verbatim in spirit from hazard_rules.py's own docstrings -- not new
# claims, just pointers to which ISO 3691-4 concern each rule is oriented at, and the same
# "disclosed, adjustable parameter, not a published standard figure" caveat every threshold here
# already carries in the code and the README.
RULE_NOTES = {
    "corridor_width_sufficient": (
        "Clearance from the route to the nearest obstacle, not full corridor width. The minimum "
        "clearance is a disclosed, adjustable parameter -- set it from your own rated speed and "
        "stopping distance, not from this report."
    ),
    "blind_corner_absent": (
        "Real ray-cast sightline in a forward cone around the route's local direction of travel. "
        "The required sightline is a disclosed, adjustable parameter -- set it from your own "
        "stopping distance."
    ),
    "intersection_flagged_for_review": (
        "Exact graph-degree analysis: flags any route-graph node where 3+ directions meet as a "
        "candidate for a personnel-detection zone, speed reduction, mirror, or right-of-way rule "
        "under ISO 3691-4's zone-classification exercise. Flags candidates; does not perform that "
        "classification for you."
    ),
}

# Reused, not reworded, from the example README's "Design boundary" and "Honest limitations"
# sections -- a report must be at least as honest as the tool it's generated from, not less.
DISCLAIMER = """\
## What this report is — and is not

This is **ISO 3691-4-*mapped* hazard flagging**, produced by automated geometric analysis of the
supplied map — never "ISO 3691-4 compliant" or "ISO 3691-4 certified." This is not a certificate and
was not produced or reviewed by a Notified Body.

- **One map, one scan, reviewed by a human before deployment.** This is a pre-deployment analysis
  artifact, not a live/runtime safety system, and must never be treated as one — see the project's
  own "Design boundary" documentation for why that distinction is load-bearing, not incidental.
- **Every threshold used in this scan is a disclosed, adjustable parameter** (listed below, under
  "Parameters used"), not a number published by the standard. ISO 3691-4 does not publish one
  universal clearance or sightline figure; a real deployment sets these from its own risk assessment
  (rated speed, stopping distance, actual site traffic) — review the parameters below against that
  assessment before acting on any finding here.
- **Semantic zones** (doors, charging stations, load-transfer points) are not inferable from raw map
  geometry alone and are out of scope for this report.
- **Ramps and inclines are out of scope.** A 2D occupancy grid carries no elevation data.
- **This report is engineering analysis evidence, not a certified or audit-ready risk assessment.**
  It has not been reviewed by a functional-safety or industrial-safety assessor. Use it as a fast
  first pass that narrows where a qualified reviewer should look first, not a replacement for that
  review.
"""


@dataclass(frozen=True)
class RuleCoverage:
    """Whether a rule was actually evaluated for this scan, or honestly skipped -- see
    scan_demo.py's own handling of intersection_flagged_for_review, which is never run against a
    stand-in graph for a real report: the coverage section below must say so either way, not just
    silently omit a rule that didn't run."""

    name: str
    evaluated: bool
    skip_reason: str = ""
    findings: tuple = field(default_factory=tuple)


def render_markdown_report(
    *,
    map_path: str,
    map_shape,  # (rows, cols)
    resolution_m: float,
    origin_xy: tuple,
    route_description: str,
    coverage: list,  # list[RuleCoverage]
    params: dict,
    generated_at: Optional[datetime.datetime] = None,
) -> str:
    generated_at = generated_at or datetime.datetime.now()
    rows, cols = map_shape
    lines = []

    lines.append("# Nav2 Hazard Scan Report")
    lines.append("")
    lines.append(f"Generated {generated_at.isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Map")
    lines.append("")
    lines.append(f"- **File:** `{map_path}`")
    lines.append(f"- **Size:** {cols} x {rows} cells @ {resolution_m}m/cell "
                  f"(~{cols * resolution_m:.1f}m x {rows * resolution_m:.1f}m)")
    lines.append(f"- **Origin (world xy):** ({origin_xy[0]:.3f}, {origin_xy[1]:.3f})")
    lines.append(f"- **Route scanned:** {route_description}")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append("| Rule | Status | OK | Flagged |")
    lines.append("|---|---|---:|---:|")
    for rc in coverage:
        label = RULE_LABELS.get(rc.name, rc.name)
        if not rc.evaluated:
            lines.append(f"| {label} | Skipped — {rc.skip_reason} | — | — |")
            continue
        ok = sum(1 for f in rc.findings if f.satisfied)
        flagged = len(rc.findings) - ok
        lines.append(f"| {label} | Evaluated | {ok} | {flagged} |")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    for rc in coverage:
        label = RULE_LABELS.get(rc.name, rc.name)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(RULE_NOTES.get(rc.name, ""))
        lines.append("")
        if not rc.evaluated:
            lines.append(f"**Not evaluated: {rc.skip_reason}**")
            lines.append("")
            continue
        flagged_findings = [f for f in rc.findings if not f.satisfied]
        if not flagged_findings:
            lines.append("No flags for this rule.")
            lines.append("")
            continue
        lines.append("| # | Location (x, y) m | Finding |")
        lines.append("|---:|---|---|")
        for i, f in enumerate(flagged_findings, 1):
            x, y = f.location_world_xy
            lines.append(f"| {i} | ({x:.2f}, {y:.2f}) | {f.reason} |")
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "Route: real 8-connected A* shortest path over the loaded occupancy grid (no planner "
        "clearance margin added — see the example README's note on why this makes the clearance "
        "and sightline rules a stricter test than a deployed route with its own margin would see). "
        "Clearance: a two-pass chamfer distance transform to the nearest obstacle cell. Sightline: "
        "real Bresenham ray casting in a forward cone around the route's local direction of travel. "
        "Junction review: exact (in+out) degree count over a supplied route graph — see Coverage "
        "above for whether one was supplied for this scan."
    )
    lines.append("")
    lines.append("## Parameters used")
    lines.append("")
    for k, v in params.items():
        lines.append(f"- `{k}` = {v}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)
