"""Helpers shared by the Isaac Lab reference adapters."""

from __future__ import annotations


def quat_xyzw_to_wxyz(q) -> tuple[float, float, float, float]:
    """Isaac Lab 3.x stores every quaternion as (x, y, z, w) -- asset data, init states, and
    isaaclab.utils.math alike (older 2.x releases used (w, x, y, z)). The harness schema's Pose is
    ``orientation_wxyz``, so adapters convert here rather than pass the raw tuple through. An earlier
    version of every Isaac Lab adapter in this package passed it through unconverted, under a
    "(w, x, y, z)" comment -- found while validating the G1 adapter, where the same mix-up had already
    produced 180-degree-wrong commanded orientations in the robot-side scripts. No precondition check
    reads orientation yet, so no decision was affected, but the field was mislabeled data."""
    x, y, z, w = (float(v) for v in q)
    return (w, x, y, z)
