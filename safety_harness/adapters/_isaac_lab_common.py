"""Helpers shared by the Isaac Lab reference adapters."""

from __future__ import annotations

import math


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


def commanded_speed_mps(action, dist_m: float, fallback_speed_mps: float) -> float:
    """The Cartesian speed the controller will actually drive toward ``target_position``.

    From the action itself when it says: ``commanded_speed_mps``, or ``duration_s`` (reach the target
    in that time, i.e. dist/duration). Only when the action states neither does it fall back to the
    adapter's configured speed -- which must then be a bound the real controller enforces. An earlier
    version always extrapolated at that fixed cap, so cartesian_speed_within_limits could never see a
    40 m/s command, and the swept-path checks under-predicted how far a faster controller moves within
    the horizon. A stated value that isn't finite and positive raises -> the gate blocks (default-deny).
    """
    params = action.params
    if "commanded_speed_mps" in params:
        v = float(params["commanded_speed_mps"])
    elif "duration_s" in params:
        d = float(params["duration_s"])
        if not (math.isfinite(d) and d > 0):
            raise ValueError(f"duration_s must be finite and positive, got {params['duration_s']!r}")
        v = dist_m / d
    else:
        v = float(fallback_speed_mps)
    if not (math.isfinite(v) and v >= 0):
        raise ValueError(f"commanded speed must be finite and non-negative, got {v!r}")
    return v


def joint_position_limits(robot, env_index: int, exempt_name_substrings=()):
    """Per-joint (min, max) soft position limits from the simulator, in joint_positions order. A
    joint whose name contains any of ``exempt_name_substrings`` gets ``None`` -- an explicit
    exemption for joints designed to rest on their stops (gripper/hand fingers), which
    joint_position_limits_respected would otherwise block on every single action.

    Returns ``None`` for the whole field -- *unreported*, which that check default-denies -- if any
    non-exempt joint has no finite limit in the simulation asset. Isaac Lab's ANYmal-C asset, for
    one, defines no position limits on any of its 12 leg joints (all -inf..inf): reporting those
    infinities as if they were limits would be data that only looks like evidence."""
    lims = robot.data.soft_joint_pos_limits.torch[env_index].tolist()
    names = list(robot.data.joint_names)
    out = []
    for name, (lo, hi) in zip(names, lims):
        if any(sub in name for sub in exempt_name_substrings):
            out.append(None)
        elif not (math.isfinite(lo) and math.isfinite(hi)):
            return None
        else:
            out.append((float(lo), float(hi)))
    return tuple(out)
