"""Generic precondition checks.

Each operates only on WorldState / PredictedTrajectory / Action -- never on a robot's native
representation -- so the same function works across every adapter (design doc: "Action Precondition
Schemas"). Each returns a PreconditionResult. Missing, stale, or low-confidence evidence must return
satisfied=False, never True -- this file is where the design doc's default-deny rule actually gets
enforced, one function at a time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .integrity import COMMAND_SEAL_PARAM, digests_match, key_from_env, try_action_digest, verify_action_seal
from .schema import (
    Action,
    AgentCategory,
    FallConsequence,
    HazardTag,
    KnownSolidRegion,
    ObservedRegion,
    PredictedTrajectory,
    PreconditionResult,
    WorldState,
)

# Hazard tags that make an object risky to have nearby even when it isn't the action's target.
_BYSTANDER_RISK_TAGS = frozenset({HazardTag.SHARP, HazardTag.HOT})
# Hazard tags that cap how hard an object may be gripped (payload_and_grip_force_within_limits).
_CRUSH_RISK_TAGS = frozenset({HazardTag.FRAGILE, HazardTag.LIQUID_CONTAINING})
GRAVITY_MPS2 = 9.81

MIN_CONFIDENCE = 0.6  # default confidence gate; override per rule via action_schema.yaml kwargs

# iso15066_separation_distance_maintained's two reaction-time defaults, named at module level (not
# just inline in the function signature) so tests/test_latency_stress.py can compare *measured*
# gate() decision latency against the same numbers the check assumes -- one source of truth instead
# of a second hardcoded 0.2 living only in the test file, silently able to drift out of sync.
DEFAULT_REACTION_TIME_S = 0.15
DEFAULT_SAMPLING_INTERVAL_S = 0.05
REACTION_INTERVAL_S = DEFAULT_REACTION_TIME_S + DEFAULT_SAMPLING_INTERVAL_S

# Timing budgets for decision_within_deadline and sensor_data_fresh, tied to the two numbers above
# rather than invented separately. Representative values, not measured on any real system -- a
# deployment sets both from its own characterized sensor and decision latency, and should pass the
# same figures to iso15066_separation_distance_maintained (sampling_interval_s / decision_latency_s)
# so that check's assumptions are enforced here instead of merely assumed there.
#
# A decision must finish within two sensing cycles: past that, the next sample was due and missed
# before this decision could act on the last one.
DEFAULT_MAX_DECISION_LATENCY_S = 2 * DEFAULT_SAMPLING_INTERVAL_S
# Sensor capture -> end of decision must fit inside the whole reaction interval the separation
# formula budgets: data older than that predates the entire window that formula reasons about.
DEFAULT_MAX_SENSOR_AGE_S = REACTION_INTERVAL_S


@dataclass(frozen=True)
class CheckContext:
    """Facts about the decision in progress that no WorldState, Action or PredictedTrajectory can
    carry -- when the decision started, on what clock, what the proposed action hashed to at that
    moment, and which configuration is running it. Supplied by ActuatorGate.gate() (or by
    ActionSchemaRegistry.run_checks when called directly) only to checks that declare
    ``wants_check_context``; YAML can never supply or override it. Every check that reads one fails
    closed when called without one: "I wasn't told" is not evidence of anything."""

    decision_started_at: float
    clock: Callable[[], float]
    checked_action_digest: Optional[str] = None
    schema_registry: Any = None


def _wants_check_context(fn):
    fn.wants_check_context = True
    return fn


def _is_real_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _distance(a, b) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points) -> tuple:
    """Andrew's monotone-chain convex hull, counter-clockwise, no external geometry library --
    dependency-free per this project's own choice (pyproject.toml declares only pyyaml)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return tuple(pts)

    def build(seq):
        hull: list = []
        for p in seq:
            while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull

    lower = build(pts)
    upper = build(list(reversed(pts)))
    return tuple(lower[:-1] + upper[:-1])


def _point_in_convex_polygon(point, hull) -> bool:
    n = len(hull)
    if n < 3:
        return False
    return all(_cross(hull[i], hull[(i + 1) % n], point) >= -1e-9 for i in range(n))


def _point_to_segment_distance(point, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return _distance((*point, 0.0), (*a, 0.0))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    closest = (ax + t * dx, ay + t * dy)
    return _distance((*point, 0.0), (*closest, 0.0))


def _distance_to_polygon_edge(point_xy, polygon_xy) -> float:
    """Signed distance from point_xy to the convex hull of polygon_xy's vertices: positive when
    inside (distance to the nearest edge), negative when outside (how far past it).

    This used to be a nearest-VERTEX distance, documented as "a conservative stand-in ... swap in a
    computational-geometry library for a real deployment." That stand-in has a real blind spot once
    the query point is actually outside the polygon: nearest-vertex distance keeps GROWING the
    further outside it goes (a center of mass 10m past every foot reported a ~9m "margin" --
    comfortably balanced, on a robot that has clearly already fallen over), because it only ever
    measures distance-from-a-vertex, never whether the point is inside or outside at all. Found by
    testing this check against a real, moving legged robot for the first time -- every prior test
    only ever probed a point already inside or barely outside, never far enough out for a
    nearest-vertex proxy's blind spot to show. Fixed with the actual computational-geometry library
    the original docstring invited, at the same call site, per its own instruction not to change it.
    """
    if not polygon_xy:
        return 0.0
    if not math.isfinite(point_xy[0]) or not math.isfinite(point_xy[1]):
        # See the design doc's NaN-Sensor Stress Test: fail explicit and non-finite here rather than
        # let an unconfirmed point flow into hull/distance arithmetic that doesn't itself guard NaN.
        return math.nan
    if len(polygon_xy) < 3:
        # Fewer than 3 contact points isn't a real support polygon to be inside of at all --
        # maximally unstable, not "vertex-distance happens to look large."
        return -min(_distance((*point_xy, 0.0), (*v, 0.0)) for v in polygon_xy)
    hull = _convex_hull(polygon_xy)
    d = min(_point_to_segment_distance(point_xy, hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))
    return d if _point_in_convex_polygon(point_xy, hull) else -d


def _ok(name: str, reason: str) -> PreconditionResult:
    return PreconditionResult(name=name, satisfied=True, reason=reason)


def _fail(name: str, reason: str) -> PreconditionResult:
    return PreconditionResult(name=name, satisfied=False, reason=reason)


# Every precondition below enforces default-deny through a `<`/`>`/`<=` comparison against a
# sensor-derived float. IEEE-754 makes any such comparison involving NaN evaluate False in *both*
# directions -- so a corrupted or never-actually-measured reading (a NaN mass, a NaN tracked
# position, a NaN confidence score) silently satisfies the very check meant to catch its absence,
# instead of failing it. Found by adversarial stress testing -- see the design doc's "NaN-Sensor
# Stress Test" section and tests/test_adversarial_stress.py -- across mass_within_force_budget,
# object_hazard_confirmed, object_pose_confirmed's confidence field, and a compound bypass of
# swept_path_clear_of_agents + iso15066_separation_distance_maintained together.
#
# These three helpers are the shared fix: every bare numeric comparison in this file that gates a
# PreconditionResult should go through one of them instead, so "unconfirmed" reliably means
# "treated as the worst case," not "silently passes." Each name matches the comparison it replaces
# (`_below` <-> `<`, `_exceeds` <-> `>`, `_at_or_within` <-> `<=`) and returns True -- the same
# truth value that comparison would need to return to correctly trigger the caller's fail path --
# whenever either operand is non-finite, regardless of which side of the comparison it's on.
def _below(value: float, threshold: float) -> bool:
    if not (math.isfinite(value) and math.isfinite(threshold)):
        return True
    return value < threshold


def _exceeds(value: float, limit: float) -> bool:
    if not (math.isfinite(value) and math.isfinite(limit)):
        return True
    return value > limit


def _at_or_within(value: float, bound: float) -> bool:
    if not (math.isfinite(value) and math.isfinite(bound)):
        return True
    return value <= bound


def _safe_max(a: float, b: float) -> float:
    """`max(a, b)`, but propagates non-finite instead of silently discarding it the way Python's
    own `max` does: `max(0.0, float('nan'))` returns `0.0`, because every comparison `max` makes
    internally against a NaN is False. Used where a worst-case running maximum must not let one
    corrupted sample vanish just because of where it fell in iteration order."""
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    return max(a, b)


def object_hazard_confirmed(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id", min_confidence: float = MIN_CONFIDENCE,
) -> PreconditionResult:
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None:
        return _fail("object_hazard_confirmed", f"object {obj_id!r} not in perceived world state")
    if _below(obj.class_confidence, min_confidence):
        return _fail(
            "object_hazard_confirmed",
            f"class confidence {obj.class_confidence:.2f} below {min_confidence}",
        )
    if HazardTag.UNKNOWN in obj.hazard_tags:
        return _fail("object_hazard_confirmed", "hazard class is unknown -- default-deny")
    return _ok("object_hazard_confirmed", f"object {obj_id!r} hazard class confirmed")


def mass_within_force_budget(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id", force_budget_kg: float = 3.0,
) -> PreconditionResult:
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None or obj.estimated_mass_kg is None:
        return _fail("mass_within_force_budget", f"no confirmed mass estimate for {obj_id!r}")
    if _exceeds(obj.estimated_mass_kg, force_budget_kg):
        return _fail(
            "mass_within_force_budget",
            f"{obj.estimated_mass_kg:.2f}kg exceeds budget {force_budget_kg}kg",
        )
    return _ok("mass_within_force_budget", f"{obj.estimated_mass_kg:.2f}kg within budget")


def swept_path_clear_of_agents(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    min_tracking_confidence: float = MIN_CONFIDENCE, margin_m: float = 0.15,
) -> PreconditionResult:
    for agent in state.agents:
        confidence_ok = agent.tracking_confidence >= min_tracking_confidence
        for point in trajectory.points:
            worst_case_r = agent.worst_case_radius_m(point.t)
            d = _distance(point.swept_volume_center, agent.pose.position)
            required_clearance = point.swept_volume_radius_m + worst_case_r + margin_m
            if not confidence_ok or _below(d, required_clearance):
                return _fail(
                    "swept_path_clear_of_agents",
                    f"agent {agent.agent_id!r} within {d:.2f}m at t={point.t:.2f}s "
                    f"(needs {required_clearance:.2f}m; tracking_confidence={agent.tracking_confidence:.2f})",
                )
    return _ok("swept_path_clear_of_agents", "no tracked agent within worst-case clearance for the full trajectory")


def _stability_margin(
    name: str, trajectory: PredictedTrajectory, *, min_margin_m: float, use_capture_point: bool,
    require_com_velocity: bool, ground_height_m: float, capture_step_allowance_m: float, gravity_mps2: float,
) -> PreconditionResult:
    """Shared body of stability_margin_maintained and its legacy name balance_margin_maintained --
    one implementation, so the two can never drift apart."""
    if not trajectory.points:
        return _fail(name, "no predicted trajectory to evaluate")
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.center_of_mass is None or robot.support_polygon is None:
            return _fail(name, "no balance state reported -- default-deny for legged/humanoid platforms")
        com = robot.center_of_mass
        margin = _distance_to_polygon_edge(com[:2], robot.support_polygon)
        if _below(margin, min_margin_m):
            return _fail(name, f"center of mass within {margin:.3f}m of support-polygon edge at t={point.t:.2f}s")
        if not use_capture_point:
            continue
        velocity = robot.center_of_mass_velocity
        if velocity is None:
            if require_com_velocity:
                return _fail(
                    name,
                    "no center-of-mass velocity reported -- a static margin alone says nothing about "
                    "whether a moving platform can stop inside its support polygon; default-deny",
                )
            continue
        if len(com) < 3 or len(velocity) < 2:
            return _fail(name, "center of mass / velocity malformed -- cannot compute a capture point")
        height = com[2] - ground_height_m
        if _at_or_within(height, 0.0) or _at_or_within(gravity_mps2, 0.0):
            return _fail(name, f"center-of-mass height {height:.3f}m above ground is non-positive or unconfirmed")
        omega = math.sqrt(gravity_mps2 / height)
        capture_point = (com[0] + velocity[0] / omega, com[1] + velocity[1] / omega)
        capture_margin = _distance_to_polygon_edge(capture_point, robot.support_polygon)
        if _below(capture_margin, min_margin_m - capture_step_allowance_m):
            return _fail(
                name,
                f"capture point {capture_margin:.3f}m from the support-polygon edge at t={point.t:.2f}s "
                f"(needs {min_margin_m - capture_step_allowance_m:.3f}m): at this center-of-mass velocity the "
                f"platform cannot stop inside its support polygon",
            )
    return _ok(name, "center of mass and capture point stayed within the safe margin throughout")


def stability_margin_maintained(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    min_margin_m: float = 0.03,
    require_com_velocity: bool = True,
    ground_height_m: float = 0.0,
    capture_step_allowance_m: float = 0.0,
    gravity_mps2: float = GRAVITY_MPS2,
) -> PreconditionResult:
    """Tip-over / fall margin for any platform that reports a center of mass and a support polygon --
    formalized from the ANYmal-C balance-margin logic
    (balance_margin_maintained, kept registered under its old name as its static-only half) into one
    named, citable check that doesn't assume anything about the robot beyond two schema fields.

    Two margins, both required at every predicted point, both against the *convex hull* of the
    reported ground contacts (_distance_to_polygon_edge -- the hull-based signed distance that
    replaced the old nearest-vertex proxy after the ANYmal-C adapter exposed its blind spot):

    * Static: the center of mass's ground projection must sit at least ``min_margin_m`` inside the
      support polygon. This is all balance_margin_maintained ever checked, and it is only a valid
      stability criterion for a platform that isn't moving.
    * Dynamic: the *capture point* (the "extrapolated center of mass", Hof et al. 2005; the
      instantaneous capture point of Pratt et al. 2006) -- ``com_xy + v_xy / omega0`` with
      ``omega0 = sqrt(g / com_height)`` under the linear-inverted-pendulum model -- must satisfy
      the same margin. It is where the center of mass has to be brought over to come to rest; if it
      lies outside the support polygon, the platform cannot stop without stepping or falling, even
      while its static projection still looks comfortably inside. That is the case a static-only
      check permits and this one blocks.

    ``require_com_velocity`` defaults to True: a platform that reports balance state but not the
    velocity needed for the dynamic half is default-denied rather than silently downgraded to the
    static half. ``capture_step_allowance_m`` (default 0.0 = must be able to stop without taking a
    step at all, the conservative case) lets a legged platform whose controller can take a
    recovery step allow the capture point that far outside its current support polygon -- a
    one-step capturability approximation, to be set from the platform's characterized step length,
    not guessed.

    Scope of the dynamic half: the capture point is an inverted-pendulum criterion, right for
    legged and humanoid platforms. A rigid wheeled base (or an arm on a cart) tips under braking by
    a different mechanism -- the zero-moment point shifting by ``com_height * deceleration / g`` --
    which this check does not model; for such a platform the static half is what applies
    (``require_com_velocity: false`` with no velocity reported), and a braking-ZMP margin is an
    open follow-up, not something the capture point stands in for.

    Center-of-mass height is ``center_of_mass[2] - ground_height_m`` in the same frame as the
    support polygon. The ANYmal-C adapter's base-position-for-center-of-mass (and base velocity
    for center-of-mass velocity) stand-in applies here unchanged -- a documented simplification,
    not a whole-body center of mass. Fewer than 3 contact points is still treated as maximally
    unstable (see _distance_to_polygon_edge), so a biped in double support must report its foot
    soles' contact corners, not two points -- the gap isaac_lab_g1.py documents remains open.
    """
    return _stability_margin(
        "stability_margin_maintained", trajectory, min_margin_m=min_margin_m, use_capture_point=True,
        require_com_velocity=require_com_velocity, ground_height_m=ground_height_m,
        capture_step_allowance_m=capture_step_allowance_m, gravity_mps2=gravity_mps2,
    )


def balance_margin_maintained(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    min_margin_m: float = 0.03,
) -> PreconditionResult:
    """Legacy name, kept registered so existing action schemas and result-name consumers keep
    working unchanged: the static half of stability_margin_maintained only, through the same
    implementation, with the same result name and the same behavior as before -- it ignores
    center-of-mass velocity even when one is reported. One deliberate tightening: an empty
    trajectory now fails instead of passing vacuously (the engine already blocked that case before
    any check ran). Wire stability_margin_maintained instead to get the capture-point margin."""
    return _stability_margin(
        "balance_margin_maintained", trajectory, min_margin_m=min_margin_m, use_capture_point=False,
        require_com_velocity=False, ground_height_m=0.0, capture_step_allowance_m=0.0, gravity_mps2=GRAVITY_MPS2,
    )


def visibility_above_threshold(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    min_visibility: float = 0.5,
) -> PreconditionResult:
    if _below(state.environment.visibility_confidence, min_visibility):
        return _fail(
            "visibility_above_threshold",
            f"visibility confidence {state.environment.visibility_confidence:.2f} below {min_visibility}",
        )
    return _ok("visibility_above_threshold", "visibility confidence sufficient")


def surface_confirmed_stable(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    surface_id_param: str = "target_surface_id", min_confidence: float = MIN_CONFIDENCE,
) -> PreconditionResult:
    """DEPRECATED -- use destination_confirmed_stable_and_clear, which checks this AND that the
    destination is stable and clear of risky objects. Kept registered so existing configs still load
    (with a DeprecationWarning, see DEPRECATED_CHECKS); not counted as a distinct active check."""
    surf_id = action.params.get(surface_id_param)
    surf = next((o for o in state.objects if o.object_id == surf_id), None)
    if surf is None or _below(surf.pose_confidence, min_confidence):
        return _fail("surface_confirmed_stable", f"surface {surf_id!r} not confirmed with sufficient confidence")
    return _ok("surface_confirmed_stable", f"surface {surf_id!r} confirmed")


def object_pose_confirmed(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id", min_confidence: float = MIN_CONFIDENCE,
) -> PreconditionResult:
    """Is the target's *position* -- not just its class -- confirmed with sufficient confidence?
    Distinct from object_hazard_confirmed, which only checks classification. Found missing by
    mutation testing: nothing previously checked this, so a target with zero pose confidence still
    permitted a grasp at whatever stale/fabricated position was on record.

    Also checks the position itself is finite, independent of the reported confidence: a high
    confidence score claimed for a NaN/Inf position is a contradiction, not a pass. Found by
    fuzzing with a NaN position -- confidence and the coordinate value are two different fields,
    and nothing previously cross-checked that they agree.

    The confidence score itself is put through the same `_below` non-finite-fails guard as every
    other confidence gate in this module -- found missing by later adversarial stress testing: this
    function's own explicit isfinite check above covered a NaN *position* under a high confidence
    score, but a NaN *confidence score* itself still passed `pose_confidence < min_confidence`
    (False either way) until now."""
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None or _below(obj.pose_confidence, min_confidence):
        return _fail("object_pose_confirmed", f"object {obj_id!r} pose confidence insufficient or object absent")
    if not all(math.isfinite(c) for c in obj.pose.position):
        return _fail("object_pose_confirmed", f"object {obj_id!r} reports a non-finite position despite a confidence score")
    return _ok("object_pose_confirmed", f"object {obj_id!r} pose confirmed")


def robot_state_confirmed(
    state: WorldState, action: Action, trajectory: PredictedTrajectory,
) -> PreconditionResult:
    """General gate, like visibility_above_threshold: is the robot's own proprioceptive state known
    AND readable? Found missing by fuzzing with robot=None. Since v0.3.1 it also checks the content:
    an earlier version passed any non-None state, so NaN joint readings sailed through it (the live
    re-validation found joint_position_limits_respected catching them instead)."""
    robot = state.robot
    if robot is None:
        return _fail("robot_state_confirmed", "no robot proprioceptive state reported")
    if len(robot.joint_positions) == 0 or len(robot.joint_positions) != len(robot.joint_velocities):
        return _fail("robot_state_confirmed", "joint positions/velocities missing or mismatched in length")
    readings = list(robot.joint_positions) + list(robot.joint_velocities)
    if robot.end_effector_pose is None:
        return _fail("robot_state_confirmed", "no end-effector pose reported")
    readings += list(robot.end_effector_pose.position)
    if not all(_is_real_number(v) and math.isfinite(v) for v in readings):
        return _fail("robot_state_confirmed", "non-finite or non-numeric proprioceptive reading")
    return _ok("robot_state_confirmed", "robot state confirmed")


def object_cleared_for_interaction(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id",
) -> PreconditionResult:
    """Is the target itself safe to approach and handle at all -- distinct from knowing its hazard
    class. An object with no confirmed clearance is off-limits, not merely handled carefully."""
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None:
        return _fail("object_cleared_for_interaction", f"object {obj_id!r} not in perceived world state")
    if not obj.cleared_for_interaction:
        return _fail("object_cleared_for_interaction", f"object {obj_id!r} has not been cleared for interaction")
    return _ok("object_cleared_for_interaction", f"object {obj_id!r} cleared for interaction")


def swept_path_clear_of_risky_objects(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id", margin_m: float = 0.15,
) -> PreconditionResult:
    """General gate, like visibility_above_threshold: no *bystander* object -- one that isn't the
    action's own target -- may be within margin_m of the swept path if it is uncleared for
    interaction or carries a bystander-risk tag (sharp, hot) or would release a hazard if struck.
    This is what blocks an action for coming near a hazardous object even when that object was
    never the target: the green cube doesn't need to be what's being grasped to matter."""
    target_id = action.params.get(object_id_param)
    for obj in state.objects:
        if obj.object_id == target_id:
            continue  # the target's own clearance is object_cleared_for_interaction's job
        risky = (
            not obj.cleared_for_interaction
            or obj.fall_consequence == FallConsequence.HAZARDOUS_RELEASE
            or bool(obj.hazard_tags & _BYSTANDER_RISK_TAGS)
        )
        if not risky:
            continue
        for point in trajectory.points:
            d = _distance(point.swept_volume_center, obj.pose.position)
            if _below(d, point.swept_volume_radius_m + margin_m):
                return _fail(
                    "swept_path_clear_of_risky_objects",
                    f"bystander object {obj.object_id!r} within {d:.2f}m at t={point.t:.2f}s "
                    f"(cleared={obj.cleared_for_interaction}, fall_consequence={obj.fall_consequence.value})",
                )
    return _ok("swept_path_clear_of_risky_objects", "no uncleared or risky bystander object within margin of the swept path")


def current_position_confirmed_stable(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id",
) -> PreconditionResult:
    """Is the object's current resting position itself safe to pick up from -- not teetering on an
    edge, not the top of a precarious stack. Unconfirmed (None) defaults to deny."""
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None or obj.supported_stably is not True:
        return _fail("current_position_confirmed_stable", f"object {obj_id!r} current position not confirmed stable")
    return _ok("current_position_confirmed_stable", f"object {obj_id!r} current position confirmed stable")


def destination_confirmed_stable_and_clear(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    surface_id_param: str = "target_surface_id", min_confidence: float = MIN_CONFIDENCE, clearance_m: float = 0.1,
) -> PreconditionResult:
    """Is the destination itself safe to place onto: confirmed stable, AND no other risky object
    close enough to be struck or damaged by the placement."""
    surf_id = action.params.get(surface_id_param)
    surf = next((o for o in state.objects if o.object_id == surf_id), None)
    if surf is None or _below(surf.pose_confidence, min_confidence):
        return _fail("destination_confirmed_stable_and_clear", f"destination {surf_id!r} not confirmed with sufficient confidence")
    if surf.supported_stably is not True:
        return _fail("destination_confirmed_stable_and_clear", f"destination {surf_id!r} itself not confirmed stable")
    for obj in state.objects:
        if obj.object_id in (surf_id, action.params.get("object_id")):
            continue
        risky = not obj.cleared_for_interaction or obj.fall_consequence == FallConsequence.HAZARDOUS_RELEASE
        if risky and _below(_distance(surf.pose.position, obj.pose.position), clearance_m):
            return _fail(
                "destination_confirmed_stable_and_clear",
                f"risky object {obj.object_id!r} within {clearance_m}m of destination {surf_id!r}",
            )
    return _ok("destination_confirmed_stable_and_clear", f"destination {surf_id!r} confirmed stable and clear")


def fall_consequence_acceptable(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id",
) -> PreconditionResult:
    """If this grasp/carry fails and the object drops, is the worst case acceptable? Reasons about
    the *consequence of a plausible failure*, not just the current state -- an object that's
    perfectly fine to grasp right now can still be unsafe to lift to a given height."""
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None:
        return _fail("fall_consequence_acceptable", f"object {obj_id!r} not in perceived world state")
    if obj.fall_consequence == FallConsequence.NONE:
        return _ok("fall_consequence_acceptable", f"object {obj_id!r} has no meaningful fall consequence")
    tolerance = obj.drop_tolerance_m if obj.drop_tolerance_m is not None else 0.0
    heights = [p.swept_volume_center[2] for p in trajectory.points] or [obj.pose.position[2]]
    max_height = heights[0]
    for h in heights[1:]:
        # Not a bare max(): Python's builtin silently drops a NaN candidate that isn't first in
        # the sequence (nan > current is False, so it never replaces the running maximum) -- the
        # same order-dependence as _distance_to_polygon_edge's min(). _safe_max propagates it.
        max_height = _safe_max(max_height, h)
    lift = max_height - obj.pose.position[2]
    if obj.fall_consequence == FallConsequence.UNKNOWN or _exceeds(lift, tolerance):
        return _fail(
            "fall_consequence_acceptable",
            f"lift of {lift:.2f}m exceeds confirmed drop tolerance ({tolerance:.2f}m; "
            f"fall_consequence={obj.fall_consequence.value})",
        )
    return _ok("fall_consequence_acceptable", f"lift of {lift:.2f}m within confirmed drop tolerance")


def joint_position_limits_respected(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    safety_margin: float = 0.02,
) -> PreconditionResult:
    """Does the predicted motion keep every joint within its own position limits, with a margin --
    not just at the destination, at every predicted point.

    Scope, stated plainly: this checks the joint state carried on each trajectory point. The
    reference adapters forward-predict the Cartesian sweep only and carry the *current* joint state
    on every point, so with them this catches "already at or near a limit", not "this command will
    drive a joint past its limit" -- that needs a dynamics adapter that predicts joint motion.

    A ``None`` entry in ``joint_position_limits`` is an explicit exemption the adapter must declare
    for a joint designed to rest on its mechanical stop (a gripper finger fully open or closed) --
    otherwise every action would block. A limits tuple whose length doesn't match the joint state
    fails closed: an earlier version zipped the two, silently leaving any unmatched joints unchecked.
    """
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.joint_position_limits is None:
            return _fail("joint_position_limits_respected", "no joint position limits reported -- default-deny")
        if len(robot.joint_position_limits) != len(robot.joint_positions):
            return _fail(
                "joint_position_limits_respected",
                f"{len(robot.joint_position_limits)} joint limits reported for {len(robot.joint_positions)} joints -- default-deny",
            )
        for j, (pos, limits) in enumerate(zip(robot.joint_positions, robot.joint_position_limits)):
            if limits is None:
                continue  # explicitly exempted by the adapter (see docstring)
            lo, hi = limits
            if _below(pos, lo + safety_margin) or _exceeds(pos, hi - safety_margin):
                return _fail(
                    "joint_position_limits_respected",
                    f"joint {j} at {pos:.3f} within {safety_margin}rad of its [{lo:.3f}, {hi:.3f}] limit at t={point.t:.2f}s",
                )
    return _ok("joint_position_limits_respected", "every joint stayed within its position limits throughout")


def joint_velocity_within_limits(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    utilization_limit: float = 0.9,
) -> PreconditionResult:
    """Does the predicted motion keep every joint's speed within a fraction of its rated limit --
    this is also where an unmodeled kinematic singularity would show up, since required joint
    speeds spike near one even for a modest commanded Cartesian speed.

    A limits tuple whose length doesn't match the joint state fails closed -- the same fix
    ``joint_position_limits_respected`` already made once for the identical flaw (zip() silently
    truncates to the shorter sequence, leaving any unmatched joint unchecked), not propagated to
    this sibling when it was wired in later. Found by an independent third-party review, 2026-09-29.
    """
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.joint_velocity_limits is None:
            return _fail("joint_velocity_within_limits", "no joint velocity limits reported -- default-deny")
        if len(robot.joint_velocity_limits) != len(robot.joint_velocities):
            return _fail(
                "joint_velocity_within_limits",
                f"{len(robot.joint_velocity_limits)} joint velocity limits reported for {len(robot.joint_velocities)} joints -- default-deny",
            )
        for j, (vel, limit) in enumerate(zip(robot.joint_velocities, robot.joint_velocity_limits)):
            if _exceeds(abs(vel), utilization_limit * limit):
                return _fail(
                    "joint_velocity_within_limits",
                    f"joint {j} speed {abs(vel):.3f} exceeds {utilization_limit:.0%} of its {limit:.3f} limit at t={point.t:.2f}s",
                )
    return _ok("joint_velocity_within_limits", "every joint speed stayed within its limit throughout")


def joint_effort_within_limits(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    utilization_limit: float = 0.9,
) -> PreconditionResult:
    """Does the predicted motion keep every joint's torque/current within its rated limit -- the
    electrical/mechanical load side, distinct from position or speed.

    A limits tuple whose length doesn't match the effort estimate fails closed -- see
    ``joint_velocity_within_limits``'s docstring for why this check needs the same guard.
    """
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.joint_effort_limits is None or robot.estimated_joint_efforts is None:
            return _fail("joint_effort_within_limits", "no joint effort limits/estimate reported -- default-deny")
        if len(robot.joint_effort_limits) != len(robot.estimated_joint_efforts):
            return _fail(
                "joint_effort_within_limits",
                f"{len(robot.joint_effort_limits)} joint effort limits reported for {len(robot.estimated_joint_efforts)} joints -- default-deny",
            )
        for j, (effort, limit) in enumerate(zip(robot.estimated_joint_efforts, robot.joint_effort_limits)):
            if _exceeds(abs(effort), utilization_limit * limit):
                return _fail(
                    "joint_effort_within_limits",
                    f"joint {j} effort {abs(effort):.3f} exceeds {utilization_limit:.0%} of its {limit:.3f} limit at t={point.t:.2f}s",
                )
    return _ok("joint_effort_within_limits", "every joint effort stayed within its limit throughout")


def motor_temperature_within_limits(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    safety_margin_c: float = 5.0,
) -> PreconditionResult:
    """Is every joint's motor currently cool enough to safely take on more sustained load? This
    checks the robot's *current* temperature, not a forward thermal simulation -- a coarse but
    honest proxy: don't add load to a motor that's already close to its limit.

    A limits tuple whose length doesn't match the temperature reading fails closed -- see
    ``joint_velocity_within_limits``'s docstring for why this check needs the same guard.
    """
    robot = state.robot
    if robot is None or robot.motor_temperature_c is None or robot.motor_temperature_limit_c is None:
        return _fail("motor_temperature_within_limits", "no motor temperature reported -- default-deny")
    if len(robot.motor_temperature_limit_c) != len(robot.motor_temperature_c):
        return _fail(
            "motor_temperature_within_limits",
            f"{len(robot.motor_temperature_limit_c)} motor temperature limits reported for {len(robot.motor_temperature_c)} motors -- default-deny",
        )
    for j, (temp, limit) in enumerate(zip(robot.motor_temperature_c, robot.motor_temperature_limit_c)):
        if _exceeds(temp, limit - safety_margin_c):
            return _fail(
                "motor_temperature_within_limits",
                f"joint {j} motor at {temp:.1f}°C, within {safety_margin_c}°C of its {limit:.1f}°C limit",
            )
    return _ok("motor_temperature_within_limits", "every motor has thermal headroom")


def cartesian_speed_within_limits(
    state: WorldState, action: Action, trajectory: PredictedTrajectory,
) -> PreconditionResult:
    """Does the swept end-effector path stay within this platform's rated Cartesian speed, computed
    directly from the predicted trajectory points -- meaningful even for a dynamics adapter that
    only predicts Cartesian motion, unlike the joint-space checks above."""
    limit = None
    points = trajectory.points
    for point in points:
        if point.robot is not None and point.robot.max_cartesian_speed_mps is not None:
            limit = point.robot.max_cartesian_speed_mps
            break
    if limit is None:
        return _fail("cartesian_speed_within_limits", "no Cartesian speed limit reported -- default-deny")
    for prev, cur in zip(points, points[1:]):
        dt = cur.t - prev.t
        if dt <= 0:
            continue
        speed = _distance(prev.swept_volume_center, cur.swept_volume_center) / dt
        # exactly-at-rating motion is within limits; the 1e-6 relative slack only absorbs float
        # round-off in distance/dt (a path at precisely 1.0 m/s computed as 1.0000001)
        if _exceeds(speed, limit * (1.0 + 1e-6)):
            return _fail(
                "cartesian_speed_within_limits",
                f"segment ending t={cur.t:.2f}s at {speed:.2f}m/s exceeds the {limit:.2f}m/s limit",
            )
    return _ok("cartesian_speed_within_limits", "swept path stayed within the Cartesian speed limit throughout")


def self_collision_clear(
    state: WorldState, action: Action, trajectory: PredictedTrajectory,
    *, min_margin_m: float = 0.02,
) -> PreconditionResult:
    """Does the predicted motion keep the robot's own links clear of each other -- reported by the
    dynamics adapter's own collision geometry, since a generic check can't recompute that from
    joint angles without duplicating the robot's link geometry."""
    for point in trajectory.points:
        if point.self_collision_margin_m is None:
            return _fail("self_collision_clear", "no self-collision margin reported -- default-deny")
        if _below(point.self_collision_margin_m, min_margin_m):
            return _fail(
                "self_collision_clear",
                f"self-collision margin {point.self_collision_margin_m:.3f}m below {min_margin_m}m at t={point.t:.2f}s",
            )
    return _ok("self_collision_clear", "robot links stayed clear of each other throughout")


def battery_charge_sufficient(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    min_charge_fraction: float = 0.15,
) -> PreconditionResult:
    """Does the robot have enough charge left not just to attempt this action, but to still be able
    to execute its fallback (retract, complete a step to a stable stance, retreat) afterward? A
    wall-powered fixed-base arm reports no battery at all -- None is not a failure specific to this
    check alone, it means the concept doesn't apply, but the check still defaults to deny rather
    than assuming "not battery powered" on the platform's behalf; a robot with no battery should
    say so explicitly by reporting a fraction of 1.0, not by reporting nothing."""
    robot = state.robot
    if robot is None or robot.battery_charge_fraction is None:
        return _fail("battery_charge_sufficient", "no battery state reported -- default-deny")
    if _below(robot.battery_charge_fraction, min_charge_fraction):
        return _fail(
            "battery_charge_sufficient",
            f"charge {robot.battery_charge_fraction:.0%} below the {min_charge_fraction:.0%} reserve needed to safely fall back",
        )
    return _ok("battery_charge_sufficient", f"charge {robot.battery_charge_fraction:.0%} has reserve for a safe fallback")


def iso15066_separation_distance_maintained(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    reaction_time_s: float = DEFAULT_REACTION_TIME_S,
    sampling_interval_s: float = DEFAULT_SAMPLING_INTERVAL_S,
    decision_latency_s: float = 0.0,
    max_deceleration_mps2: float = 5.0,
    intrusion_distance_m: float = 0.85,
    robot_position_uncertainty_m: float = 0.02,
) -> PreconditionResult:
    """Structurally faithful to ISO/TS 15066:2016 Annex A's protective separation distance formula
    for Speed and Separation Monitoring: S(t0) = Sh + Sr + Ss + C + Zd + Zr -- the human's own
    reach during the reaction interval, the robot's travel during that same interval, the robot's
    stopping distance once decelerating, a fixed intrusion allowance, and position-uncertainty
    terms for both. This supersedes swept_path_clear_of_agents' flat margin for human/animal
    proximity specifically -- both stay registered; either can block, same as
    destination_confirmed_stable_and_clear superseding surface_confirmed_stable for `place`.

    The numeric defaults (reaction time, intrusion distance, deceleration) are representative
    literature values for the *structure* of the calculation, not a certified figure for any real
    deployment -- ISO/TS 15066 compliance requires a qualified safety engineer to set these from
    the actual system's characterized reaction time and the standard's current edition. See the
    design doc's Regulatory Mapping section.

    ``reaction_time_s`` models only the robot's own hardware stop-response lag (ISO/TS 15066's
    definition). It says nothing about how long ActuatorGate.gate() itself takes to decide BLOCK --
    tests/test_latency_stress.py measures that under injected adapter jitter and found it can run
    to hundreds of milliseconds, on top of (not instead of) the robot's hardware lag. ``
    decision_latency_s`` exists to let a deployment add its own *measured* worst-case gate()
    latency (e.g. a rolling p99 from its own Logger) into the required separation distance.
    Defaulting it to 0.0 rather than a fabricated nonzero number is deliberate -- this codebase's
    own rule is that a number nobody measured is worse than an explicit gap (see the design doc's
    "Design Principle"). Leaving this at 0.0 in a real deployment means decision latency has not
    been accounted for; that is a gap to close before relying on this check's number, not a safe
    default.
    """
    points = trajectory.points
    if not points:
        return _fail("iso15066_separation_distance_maintained", "no predicted trajectory to evaluate")
    reaction_interval = reaction_time_s + sampling_interval_s + decision_latency_s
    for agent in state.agents:
        prev = None
        for point in points:
            robot_speed = 0.0 if prev is None else _speed(prev, point)
            human_reach = agent.worst_case_speed_mps * reaction_interval  # Sh
            robot_reach = robot_speed * reaction_interval  # Sr
            robot_stopping = (robot_speed ** 2) / (2 * max_deceleration_mps2) if max_deceleration_mps2 > 0 else float("inf")  # Ss
            # Position uncertainty for the human grows with tracking staleness -- same principle as
            # every other confidence-gated check in this module: a less-recently-confirmed position
            # is a less certain one, not a free pass.
            human_position_uncertainty = agent.worst_case_speed_mps * agent.time_since_confirmed_s  # Zd
            required = (
                human_reach + robot_reach + robot_stopping + intrusion_distance_m
                + human_position_uncertainty + robot_position_uncertainty_m  # C, Zr
            )
            d = _distance(point.swept_volume_center, agent.pose.position)
            if _below(d, required):
                return _fail(
                    "iso15066_separation_distance_maintained",
                    f"separation {d:.2f}m below ISO/TS 15066 S(t0)={required:.2f}m at t={point.t:.2f}s "
                    f"(agent {agent.agent_id!r}, robot_speed={robot_speed:.2f}m/s)",
                )
            prev = point
    return _ok("iso15066_separation_distance_maintained", "separation distance maintained for every tracked agent throughout")


def _speed(prev_point, point) -> float:
    dt = point.t - prev_point.t
    if dt <= 0:
        return 0.0
    return _distance(prev_point.swept_volume_center, point.swept_volume_center) / dt


def iso15066_power_force_limiting(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    effective_mass_kg: float = 5.0,
    assumed_contact_time_s: float = 0.015,
    max_transient_force_n: float = 150.0,
    contact_plausible_range_m: float = 0.3,
) -> PreconditionResult:
    """A simplified proxy for ISO/TS 15066 Annex A's Power and Force Limiting intent: if the robot
    struck a human at its current commanded speed, would the estimated transient contact force
    exceed a biomechanical limit? Estimated as an impulse over an assumed short contact/deceleration
    time (F ~= effective_mass * relative_speed / contact_time), not the standard's full spring-
    constant collision model -- deliberately simpler and more transparent, at the cost of being an
    approximation rather than a reproduction of the annex's exact formula.

    Every numeric default here (effective mass, contact time, force limit) is a representative
    literature value for illustrating the *shape* of a force-limiting check, not a per-body-region
    figure from the standard's own tables (which vary by body part -- hand/finger, forearm, chest,
    abdomen -- and by whether the contact is constrained or free). A real deployment needs a
    qualified safety engineer setting these per body region from the current standard, the same
    caveat as iso15066_separation_distance_maintained. See the design doc's Regulatory Mapping.

    Only evaluated when an agent is within contact_plausible_range_m of the swept path at that
    point -- a collision force estimate for an agent nowhere near contact isn't meaningful, and
    without this gate every tracked agent anywhere in the scene would trigger a "would this hurt
    them" calculation regardless of distance. Found by the first pass of local tests: a far_agent()
    fixture 7m away still failed this check before the gate was added.

    That "too far, skip" gate is deliberately *not* run through `_exceeds` the way every other
    comparison in this module is: `_exceeds` treats a non-finite distance as automatically
    exceeding the range, which here would mean automatically *skipping* an agent whose position is
    unconfirmed -- exactly backwards for a gate whose only job is to decide whether to bother
    checking at all. An unconfirmed distance must never license skipping; it must fall through to
    be evaluated, same as `math.isfinite` gates the skip explicitly below. Found by adversarial
    stress testing: with the naive `> contact_plausible_range_m` comparison, a NaN coordinate made
    the gate stop skipping (since `nan > range` is also False) as a side effect rather than by
    design, which happened to fail closed only for this check's specific default numbers -- see the
    design doc's "NaN-Sensor Stress Test".
    """
    points = trajectory.points
    if not points:
        return _fail("iso15066_power_force_limiting", "no predicted trajectory to evaluate")
    if not state.agents:
        return _ok("iso15066_power_force_limiting", "no tracked agents to estimate a collision force against")
    prev = None
    worst_force = 0.0
    for point in points:
        robot_speed = 0.0 if prev is None else _speed(prev, point)
        for agent in state.agents:
            d = _distance(point.swept_volume_center, agent.pose.position)
            if math.isfinite(d) and d > contact_plausible_range_m:
                continue  # confirmed too far for contact to be physically plausible at this point
            relative_speed = robot_speed + agent.worst_case_speed_mps  # worst case: closing head-on
            force = effective_mass_kg * relative_speed / assumed_contact_time_s
            worst_force = _safe_max(worst_force, force)
        prev = point
    if _exceeds(worst_force, max_transient_force_n):
        return _fail(
            "iso15066_power_force_limiting",
            f"estimated transient contact force {worst_force:.0f}N exceeds the {max_transient_force_n:.0f}N limit",
        )
    return _ok("iso15066_power_force_limiting", f"estimated transient contact force {worst_force:.0f}N within limit")


def reduced_speed_near_human(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    collaborative_zone_radius_m: float = 1.5,
    max_speed_in_zone_mps: float = 0.25,
) -> PreconditionResult:
    """ISO 10218 / ANSI-RIA R15.06 style reduced-speed collaborative zone: whenever a tracked
    human-tagged agent is within collaborative_zone_radius_m of any point on the swept path, the
    commanded speed anywhere on that path must not exceed max_speed_in_zone_mps. Distinct from and
    complementary to the ISO/TS 15066 separation-distance formula above: that one computes how much
    distance a given speed requires, this one instead caps the speed outright once a human is
    nearby, which is how many real collaborative cells implement the requirement in practice."""
    points = trajectory.points
    if not points:
        return _fail("reduced_speed_near_human", "no predicted trajectory to evaluate")
    # TrackedAgent has no hazard-class field of its own to distinguish "human" from "animal" --
    # every tracked agent is treated as human-or-equivalent for this rule, the conservative
    # reading: unless a deployment can positively confirm "definitely not human," assume it could be.
    human_agents = list(state.agents)
    if not human_agents:
        return _ok("reduced_speed_near_human", "no tracked agents")
    prev = None
    for point in points:
        robot_speed = 0.0 if prev is None else _speed(prev, point)
        for agent in human_agents:
            d = _distance(point.swept_volume_center, agent.pose.position)
            if _at_or_within(d, collaborative_zone_radius_m) and _exceeds(robot_speed, max_speed_in_zone_mps):
                return _fail(
                    "reduced_speed_near_human",
                    f"agent {agent.agent_id!r} within {d:.2f}m (zone={collaborative_zone_radius_m}m) while "
                    f"commanded speed {robot_speed:.2f}m/s exceeds the {max_speed_in_zone_mps}m/s collaborative cap",
                )
        prev = point
    return _ok("reduced_speed_near_human", "commanded speed stayed within the collaborative cap near every tracked agent")


def environment_hazard_clear(
    state: WorldState, action: Action, trajectory: PredictedTrajectory,
) -> PreconditionResult:
    """Is the environment itself free of a detected hazard (spill, smoke, ...)? Found missing while
    scoping a broader environmental hazard sweep: EnvironmentSignals.surface_hazards existed in the
    schema this whole time and nothing anywhere ever read it -- a silently dead field."""
    if state.environment.surface_hazards:
        return _fail("environment_hazard_clear", f"environment hazards detected: {sorted(state.environment.surface_hazards)}")
    return _ok("environment_hazard_clear", "no environment hazard detected")


# ---------------------------------------------------------------------------------------------
# Next safety checks roadmap (2026-09): liveness, payload/grip force, stability (above), sensor
# staleness and coverage, command and config integrity, vulnerable bystanders. Each closes a case
# that every check above permits -- see tests/test_adversarial_next_checks.py for the scenario
# each one was written against, demonstrated passing the pre-existing checks first.
# ---------------------------------------------------------------------------------------------


@_wants_check_context
def decision_within_deadline(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    max_decision_latency_s: float = DEFAULT_MAX_DECISION_LATENCY_S,
    context: Optional[CheckContext] = None,
) -> PreconditionResult:
    """Software liveness, half one of two: was this decision made within its deadline?

    Every other check assumes the checker itself is running normally. It may not be: a dynamics
    adapter that stalls for half a second, a GC pause, a perception call that blocks on a dead
    socket and eventually returns. A decision like that still reaches PERMIT on every other check --
    each one evaluates the world *as it was when the decision started*, which is no longer the world
    the PERMIT will be acted in. This check measures the decision's own elapsed time, from the
    moment ActuatorGate.gate() began (before perception) to the moment this check runs, on the
    gate's own monotonic clock, and fails once it exceeds ``max_decision_latency_s``. List it LAST
    in an action type's checks so it covers every check before it.

    Half two is DecisionWatchdog (safety_harness/watchdog.py). This check can only catch a decision
    that is late; it cannot catch one that never finishes, or a checker that has crashed -- a
    check that never runs can't fail. The watchdog sits on the actuator side and freezes the robot
    when no fresh PERMIT has arrived within its own deadline, whatever the reason. Neither half is a
    substitute for a hardware watchdog timer: if the whole process hangs, only hardware can act.

    Needs a CheckContext (supplied by the engine); called without one, it fails closed -- there is
    no decision start time to measure from. A non-finite or negative elapsed time (a clock that ran
    backwards, a corrupted start stamp) fails closed too.
    """
    name = "decision_within_deadline"
    if context is None or not _is_real_number(context.decision_started_at):
        return _fail(name, "no decision start time supplied -- not running under ActuatorGate; default-deny")
    elapsed = context.clock() - context.decision_started_at
    if _below(elapsed, 0.0):
        return _fail(name, f"decision elapsed time {elapsed!r}s is negative or non-finite -- clock unconfirmed")
    if _exceeds(elapsed, max_decision_latency_s):
        return _fail(
            name,
            f"decision took {elapsed * 1000:.1f}ms, over its {max_decision_latency_s * 1000:.1f}ms deadline -- "
            f"every check above evaluated a world that has since moved on",
        )
    return _ok(name, f"decision made in {elapsed * 1000:.1f}ms, within its {max_decision_latency_s * 1000:.1f}ms deadline")


def payload_and_grip_force_within_limits(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    object_id_param: str = "object_id",
    grip_force_param: str = "grip_force_n",
    rated_payload_kg: Optional[float] = None,
    max_grip_force_n: float = 70.0,
    max_grip_force_fragile_n: float = 15.0,
    friction_coefficient: float = 0.3,
    contact_count: int = 2,
    grip_safety_factor: float = 2.0,
) -> PreconditionResult:
    """The object-side counterpart of ISO/TS 15066's human-contact force limits: can the robot
    carry this object at all, and is the commanded grip force neither too weak to hold it nor too
    strong for it?

    * Payload: the object's confirmed mass must not exceed the robot's rated payload. The rating
      comes from the robot itself (``RobotProprioception.rated_payload_kg``, its datasheet figure)
      and/or from ``rated_payload_kg`` here; when both are present the lower one governs, when
      neither is, default-deny. Distinct from mass_within_force_budget, whose flat per-config
      budget knows nothing about which robot is running it -- a config written for a 3kg-payload arm
      and reused on a 0.5kg-payload one keeps permitting 3kg lifts until this check says otherwise.
    * Too weak: ``action.params[grip_force_param]`` must reach the minimum friction grip that holds
      the object against gravity, ``m * g * safety_factor / (mu * contact_count)`` -- below that the
      object slips mid-carry, a drop hazard no other check reasons about.
    * Too strong: it must not exceed ``max_grip_force_n`` (a representative parallel-gripper
      continuous-force figure), or ``max_grip_force_fragile_n`` for an object tagged FRAGILE or
      LIQUID_CONTAINING (a crushed glass is a sharp-edged, possibly liquid-releasing hazard), or the
      object's own ``max_safe_grip_force_n`` if perception knows one. The lowest cap governs.

    Missing object, mass, payload rating or commanded grip force; a NaN/negative/non-numeric value
    anywhere; or an UNKNOWN hazard class (fragility unconfirmed): all default-deny. The friction and
    force-cap defaults are representative values for illustrating the check's shape, not a
    characterized figure for any real gripper or object -- same caveat as every ISO-derived default
    in this module.
    """
    name = "payload_and_grip_force_within_limits"
    obj_id = action.params.get(object_id_param)
    obj = next((o for o in state.objects if o.object_id == obj_id), None)
    if obj is None:
        return _fail(name, f"object {obj_id!r} not in perceived world state")
    mass = obj.estimated_mass_kg
    if mass is None or not _is_real_number(mass) or _below(mass, 0.0):
        return _fail(name, f"no confirmed, finite, non-negative mass for {obj_id!r} (got {mass!r})")

    robot_rating = state.robot.rated_payload_kg if state.robot is not None else None
    ratings = [r for r in (rated_payload_kg, robot_rating) if r is not None]
    if not ratings:
        return _fail(name, "no rated payload reported by the robot or configured -- default-deny")
    for rating in ratings:
        if not _is_real_number(rating) or _exceeds(mass, rating):
            return _fail(name, f"{mass:.2f}kg exceeds (or cannot be confirmed within) the {rating!r}kg rated payload")

    if HazardTag.UNKNOWN in obj.hazard_tags:
        return _fail(name, f"hazard class of {obj_id!r} unknown -- no grip force can be confirmed safe for it")
    grip = action.params.get(grip_force_param)
    if grip is None or not _is_real_number(grip) or not math.isfinite(grip) or grip <= 0:
        return _fail(name, f"no finite, positive commanded grip force in params[{grip_force_param!r}] (got {grip!r}) -- default-deny")

    model = (friction_coefficient, contact_count, grip_safety_factor)
    if not all(_is_real_number(v) and math.isfinite(v) and v > 0 for v in model):
        return _fail(name, f"grip model parameters {model!r} must be finite and positive")
    min_hold_n = mass * GRAVITY_MPS2 * grip_safety_factor / (friction_coefficient * contact_count)
    if _below(grip, min_hold_n):
        return _fail(
            name,
            f"grip {grip:.1f}N below the {min_hold_n:.1f}N needed to hold {mass:.2f}kg "
            f"(mu={friction_coefficient}, safety factor {grip_safety_factor}) -- the object would slip",
        )

    caps = [(max_grip_force_n, "platform grip-force cap")]
    if obj.hazard_tags & _CRUSH_RISK_TAGS:
        caps.append((max_grip_force_fragile_n, "fragile/liquid-containing grip-force cap"))
    if obj.max_safe_grip_force_n is not None:
        caps.append((obj.max_safe_grip_force_n, f"{obj_id!r}'s own crush limit"))
    for cap, label in caps:
        if not _is_real_number(cap) or _exceeds(grip, cap):
            return _fail(name, f"grip {grip:.1f}N exceeds (or cannot be confirmed within) the {cap!r}N {label}")
    return _ok(name, f"{mass:.2f}kg within rated payload; grip {grip:.1f}N holds it without exceeding any cap")


def sensor_data_fresh(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    max_sensor_age_s: float = DEFAULT_MAX_SENSOR_AGE_S,
    max_future_skew_s: float = 0.0,
    now: Optional[float] = None,
) -> PreconditionResult:
    """Is the sensor data this WorldState was built from recent enough to act on?

    ``WorldState.timestamp`` can't answer this: it records when the state object was *assembled*,
    so a state built right now from a camera frame captured two seconds ago looks perfectly fresh
    to it, and to every check above -- a person who walked into the path in those two seconds is
    simply absent. ``WorldState.sensor_timestamp`` is the capture time of the OLDEST sensor reading
    the state was built from; this check compares it to the wall clock (``time.time()``, the same
    epoch sensor drivers stamp with) and fails once it is older than ``max_sensor_age_s``.

    Unreported (None), non-numeric or non-finite: default-deny -- an adapter has to report a
    capture time to pass, there is no default that assumes one. A capture time in the future by more
    than ``max_future_skew_s`` (default 0: none tolerated) also fails: it means the sensor's clock
    and this host's disagree, and freshness can't be established across clocks that disagree. A
    deployment with sensors on separately-synchronized hosts sets the tolerance from its measured
    clock sync, rather than this check assuming one. ``now`` exists for deterministic tests.
    """
    name = "sensor_data_fresh"
    ts = state.sensor_timestamp
    if ts is None:
        return _fail(name, "no sensor capture timestamp reported -- freshness unconfirmed, default-deny")
    if not _is_real_number(ts) or not math.isfinite(ts):
        return _fail(name, f"sensor capture timestamp {ts!r} is non-numeric or non-finite")
    now = time.time() if now is None else now
    age = now - ts
    if _below(age, -max_future_skew_s):
        return _fail(name, f"sensor timestamp is {-age:.3f}s in the future -- sensor and host clocks disagree")
    if _exceeds(age, max_sensor_age_s):
        return _fail(name, f"sensor data is {age:.3f}s old, over the {max_sensor_age_s:.3f}s limit")
    return _ok(name, f"sensor data is {age:.3f}s old, within the {max_sensor_age_s:.3f}s limit")


def _sphere_inside_region(center, radius: float, region: ObservedRegion) -> bool:
    """Is the whole sphere inside the box -- every coordinate finite, box well-formed? Any doubt is
    False (not observed), never True."""
    try:
        lo, hi = tuple(region.min_corner), tuple(region.max_corner)
        center = tuple(center)
    except TypeError:
        return False
    if len(lo) != 3 or len(hi) != 3 or len(center) != 3:
        return False
    if not (_is_real_number(radius) and math.isfinite(radius) and radius >= 0):
        return False
    for c, a, b in zip(center, lo, hi):
        if not all(_is_real_number(v) and math.isfinite(v) for v in (c, a, b)):
            return False
        if not (a + radius <= c <= b - radius):
            return False
    return True


def _finite_box(region):
    """(lo, hi) as float triples if the region is a well-formed finite box, else None."""
    try:
        lo, hi = tuple(region.min_corner), tuple(region.max_corner)
    except (TypeError, AttributeError):
        return None
    if len(lo) != 3 or len(hi) != 3:
        return None
    if not all(_is_real_number(v) and math.isfinite(v) for v in lo + hi):
        return None
    if any(a > b for a, b in zip(lo, hi)):
        return None
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _aabb_covered(lo, hi, boxes) -> bool:
    """Is the axis-aligned box [lo, hi] entirely inside the union of ``boxes``? Exact, by coordinate
    compression: split [lo, hi] at every box face inside it, and require the midpoint of every
    resulting cell to lie in some box. (Cells are either wholly inside a box or wholly outside it.)"""
    if not boxes:
        return False
    cuts = []
    for ax in range(3):
        pts = {lo[ax], hi[ax]}
        for blo, bhi in boxes:
            for v in (blo[ax], bhi[ax]):
                if lo[ax] < v < hi[ax]:
                    pts.add(v)
        cuts.append(sorted(pts))
    for x0, x1 in zip(cuts[0], cuts[0][1:] or cuts[0]):
        for y0, y1 in zip(cuts[1], cuts[1][1:] or cuts[1]):
            for z0, z1 in zip(cuts[2], cuts[2][1:] or cuts[2]):
                m = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
                if not any(all(b[0][k] <= m[k] <= b[1][k] for k in range(3)) for b in boxes):
                    return False
    return True


def swept_path_observed(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    margin_m: float = 0.15,
) -> PreconditionResult:
    """Was the space the robot is about to sweep through actually *observed* this cycle?

    Every human- and object-proximity check above reasons only about what perception reported.
    "No agent within clearance" is evidence the path is clear only where a sensor could see; in
    space occluded by a cabinet, or outside every camera's field of view, "no agent reported" is
    the absence of evidence, not evidence of absence. visibility_above_threshold can't tell these
    apart either: it is one scene-wide confidence number, happily 1.0 while the camera sees its
    own half of the table perfectly and nothing at all of the half the arm is about to reach into.

    This is coverage, deliberately distinct from confidence. Observed-but-uncertain (fog, glare, a
    low-confidence track) is still visibility_above_threshold's and the tracking checks' job;
    unobserved is this one's. Every predicted point's swept sphere, inflated by ``margin_m``, must
    lie entirely inside the union of the ``WorldState.observed_regions`` boxes and any declared
    ``WorldState.solid_regions`` (known static solid geometry, where no agent can be). The union is
    tested exactly over the sphere's bounding box, which contains the sphere, so this stays
    conservative. Coverage unreported (None), empty, or made only of malformed/non-finite regions:
    default-deny. Solid regions can only add coverage next to observed space; they never stand in
    for observation where an agent could actually be.

    ``margin_m`` defaults to the same 0.15m as the other swept-path checks. In a speed-and-
    separation-monitored cell, set it to at least the protective separation distance: an unobserved
    person just outside the observed box could otherwise reach the path before being seen.
    """
    name = "swept_path_observed"
    regions = state.observed_regions
    if regions is None:
        return _fail(name, "no sensor coverage reported -- whether the swept path was observed at all is unknown; default-deny")
    valid = tuple(r for r in regions if isinstance(r, ObservedRegion)) if isinstance(regions, (tuple, list)) else ()
    if not valid:
        return _fail(name, "no observed region reported -- nothing was observed, so no path through it is confirmed clear")
    if not trajectory.points:
        return _fail(name, "no predicted trajectory to evaluate")
    solids = state.solid_regions if isinstance(state.solid_regions, (tuple, list)) else ()
    boxes = [b for b in (_finite_box(r) for r in valid) if b is not None]
    boxes += [b for b in (_finite_box(r) for r in solids if isinstance(r, KnownSolidRegion)) if b is not None]
    for point in trajectory.points:
        radius = point.swept_volume_radius_m + margin_m
        if any(_sphere_inside_region(point.swept_volume_center, radius, region) for region in valid):
            continue  # fast path: wholly inside one observed box
        c = point.swept_volume_center
        try:
            ok_geom = (len(c) == 3 and all(_is_real_number(v) and math.isfinite(v) for v in c)
                       and _is_real_number(radius) and math.isfinite(radius) and radius >= 0)
        except TypeError:
            ok_geom = False
        if not (ok_geom and _aabb_covered(tuple(v - radius for v in c), tuple(v + radius for v in c), boxes)):
            return _fail(
                name,
                f"swept volume at t={point.t:.2f}s (center {point.swept_volume_center}, radius+margin {radius:.2f}m) "
                f"extends outside every observed region -- unobserved, not merely uncertain",
            )
    return _ok(name, "every point of the swept path lies inside an observed region")


@_wants_check_context
def command_integrity_verified(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    require_seal: bool = False,
    seal_param: str = COMMAND_SEAL_PARAM,
    hmac_key_env: Optional[str] = None,
    context: Optional[CheckContext] = None,
) -> PreconditionResult:
    """Is the action being checked bit-identical to the one that will be executed?

    ``Action`` is frozen but its ``params`` dict isn't, so a checked action can be rewritten by
    anyone holding a reference to it -- after the decision (closed by the engine binding a digest
    into ``Decision.action_digest`` and the executor re-verifying it: integrity.
    verify_decision_action / DecisionWatchdog.command()) or *during* it, between one check and the
    next. The during-case is this check's job. The engine hashes the proposed action when gate()
    begins; this check re-hashes it and fails on any difference: a rewrite part-way through means
    the checks earlier in the list evaluated a different command than the one that would execute.
    List it at the END of an action type's checks (only decision_within_deadline after it) so it
    covers every check before it.

    Fails closed on: no CheckContext or no start-of-decision digest; an action with no canonical
    encoding (an opaque or lazily-evaluated param object whose value can't be pinned -- if it
    can't be hashed it can't be proven unchanged); a digest mismatch.

    Optional upstream layer: a proposer may seal its command (integrity.seal_action, HMAC-keyed if
    ``hmac_key_env`` names an environment variable holding the key) so tampering *before* the gate
    is caught too. A seal that is present is always verified, and a wrong one always fails.
    ``require_seal`` makes an unsealed command fail as well -- set it wherever proposers seal, or
    stripping the seal would be a way around it. A named key that isn't set fails closed rather
    than silently verifying unkeyed.
    """
    name = "command_integrity_verified"
    current = try_action_digest(action)
    if current is None:
        return _fail(
            name,
            "action has no canonical encoding (an opaque, lazily-evaluated or self-referencing param?) -- "
            "it can't be bound to a digest, so it can't be proven unchanged at execution; default-deny",
        )
    if context is None or context.checked_action_digest is None:
        return _fail(name, "no digest of the action as it was when the decision began -- default-deny")
    if not digests_match(context.checked_action_digest, current):
        return _fail(
            name,
            "action was modified while it was being checked -- earlier checks evaluated a different "
            "command than the one that would be executed",
        )
    has_seal = seal_param in action.params
    if require_seal or has_seal:
        key = None
        if hmac_key_env:
            key = key_from_env(hmac_key_env)
            if key is None:
                return _fail(name, f"command-seal key variable {hmac_key_env!r} is unset -- cannot authenticate the seal")
        if not has_seal:
            return _fail(name, f"command carries no seal in params[{seal_param!r}] and this schema requires one")
        if not verify_action_seal(action, key=key, seal_param=seal_param):
            return _fail(name, "command seal does not match its content -- altered after it was issued, or sealed with another key")
    return _ok(name, "action unchanged since the decision began" + (" and its seal verifies" if has_seal else ""))


@_wants_check_context
def config_integrity_verified(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    context: Optional[CheckContext] = None,
) -> PreconditionResult:
    """Does the configuration running this decision still match its known-good digest?

    The action schema (which checks each action type runs) and every check's parameters are what
    was validated. A silently edited YAML -- a force budget raised, a check deleted -- an
    in-process rebinding of a REGISTRY entry, or a threshold default changed in code each change
    validated behavior while every check still reports satisfied, because every check is only
    checking what it was told to. ActionSchemaRegistry.config_digest() hashes the *effective*
    running configuration (see its docstring for exactly what's covered); this check compares it,
    on every single decision, to the digest pinned when the registry was constructed. So a tamper
    after load fails here even though the load-time verification passed.

    The pin can't live inside the config it protects -- whoever edits one edits both -- so it is
    supplied to ActionSchemaRegistry(expected_digest=...) from outside (a release manifest, a
    secrets store, ``configs/example_action_schema.yaml.sha256`` for the example). With an HMAC key
    (hmac_key=...), held where the config's editor can't read it, the pin is a signature; without
    one it detects corruption and uncoordinated edits but not an attacker who rewrites both.

    No pinned digest at all: default-deny, because an unpinned config is an unvalidated one. No
    CheckContext / no registry supplied: default-deny.
    """
    name = "config_integrity_verified"
    registry = context.schema_registry if context is not None else None
    if registry is None:
        return _fail(name, "no running configuration supplied to verify -- default-deny")
    if getattr(registry, "expected_digest", None) is None:
        return _fail(name, "running configuration has no known-good digest pinned -- unvalidated config, default-deny")
    if not registry.verify_config():
        return _fail(name, "running configuration no longer matches its pinned known-good digest -- changed since it was validated")
    return _ok(name, "running configuration matches its pinned known-good digest")


# ISO/TS 15066:2016 Annex A, Table A.2: maximum permissible quasi-static contact force by body
# region, in newtons. The standard allows transient contact up to twice the quasi-static value
# (``transient_multiplier``) -- except for the head regions (skull/forehead, face), where contact is
# to be avoided and this module applies no multiplier at all, conservatively. The height bands are
# NOT from the standard, which defines regions anatomically: they are this module's own coarse
# standing-posture approximation, as fractions of stature, of where each region can be. Hands are
# assumed able to be anywhere from the floor to overhead reach. Every figure here needs a qualified
# safety engineer to confirm against the current edition before any real deployment relies on it --
# the same caveat as iso15066_power_force_limiting.
_ISO15066_BODY_REGIONS = (
    # (region, quasi-static max force N, (band low, band high) as a fraction of stature, transient contact allowed)
    ("skull_and_forehead", 130.0, (0.89, 1.00), False),
    ("face", 65.0, (0.86, 0.95), False),
    ("neck", 150.0, (0.81, 0.88), True),
    ("back_and_shoulders", 210.0, (0.70, 0.84), True),
    ("chest", 140.0, (0.70, 0.82), True),
    ("abdomen", 110.0, (0.56, 0.71), True),
    ("pelvis", 180.0, (0.46, 0.58), True),
    ("upper_arms_and_elbows", 150.0, (0.58, 0.84), True),
    ("lower_arms_and_wrists", 160.0, (0.38, 0.66), True),
    ("hands_and_fingers", 140.0, (0.00, 1.25), True),
    ("thighs_and_knees", 220.0, (0.25, 0.52), True),
    ("lower_legs", 130.0, (0.00, 0.30), True),
)


def _body_region_force_limit(swept_z, swept_r, floor_height_m, stature_m, transient_multiplier):
    """(lowest permissible force, region name) among the body regions the swept sphere's height band
    overlaps. Unknown/non-finite stature or height: every region counts (the face included) --
    fail-closed. (inf, None) only when stature and height are confirmed and no region overlaps."""
    known = all(_is_real_number(v) and math.isfinite(v) for v in (swept_z, swept_r, floor_height_m, stature_m)) and stature_m > 0
    best_limit, best_region = math.inf, None
    for region, force, (lo, hi), transient in _ISO15066_BODY_REGIONS:
        if known:
            band_lo = (swept_z - swept_r - floor_height_m) / stature_m
            band_hi = (swept_z + swept_r - floor_height_m) / stature_m
            if band_hi < lo or band_lo > hi:
                continue
        limit = force * (transient_multiplier if transient else 1.0)
        if limit < best_limit:
            best_limit, best_region = limit, region
    return best_limit, best_region


def vulnerable_bystander_protected(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    adult_min_clearance_m: float = 0.15,
    vulnerable_min_clearance_m: float = 1.0,
    crowd_size: int = 2,
    crowd_radius_m: float = 2.0,
    crowd_clearance_scale: float = 2.0,
    crowd_force_scale: float = 0.75,
    contact_plausible_range_m: float = 0.3,
    effective_mass_kg: float = 5.0,
    assumed_contact_time_s: float = 0.015,
    transient_multiplier: float = 2.0,
    floor_height_m: float = 0.0,
) -> PreconditionResult:
    """Different distance and force limits for who is actually there, instead of one "human" limit.

    Every human-proximity check above treats every tracked agent identically -- the same margin,
    the same flat 150N force limit -- whether it is one adult, a child, or three people at once.
    ISO/TS 15066 itself doesn't: its biomechanical limits differ by body region (65N at the face,
    220N at the thighs; see _ISO15066_BODY_REGIONS), and they were derived for adult workers only.

    * Who: ``TrackedAgent.category``. ADULT gets the adult limits below. CHILD, ANIMAL and UNKNOWN
      (the default -- unclassified is never assumed to be an adult) are *vulnerable*: no contact
      is permitted at all, because no standard's force limits cover them, and a larger clearance,
      ``vulnerable_min_clearance_m``, is required.
    * Distance: every agent must stay at least swept radius + its own worst-case reach by that
      point's time (same model as swept_path_clear_of_agents) + its category's clearance away from
      the swept path. For one tracked adult with the defaults, this reproduces
      swept_path_clear_of_agents' own 0.15m margin exactly -- the baseline is unchanged.
    * Crowd: once ``crowd_size`` or more agents are within ``crowd_radius_m`` of the swept path,
      every clearance is multiplied by ``crowd_clearance_scale`` and every force limit by
      ``crowd_force_scale``. With several people, one can push or be pushed into the robot, one can
      occlude another from the tracker, and a reactive behavior that slows for the nearest person
      says nothing about the second.
    * Force (adults only, within ``contact_plausible_range_m``): the same transient-impact estimate
      as iso15066_power_force_limiting (``m_eff * closing_speed / contact_time``), but compared
      against the lowest limit among the ISO/TS 15066 body regions the swept volume can reach *at
      its height*, given the agent's ``stature_m``. A robot reaching up to a shelf at an adult's face
      height gets the 65N face limit, not the flat 150N -- the case this check blocks and that one
      permits. Unknown stature means every region, face included, is in reach.

    The clearance and scaling defaults are representative values chosen for the structure of the
    check, not figures from any standard -- ISO/TS 15066 has no child or crowd provisions to take
    them from, which is exactly the gap. A qualified safety engineer sets them for a deployment, the
    same caveat as every ISO-derived default in this module. A NaN distance or position fails
    closed through ``_below``, and a NaN distance is never used to skip the force estimate, same as
    iso15066_power_force_limiting.
    """
    name = "vulnerable_bystander_protected"
    points = trajectory.points
    if not points:
        return _fail(name, "no predicted trajectory to evaluate")
    if not state.agents:
        return _ok(name, "no tracked agents")
    # A NaN multiplier or scale would make every "limit < best" comparison below False and silently
    # select no body region at all -- validate the force model up front instead.
    force_model = (effective_mass_kg, assumed_contact_time_s, transient_multiplier, crowd_force_scale)
    if not all(_is_real_number(v) and math.isfinite(v) and v > 0 for v in force_model):
        return _fail(name, f"force model parameters {force_model!r} must be finite and positive")

    near = sum(
        1 for agent in state.agents
        if any(_at_or_within(_distance(p.swept_volume_center, agent.pose.position), crowd_radius_m) for p in points)
    )
    # Written as "not fewer than" so a NaN crowd_size makes this a crowd (the stricter case).
    crowd = not (near < crowd_size)
    clearance_scale = crowd_clearance_scale if crowd else 1.0
    force_scale = crowd_force_scale if crowd else 1.0
    crowd_note = f"; crowd of {near} within {crowd_radius_m}m" if crowd else ""

    prev = None
    for point in points:
        robot_speed = 0.0 if prev is None else _speed(prev, point)
        for agent in state.agents:
            vulnerable = agent.category != AgentCategory.ADULT
            who = getattr(agent.category, "value", agent.category)
            d = _distance(point.swept_volume_center, agent.pose.position)
            clearance = (vulnerable_min_clearance_m if vulnerable else adult_min_clearance_m) * clearance_scale
            required = point.swept_volume_radius_m + agent.worst_case_radius_m(point.t) + clearance
            if _below(d, required):
                return _fail(
                    name,
                    f"{who} agent {agent.agent_id!r} within {d:.2f}m at t={point.t:.2f}s "
                    f"(needs {required:.2f}m for a {'vulnerable' if vulnerable else 'adult'} bystander{crowd_note})",
                )
            if math.isfinite(d) and d > contact_plausible_range_m:
                continue  # confirmed too far for contact to be physically plausible at this point
            if vulnerable:
                return _fail(
                    name,
                    f"{who} agent {agent.agent_id!r} within contact range at t={point.t:.2f}s -- ISO/TS 15066's "
                    f"force limits are for adults only; no contact is permitted with a {who} bystander",
                )
            force = effective_mass_kg * (robot_speed + agent.worst_case_speed_mps) / assumed_contact_time_s
            limit, region = _body_region_force_limit(
                point.swept_volume_center[2], point.swept_volume_radius_m, floor_height_m, agent.stature_m, transient_multiplier,
            )
            if region is None:
                continue  # confirmed: no body region at the swept height (e.g. far above the head)
            limit *= force_scale
            if _exceeds(force, limit):
                return _fail(
                    name,
                    f"estimated {force:.0f}N contact with adult {agent.agent_id!r}'s {region} at t={point.t:.2f}s "
                    f"exceeds its {limit:.0f}N ISO/TS 15066 body-region limit{crowd_note}",
                )
        prev = point
    return _ok(name, "every tracked agent kept its category's clearance and body-region force limit" + crowd_note)


# Registered for backward compatibility only; ActionSchemaRegistry warns when a config uses one.
DEPRECATED_CHECKS = {"surface_confirmed_stable": "destination_confirmed_stable_and_clear"}

REGISTRY = {
    "object_hazard_confirmed": object_hazard_confirmed,
    "mass_within_force_budget": mass_within_force_budget,
    "swept_path_clear_of_agents": swept_path_clear_of_agents,
    "balance_margin_maintained": balance_margin_maintained,
    "visibility_above_threshold": visibility_above_threshold,
    "surface_confirmed_stable": surface_confirmed_stable,
    "object_cleared_for_interaction": object_cleared_for_interaction,
    "swept_path_clear_of_risky_objects": swept_path_clear_of_risky_objects,
    "current_position_confirmed_stable": current_position_confirmed_stable,
    "destination_confirmed_stable_and_clear": destination_confirmed_stable_and_clear,
    "fall_consequence_acceptable": fall_consequence_acceptable,
    "joint_position_limits_respected": joint_position_limits_respected,
    "joint_velocity_within_limits": joint_velocity_within_limits,
    "joint_effort_within_limits": joint_effort_within_limits,
    "motor_temperature_within_limits": motor_temperature_within_limits,
    "cartesian_speed_within_limits": cartesian_speed_within_limits,
    "self_collision_clear": self_collision_clear,
    "battery_charge_sufficient": battery_charge_sufficient,
    "object_pose_confirmed": object_pose_confirmed,
    "robot_state_confirmed": robot_state_confirmed,
    "environment_hazard_clear": environment_hazard_clear,
    "iso15066_separation_distance_maintained": iso15066_separation_distance_maintained,
    "iso15066_power_force_limiting": iso15066_power_force_limiting,
    "reduced_speed_near_human": reduced_speed_near_human,
    # Next safety checks roadmap. balance_margin_maintained above is now the legacy name for
    # stability_margin_maintained's static-only half (same implementation, unchanged behavior):
    # 32 registry names, 31 distinct checks.
    "decision_within_deadline": decision_within_deadline,
    "payload_and_grip_force_within_limits": payload_and_grip_force_within_limits,
    "stability_margin_maintained": stability_margin_maintained,
    "sensor_data_fresh": sensor_data_fresh,
    "swept_path_observed": swept_path_observed,
    "command_integrity_verified": command_integrity_verified,
    "config_integrity_verified": config_integrity_verified,
    "vulnerable_bystander_protected": vulnerable_bystander_protected,
}
