"""Generic precondition checks.

Each operates only on WorldState / PredictedTrajectory / Action -- never on a robot's native
representation -- so the same function works across every adapter (design doc: "Action Precondition
Schemas"). Each returns a PreconditionResult. Missing, stale, or low-confidence evidence must return
satisfied=False, never True -- this file is where the design doc's default-deny rule actually gets
enforced, one function at a time.
"""

from __future__ import annotations

import math

from .schema import Action, FallConsequence, HazardTag, PredictedTrajectory, PreconditionResult, WorldState

# Hazard tags that make an object risky to have nearby even when it isn't the action's target.
_BYSTANDER_RISK_TAGS = frozenset({HazardTag.SHARP, HazardTag.HOT})

MIN_CONFIDENCE = 0.6  # default confidence gate; override per rule via action_schema.yaml kwargs

# iso15066_separation_distance_maintained's two reaction-time defaults, named at module level (not
# just inline in the function signature) so tests/test_latency_stress.py can compare *measured*
# gate() decision latency against the same numbers the check assumes -- one source of truth instead
# of a second hardcoded 0.2 living only in the test file, silently able to drift out of sync.
DEFAULT_REACTION_TIME_S = 0.15
DEFAULT_SAMPLING_INTERVAL_S = 0.05
REACTION_INTERVAL_S = DEFAULT_REACTION_TIME_S + DEFAULT_SAMPLING_INTERVAL_S


def _distance(a, b) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _distance_to_polygon_edge(point_xy, polygon_xy) -> float:
    """Nearest-vertex distance as a conservative stand-in for true polygon-edge distance. Swap in a
    computational-geometry library for a real deployment without changing the call site."""
    if not polygon_xy:
        return 0.0
    if not math.isfinite(point_xy[0]) or not math.isfinite(point_xy[1]):
        # Python's builtin min() silently drops a NaN candidate depending on iteration order
        # (nan < current is False, so a NaN that isn't first in the sequence never replaces a
        # normal running minimum) -- see the design doc's NaN-Sensor Stress Test. Fail explicit and
        # non-finite here rather than let that order-dependence decide the result.
        return math.nan
    return min(_distance((*point_xy, 0.0), (*v, 0.0)) for v in polygon_xy)


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


def balance_margin_maintained(
    state: WorldState, action: Action, trajectory: PredictedTrajectory, *,
    min_margin_m: float = 0.03,
) -> PreconditionResult:
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.center_of_mass is None or robot.support_polygon is None:
            return _fail(
                "balance_margin_maintained",
                "no balance state reported -- default-deny for legged/humanoid platforms",
            )
        margin = _distance_to_polygon_edge(robot.center_of_mass[:2], robot.support_polygon)
        if _below(margin, min_margin_m):
            return _fail(
                "balance_margin_maintained",
                f"center of mass within {margin:.3f}m of support-polygon edge at t={point.t:.2f}s",
            )
    return _ok("balance_margin_maintained", "center of mass stayed within the safe margin throughout")


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
    """General gate, like visibility_above_threshold: is the robot's own proprioceptive state even
    known? Found missing by fuzzing with robot=None: nothing currently wired reads state.robot
    directly, so its absence went completely unnoticed until this test specifically tried it."""
    if state.robot is None:
        return _fail("robot_state_confirmed", "no robot proprioceptive state reported")
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
    not just at the destination, at every predicted point."""
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.joint_position_limits is None:
            return _fail("joint_position_limits_respected", "no joint position limits reported -- default-deny")
        for j, (pos, (lo, hi)) in enumerate(zip(robot.joint_positions, robot.joint_position_limits)):
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
    speeds spike near one even for a modest commanded Cartesian speed."""
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.joint_velocity_limits is None:
            return _fail("joint_velocity_within_limits", "no joint velocity limits reported -- default-deny")
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
    electrical/mechanical load side, distinct from position or speed."""
    for point in trajectory.points:
        robot = point.robot
        if robot is None or robot.joint_effort_limits is None or robot.estimated_joint_efforts is None:
            return _fail("joint_effort_within_limits", "no joint effort limits/estimate reported -- default-deny")
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
    honest proxy: don't add load to a motor that's already close to its limit."""
    robot = state.robot
    if robot is None or robot.motor_temperature_c is None or robot.motor_temperature_limit_c is None:
        return _fail("motor_temperature_within_limits", "no motor temperature reported -- default-deny")
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
        if _exceeds(speed, limit):
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
}
