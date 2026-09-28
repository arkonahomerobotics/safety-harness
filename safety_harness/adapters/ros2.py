"""Reference PerceptionAdapter + DynamicsAdapter against a real ROS 2 system.

Development Roadmap stage 3, second reference platform pattern (after Isaac Lab): a robot
integration that isn't a simulator at all -- an actual `rclpy` node graph, real topics, real
messages, real timing. See ``examples/ros2_hooks/`` for the node that actually talks to ROS 2 and a
runnable end-to-end demonstration against it.

Like every other adapter in this package (``isaac_lab.py``, ``isaac_lab_g1.py``,
``isaac_lab_anymal.py``), this module imports NOTHING robot- or framework-specific -- no `rclpy`
here, same as those import nothing from `isaaclab`. That is deliberate, not an oversight: it is what
keeps ``pip install safety-harness`` dependency-free (see ``pyproject.toml``'s single declared
dependency, PyYAML, and the SBOM's own drift test). Every field this adapter reads comes off a
``bridge`` object passed into the constructor, duck-typed against real ROS 2 message shapes:

* ``bridge.joint_state``: ``None``, or an object shaped exactly like ``sensor_msgs/msg/JointState``
  -- ``.name`` (sequence of str), ``.position``/``.velocity`` (sequences of float, same order as
  ``.name``), ``.header.stamp`` (``.sec``/``.nanosec``, i.e. ``builtin_interfaces/msg/Time``).
* ``bridge.ee_pose``: ``None``, or an object shaped like ``geometry_msgs/msg/PoseStamped`` --
  ``.pose.position.{x,y,z}``, ``.pose.orientation.{x,y,z,w}``, ``.header.stamp``.
* ``bridge.joint_position_limits`` / ``bridge.joint_velocity_limits`` / ``bridge.joint_effort_limits``:
  ``None``, or a tuple in the same order as ``joint_state.name`` (position: ``(lo, hi)`` pairs, or
  ``None`` entries for an explicitly exempted joint -- a gripper finger resting on its own stop, same
  convention as every other adapter; velocity/effort: a single max-magnitude float per joint, or
  ``None``). No single ROS message carries these; a real bridge node gets all three from the same
  place -- the robot's own URDF, published on ``/robot_description``, whose ``<limit>`` tag carries
  ``lower``/``upper``/``velocity``/``effort`` together. ``joint_state.effort``, when the publisher
  fills it in (``sensor_msgs/msg/JointState`` has the field; not every publisher populates it), maps
  straight to ``estimated_joint_efforts``.
* ``bridge.max_cartesian_speed_mps`` / ``bridge.rated_payload_kg``: floats or ``None`` -- static,
  per-robot figures a real bridge would read from ROS 2 parameters, not a topic.
* ``bridge.tracked_objects`` / ``bridge.tracked_agents``: sequences of small objects carrying this
  project's own safety-relevant fields (``object_id``, ``position``, ``mass_kg``, ``hazard_tags``,
  ``cleared_for_interaction``, ``supported_stably``, ``fall_consequence``, ... / ``agent_id``,
  ``position``, ``velocity``, ``tracking_confidence``, ``category``, ``stature_m``, ...) plus a
  ``stamp``. Deliberately NOT a standard ROS message (unlike the two above): no existing message
  type carries "this object's fall consequence is HAZARDOUS_RELEASE" or "this agent is a CHILD" --
  those are safety_harness's own semantic additions, and any real integration needs its own
  perception-to-WorldState bridge for them regardless of which ROS message library it started from.
  ``examples/ros2_hooks/`` publishes these as JSON on plain ``std_msgs/String`` topics, documented as
  a project-specific minimal contract, not a ROS standard -- see that example's README for why.
* ``bridge.visibility_confidence`` (float, default 1.0 if absent) / ``bridge.observed_regions``
  (``None``, or a sequence of ``(min_corner, max_corner)`` pairs).

A field genuinely not yet received (the topic hasn't published since the bridge node started) reads
as ``None``/empty, which flows straight into this project's existing default-deny handling --
``robot=None`` when no ``joint_state`` has arrived yet, an empty ``objects``/``agents`` tuple when
nothing has been tracked yet. This adapter adds no new default-deny logic of its own; it just doesn't
fabricate data that hasn't actually arrived over the wire.
"""

from __future__ import annotations

from ..schema import (
    AgentCategory,
    EnvironmentSignals,
    FallConsequence,
    HazardTag,
    ObservedRegion,
    Pose,
    PredictedTrajectory,
    RobotProprioception,
    TrackedAgent,
    TrackedObject,
    TrajectoryPoint,
    WorldState,
)
from .base import DynamicsAdapter, PerceptionAdapter


def _stamp_to_epoch_s(stamp) -> float:
    """``builtin_interfaces/msg/Time`` (``.sec``, ``.nanosec``) -> a ``time.time()``-comparable
    float. Only correct when the publishing node's clock is the wall clock, i.e. NOT running with
    ``use_sim_time`` -- see the module and example README for that caveat."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _dist(a, b) -> float:
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5


def _commanded_speed_mps(action, dist_m: float, fallback_speed_mps: float) -> float:
    """Same convention as the Isaac Lab adapters' ``commanded_speed_mps`` helper: prefer what the
    action itself states (``commanded_speed_mps``, or ``duration_s`` implying dist/duration) over a
    configured fallback cap, so a faster-than-expected real command is actually seen as fast by
    ``cartesian_speed_within_limits`` and the swept-path checks, not silently clipped to the cap."""
    params = action.params
    if "commanded_speed_mps" in params:
        v = float(params["commanded_speed_mps"])
    elif "duration_s" in params and float(params["duration_s"]) > 0:
        v = dist_m / float(params["duration_s"])
    else:
        v = fallback_speed_mps
    if v < 0 or v != v:  # negative or NaN
        raise ValueError(f"non-finite/negative commanded speed {v!r}")
    return v


_FALL_CONSEQUENCE = {"none": FallConsequence.NONE, "mess": FallConsequence.MESS, "hazardous_release": FallConsequence.HAZARDOUS_RELEASE}
_AGENT_CATEGORY = {"adult": AgentCategory.ADULT, "child": AgentCategory.CHILD, "animal": AgentCategory.ANIMAL}


class ROS2PerceptionAdapter(PerceptionAdapter):
    """Reads the current WorldState off a ``bridge`` object -- see the module docstring for the
    exact duck-typed contract every field below is read from."""

    def __init__(self, bridge, joint_limit_exempt: tuple = ()):
        self._bridge = bridge
        self._exempt = joint_limit_exempt

    def get_world_state(self) -> WorldState:
        b = self._bridge
        stamps = []

        robot = None
        js = getattr(b, "joint_state", None)
        if js is not None:
            stamps.append(_stamp_to_epoch_s(js.header.stamp))
            ee = getattr(b, "ee_pose", None)
            if ee is not None:
                stamps.append(_stamp_to_epoch_s(ee.header.stamp))
                p, o = ee.pose.position, ee.pose.orientation
                ee_pose = Pose(position=(p.x, p.y, p.z), orientation_wxyz=(o.w, o.x, o.y, o.z))
            else:
                ee_pose = Pose(position=(0.0, 0.0, 0.0))
            limits = getattr(b, "joint_position_limits", None)
            if limits is not None and self._exempt:
                limits = tuple(None if any(tag in name for tag in self._exempt) else lim for name, lim in zip(js.name, limits))
            effort = getattr(js, "effort", None)
            robot = RobotProprioception(
                joint_positions=tuple(js.position),
                joint_velocities=tuple(js.velocity),
                end_effector_pose=ee_pose,
                gripper_state=0.5,  # unreported by this bridge's minimal contract -- a real integration reads its own gripper topic
                joint_position_limits=limits,
                joint_velocity_limits=getattr(b, "joint_velocity_limits", None),
                joint_effort_limits=getattr(b, "joint_effort_limits", None),
                estimated_joint_efforts=tuple(effort) if effort else None,
                max_cartesian_speed_mps=getattr(b, "max_cartesian_speed_mps", None),
                rated_payload_kg=getattr(b, "rated_payload_kg", None),
            )

        objects = []
        for o in getattr(b, "tracked_objects", ()) or ():
            stamps.append(o.stamp)
            objects.append(
                TrackedObject(
                    object_id=o.object_id,
                    object_class=getattr(o, "object_class", "unknown"),
                    pose=Pose(position=tuple(o.position)),
                    velocity=tuple(getattr(o, "velocity", (0.0, 0.0, 0.0))),
                    estimated_mass_kg=getattr(o, "mass_kg", None),
                    hazard_tags=frozenset(HazardTag(t) for t in getattr(o, "hazard_tags", ("unknown",))),
                    pose_confidence=getattr(o, "pose_confidence", 0.0),
                    class_confidence=getattr(o, "class_confidence", 0.0),
                    cleared_for_interaction=getattr(o, "cleared_for_interaction", False),
                    supported_stably=getattr(o, "supported_stably", None),
                    fall_consequence=_FALL_CONSEQUENCE.get(getattr(o, "fall_consequence", "unknown"), FallConsequence.UNKNOWN),
                )
            )

        agents = []
        for a in getattr(b, "tracked_agents", ()) or ():
            stamps.append(a.stamp)
            agents.append(
                TrackedAgent(
                    agent_id=a.agent_id,
                    pose=Pose(position=tuple(a.position)),
                    velocity=tuple(getattr(a, "velocity", (0.0, 0.0, 0.0))),
                    time_since_confirmed_s=getattr(a, "time_since_confirmed_s", 0.0),
                    tracking_confidence=getattr(a, "tracking_confidence", 0.0),
                    worst_case_speed_mps=getattr(a, "worst_case_speed_mps", 1.5),
                    category=_AGENT_CATEGORY.get(getattr(a, "category", "unknown"), AgentCategory.UNKNOWN),
                    stature_m=getattr(a, "stature_m", None),
                )
            )

        regions = getattr(b, "observed_regions", None)
        observed = tuple(ObservedRegion(min_corner=lo, max_corner=hi) for lo, hi in regions) if regions is not None else None

        return WorldState(
            objects=tuple(objects),
            agents=tuple(agents),
            robot=robot,
            environment=EnvironmentSignals(visibility_confidence=getattr(b, "visibility_confidence", 1.0)),
            # The OLDEST reading behind this state, per WorldState.sensor_timestamp's own contract --
            # not "now": a state built from a 2-second-stale joint_state must not look fresh.
            sensor_timestamp=min(stamps) if stamps else None,
            observed_regions=observed,
        )


class ROS2DynamicsAdapter(DynamicsAdapter):
    """Forward-predicts the swept end-effector path toward the action's target at the real
    commanded speed (or a conservative configured cap when the action states none) -- the same
    Cartesian-sweep-only convention as the Isaac Lab adapters use, carrying the current joint state
    on every point (see ``joint_position_limits_respected``'s own docstring on that scope)."""

    def __init__(self, max_ee_speed_mps: float = 0.5, ee_radius_m: float = 0.05, n_points: int = 10):
        self._max_speed = max_ee_speed_mps
        self._ee_radius = ee_radius_m
        self._n_points = n_points

    def predict_trajectory(self, state: WorldState, action, horizon_s: float) -> PredictedTrajectory:
        start = state.robot.end_effector_pose.position
        if "target_position" not in action.params:
            raise ValueError(f"action {action.action_type!r} has no target_position; cannot predict a trajectory")
        target = action.params["target_position"]
        dist = _dist(start, target)
        speed = _commanded_speed_mps(action, dist, self._max_speed)
        n = max(2, self._n_points)
        points = []
        for k in range(n):
            t = horizon_s * k / (n - 1)
            frac = 0.0 if dist < 1e-6 else min(1.0, (speed * t) / dist)
            center = tuple(s + frac * (g - s) for s, g in zip(start, target))
            points.append(TrajectoryPoint(t=t, robot=state.robot, swept_volume_center=center, swept_volume_radius_m=self._ee_radius))
        return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)
