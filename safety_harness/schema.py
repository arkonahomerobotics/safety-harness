"""Core data contract for the safety harness.

Stable and versioned: every adapter targets this shape regardless of what robot or perception stack
produced it. See the design doc's "Perception -> Structured World State" and "Reference Module
Design" sections. Changing this file after adapters exist against it is the expensive move the
design doc's Development Roadmap warns about -- treat SCHEMA_VERSION bumps deliberately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

SCHEMA_VERSION = "0.2.0"


class HazardTag(str, Enum):
    FRAGILE = "fragile"
    HOT = "hot"
    SHARP = "sharp"
    HUMAN = "human"
    ANIMAL = "animal"
    LIQUID_CONTAINING = "liquid_containing"
    UNKNOWN = "unknown"  # the default -- treated as the most restrictive assumption downstream


class FallConsequence(str, Enum):
    """What happens if this object falls or is struck -- distinct from whether it's currently in a
    stable position. Gates whether lifting it to a given height is acceptable at all (see
    ``fall_consequence_acceptable`` in preconditions.py)."""

    NONE = "none"  # confirmed: falling has no meaningful consequence
    MESS = "mess"  # breaks or spills, but nothing hazardous released
    HAZARDOUS_RELEASE = "hazardous_release"  # breaking releases something dangerous (sharp, chemical, biohazard)
    UNKNOWN = "unknown"  # the default -- unconfirmed, treated as the most restrictive assumption


class DecisionVerdict(str, Enum):
    PERMIT = "permit"
    BLOCK = "block"


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TrackedObject:
    """A perceived, roughly static-to-slow-moving object -- a cube, a cup, a tool."""

    object_id: str
    object_class: str
    pose: Pose
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    estimated_mass_kg: Optional[float] = None
    hazard_tags: frozenset = field(default_factory=lambda: frozenset({HazardTag.UNKNOWN}))
    pose_confidence: float = 0.0  # [0, 1]; 0 = no confirming evidence
    class_confidence: float = 0.0
    # Whether this specific object has been affirmatively confirmed safe to approach/handle at all --
    # distinct from hazard_tags, which describe how to handle it once cleared. Defaults to False:
    # an object perception hasn't explicitly cleared is off-limits, not merely "handle with care."
    # This is what lets a *bystander* object -- not the action's target -- block an action just by
    # being near the swept path (see swept_path_clear_of_risky_objects).
    cleared_for_interaction: bool = False
    # Whether perception has confirmed this object's current resting position is itself stable --
    # not teetering on an edge, not the top of a precarious stack. None = unconfirmed -> deny.
    supported_stably: Optional[bool] = None
    # What breaking or falling releases; gates how high it may safely be lifted (fall_consequence_
    # acceptable). Unconfirmed defaults to UNKNOWN, the most restrictive case.
    fall_consequence: FallConsequence = FallConsequence.UNKNOWN
    # Max fall height this object tolerates without breaking/releasing a hazard. None = unconfirmed;
    # treated as 0.0 (no lift height is acceptable) whenever fall_consequence != NONE.
    drop_tolerance_m: Optional[float] = None


@dataclass(frozen=True)
class TrackedAgent:
    """A human or animal -- always assumed capable of moving, bounded by a worst-case velocity."""

    agent_id: str
    pose: Pose
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    time_since_confirmed_s: float = 0.0
    tracking_confidence: float = 0.0
    worst_case_speed_mps: float = 1.5  # conservative human walking speed unless perception says otherwise

    def worst_case_radius_m(self, horizon_s: float) -> float:
        """How far this agent could plausibly be by ``horizon_s`` from now, given tracking staleness."""
        return self.worst_case_speed_mps * (self.time_since_confirmed_s + horizon_s)


@dataclass(frozen=True)
class RobotProprioception:
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    end_effector_pose: Pose
    gripper_state: float  # 0 = closed, 1 = open, continuous in between
    center_of_mass: Optional[tuple[float, float, float]] = None  # legged/humanoid only
    support_polygon: Optional[tuple] = None  # legged/humanoid only: tuple of (x, y) ground points
    # Robot self-limits -- fixed per robot model (from its URDF/datasheet), not something perception
    # observes freshly each cycle, but carried alongside state so the same per-point trajectory
    # checking machinery used for balance/clearance can also check the robot's own physical limits.
    # None means "not reported" and every check below treats that as default-deny, same as elsewhere.
    joint_position_limits: Optional[tuple] = None  # tuple of (min, max) pairs, same order as joint_positions
    joint_velocity_limits: Optional[tuple] = None  # tuple of max |velocity|, same order as joint_velocities
    joint_effort_limits: Optional[tuple] = None  # tuple of max |torque or current| per joint
    estimated_joint_efforts: Optional[tuple] = None  # current estimated/commanded effort per joint
    motor_temperature_c: Optional[tuple] = None  # current per-joint motor temperature, if sensed
    motor_temperature_limit_c: Optional[tuple] = None  # safe upper bound per joint
    max_cartesian_speed_mps: Optional[float] = None  # this platform's end-effector speed limit
    battery_charge_fraction: Optional[float] = None  # [0, 1] current state of charge; None = not battery-powered or unreported


@dataclass(frozen=True)
class EnvironmentSignals:
    visibility_confidence: float = 1.0  # [0, 1]; degraded lighting/occlusion lowers this
    surface_hazards: frozenset = field(default_factory=frozenset)  # e.g. {"spill", "smoke"}


@dataclass(frozen=True)
class WorldState:
    objects: tuple = ()
    agents: tuple = ()
    robot: Optional[RobotProprioception] = None
    environment: EnvironmentSignals = field(default_factory=EnvironmentSignals)
    schema_version: str = SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Action:
    action_type: str  # e.g. "grasp", "place", "step", "reach"
    params: dict = field(default_factory=dict)  # action-type-specific parameters


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    robot: Optional[RobotProprioception]
    swept_volume_center: tuple[float, float, float]
    swept_volume_radius_m: float  # conservative bounding-sphere radius for the moving part, per point
    # Minimum clearance between any two non-adjacent robot links at this point, as reported by the
    # dynamics adapter's own collision geometry -- not something a generic check can recompute from
    # joint angles alone. None = not reported -> default-deny, same convention as everywhere else.
    self_collision_margin_m: Optional[float] = None


@dataclass(frozen=True)
class PredictedTrajectory:
    points: tuple = ()
    horizon_s: float = 0.0


@dataclass(frozen=True)
class PreconditionResult:
    name: str
    satisfied: bool
    reason: str


@dataclass(frozen=True)
class Decision:
    verdict: DecisionVerdict
    action: Action  # the action to actually execute: original if PERMIT, fallback's if BLOCK
    precondition_results: tuple = ()
    triggered_fallback: bool = False
