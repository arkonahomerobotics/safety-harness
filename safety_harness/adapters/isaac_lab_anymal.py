"""Reference PerceptionAdapter + DynamicsAdapter against a legged, moving-base robot -- ANYmal-C
running Isaac Lab's navigation task (Isaac-Navigation-Flat-Anymal-C-v0) -- built specifically to
test whether the engine and precondition library, designed and validated against a fixed-base arm,
generalize to a robot whose own body is the thing sweeping through space (Development Roadmap
stage 4: "Build a second adapter against a genuinely different stack ... specifically to test the
abstraction, not to ship an integration").

Where the Franka cube-stack adapter sweeps an end-effector toward a grasp target, this one sweeps
the robot's own base toward a navigation goal -- the swept_volume_center/radius every human-proximity
check reasons about is now the robot's body, not its gripper. It also activates
balance_margin_maintained for the first time against real robot geometry: this quadruped's four foot
contact positions are its real, measured support polygon. Before this adapter, that check had only
ever seen hand-built fixtures or a NaN-fault-injection probe (design doc: Robot Self-Limits,
NaN-Sensor Stress Test) -- never a real legged robot standing on real ground.

A naming finding worth keeping, not hiding: this adapter reports the robot's base pose through
`RobotProprioception.end_effector_pose`, the same field the Franka adapter uses for its gripper.
There is no schema field for "the point this robot sweeps through space" that isn't named after an
end effector -- the engine and every precondition only ever read it as exactly that (a swept
reference point), so nothing breaks, but the field name itself is Franka-specific and doesn't
describe what it means here. This is precisely the kind of thing the roadmap step "build a second
adapter ... to test the abstraction" is meant to surface. Deliberately not renamed as part of this
adapter -- that would be a schema change (SCHEMA_VERSION bump, both adapters touched) beyond what
adding a second adapter should force; flagged here and in the design doc as a follow-up instead.

NOT YET RUN against a live environment beyond this project's own validation pass -- see the design
doc's Live Validation Results for exactly what was run and what wasn't.
"""

from __future__ import annotations

import math
import time

from ..schema import (
    Action,
    EnvironmentSignals,
    ObservedRegion,
    Pose,
    PredictedTrajectory,
    RobotProprioception,
    TrajectoryPoint,
    WorldState,
)
from ._isaac_lab_common import quat_xyzw_to_wxyz
from .base import DynamicsAdapter, PerceptionAdapter

# ANYmal-C's four foot body names, in this task's fixed body ordering (LF/LH/RF/RH = left-front,
# left-hind, right-front, right-hind). Resolved by name in __init__, not trusted by index order --
# the same discipline the Franka adapter's finger-joint lookup already established, for the same
# reason: index order is an assumption, not a verified fact.
FOOT_BODY_NAMES = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")

# Conservative bounding-circle radius for ANYmal-C's body plus splayed legs, in meters -- a real,
# documented published-footprint fact, not a fabricated guess. Same "conservative bounding sphere"
# convention as the Franka adapter's own ee_radius_m for the gripper.
BODY_FOOTPRINT_RADIUS_M = 0.35

DEFAULT_MAX_BASE_SPEED_MPS = 1.0  # ANYmal-C's typical commanded walking speed; conservative vs. its rated max

# Privileged simulator state is omniscient -- see isaac_lab.py's PRIVILEGED_STATE_OBSERVED_REGION for
# why the whole env-local area counts as observed here and why a real sensor adapter must not copy
# this. Larger than the Franka box: navigation goals are meters away, not centimeters.
PRIVILEGED_STATE_OBSERVED_REGION = ObservedRegion(min_corner=(-20.0, -20.0, -1.0), max_corner=(20.0, 20.0, 3.0))


class IsaacLabAnymalNavPerceptionAdapter(PerceptionAdapter):
    """Reads ANYmal-C's own privileged simulator state -- base pose/velocity, joint state, and each
    foot's real world position -- directly from Isaac Lab's navigation task. No perception noise
    yet, same caveat as the Franka adapter: this validates the engine/schema against a genuinely
    different robot's real geometry before real perception uncertainty enters the picture."""

    def __init__(self, env, env_index: int = 0):
        self._env = env
        self._env_index = env_index
        robot = env.scene["robot"]
        foot_ids, _ = robot.find_bodies(list(FOOT_BODY_NAMES))
        self._foot_ids = foot_ids

    def get_world_state(self) -> WorldState:
        sensor_timestamp = time.time()  # synchronous privileged read: capture time = read start
        env = self._env
        i = self._env_index
        origin = env.scene.env_origins[i]
        robot = env.scene["robot"]

        base_pos = (robot.data.root_pos_w.torch[i] - origin).tolist()
        # Base linear velocity stands in for center-of-mass velocity, the same documented
        # simplification as base position for center of mass below; feeds the capture-point half of
        # stability_margin_maintained. Same data field the Franka adapter reads for its cubes.
        base_vel = robot.data.root_lin_vel_w.torch[i].tolist()
        base_quat = quat_xyzw_to_wxyz(robot.data.root_quat_w.torch[i].tolist())  # Isaac Lab 3.x stores (x, y, z, w)
        joint_pos = robot.data.joint_pos.torch[i].tolist()
        joint_vel = robot.data.joint_vel.torch[i].tolist()

        # Real, measured foot contact points -- not a kinematic approximation -- projected to (x, y)
        # for the same nearest-vertex support-polygon distance used everywhere else in this module.
        foot_positions_w = robot.data.body_pos_w.torch[i, self._foot_ids]
        support_polygon = tuple(
            (float(p[0] - origin[0]), float(p[1] - origin[1])) for p in foot_positions_w
        )

        robot_state = RobotProprioception(
            joint_positions=tuple(joint_pos),
            joint_velocities=tuple(joint_vel),
            # The base's own pose, not a gripper's -- see this module's docstring for why this field
            # is the closest fit anyway and why that's a real naming finding, not an oversight.
            end_effector_pose=Pose(position=tuple(base_pos), orientation_wxyz=tuple(base_quat)),
            gripper_state=0.0,  # no gripper on this platform; unused by every check wired for `navigate`
            # Base position stands in for center of mass -- documented simplification, not a true
            # whole-body CoM (which needs every link's own mass and pose) -- same "conservative
            # stand-in, swap in a real one for production" convention _distance_to_polygon_edge's own
            # docstring uses for the support-polygon distance calculation itself.
            center_of_mass=tuple(base_pos),
            support_polygon=support_polygon,
            center_of_mass_velocity=tuple(base_vel),
        )

        return WorldState(
            objects=(),  # no tracked objects in this navigation task
            agents=(),  # no humans/agents tracked by this task's own perception; inject via a wrapper adapter
            robot=robot_state,
            environment=EnvironmentSignals(visibility_confidence=1.0),
            sensor_timestamp=sensor_timestamp,
            observed_regions=(PRIVILEGED_STATE_OBSERVED_REGION,),
        )


class IsaacLabAnymalNavDynamicsAdapter(DynamicsAdapter):
    """Forward-predicts the swept path of the robot's own BASE toward a navigation target at a
    conservative constant-speed bound -- structurally identical to the Franka adapter's end-effector
    extrapolation, deliberately: the point of this adapter is to test whether that same simple
    design generalizes to a different actuation modality (legged locomotion vs. a fixed arm), not to
    invent a new one. Reuses the current robot state (support polygon, center of mass) unchanged at
    every predicted point -- it doesn't forward-simulate the gait, only the base's translation,
    exactly as the Franka adapter never forward-simulates joint angles, only Cartesian position."""

    def __init__(
        self, max_base_speed_mps: float = DEFAULT_MAX_BASE_SPEED_MPS,
        body_radius_m: float = BODY_FOOTPRINT_RADIUS_M, n_points: int = 10,
    ):
        self._max_speed = max_base_speed_mps
        self._body_radius = body_radius_m
        self._n_points = n_points

    def predict_trajectory(self, state: WorldState, action: Action, horizon_s: float) -> PredictedTrajectory:
        start = state.robot.end_effector_pose.position  # the base's own position -- see perception adapter above
        if "target_position" not in action.params:
            # Same default-deny reasoning as the Franka adapter: no target means no basis for a
            # trajectory prediction, and assuming "no motion" would look trivially safe to every
            # clearance check -- raise and let the engine's default-deny wrapper block instead.
            raise ValueError(f"action {action.action_type!r} has no target_position; cannot predict a trajectory")
        target = action.params["target_position"]
        dist = _dist(start, target)
        n = max(2, self._n_points)
        points = []
        for k in range(n):
            t = horizon_s * k / (n - 1)
            frac = 0.0 if dist < 1e-6 else min(1.0, (self._max_speed * t) / dist)
            center = tuple(s + frac * (g - s) for s, g in zip(start, target))
            points.append(
                TrajectoryPoint(
                    t=t,
                    robot=state.robot,
                    swept_volume_center=center,
                    swept_volume_radius_m=self._body_radius,
                )
            )
        return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)


def _dist(a, b) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
