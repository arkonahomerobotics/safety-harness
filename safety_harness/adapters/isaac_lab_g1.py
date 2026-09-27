"""Reference PerceptionAdapter + DynamicsAdapter against a dexterous humanoid manipulator -- the
Unitree G1 running Isaac Lab's locomanipulation pick-place task (Isaac-PickPlace-Locomanipulation-
G1-Abs-v0), reaching for a real object (a steering wheel prop from Isaac Lab's own Mimic pick-place
assets) on a real packing table. Built after the ANYmal-C navigation adapter, for a different reason:
that one tested whether the engine generalizes to a robot whose BASE sweeps through space; this one
tests whether it generalizes to a second, very different ARM -- a 14-joint dexterous hand on a
whole-body Pink-IK-controlled humanoid, instead of a parallel gripper on a fixed Franka arm.

Unlike the ANYmal adapter, this one needed no naming workaround: `RobotProprioception.
end_effector_pose` is used for exactly what its name says, a real end effector (the left wrist,
chosen because the task's object starts on that side of the robot's body -- the first version of
this adapter's validation script reached with the RIGHT wrist out of an unchecked assumption and the
arm never got near the object at all, confirmed by the object not moving through an entire scripted
sequence). And the existing `grasp` action type in configs/example_action_schema.yaml -- the same
checks already validated against the Franka gripper -- applies here with no new action type. (Since
schema 0.3.0 it additionally needs this adapter's sensor_timestamp / observed_regions /
rated_payload_kg, reported below, and the proposer must send ``grip_force_n`` with a grasp.)

Scope this adapter does NOT claim: balance_margin_maintained is not wired for this robot. G1 stands
on two feet, not four -- a two-point base is a genuinely different stability problem (a line, not a
polygon) that this project's convex-hull margin check isn't built for; the fallback for a support
polygon under 3 points currently treats it as always maximally unstable, which would make the check
fire on every single call regardless of real balance and teach nothing. Left unwired rather than
wired to a check that can't say anything meaningful yet -- a real gap, not a silent one.

Also NOT a claim: this adapter's own validation script scripts the grasp with simple linear
end-effector waypoints (no grasp planner), which makes real contact with the real object but does
not reliably lift it. That is a robot-control-fidelity limit of the scripted demo, not a
safety-harness finding -- the harness gates whatever the dynamics adapter predicts and whatever the
perception adapter reports, regardless of how good the underlying manipulation policy is.
"""

from __future__ import annotations

import math
import time

from ..schema import (
    Action,
    EnvironmentSignals,
    FallConsequence,
    HazardTag,
    ObservedRegion,
    Pose,
    PredictedTrajectory,
    RobotProprioception,
    TrackedObject,
    TrajectoryPoint,
    WorldState,
)
from ._isaac_lab_common import commanded_speed_mps, joint_position_limits, quat_xyzw_to_wxyz
from .base import DynamicsAdapter, PerceptionAdapter

# The object is a real prop from Isaac Lab's own Mimic pick-place assets (steering_wheel.usd), not
# a fabricated stand-in. Hazard tag defaults to a conservative FRAGILE for an unrecognized rigid
# prop, matching the Franka adapter's own CUBE_HAZARD_TAGS convention.
OBJECT_HAZARD_TAGS = frozenset({HazardTag.FRAGILE})
OBJECT_CLEARED_FOR_INTERACTION = True  # a known, controlled task prop -- see isaac_lab.py's own caveat about this assumption
OBJECT_FALL_CONSEQUENCE = FallConsequence.NONE  # a steering wheel prop tolerates a modest drop without hazardous release

HAND_JOINT_NAMES = (
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
)
# This task's real, measured object mass (0.58kg) is comfortably under a modest force budget; a
# conservative reference value, not a robot-specific rating -- override via action_schema.yaml
# kwargs for a real deployment's actual rated payload.
DEFAULT_FORCE_BUDGET_KG = 3.0

# Conservative bounding-circle radius for the hand/wrist's swept volume during a reach, in meters --
# same "conservative bounding sphere" convention as every other adapter's own ee_radius_m.
HAND_SWEEP_RADIUS_M = 0.08
DEFAULT_MAX_HAND_SPEED_MPS = 0.4
# Conservative single-arm payload for the G1 (a reference figure, not a datasheet guarantee --
# override for a real deployment). Reported so payload_and_grip_force_within_limits applies it.
G1_ARM_RATED_PAYLOAD_KG = 2.0
# Hand speed the harness holds the G1 to (cartesian_speed_within_limits); a conservative reference
# figure, not a datasheet rating.
G1_MAX_HAND_SPEED_MPS = 1.0
# Dex3 finger joints rest at their stops when open (every hand joint reads 0.0 at rest, at or next to
# a soft limit) -- exempted explicitly, like the Franka's gripper fingers.
G1_JOINT_LIMIT_EXEMPT = ("_hand_",)
# Privileged simulator state is omniscient, so the whole env-local volume counts as observed
# (swept_path_observed). A camera-based adapter must replace this with real, occlusion-aware coverage.
# Note the floor at z=-1: a table-top-only region made swept_path_observed block ~98% of nominal
# near-table reaches in the G1 stacking test, because the margin sphere around any near-table path
# dips into the space under the tabletop.
PRIVILEGED_STATE_OBSERVED_REGION = ObservedRegion(min_corner=(-5.0, -5.0, -1.0), max_corner=(5.0, 5.0, 3.0))


class IsaacLabG1PickPlacePerceptionAdapter(PerceptionAdapter):
    """Reads the G1's own privileged simulator state -- left-wrist pose, joint state, and the real
    pick-place object's pose/velocity/mass -- directly from Isaac Lab's locomanipulation task. No
    perception noise yet, same caveat as every other adapter here: this validates the engine/schema
    against a genuinely different manipulator's real geometry before real perception uncertainty
    enters the picture."""

    def __init__(self, env, object_id: str = "object", env_index: int = 0):
        self._env = env
        self._object_id = object_id
        self._env_index = env_index
        robot = env.scene["robot"]
        hand_ids, _ = robot.find_joints(list(HAND_JOINT_NAMES))
        self._hand_ids = hand_ids
        left_index_0_id, _ = robot.find_joints(["left_hand_index_0_joint"])
        self._left_index_0_id = left_index_0_id[0]
        self._left_index_0_limits = robot.data.soft_joint_pos_limits.torch[env_index, left_index_0_id[0]].tolist()

    def get_world_state(self) -> WorldState:
        sensor_timestamp = time.time()  # stamped before any read -- simulator state is read synchronously
        env = self._env
        i = self._env_index
        origin = env.scene.env_origins[i]
        robot = env.scene["robot"]
        obj = env.scene[self._object_id]

        # Real end-effector pose -- the left wrist, since this task's object starts on that side of
        # the robot's body. See module docstring for why the first version of this adapter's own
        # validation script had this on the wrong side entirely.
        left_wrist_id, _ = robot.find_bodies(["left_wrist_yaw_link"])
        ee_pos = (robot.data.body_pos_w.torch[i, left_wrist_id[0]] - origin).tolist()
        ee_quat = quat_xyzw_to_wxyz(robot.data.body_quat_w.torch[i, left_wrist_id[0]].tolist())

        joint_pos = robot.data.joint_pos.torch[i].tolist()
        joint_vel = robot.data.joint_vel.torch[i].tolist()

        lo, hi = self._left_index_0_limits
        raw = float(robot.data.joint_pos.torch[i, self._left_index_0_id])
        gripper_state = 1.0 - max(0.0, min(1.0, (raw - lo) / (hi - lo)))  # 1=open, matching the schema's own convention

        robot_state = RobotProprioception(
            joint_positions=tuple(joint_pos),
            joint_velocities=tuple(joint_vel),
            end_effector_pose=Pose(position=tuple(ee_pos), orientation_wxyz=tuple(ee_quat)),
            gripper_state=gripper_state,
            center_of_mass=None,  # not wired -- see module docstring on the two-point-stance gap
            support_polygon=None,
            rated_payload_kg=G1_ARM_RATED_PAYLOAD_KG,
            joint_position_limits=joint_position_limits(robot, i, G1_JOINT_LIMIT_EXEMPT),
            max_cartesian_speed_mps=G1_MAX_HAND_SPEED_MPS,
        )

        obj_pos = (obj.data.root_pos_w.torch[i] - origin).tolist()
        obj_quat = quat_xyzw_to_wxyz(obj.data.root_quat_w.torch[i].tolist())
        obj_vel = obj.data.root_lin_vel_w.torch[i].tolist()
        # Real, measured mass from the physics engine's own rigid-body properties -- not fabricated.
        # get_masses() returns an NVIDIA Warp array (not numpy or torch -- neither supports direct
        # item indexing), shaped (num_envs, num_bodies_per_env); .numpy() converts once, then a
        # flat reshape sidesteps caring about the exact per-env/per-body layout for a single object.
        obj_mass = float(obj.root_physx_view.get_masses().numpy().reshape(-1)[0])
        speed = sum(v * v for v in obj_vel) ** 0.5
        supported_stably = speed < 0.05  # same "total speed, not just vertical" convention as isaac_lab.py's own fix

        target_object = TrackedObject(
            object_id=self._object_id,
            object_class="steering_wheel",
            pose=Pose(position=tuple(obj_pos), orientation_wxyz=tuple(obj_quat)),
            velocity=tuple(obj_vel),
            estimated_mass_kg=obj_mass,
            hazard_tags=OBJECT_HAZARD_TAGS,
            pose_confidence=1.0,
            class_confidence=1.0,
            cleared_for_interaction=OBJECT_CLEARED_FOR_INTERACTION,
            supported_stably=supported_stably,
            fall_consequence=OBJECT_FALL_CONSEQUENCE,
        )

        return WorldState(
            objects=(target_object,),
            agents=(),  # no humans/agents tracked by this task's own perception; inject via a wrapper adapter
            robot=robot_state,
            environment=EnvironmentSignals(visibility_confidence=1.0),
            sensor_timestamp=sensor_timestamp,
            observed_regions=(PRIVILEGED_STATE_OBSERVED_REGION,),
        )


class IsaacLabG1HandDynamicsAdapter(DynamicsAdapter):
    """Forward-predicts the swept path of the left hand/wrist toward a reach target at a
    conservative constant-speed bound -- structurally identical to the Franka adapter's own
    end-effector extrapolation. Reuses the current robot state unchanged at every predicted point,
    exactly as every other adapter in this project does: it forward-simulates the Cartesian sweep,
    never the underlying joint/IK dynamics."""

    def __init__(
        self, max_hand_speed_mps: float = DEFAULT_MAX_HAND_SPEED_MPS,
        hand_radius_m: float = HAND_SWEEP_RADIUS_M, n_points: int = 10,
    ):
        self._max_speed = max_hand_speed_mps
        self._hand_radius = hand_radius_m
        self._n_points = n_points

    def predict_trajectory(self, state: WorldState, action: Action, horizon_s: float) -> PredictedTrajectory:
        start = state.robot.end_effector_pose.position
        if "target_position" not in action.params:
            raise ValueError(f"action {action.action_type!r} has no target_position; cannot predict a trajectory")
        target = action.params["target_position"]
        dist = _dist(start, target)
        speed = commanded_speed_mps(action, dist, self._max_speed)  # commanded, not assumed (see helper)
        n = max(2, self._n_points)
        points = []
        for k in range(n):
            t = horizon_s * k / (n - 1)
            frac = 0.0 if dist < 1e-6 else min(1.0, (speed * t) / dist)
            center = tuple(s + frac * (g - s) for s, g in zip(start, target))
            points.append(
                TrajectoryPoint(
                    t=t,
                    robot=state.robot,
                    swept_volume_center=center,
                    swept_volume_radius_m=self._hand_radius,
                )
            )
        return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)


def _dist(a, b) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
