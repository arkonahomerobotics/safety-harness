"""Reference PerceptionAdapter + DynamicsAdapter against this project's Isaac Lab cube-stack task.

Development Roadmap stage 3: "Build one reference adapter against a simulator already in hand."
Uses privileged simulator state directly -- no perception noise yet -- which validates the engine's
logic before real perception uncertainty enters the picture. A later real-camera adapter reuses this
same WorldState shape with confidence fields below 1.0 instead of exactly 1.0.

NOT YET RUN against a live environment: this file imports Isaac Lab APIs (``env.scene[...]``,
``.data.root_pos_w.torch``, and friends) that only resolve inside the ``vscode`` container with the
GPU instance running and Isaac Sim launched. It is written to the same conventions used throughout
this project's other Isaac Lab scripts, but it has not been executed or unit-tested against a real
``env`` object yet -- do that before trusting it.
"""

from __future__ import annotations

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
from ._isaac_lab_common import quat_xyzw_to_wxyz
from .base import DynamicsAdapter, PerceptionAdapter

# The training cubes are plastic/foam props with no real fragility, but there is no perception
# signal confirming that for an arbitrary object -- tag them fragile as the conservative default a
# real deployment would apply to an unrecognized rigid object of this size.
CUBE_HAZARD_TAGS = frozenset({HazardTag.FRAGILE})
CUBE_MASS_KG = 0.05
# These specific cubes are known, controlled props in a fixed simulated training task, confirmed
# harmless -- distinct from HazardTag above, which is a conservative default for an *unrecognized*
# object. A real camera-based adapter must not carry this assumption over; it has to earn these
# fields from actual perception, per object, the way this one does from privileged simulator state.
CUBE_CLEARED_FOR_INTERACTION = True
CUBE_FALL_CONSEQUENCE = FallConsequence.NONE
STABLE_REST_HEIGHT_M = 0.0203  # established scene fact: cube center height at rest, above env origin
STABLE_HEIGHT_TOLERANCE_M = 0.01
# Franka Emika Panda datasheet payload -- a published robot rating, reported so
# payload_and_grip_force_within_limits holds this robot to its own figure, not only a config's.
FRANKA_PANDA_RATED_PAYLOAD_KG = 3.0
# Privileged simulator state is omniscient: every body in the scene is known exactly, so the whole
# env-local workspace counts as observed (swept_path_observed). A generous but finite box around the
# env origin -- a real camera-based adapter must replace this with its sensors' actual, occlusion-
# aware coverage, never carry this assumption over.
PRIVILEGED_STATE_OBSERVED_REGION = ObservedRegion(min_corner=(-5.0, -5.0, -1.0), max_corner=(5.0, 5.0, 3.0))


class IsaacLabCubeStackPerceptionAdapter(PerceptionAdapter):
    """Reads cube and robot state directly from the simulator's privileged state."""

    def __init__(self, env, cube_names: tuple = ("cube_1", "cube_2", "cube_3"), env_index: int = 0):
        self._env = env
        self._cube_names = cube_names
        self._env_index = env_index
        # Resolve the finger joints by name, not by trusting joint-index order -- the checklist item
        # that caught this: index order is an assumption, not a verified fact, and every other script
        # in this project confirmed it this way instead.
        finger_ids, _ = env.scene["robot"].find_joints("panda_finger_joint.*")
        self._finger_ids = finger_ids

    def get_world_state(self) -> WorldState:
        # Stamped before any read: simulator state is read synchronously, so the moment reading
        # starts is the capture time of the oldest value in this state (sensor_data_fresh).
        sensor_timestamp = time.time()
        env = self._env
        i = self._env_index
        origin = env.scene.env_origins[i]

        objects = []
        for name in self._cube_names:
            handle = env.scene[name]
            pos = (handle.data.root_pos_w.torch[i] - origin).tolist()
            quat = quat_xyzw_to_wxyz(handle.data.root_quat_w.torch[i].tolist())  # Isaac Lab 3.x stores (x, y, z, w)
            vel = handle.data.root_lin_vel_w.torch[i].tolist()
            # A real, measured signal, not a fabricated assumption: not currently moving in any
            # direction, not just not-falling. The first version of this checked only vertical
            # velocity and missed a cube sliding sideways at 1.5 m/s -- found by driving a real
            # lateral disturbance through the live simulator, not by unit-testing the formula in
            # isolation. Total speed is what "at rest" actually means.
            speed = sum(v * v for v in vel) ** 0.5
            supported_stably = speed < 0.05
            objects.append(
                TrackedObject(
                    object_id=name,
                    object_class="cube",
                    pose=Pose(position=tuple(pos), orientation_wxyz=tuple(quat)),
                    velocity=tuple(vel),
                    estimated_mass_kg=CUBE_MASS_KG,
                    hazard_tags=CUBE_HAZARD_TAGS,
                    pose_confidence=1.0,  # privileged state: no perception noise yet
                    class_confidence=1.0,
                    # Known-good assumptions for THIS controlled task's props -- see the module-level
                    # comment on why a real camera-based adapter must earn these per object instead.
                    cleared_for_interaction=CUBE_CLEARED_FOR_INTERACTION,
                    supported_stably=supported_stably,
                    fall_consequence=CUBE_FALL_CONSEQUENCE,
                )
            )

        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        joint_pos = robot.data.joint_pos.torch[i].tolist()
        joint_vel = robot.data.joint_vel.torch[i].tolist()
        ee_pos = (ee_frame.data.target_pos_w.torch[i, 0] - origin).tolist()
        # Mean finger-joint aperture as a 0 (closed) - 1 (open) proxy, resolved by name in __init__.
        gripper = float(robot.data.joint_pos.torch[i, self._finger_ids].mean())

        robot_state = RobotProprioception(
            joint_positions=tuple(joint_pos),
            joint_velocities=tuple(joint_vel),
            end_effector_pose=Pose(position=tuple(ee_pos)),
            gripper_state=gripper,
            center_of_mass=None,  # fixed-base arm: no whole-body balance constraint
            support_polygon=None,
            rated_payload_kg=FRANKA_PANDA_RATED_PAYLOAD_KG,
        )

        return WorldState(
            objects=tuple(objects),
            agents=(),  # no humans/agents tracked in this sim task
            robot=robot_state,
            environment=EnvironmentSignals(visibility_confidence=1.0),
            sensor_timestamp=sensor_timestamp,
            observed_regions=(PRIVILEGED_STATE_OBSERVED_REGION,),
        )


class IsaacLabCubeStackDynamicsAdapter(DynamicsAdapter):
    """Forward-predicts the swept end-effector path toward the action's target at a conservative
    constant-speed bound, rather than stepping the live simulator (cheap, no side effects on the
    running episode). A tighter version could step a cloned env instead of extrapolating."""

    def __init__(self, max_ee_speed_mps: float = 0.5, ee_radius_m: float = 0.05, n_points: int = 10):
        self._max_speed = max_ee_speed_mps
        self._ee_radius = ee_radius_m
        self._n_points = n_points

    def predict_trajectory(self, state: WorldState, action: Action, horizon_s: float) -> PredictedTrajectory:
        start = state.robot.end_effector_pose.position
        if "target_position" not in action.params:
            # No target -> no basis for a trajectory prediction. Defaulting to "assume no motion"
            # would look trivially safe to every clearance check, which is a default-deny violation:
            # the adapter doesn't know what the robot will do, so it must not manufacture the
            # safest-looking answer. Raise, and let the engine's default-deny wrapper block instead.
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
                    swept_volume_radius_m=self._ee_radius,
                )
            )
        return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)


def _dist(a, b) -> float:
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5
