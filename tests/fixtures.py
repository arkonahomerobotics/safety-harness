"""Hand-built WorldState fixtures for testing the engine with no robot, no adapter, and no
perception noise involved (Development Roadmap stage 2; Testing and Validation Strategy in the
design doc).
"""

from __future__ import annotations

import time
from dataclasses import replace

from safety_harness.schema import (
    Action,
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


# What perception actually observed, for swept_path_observed: the whole tabletop workspace around the
# fixture trajectory (0.5, 0, 0.3) -> (0.5, 0, 0.05), and a nav-sized area for the quadruped.
WORKSPACE_OBSERVED = (ObservedRegion(min_corner=(-1.0, -1.0, -0.5), max_corner=(2.0, 1.0, 1.5)),)
NAV_AREA_OBSERVED = (ObservedRegion(min_corner=(-2.0, -3.0, -0.5), max_corner=(6.0, 3.0, 2.0)),)

# A safe commanded grip for the 0.05kg fragile training cube: above the ~1.6N needed to hold it,
# below the 15N fragile-object cap (payload_and_grip_force_within_limits).
SAFE_GRIP_FORCE_N = 5.0


def robot_state(ee_position=(0.5, 0.0, 0.3)) -> RobotProprioception:
    """No kinematic/electrical limits reported -- the sparse default. Every limit check must
    default-deny against this, matching every other "unconfirmed" case in the module."""
    return RobotProprioception(
        joint_positions=(0.0,) * 7,
        joint_velocities=(0.0,) * 7,
        end_effector_pose=Pose(position=ee_position),
        gripper_state=1.0,
    )


def instrumented_robot_state(
    ee_position=(0.5, 0.0, 0.3), joint_positions=(0.0,) * 7, joint_velocities=(0.0,) * 7,
    joint_efforts=(0.0,) * 7, max_cartesian_speed_mps=1.0,
) -> RobotProprioception:
    """A fully-instrumented robot state: joint limits, effort limits, and thermal limits all
    reported and comfortably satisfied -- the positive case for every kinematic/electrical check."""
    return RobotProprioception(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        end_effector_pose=Pose(position=ee_position),
        gripper_state=1.0,
        joint_position_limits=((-2.9, 2.9),) * 7,
        joint_velocity_limits=(2.5,) * 7,
        joint_effort_limits=(87.0,) * 7,
        estimated_joint_efforts=joint_efforts,
        motor_temperature_c=(40.0,) * 7,
        motor_temperature_limit_c=(80.0,) * 7,
        max_cartesian_speed_mps=max_cartesian_speed_mps,
        battery_charge_fraction=0.9,
    )


def confirmed_object(object_id="cube_2", hazard_tags=frozenset(), mass_kg=0.05, position=(0.5, 0.0, 0.05)) -> TrackedObject:
    """A fully cleared, harmless foam training cube: safe to approach, currently stable, and
    tolerates being dropped (fall_consequence=NONE)."""
    return TrackedObject(
        object_id=object_id,
        object_class="cube",
        pose=Pose(position=position),
        estimated_mass_kg=mass_kg,
        hazard_tags=hazard_tags or frozenset({HazardTag.FRAGILE}),
        pose_confidence=1.0,
        class_confidence=1.0,
        cleared_for_interaction=True,
        supported_stably=True,
        fall_consequence=FallConsequence.NONE,
    )


def uncleared_bystander_object(object_id="cube_3", position=(0.55, 0.0, 0.05)) -> TrackedObject:
    """An object perception has NOT cleared for interaction and confirms would release a hazard if
    struck -- e.g. the user's "the green cube is actually hazardous" case. Never the action's own
    target in these tests; it blocks purely by being near the swept path."""
    return TrackedObject(
        object_id=object_id,
        object_class="cube",
        pose=Pose(position=position),
        estimated_mass_kg=0.05,
        hazard_tags=frozenset({HazardTag.UNKNOWN}),
        pose_confidence=1.0,
        class_confidence=1.0,
        cleared_for_interaction=False,
        supported_stably=True,
        fall_consequence=FallConsequence.HAZARDOUS_RELEASE,
        drop_tolerance_m=0.0,
    )


def unstable_object(object_id="cube_2", position=(0.5, 0.0, 0.05)) -> TrackedObject:
    """Cleared for interaction, but its current resting position is confirmed NOT stable -- e.g.
    teetering on the edge of a table."""
    obj = confirmed_object(object_id=object_id, position=position)
    return replace(obj, supported_stably=False)


def fragile_high_lift_object(object_id="cube_2", position=(0.5, 0.0, 0.05), drop_tolerance_m=0.02) -> TrackedObject:
    """Cleared and currently stable, but confirmed fragile enough that only a very small lift is
    tolerable -- exercises fall_consequence_acceptable independent of every other check."""
    obj = confirmed_object(object_id=object_id, position=position)
    return replace(obj, fall_consequence=FallConsequence.MESS, drop_tolerance_m=drop_tolerance_m)


def unknown_object(object_id="mystery_object", position=(0.5, 0.0, 0.05)) -> TrackedObject:
    """An object perception hasn't classified -- must default-deny, not default-permit."""
    return TrackedObject(
        object_id=object_id,
        object_class="unknown",
        pose=Pose(position=position),
        estimated_mass_kg=None,
        hazard_tags=frozenset({HazardTag.UNKNOWN}),
        pose_confidence=0.9,
        class_confidence=0.1,
    )


def far_agent(agent_id="person_1") -> TrackedAgent:
    return TrackedAgent(
        agent_id=agent_id,
        pose=Pose(position=(5.0, 5.0, 0.0)),
        tracking_confidence=0.95,
        time_since_confirmed_s=0.0,
        worst_case_speed_mps=1.5,
    )


def close_agent(agent_id="person_1") -> TrackedAgent:
    return TrackedAgent(
        agent_id=agent_id,
        pose=Pose(position=(0.55, 0.0, 0.05)),  # right next to the target below
        tracking_confidence=0.95,
        time_since_confirmed_s=0.0,
        worst_case_speed_mps=1.5,
    )


def stale_agent(agent_id="person_1") -> TrackedAgent:
    """Tracked once, far away, but long enough ago that the worst-case radius now reaches the
    action -- exercises the staleness-widens-the-uncertainty-region path, not just distance."""
    return TrackedAgent(
        agent_id=agent_id,
        pose=Pose(position=(2.0, 0.0, 0.0)),
        tracking_confidence=0.95,
        time_since_confirmed_s=5.0,
        worst_case_speed_mps=1.5,
    )


def quadruped_robot_state(
    base_position=(0.0, 0.0, 0.6), support_polygon=((0.3, 0.2), (0.3, -0.2), (-0.3, 0.2), (-0.3, -0.2)),
    com_velocity=(0.0, 0.0, 0.0),
) -> RobotProprioception:
    """A legged robot's own state: base position standing in for center of mass (documented
    simplification -- see isaac_lab_anymal.py), and a real four-foot support polygon centered under
    it, comfortably stable. No kinematic/electrical limits reported, matching robot_state()'s own
    sparse-default convention -- this robot doesn't wire those checks either."""
    return RobotProprioception(
        joint_positions=(0.0,) * 12,
        joint_velocities=(0.0,) * 12,
        end_effector_pose=Pose(position=base_position),  # the base's own pose -- see isaac_lab_anymal.py
        gripper_state=0.0,
        center_of_mass=base_position,
        support_polygon=support_polygon,
        center_of_mass_velocity=com_velocity,
    )


def tipping_quadruped_robot_state(base_position=(0.0, 0.0, 0.6)) -> RobotProprioception:
    """Same robot, but with its support polygon shifted well clear of its center of mass -- e.g. a
    real foothold slip, or a slope steep enough that its feet are no longer under it. Exercises
    balance_margin_maintained with real-shaped geometry, not just a missing-state default-deny."""
    return quadruped_robot_state(
        base_position=base_position,
        support_polygon=((0.9, 0.2), (0.9, -0.2), (0.6, 0.2), (0.6, -0.2)),
    )


def navigate_action(target_position=(3.0, 0.0, 0.6)) -> Action:
    return Action(action_type="navigate", params={"target_position": target_position})


def grasp_action(object_id="cube_2", target_position=(0.5, 0.0, 0.05), grip_force_n=SAFE_GRIP_FORCE_N) -> Action:
    params = {"object_id": object_id, "target_position": target_position}
    if grip_force_n is not None:
        params["grip_force_n"] = grip_force_n
    return Action(action_type="grasp", params=params)


def place_action(object_id="cube_2", target_surface_id="cube_1", target_position=(0.5, 0.0, 0.05)) -> Action:
    return Action(
        action_type="place",
        params={"object_id": object_id, "target_surface_id": target_surface_id, "target_position": target_position},
    )


def straight_line_trajectory(start=(0.5, 0.0, 0.3), end=(0.5, 0.0, 0.05), horizon_s=1.0, n_points=5, radius_m=0.05) -> PredictedTrajectory:
    points = []
    for k in range(n_points):
        t = horizon_s * k / (n_points - 1)
        frac = k / (n_points - 1)
        center = tuple(s + frac * (e - s) for s, e in zip(start, end))
        points.append(TrajectoryPoint(t=t, robot=robot_state(center), swept_volume_center=center, swept_volume_radius_m=radius_m))
    return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)


def base_world_state(objects=(), agents=(), visibility=1.0, sensor_age_s=0.0, observed_regions=WORKSPACE_OBSERVED) -> WorldState:
    """Sensor data captured ``sensor_age_s`` ago (fresh by default) with the whole tabletop
    workspace observed -- the positive case for sensor_data_fresh and swept_path_observed."""
    return WorldState(
        objects=objects,
        agents=agents,
        robot=robot_state(),
        environment=EnvironmentSignals(visibility_confidence=visibility),
        sensor_timestamp=time.time() - sensor_age_s,
        observed_regions=observed_regions,
    )


def quadruped_world_state(agents=(), visibility=1.0, robot=None) -> WorldState:
    return WorldState(
        objects=(),
        agents=agents,
        robot=robot if robot is not None else quadruped_robot_state(),
        environment=EnvironmentSignals(visibility_confidence=visibility),
        sensor_timestamp=time.time(),
        observed_regions=NAV_AREA_OBSERVED,
    )
