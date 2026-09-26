"""Conformance-style test suite for the engine, run entirely against hand-built fixtures -- no
robot, no adapter, no perception noise (Development Roadmap stage 2; design doc's "Testing and
Validation Strategy": positive/negative per precondition, boundary cases, unregistered types).

Run with: python3 -m unittest discover -s tests -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict, HazardTag, PerceptionFailure  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402
from safety_harness.schema import PredictedTrajectory, TrajectoryPoint, WorldState  # noqa: E402
from safety_harness import preconditions as pc  # noqa: E402

SCHEMA_DICT = {
    "action_types": {
        "grasp": {
            "checks": [
                {"name": "robot_state_confirmed", "kwargs": {}},
                {"name": "object_hazard_confirmed", "kwargs": {"min_confidence": 0.6}},
                {"name": "object_pose_confirmed", "kwargs": {"min_confidence": 0.6}},
                {"name": "object_cleared_for_interaction", "kwargs": {}},
                {"name": "current_position_confirmed_stable", "kwargs": {}},
                {"name": "fall_consequence_acceptable", "kwargs": {}},
                {"name": "mass_within_force_budget", "kwargs": {"force_budget_kg": 3.0}},
                {"name": "swept_path_clear_of_agents", "kwargs": {"margin_m": 0.15}},
                {"name": "iso15066_separation_distance_maintained", "kwargs": {}},
                {"name": "iso15066_power_force_limiting", "kwargs": {}},
                {"name": "reduced_speed_near_human", "kwargs": {}},
                {"name": "swept_path_clear_of_risky_objects", "kwargs": {"margin_m": 0.15}},
                {"name": "visibility_above_threshold", "kwargs": {"min_visibility": 0.5}},
                {"name": "environment_hazard_clear", "kwargs": {}},
            ]
        },
        "place": {
            "checks": [
                {"name": "robot_state_confirmed", "kwargs": {}},
                {"name": "destination_confirmed_stable_and_clear", "kwargs": {"clearance_m": 0.1}},
                {"name": "fall_consequence_acceptable", "kwargs": {}},
                {"name": "swept_path_clear_of_agents", "kwargs": {"margin_m": 0.15}},
                {"name": "iso15066_separation_distance_maintained", "kwargs": {}},
                {"name": "iso15066_power_force_limiting", "kwargs": {}},
                {"name": "reduced_speed_near_human", "kwargs": {}},
                {"name": "swept_path_clear_of_risky_objects", "kwargs": {"margin_m": 0.15}},
                {"name": "visibility_above_threshold", "kwargs": {"min_visibility": 0.5}},
                {"name": "environment_hazard_clear", "kwargs": {}},
            ]
        },
        "no_op_registered_empty": {"checks": []},
    }
}


class _StubPerception(PerceptionAdapter):
    def __init__(self, state):
        self._state = state

    def get_world_state(self):
        return self._state


class _StubDynamics(DynamicsAdapter):
    def __init__(self, trajectory):
        self._trajectory = trajectory

    def predict_trajectory(self, state, action, horizon_s):
        return self._trajectory


def make_gate(state, trajectory=None):
    trajectory = trajectory or fixtures.straight_line_trajectory()
    return ActuatorGate(
        perception=_StubPerception(state),
        dynamics=_StubDynamics(trajectory),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=ActionSchemaRegistry.from_dict(SCHEMA_DICT),
    )


class EnginePermitBlockTests(unittest.TestCase):
    def test_permit_when_all_confirmed(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.far_agent(),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)
        self.assertFalse(decision.triggered_fallback)
        self.assertTrue(all(r.satisfied for r in decision.precondition_results))

    def test_block_when_object_unknown(self):
        state = fixtures.base_world_state(objects=(fixtures.unknown_object(object_id="cube_2"),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertTrue(decision.triggered_fallback)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("object_hazard_confirmed", failed)

    def test_block_when_object_absent_from_world_state(self):
        # The grasp targets "cube_2" but perception reports nothing by that id at all.
        state = fixtures.base_world_state(objects=())
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_block_when_mass_exceeds_budget(self):
        heavy = fixtures.confirmed_object(mass_kg=5.0)
        state = fixtures.base_world_state(objects=(heavy,))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("mass_within_force_budget", failed)

    def test_permit_when_mass_exactly_at_budget_boundary(self):
        at_budget = fixtures.confirmed_object(mass_kg=3.0)  # boundary: budget is 3.0kg, "exceeds" is strict >
        state = fixtures.base_world_state(objects=(at_budget,))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)

    def test_block_when_agent_close_to_swept_path(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.close_agent(),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("swept_path_clear_of_agents", failed)

    def test_block_when_agent_tracking_confidence_low_even_if_far(self):
        from safety_harness.schema import Pose, TrackedAgent

        agent = TrackedAgent(agent_id="p1", pose=Pose(position=(5.0, 5.0, 0.0)), tracking_confidence=0.1)
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(agent,))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("swept_path_clear_of_agents", failed)

    def test_block_when_stale_tracking_widens_worst_case_radius(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.stale_agent(),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_block_when_visibility_low(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), visibility=0.2)
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("visibility_above_threshold", failed)

    def test_block_when_action_type_unregistered(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        gate = make_gate(state)
        from safety_harness.schema import Action

        decision = gate.gate(Action(action_type="somersault", params={}))
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertEqual(decision.precondition_results[0].name, "action_type_registered")

    def test_block_when_schema_registered_with_empty_checks(self):
        # Registered, but with no checks at all -> still unsafe by default, not a free pass.
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        gate = make_gate(state)
        from safety_harness.schema import Action

        decision = gate.gate(Action(action_type="no_op_registered_empty", params={}))
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_fallback_freezes_at_current_joint_positions(self):
        state = fixtures.base_world_state(objects=(fixtures.unknown_object(object_id="cube_2"),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.action.action_type, "freeze")
        self.assertEqual(decision.action.params["hold_joint_positions"], state.robot.joint_positions)

    def test_decision_is_logged(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        logger = InMemoryLogger()
        gate = ActuatorGate(
            perception=_StubPerception(state),
            dynamics=_StubDynamics(fixtures.straight_line_trajectory()),
            fallback=FreezeInPlaceFallback(),
            logger=logger,
            action_schema=ActionSchemaRegistry.from_dict(SCHEMA_DICT),
        )
        gate.gate(fixtures.grasp_action())
        self.assertEqual(len(logger.records), 1)


class ObjectAndPlacementSafetyTests(unittest.TestCase):
    """Answers to: is the object itself safe to touch, is its current spot safe, is the destination
    safe, will it break if dropped, and can a hazardous *bystander* object -- not the action's own
    target -- block an action just by being nearby."""

    def test_block_when_bystander_object_is_uncleared_and_hazardous(self):
        # Grasping cube_2 (fully cleared); cube_3 sits right next to the swept path, uncleared and
        # confirmed to release a hazard if struck. Never the target -- must still block.
        state = fixtures.base_world_state(
            objects=(fixtures.confirmed_object(object_id="cube_2"), fixtures.uncleared_bystander_object(object_id="cube_3")),
        )
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action(object_id="cube_2"))
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("swept_path_clear_of_risky_objects", failed)

    def test_permit_when_bystander_object_is_cleared_and_far(self):
        far_cube = fixtures.confirmed_object(object_id="cube_3", position=(5.0, 5.0, 0.05))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(object_id="cube_2"), far_cube))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action(object_id="cube_2"))
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)

    def test_block_when_target_not_cleared_for_interaction(self):
        # cleared_for_interaction defaults to False -- an object nobody explicitly cleared.
        uncleared_target = fixtures.uncleared_bystander_object(object_id="cube_2")
        state = fixtures.base_world_state(objects=(uncleared_target,))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action(object_id="cube_2"))
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("object_cleared_for_interaction", failed)

    def test_block_when_current_position_not_confirmed_stable(self):
        state = fixtures.base_world_state(objects=(fixtures.unstable_object(),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("current_position_confirmed_stable", failed)

    def test_block_when_lift_exceeds_confirmed_drop_tolerance(self):
        # Confirmed fragile with only a 2cm drop tolerance; the fixture trajectory lifts to 0.3m.
        state = fixtures.base_world_state(objects=(fixtures.fragile_high_lift_object(drop_tolerance_m=0.02),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("fall_consequence_acceptable", failed)

    def test_permit_when_lift_within_confirmed_drop_tolerance(self):
        state = fixtures.base_world_state(objects=(fixtures.fragile_high_lift_object(drop_tolerance_m=1.0),))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)

    def test_block_when_fall_consequence_unconfirmed(self):
        # fall_consequence defaults to UNKNOWN -- must block even with a generous drop_tolerance_m,
        # since "unknown consequence" is not the same as "confirmed harmless."
        obj = fixtures.confirmed_object()
        from dataclasses import replace
        from safety_harness.schema import FallConsequence

        unconfirmed = replace(obj, fall_consequence=FallConsequence.UNKNOWN, drop_tolerance_m=10.0)
        state = fixtures.base_world_state(objects=(unconfirmed,))
        gate = make_gate(state)
        decision = gate.gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_block_place_when_destination_not_confirmed_stable(self):
        unstable_surface = fixtures.unstable_object(object_id="cube_1")
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(object_id="cube_2"), unstable_surface))
        gate = make_gate(state)
        decision = gate.gate(fixtures.place_action(object_id="cube_2", target_surface_id="cube_1"))
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("destination_confirmed_stable_and_clear", failed)

    def test_block_place_when_hazardous_object_too_close_to_destination(self):
        surface = fixtures.confirmed_object(object_id="cube_1", position=(0.5, 0.0, 0.02))
        nearby_hazard = fixtures.uncleared_bystander_object(object_id="cube_3", position=(0.55, 0.0, 0.02))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(object_id="cube_2"), surface, nearby_hazard))
        gate = make_gate(state)
        decision = gate.gate(fixtures.place_action(object_id="cube_2", target_surface_id="cube_1"))
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = [r.name for r in decision.precondition_results if not r.satisfied]
        self.assertIn("destination_confirmed_stable_and_clear", failed)

    def test_permit_place_when_destination_and_neighbors_all_clear(self):
        surface = fixtures.confirmed_object(object_id="cube_1", position=(0.5, 0.0, 0.02))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(object_id="cube_2"), surface))
        gate = make_gate(state)
        decision = gate.gate(fixtures.place_action(object_id="cube_2", target_surface_id="cube_1"))
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)


class AdapterFailureTests(unittest.TestCase):
    """Failures inside an adapter must never crash the control loop or silently permit -- they must
    degrade to BLOCK (if state was already available) or a distinct hard failure (if not)."""

    def test_dynamics_exception_degrades_to_block_not_crash(self):
        class _BrokenDynamics(DynamicsAdapter):
            def predict_trajectory(self, state, action, horizon_s):
                raise ValueError("no target_position provided")

        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        gate = ActuatorGate(
            perception=_StubPerception(state),
            dynamics=_BrokenDynamics(),
            fallback=FreezeInPlaceFallback(),
            logger=InMemoryLogger(),
            action_schema=ActionSchemaRegistry.from_dict(SCHEMA_DICT),
        )
        decision = gate.gate(fixtures.grasp_action())  # must not raise
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertEqual(decision.precondition_results[0].name, "adapter_error")

    def test_perception_exception_raises_perception_failure_not_a_decision(self):
        class _BrokenPerception(PerceptionAdapter):
            def get_world_state(self):
                raise RuntimeError("camera feed lost")

        gate = ActuatorGate(
            perception=_BrokenPerception(),
            dynamics=_StubDynamics(fixtures.straight_line_trajectory()),
            fallback=FreezeInPlaceFallback(),
            logger=InMemoryLogger(),
            action_schema=ActionSchemaRegistry.from_dict(SCHEMA_DICT),
        )
        with self.assertRaises(PerceptionFailure):
            gate.gate(fixtures.grasp_action())


class PreconditionUnitTests(unittest.TestCase):
    """Direct tests of individual precondition functions, independent of the engine."""

    def test_object_hazard_confirmed_fails_below_confidence_threshold(self):
        low_conf = fixtures.confirmed_object()
        object.__setattr__(low_conf, "class_confidence", 0.3)  # frozen dataclass: mutate for the test
        state = fixtures.base_world_state(objects=(low_conf,))
        result = pc.object_hazard_confirmed(state, fixtures.grasp_action(), fixtures.straight_line_trajectory())
        self.assertFalse(result.satisfied)

    def test_balance_margin_defaults_to_deny_without_balance_state(self):
        # No center_of_mass / support_polygon reported at all -> must fail, never silently pass.
        traj = fixtures.straight_line_trajectory()
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        result = pc.balance_margin_maintained(state, fixtures.grasp_action(), traj)
        self.assertFalse(result.satisfied)

    def test_iso15066_blocks_within_computed_separation_distance(self):
        # far_agent() (5m away) passes the flat-margin check easily; the ISO formula's own reach,
        # stopping-distance and intrusion terms should still catch a MUCH closer agent that the
        # flat margin alone would have missed at this particular distance.
        close_ish = replace(fixtures.close_agent(), pose=fixtures.Pose(position=(1.0, 0.0, 0.05)))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(close_ish,))
        traj = fixtures.straight_line_trajectory()
        result = pc.iso15066_separation_distance_maintained(state, fixtures.grasp_action(), traj)
        self.assertFalse(result.satisfied)

    def test_iso15066_permits_well_beyond_computed_separation_distance(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.far_agent(),))
        traj = fixtures.straight_line_trajectory()
        result = pc.iso15066_separation_distance_maintained(state, fixtures.grasp_action(), traj)
        self.assertTrue(result.satisfied)

    def test_power_force_limiting_ignores_agent_far_from_contact(self):
        # Regression test for a real bug: the first version estimated a collision force for any
        # tracked agent regardless of distance, so a person 7m away still failed the check.
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.far_agent(),))
        traj = fixtures.straight_line_trajectory()
        result = pc.iso15066_power_force_limiting(state, fixtures.grasp_action(), traj)
        self.assertTrue(result.satisfied, result.reason)

    def test_power_force_limiting_blocks_high_force_at_close_contact(self):
        agent = replace(fixtures.close_agent(), pose=fixtures.Pose(position=(0.5, 0.0, 0.05)))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(agent,))
        traj = fixtures.straight_line_trajectory()
        result = pc.iso15066_power_force_limiting(state, fixtures.grasp_action(), traj, effective_mass_kg=5.0, assumed_contact_time_s=0.015, max_transient_force_n=150.0)
        self.assertFalse(result.satisfied)

    def test_iso15066_defaults_to_deny_on_empty_trajectory(self):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.far_agent(),))
        result = pc.iso15066_separation_distance_maintained(state, fixtures.grasp_action(), PredictedTrajectory(points=(), horizon_s=1.0))
        self.assertFalse(result.satisfied)


class RobotKinematicElectricalLimitTests(unittest.TestCase):
    """Checks for the robot's own physical limits: joint position/velocity/effort, motor
    temperature, Cartesian speed, and self-collision. Every one defaults to deny when the adapter
    hasn't reported the data the check needs -- exercised directly here, unit-style, the same way
    balance_margin_maintained already is."""

    def _traj(self, robot, n=3, horizon_s=0.2):
        points = [TrajectoryPoint(t=horizon_s * k / (n - 1), robot=robot,
                                   swept_volume_center=(0.5, 0.0, 0.3), swept_volume_radius_m=0.05,
                                   self_collision_margin_m=0.10)
                  for k in range(n)]
        return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)

    def test_joint_position_limits_default_deny_without_limits(self):
        result = pc.joint_position_limits_respected(None, None, self._traj(fixtures.robot_state()))
        self.assertFalse(result.satisfied)

    def test_joint_position_limits_permit_within_range(self):
        result = pc.joint_position_limits_respected(None, None, self._traj(fixtures.instrumented_robot_state()))
        self.assertTrue(result.satisfied)

    def test_joint_position_limits_block_near_edge(self):
        near_limit = fixtures.instrumented_robot_state(joint_positions=(2.89, 0, 0, 0, 0, 0, 0))
        result = pc.joint_position_limits_respected(None, None, self._traj(near_limit))
        self.assertFalse(result.satisfied)

    def test_joint_velocity_default_deny_without_limits(self):
        result = pc.joint_velocity_within_limits(None, None, self._traj(fixtures.robot_state()))
        self.assertFalse(result.satisfied)

    def test_joint_velocity_block_over_utilization(self):
        fast = fixtures.instrumented_robot_state(joint_velocities=(2.4, 0, 0, 0, 0, 0, 0))  # limit is 2.5
        result = pc.joint_velocity_within_limits(None, None, self._traj(fast))
        self.assertFalse(result.satisfied)

    def test_joint_velocity_permit_within_utilization(self):
        result = pc.joint_velocity_within_limits(None, None, self._traj(fixtures.instrumented_robot_state()))
        self.assertTrue(result.satisfied)

    def test_joint_effort_default_deny_without_estimate(self):
        no_effort = fixtures.instrumented_robot_state()
        no_effort = replace(no_effort, estimated_joint_efforts=None)
        result = pc.joint_effort_within_limits(None, None, self._traj(no_effort))
        self.assertFalse(result.satisfied)

    def test_joint_effort_block_over_utilization(self):
        overloaded = fixtures.instrumented_robot_state(joint_efforts=(85.0, 0, 0, 0, 0, 0, 0))  # limit 87.0
        result = pc.joint_effort_within_limits(None, None, self._traj(overloaded))
        self.assertFalse(result.satisfied)

    def test_motor_temperature_default_deny_without_reading(self):
        state = fixtures.base_world_state()  # robot_state() has no temperature reported
        result = pc.motor_temperature_within_limits(state, None, None)
        self.assertFalse(result.satisfied)

    def test_motor_temperature_block_near_limit(self):
        hot = fixtures.instrumented_robot_state()
        hot = replace(hot, motor_temperature_c=(78.0,) * 7)  # limit 80, margin 5
        state = WorldState(robot=hot)
        result = pc.motor_temperature_within_limits(state, None, None)
        self.assertFalse(result.satisfied)

    def test_motor_temperature_permit_with_headroom(self):
        state = WorldState(robot=fixtures.instrumented_robot_state())
        result = pc.motor_temperature_within_limits(state, None, None)
        self.assertTrue(result.satisfied)

    def test_cartesian_speed_default_deny_without_limit(self):
        result = pc.cartesian_speed_within_limits(None, None, self._traj(fixtures.robot_state()))
        self.assertFalse(result.satisfied)

    def test_cartesian_speed_block_when_too_fast(self):
        traj = PredictedTrajectory(points=(
            TrajectoryPoint(t=0.0, robot=fixtures.instrumented_robot_state(max_cartesian_speed_mps=0.5),
                             swept_volume_center=(0.0, 0.0, 0.0), swept_volume_radius_m=0.05),
            TrajectoryPoint(t=0.1, robot=fixtures.instrumented_robot_state(max_cartesian_speed_mps=0.5),
                             swept_volume_center=(0.2, 0.0, 0.0), swept_volume_radius_m=0.05),  # 2.0 m/s
        ))
        result = pc.cartesian_speed_within_limits(None, None, traj)
        self.assertFalse(result.satisfied)

    def test_cartesian_speed_permit_when_within_limit(self):
        result = pc.cartesian_speed_within_limits(None, None, self._traj(fixtures.instrumented_robot_state()))
        self.assertTrue(result.satisfied)

    def test_self_collision_default_deny_without_margin(self):
        traj = PredictedTrajectory(points=(
            TrajectoryPoint(t=0.0, robot=fixtures.robot_state(), swept_volume_center=(0, 0, 0),
                             swept_volume_radius_m=0.05, self_collision_margin_m=None),
        ))
        result = pc.self_collision_clear(None, None, traj)
        self.assertFalse(result.satisfied)

    def test_self_collision_block_when_links_too_close(self):
        traj = PredictedTrajectory(points=(
            TrajectoryPoint(t=0.0, robot=fixtures.robot_state(), swept_volume_center=(0, 0, 0),
                             swept_volume_radius_m=0.05, self_collision_margin_m=0.005),
        ))
        result = pc.self_collision_clear(None, None, traj)
        self.assertFalse(result.satisfied)

    def test_self_collision_permit_with_clearance(self):
        result = pc.self_collision_clear(None, None, self._traj(fixtures.robot_state()))
        self.assertTrue(result.satisfied)

    def test_battery_default_deny_without_reading(self):
        state = fixtures.base_world_state()  # robot_state() has no battery reported
        result = pc.battery_charge_sufficient(state, None, None)
        self.assertFalse(result.satisfied)

    def test_battery_block_when_low(self):
        low = replace(fixtures.instrumented_robot_state(), battery_charge_fraction=0.10)  # below 0.15 default
        state = WorldState(robot=low)
        result = pc.battery_charge_sufficient(state, None, None)
        self.assertFalse(result.satisfied)

    def test_battery_permit_with_reserve(self):
        state = WorldState(robot=fixtures.instrumented_robot_state())  # 0.9 charge
        result = pc.battery_charge_sufficient(state, None, None)
        self.assertTrue(result.satisfied)


class IsaacLabDynamicsAdapterTests(unittest.TestCase):
    """Exercises the Isaac Lab reference DynamicsAdapter's own logic directly -- no live env needed,
    since it only touches WorldState/Action, not the simulator, until predict_trajectory is called
    with real cube/robot state from a real env (Development Roadmap stage 3, not yet run)."""

    def test_missing_target_position_raises_instead_of_predicting_no_motion(self):
        from safety_harness.adapters.isaac_lab import IsaacLabCubeStackDynamicsAdapter

        adapter = IsaacLabCubeStackDynamicsAdapter()
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        from safety_harness.schema import Action

        with self.assertRaises(ValueError):
            adapter.predict_trajectory(state, Action(action_type="grasp", params={"object_id": "cube_2"}), 1.0)

    def test_with_target_position_predicts_toward_it(self):
        from safety_harness.adapters.isaac_lab import IsaacLabCubeStackDynamicsAdapter

        adapter = IsaacLabCubeStackDynamicsAdapter(max_ee_speed_mps=10.0)  # fast enough to reach target within horizon
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        traj = adapter.predict_trajectory(state, fixtures.grasp_action(target_position=(0.5, 0.0, 0.05)), 1.0)
        self.assertAlmostEqual(traj.points[0].swept_volume_center[2], state.robot.end_effector_pose.position[2], places=3)
        self.assertAlmostEqual(traj.points[-1].swept_volume_center[2], 0.05, places=3)


if __name__ == "__main__":
    unittest.main()
