"""Tests the ANYmal-C navigation adapter (safety_harness/adapters/isaac_lab_anymal.py) and the new
`navigate` action type against hand-built fixtures -- no live env needed, mirroring
test_engine.py's own IsaacLabDynamicsAdapterTests convention for the Franka adapter. This is the
second-adapter abstraction test the design doc's Development Roadmap calls for: same engine, same
precondition library, a robot whose own body is the thing sweeping through space instead of an
end-effector, and -- for the first time anywhere in this suite -- a real support polygon feeding
balance_margin_maintained through a real ActuatorGate.gate() call, not just a hand-built fixture
probing the function directly.

Run with: python3 -m unittest discover -s tests -p "test_*.py" -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402


def _failed_names(decision):
    return [r.name for r in decision.precondition_results if not r.satisfied]


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


NAVIGATE_SCHEMA_DICT = {
    "action_types": {
        "navigate": {
            "checks": [
                {"name": "robot_state_confirmed", "kwargs": {}},
                {"name": "balance_margin_maintained", "kwargs": {"min_margin_m": 0.03}},
                {"name": "swept_path_clear_of_agents", "kwargs": {"margin_m": 0.15}},
                {"name": "iso15066_separation_distance_maintained", "kwargs": {}},
                {"name": "iso15066_power_force_limiting", "kwargs": {}},
                {"name": "reduced_speed_near_human", "kwargs": {}},
                {"name": "swept_path_clear_of_risky_objects", "kwargs": {"margin_m": 0.15}},
                {"name": "visibility_above_threshold", "kwargs": {"min_visibility": 0.5}},
                {"name": "environment_hazard_clear", "kwargs": {}},
            ]
        }
    }
}


def _base_footprint_trajectory(robot, start=(0.0, 0.0, 0.6), end=(3.0, 0.0, 0.6), horizon_s=3.0, n_points=6, radius_m=0.35):
    """A moving BASE's swept footprint, not an end-effector's -- same shape as
    fixtures.straight_line_trajectory, at a robot-body radius instead of a gripper radius. Carries
    the GIVEN robot state (support polygon, center of mass) at every point, exactly as
    IsaacLabAnymalNavDynamicsAdapter does against a real env -- it reuses whatever robot state the
    perception adapter reported, never a default of its own. (An earlier version of this helper
    built its own default robot internally, silently ignoring the one the state under test carried;
    two tests below caught it by asserting BLOCK and getting PERMIT instead.)"""
    from safety_harness.schema import PredictedTrajectory, TrajectoryPoint

    points = []
    for k in range(n_points):
        t = horizon_s * k / (n_points - 1)
        frac = k / (n_points - 1)
        center = tuple(s + frac * (e - s) for s, e in zip(start, end))
        points.append(TrajectoryPoint(t=t, robot=robot, swept_volume_center=center, swept_volume_radius_m=radius_m))
    return PredictedTrajectory(points=tuple(points), horizon_s=horizon_s)


def make_nav_gate(state, trajectory=None):
    trajectory = trajectory or _base_footprint_trajectory(state.robot)
    return ActuatorGate(
        perception=_StubPerception(state),
        dynamics=_StubDynamics(trajectory),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=ActionSchemaRegistry.from_dict(NAVIGATE_SCHEMA_DICT),
    )


class DynamicsAdapterGeneralizesToABaseSweep(unittest.TestCase):
    """IsaacLabAnymalNavDynamicsAdapter's own logic, no live env needed -- it only touches
    WorldState/Action until predict_trajectory is called with real state from a real env."""

    def test_missing_target_position_raises(self):
        from safety_harness.adapters.isaac_lab_anymal import IsaacLabAnymalNavDynamicsAdapter
        from safety_harness.schema import Action

        adapter = IsaacLabAnymalNavDynamicsAdapter()
        state = fixtures.quadruped_world_state()
        with self.assertRaises(ValueError):
            adapter.predict_trajectory(state, Action(action_type="navigate", params={}), 1.0)

    def test_predicts_the_base_sweeping_toward_the_target(self):
        from safety_harness.adapters.isaac_lab_anymal import IsaacLabAnymalNavDynamicsAdapter

        adapter = IsaacLabAnymalNavDynamicsAdapter(max_base_speed_mps=10.0)  # fast enough to reach target within horizon
        state = fixtures.quadruped_world_state()
        traj = adapter.predict_trajectory(state, fixtures.navigate_action(target_position=(3.0, 0.0, 0.6)), 1.0)
        self.assertAlmostEqual(traj.points[0].swept_volume_center[0], 0.0, places=3)
        self.assertAlmostEqual(traj.points[-1].swept_volume_center[0], 3.0, places=3)
        # It's the robot's own body sweeping, not a gripper -- a real, robot-scale radius, not the
        # Franka adapter's gripper-scale one.
        self.assertGreater(traj.points[0].swept_volume_radius_m, 0.2)

    def test_reuses_current_robot_state_at_every_point_no_gait_forward_sim(self):
        from safety_harness.adapters.isaac_lab_anymal import IsaacLabAnymalNavDynamicsAdapter

        adapter = IsaacLabAnymalNavDynamicsAdapter(max_base_speed_mps=1.0)
        state = fixtures.quadruped_world_state()
        traj = adapter.predict_trajectory(state, fixtures.navigate_action(target_position=(3.0, 0.0, 0.6)), 1.0)
        self.assertTrue(all(p.robot is state.robot for p in traj.points))


class BalanceMarginNowWiredThroughARealGateCall(unittest.TestCase):
    """First time balance_margin_maintained is reached through ActuatorGate.gate() with anything
    other than a hand-built probe of the function itself or a NaN fault-injection test -- a real
    four-foot support polygon, comfortably under the robot vs. shifted well clear of it."""

    def test_permit_when_comfortably_balanced_and_path_clear(self):
        state = fixtures.quadruped_world_state()
        decision = make_nav_gate(state).gate(fixtures.navigate_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)
        self.assertTrue(all(r.satisfied for r in decision.precondition_results))

    def test_block_when_support_polygon_shifted_clear_of_center_of_mass(self):
        state = fixtures.quadruped_world_state(robot=fixtures.tipping_quadruped_robot_state())
        decision = make_nav_gate(state).gate(fixtures.navigate_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn("balance_margin_maintained", _failed_names(decision))

    def test_default_denies_without_any_balance_state_reported(self):
        """Same convention as everywhere else: unconfirmed is the most restrictive case."""
        state = fixtures.quadruped_world_state(robot=replace(fixtures.quadruped_robot_state(), support_polygon=None))
        decision = make_nav_gate(state).gate(fixtures.navigate_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn("balance_margin_maintained", _failed_names(decision))


class HumanProximityChecksGeneralizeToTheRobotsOwnBodySweep(unittest.TestCase):
    """The exact checks validated against a Franka gripper's swept path now gate the ANYmal's own
    body -- same functions, same config keys, a different physical thing sweeping through space."""

    def test_permit_when_agent_far_from_the_bodys_path(self):
        state = fixtures.quadruped_world_state(agents=(fixtures.far_agent(),))
        decision = make_nav_gate(state).gate(fixtures.navigate_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)

    def test_block_when_agent_close_to_the_bodys_path(self):
        close_to_base_path = replace(fixtures.close_agent(), pose=replace(fixtures.close_agent().pose, position=(1.5, 0.0, 0.0)))
        state = fixtures.quadruped_world_state(agents=(close_to_base_path,))
        decision = make_nav_gate(state).gate(fixtures.navigate_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        failed = _failed_names(decision)
        self.assertIn("swept_path_clear_of_agents", failed)
        self.assertIn("iso15066_separation_distance_maintained", failed)


if __name__ == "__main__":
    unittest.main()
