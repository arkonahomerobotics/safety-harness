"""safety_harness.adapters.ros2 -- pure-Python, no rclpy required (the adapter itself never imports
it, same convention as the Isaac Lab adapters not importing isaaclab -- see that module's docstring).
Bridge objects here are hand-built SimpleNamespaces shaped exactly like the real ROS 2 messages the
adapter documents (sensor_msgs/JointState, geometry_msgs/PoseStamped, builtin_interfaces/Time), the
same mocking convention as tests/test_wired_self_limits.py uses for Isaac Lab's `robot.data`.

Run with: python3 -m unittest discover -s tests -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_harness.adapters.ros2 import ROS2DynamicsAdapter, ROS2PerceptionAdapter  # noqa: E402
from safety_harness.schema import Action, FallConsequence, HazardTag, AgentCategory  # noqa: E402


def _stamp(epoch_s: float):
    sec = int(epoch_s)
    return SimpleNamespace(sec=sec, nanosec=int(round((epoch_s - sec) * 1e9)))


def _joint_state(names, positions, velocities, stamp_s, effort=None):
    kw = {}
    if effort is not None:
        kw["effort"] = list(effort)
    return SimpleNamespace(name=list(names), position=list(positions), velocity=list(velocities),
                            header=SimpleNamespace(stamp=_stamp(stamp_s)), **kw)


def _ee_pose(xyz, wxyz, stamp_s):
    w, x, y, z = wxyz
    return SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
                              orientation=SimpleNamespace(x=x, y=y, z=z, w=w)),
        header=SimpleNamespace(stamp=_stamp(stamp_s)),
    )


def _tracked_object(**kw):
    kw.setdefault("object_class", "cube")
    kw.setdefault("velocity", (0.0, 0.0, 0.0))
    return SimpleNamespace(**kw)


def _tracked_agent(**kw):
    kw.setdefault("velocity", (0.0, 0.0, 0.0))
    return SimpleNamespace(**kw)


class NoDataYetTest(unittest.TestCase):
    """Nothing published yet on any bridge topic -- must read as absent, never fabricated."""

    def test_empty_bridge_gives_no_robot_and_no_sensor_timestamp(self):
        state = ROS2PerceptionAdapter(SimpleNamespace()).get_world_state()
        self.assertIsNone(state.robot)
        self.assertEqual(state.objects, ())
        self.assertEqual(state.agents, ())
        self.assertIsNone(state.sensor_timestamp)

    def test_ee_pose_without_joint_state_is_ignored(self):
        # a real bridge could receive /ee_pose before /joint_states -- robot needs both to exist
        bridge = SimpleNamespace(joint_state=None, ee_pose=_ee_pose((1, 2, 3), (1, 0, 0, 0), 100.0))
        state = ROS2PerceptionAdapter(bridge).get_world_state()
        self.assertIsNone(state.robot)


class RobotStateTest(unittest.TestCase):
    def test_joint_state_and_ee_pose_populate_robot(self):
        bridge = SimpleNamespace(
            joint_state=_joint_state(["j1", "j2"], [0.1, 0.2], [0.0, 0.0], 1000.0),
            ee_pose=_ee_pose((0.5, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0), 1000.5),
            joint_position_limits=((-1.0, 1.0), (-2.0, 2.0)),
            max_cartesian_speed_mps=1.5,
            rated_payload_kg=3.0,
        )
        state = ROS2PerceptionAdapter(bridge).get_world_state()
        self.assertEqual(state.robot.joint_positions, (0.1, 0.2))
        self.assertEqual(state.robot.end_effector_pose.position, (0.5, 0.0, 0.3))
        self.assertEqual(state.robot.joint_position_limits, ((-1.0, 1.0), (-2.0, 2.0)))
        self.assertEqual(state.robot.max_cartesian_speed_mps, 1.5)
        self.assertEqual(state.robot.rated_payload_kg, 3.0)
        # oldest of the two stamps (1000.0), not the newest and not "now" -- sensor_data_fresh's whole point
        self.assertEqual(state.sensor_timestamp, 1000.0)

    def test_missing_ee_pose_defaults_to_origin_not_a_crash(self):
        bridge = SimpleNamespace(joint_state=_joint_state(["j1"], [0.0], [0.0], 5.0), ee_pose=None)
        state = ROS2PerceptionAdapter(bridge).get_world_state()
        self.assertIsNotNone(state.robot)
        self.assertEqual(state.robot.end_effector_pose.position, (0.0, 0.0, 0.0))

    def test_velocity_and_effort_limits_and_estimate_mapped(self):
        bridge = SimpleNamespace(
            joint_state=_joint_state(["j1", "j2"], [0.0, 0.0], [0.1, 0.2], 3.0, effort=(1.5, 2.5)),
            ee_pose=None,
            joint_velocity_limits=(2.0, 2.0),
            joint_effort_limits=(50.0, 50.0),
        )
        state = ROS2PerceptionAdapter(bridge).get_world_state()
        self.assertEqual(state.robot.joint_velocity_limits, (2.0, 2.0))
        self.assertEqual(state.robot.joint_effort_limits, (50.0, 50.0))
        self.assertEqual(state.robot.estimated_joint_efforts, (1.5, 2.5))

    def test_missing_effort_field_reports_unestimated(self):
        bridge = SimpleNamespace(joint_state=_joint_state(["j1"], [0.0], [0.0], 1.0), ee_pose=None)
        state = ROS2PerceptionAdapter(bridge).get_world_state()
        self.assertIsNone(state.robot.estimated_joint_efforts)

    def test_joint_limit_exemption_by_name_substring(self):
        bridge = SimpleNamespace(
            joint_state=_joint_state(["shoulder", "gripper_left", "gripper_right"], [0.0, 0.04, 0.04], [0, 0, 0], 1.0),
            ee_pose=None,
            joint_position_limits=((-2.9, 2.9), (0.0, 0.04), (0.0, 0.04)),
        )
        state = ROS2PerceptionAdapter(bridge, joint_limit_exempt=("gripper",)).get_world_state()
        self.assertEqual(state.robot.joint_position_limits, ((-2.9, 2.9), None, None))


class TrackedObjectsAndAgentsTest(unittest.TestCase):
    def test_object_fields_mapped_correctly(self):
        obj = _tracked_object(object_id="block_a", position=(0.1, 0.2, 0.3), mass_kg=0.4,
                               hazard_tags=("fragile",), pose_confidence=0.9, class_confidence=0.8,
                               cleared_for_interaction=True, supported_stably=True,
                               fall_consequence="mess", stamp=42.0)
        bridge = SimpleNamespace(tracked_objects=[obj])
        state = ROS2PerceptionAdapter(bridge).get_world_state()
        self.assertEqual(len(state.objects), 1)
        o = state.objects[0]
        self.assertEqual(o.object_id, "block_a")
        self.assertEqual(o.estimated_mass_kg, 0.4)
        self.assertEqual(o.hazard_tags, frozenset({HazardTag.FRAGILE}))
        self.assertEqual(o.fall_consequence, FallConsequence.MESS)
        self.assertTrue(o.cleared_for_interaction)
        self.assertEqual(state.sensor_timestamp, 42.0)

    def test_object_missing_hazard_tags_defaults_to_unknown_not_permit(self):
        obj = _tracked_object(object_id="mystery", position=(0, 0, 0), mass_kg=None,
                               pose_confidence=0.5, class_confidence=0.5,
                               cleared_for_interaction=False, supported_stably=None, stamp=1.0)
        state = ROS2PerceptionAdapter(SimpleNamespace(tracked_objects=[obj])).get_world_state()
        self.assertEqual(state.objects[0].hazard_tags, frozenset({HazardTag.UNKNOWN}))

    def test_agent_fields_and_category_mapped(self):
        agent = _tracked_agent(agent_id="person_1", position=(1.0, 0.0, 0.0), tracking_confidence=0.9,
                                category="child", stature_m=1.1, time_since_confirmed_s=0.2, stamp=7.0)
        state = ROS2PerceptionAdapter(SimpleNamespace(tracked_agents=[agent])).get_world_state()
        self.assertEqual(len(state.agents), 1)
        a = state.agents[0]
        self.assertEqual(a.agent_id, "person_1")
        self.assertEqual(a.category, AgentCategory.CHILD)
        self.assertEqual(a.stature_m, 1.1)

    def test_agent_missing_category_defaults_unknown(self):
        agent = _tracked_agent(agent_id="p", position=(0, 0, 0), tracking_confidence=0.5, stamp=1.0)
        state = ROS2PerceptionAdapter(SimpleNamespace(tracked_agents=[agent])).get_world_state()
        self.assertEqual(state.agents[0].category, AgentCategory.UNKNOWN)


class DynamicsAdapterTest(unittest.TestCase):
    def _state(self, ee=(0.0, 0.0, 0.0)):
        bridge = SimpleNamespace(joint_state=_joint_state(["j1"], [0.0], [0.0], 1.0),
                                  ee_pose=_ee_pose(ee, (1, 0, 0, 0), 1.0))
        return ROS2PerceptionAdapter(bridge).get_world_state()

    def test_straight_line_sweep_toward_target(self):
        state = self._state(ee=(0.0, 0.0, 0.0))
        action = Action("reach", {"target_position": (1.0, 0.0, 0.0), "commanded_speed_mps": 1.0})
        traj = ROS2DynamicsAdapter(n_points=3).predict_trajectory(state, action, horizon_s=1.0)
        self.assertAlmostEqual(traj.points[0].swept_volume_center[0], 0.0)
        self.assertAlmostEqual(traj.points[-1].swept_volume_center[0], 1.0)

    def test_prefers_commanded_speed_over_configured_cap(self):
        state = self._state()
        action = Action("reach", {"target_position": (10.0, 0.0, 0.0), "commanded_speed_mps": 40.0})
        traj = ROS2DynamicsAdapter(max_ee_speed_mps=0.5, n_points=2).predict_trajectory(state, action, horizon_s=0.1)
        # at 40 m/s for 0.1s -> 4m covered, not the 0.05m a 0.5 m/s cap would predict
        self.assertAlmostEqual(traj.points[-1].swept_volume_center[0], 4.0)

    def test_duration_s_implies_speed_when_no_explicit_speed_given(self):
        state = self._state()
        action = Action("reach", {"target_position": (2.0, 0.0, 0.0), "duration_s": 1.0})
        traj = ROS2DynamicsAdapter(max_ee_speed_mps=0.5, n_points=2).predict_trajectory(state, action, horizon_s=1.0)
        self.assertAlmostEqual(traj.points[-1].swept_volume_center[0], 2.0)

    def test_missing_target_position_raises(self):
        state = self._state()
        with self.assertRaises(ValueError):
            ROS2DynamicsAdapter().predict_trajectory(state, Action("reach", {}), horizon_s=1.0)

    def test_negative_commanded_speed_raises(self):
        state = self._state()
        action = Action("reach", {"target_position": (1.0, 0.0, 0.0), "commanded_speed_mps": -5.0})
        with self.assertRaises(ValueError):
            ROS2DynamicsAdapter().predict_trajectory(state, action, horizon_s=1.0)


if __name__ == "__main__":
    unittest.main()
