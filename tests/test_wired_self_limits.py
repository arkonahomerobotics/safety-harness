"""joint_position_limits_respected and cartesian_speed_within_limits, now wired into every action
type of the example schema -- and the adapter-side changes that make them able to fire at all.

Both were live-demonstrated as missing: a Franka joint driven to within 0.01 rad of its limit, and a
command implying ~40 m/s, each PERMITted end to end because neither check was wired. Wiring alone
would not have been enough for the second: the reference dynamics adapters extrapolated every
motion at a fixed capped speed, so the speed check could only ever see the cap."""

import math
import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

from safety_harness import preconditions as pc
from safety_harness.adapters._isaac_lab_common import commanded_speed_mps
from safety_harness.adapters.isaac_lab import IsaacLabCubeStackDynamicsAdapter
from safety_harness.adapters.isaac_lab_anymal import IsaacLabAnymalNavDynamicsAdapter
from safety_harness.schema import Action, PredictedTrajectory, TrajectoryPoint, WorldState

sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402


def _traj(robot, n=3, horizon_s=0.2):
    return PredictedTrajectory(points=tuple(
        TrajectoryPoint(t=horizon_s * k / (n - 1), robot=robot, swept_volume_center=(0.5, 0.0, 0.3),
                        swept_volume_radius_m=0.05) for k in range(n)), horizon_s=horizon_s)


class JointLimitExemptionsTest(unittest.TestCase):
    def test_gripper_at_stop_blocks_without_exemption(self):
        # 7 arm joints mid-range + 2 Franka fingers fully open at their 0.04m stop
        r = replace(fixtures.robot_state(), joint_positions=(0.0,) * 7 + (0.04, 0.04),
                    joint_position_limits=((-2.9, 2.9),) * 7 + ((0.0, 0.04),) * 2)
        self.assertFalse(pc.joint_position_limits_respected(None, None, _traj(r)).satisfied)

    def test_gripper_at_stop_permits_with_explicit_exemption(self):
        r = replace(fixtures.robot_state(), joint_positions=(0.0,) * 7 + (0.04, 0.04),
                    joint_position_limits=((-2.9, 2.9),) * 7 + (None, None))
        self.assertTrue(pc.joint_position_limits_respected(None, None, _traj(r)).satisfied)

    def test_exemption_does_not_hide_an_arm_joint_at_its_limit(self):
        r = replace(fixtures.robot_state(), joint_positions=(2.89,) + (0.0,) * 6 + (0.04, 0.04),
                    joint_position_limits=((-2.9, 2.9),) * 7 + (None, None))
        self.assertFalse(pc.joint_position_limits_respected(None, None, _traj(r)).satisfied)

    def test_fewer_limits_than_joints_fails_closed(self):
        # previously zip() silently skipped the unmatched joints -- here joint 7 is past its limit
        r = replace(fixtures.robot_state(), joint_positions=(0.0,) * 7 + (9.9,),
                    joint_position_limits=((-2.9, 2.9),) * 7)
        self.assertFalse(pc.joint_position_limits_respected(None, None, _traj(r)).satisfied)

    def test_nan_limit_fails_closed(self):
        r = replace(fixtures.robot_state(), joint_position_limits=((float("nan"), 2.9),) + ((-2.9, 2.9),) * 6)
        self.assertFalse(pc.joint_position_limits_respected(None, None, _traj(r)).satisfied)

    def test_within_001_rad_of_limit_blocks(self):
        # the exact live condition that previously slipped through
        r = replace(fixtures.robot_state(), joint_positions=(2.89,) + (0.0,) * 6)
        self.assertFalse(pc.joint_position_limits_respected(None, None, _traj(r)).satisfied)


class CommandedSpeedTest(unittest.TestCase):
    def test_duration_sets_speed(self):
        self.assertAlmostEqual(commanded_speed_mps(Action("reach", {"duration_s": 0.05}), 2.0, 0.5), 40.0)

    def test_explicit_speed_wins(self):
        self.assertEqual(commanded_speed_mps(Action("reach", {"commanded_speed_mps": 3.0, "duration_s": 9.0}), 2.0, 0.5), 3.0)

    def test_fallback_when_unstated(self):
        self.assertEqual(commanded_speed_mps(Action("reach", {}), 2.0, 0.5), 0.5)

    def test_bad_duration_raises(self):
        for d in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                commanded_speed_mps(Action("reach", {"duration_s": d}), 2.0, 0.5)

    def test_bad_speed_raises(self):
        for v in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                commanded_speed_mps(Action("reach", {"commanded_speed_mps": v}), 2.0, 0.5)


def _state(robot):
    return WorldState(robot=robot)


class DynamicsAdapterSpeedTest(unittest.TestCase):
    """The reference dynamics adapters now predict at the commanded speed, so the speed check (and
    every swept-path check) sees the motion that will actually happen."""

    def _check(self, adapter, robot, action):
        traj = adapter.predict_trajectory(_state(robot), action, horizon_s=0.5)
        return traj, pc.cartesian_speed_within_limits(None, action, traj)

    def test_franka_40mps_command_blocks(self):
        robot = replace(fixtures.robot_state(), max_cartesian_speed_mps=1.7)
        a = Action("reach", {"target_position": (2.5, 0.0, 0.3), "duration_s": 0.05})  # 2.0m in 0.05s
        _, res = self._check(IsaacLabCubeStackDynamicsAdapter(), robot, a)
        self.assertFalse(res.satisfied, res.reason)

    def test_franka_same_target_capped_fallback_would_have_permitted(self):
        # documents the old behavior: without a stated command speed the adapter falls back to its
        # 0.5 m/s cap, which can never exceed the 1.7 m/s rating
        robot = replace(fixtures.robot_state(), max_cartesian_speed_mps=1.7)
        a = Action("reach", {"target_position": (2.5, 0.0, 0.3)})
        _, res = self._check(IsaacLabCubeStackDynamicsAdapter(), robot, a)
        self.assertTrue(res.satisfied)

    def test_franka_normal_command_permits(self):
        robot = replace(fixtures.robot_state(), max_cartesian_speed_mps=1.7)
        a = Action("reach", {"target_position": (0.5, 0.0, 0.05), "duration_s": 1.0})  # 0.25 m/s
        _, res = self._check(IsaacLabCubeStackDynamicsAdapter(), robot, a)
        self.assertTrue(res.satisfied, res.reason)

    def test_faster_command_sweeps_farther_within_horizon(self):
        robot = fixtures.robot_state()
        a_slow = Action("reach", {"target_position": (2.5, 0.0, 0.3)})
        a_fast = Action("reach", {"target_position": (2.5, 0.0, 0.3), "commanded_speed_mps": 2.0})
        ad = IsaacLabCubeStackDynamicsAdapter()
        slow = ad.predict_trajectory(_state(robot), a_slow, 0.5).points[-1].swept_volume_center
        fast = ad.predict_trajectory(_state(robot), a_fast, 0.5).points[-1].swept_volume_center
        self.assertGreater(math.dist(robot.end_effector_pose.position, fast),
                           math.dist(robot.end_effector_pose.position, slow) * 3)

    def test_anymal_overspeed_navigate_blocks(self):
        robot = fixtures.quadruped_robot_state()  # rated 1.0 m/s
        a = Action("navigate", {"target_position": (3.0, 0.0, 0.6), "commanded_speed_mps": 3.0})
        _, res = self._check(IsaacLabAnymalNavDynamicsAdapter(), robot, a)
        self.assertFalse(res.satisfied, res.reason)

    def test_anymal_normal_navigate_permits(self):
        robot = fixtures.quadruped_robot_state()
        a = Action("navigate", {"target_position": (3.0, 0.0, 0.6)})  # fallback 1.0 m/s, not above rating
        _, res = self._check(IsaacLabAnymalNavDynamicsAdapter(), robot, a)
        self.assertTrue(res.satisfied, res.reason)


class JointLimitsFromSimTest(unittest.TestCase):
    def test_exempt_substring_gives_none(self):
        from safety_harness.adapters._isaac_lab_common import joint_position_limits

        class _T:
            def __init__(self, v): self.v = v
            def tolist(self): return self.v
        lims = [[-2.9, 2.9]] * 7 + [[0.0, 0.04]] * 2
        robot = SimpleNamespace(data=SimpleNamespace(
            soft_joint_pos_limits=SimpleNamespace(torch=[_T(lims)]),
            joint_names=[f"panda_joint{k}" for k in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]))
        out = joint_position_limits(robot, 0, ("panda_finger",))
        self.assertEqual(out[:7], ((-2.9, 2.9),) * 7)
        self.assertEqual(out[7:], (None, None))

    def test_non_finite_limit_reports_unreported(self):
        # Isaac Lab's ANYmal-C asset: every leg joint -inf..inf -> the whole field is unreported
        from safety_harness.adapters._isaac_lab_common import joint_position_limits

        class _T:
            def __init__(self, v): self.v = v
            def tolist(self): return self.v
        inf = float("inf")
        robot = SimpleNamespace(data=SimpleNamespace(
            soft_joint_pos_limits=SimpleNamespace(torch=[_T([[-inf, inf]] * 12)]),
            joint_names=[f"LF_HAA{k}" for k in range(12)]))
        self.assertIsNone(joint_position_limits(robot, 0))


class ExampleSchemaWiringTest(unittest.TestCase):
    def test_wired_where_meaningful(self):
        import yaml
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "example_action_schema.yaml")))
        names = {a: [c["name"] for c in s["checks"]] for a, s in cfg["action_types"].items()}
        for a in ("grasp", "place", "reach"):
            self.assertIn("joint_position_limits_respected", names[a])
            self.assertIn("cartesian_speed_within_limits", names[a])
        self.assertIn("cartesian_speed_within_limits", names["navigate"])
        # the ANYmal-C sim asset has no joint limits to check against -- see the config comment
        self.assertNotIn("joint_position_limits_respected", names["navigate"])


if __name__ == "__main__":
    unittest.main()
