"""Pre-outreach fixes in 0.3.1: robot_state_confirmed validates content; surface_confirmed_stable is
deprecated; and a regression test pinning that destination_confirmed_stable_and_clear never counts
the object being placed as clutter at its own destination (a hypothesis in an earlier write-up that
turned out to be wrong -- the check already excluded it)."""

import math
import os
import sys
import unittest
import warnings
from dataclasses import replace

from safety_harness import preconditions as pc
from safety_harness.action_schema import ActionSchemaRegistry
from safety_harness.schema import Action, FallConsequence, Pose, WorldState

sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402


class RobotStateContentTest(unittest.TestCase):
    def _state(self, **kw):
        return WorldState(robot=replace(fixtures.robot_state(), **kw))

    def test_valid_state_passes(self):
        self.assertTrue(pc.robot_state_confirmed(WorldState(robot=fixtures.robot_state()), None, None).satisfied)

    def test_missing_state_fails(self):
        self.assertFalse(pc.robot_state_confirmed(WorldState(robot=None), None, None).satisfied)

    def test_nan_joint_fails(self):
        self.assertFalse(pc.robot_state_confirmed(self._state(joint_positions=(math.nan,) + (0.0,) * 6), None, None).satisfied)

    def test_inf_velocity_fails(self):
        self.assertFalse(pc.robot_state_confirmed(self._state(joint_velocities=(math.inf,) + (0.0,) * 6), None, None).satisfied)

    def test_nan_end_effector_fails(self):
        self.assertFalse(pc.robot_state_confirmed(self._state(end_effector_pose=Pose(position=(math.nan, 0.0, 0.3))), None, None).satisfied)

    def test_length_mismatch_fails(self):
        self.assertFalse(pc.robot_state_confirmed(self._state(joint_velocities=(0.0,) * 6), None, None).satisfied)

    def test_empty_joints_fails(self):
        self.assertFalse(pc.robot_state_confirmed(self._state(joint_positions=(), joint_velocities=()), None, None).satisfied)

    def test_non_numeric_fails(self):
        self.assertFalse(pc.robot_state_confirmed(self._state(joint_positions=("0.1",) + (0.0,) * 6), None, None).satisfied)


class DeprecationTest(unittest.TestCase):
    def test_config_using_deprecated_check_warns_but_loads(self):
        raw = {"action_types": {"place": {"checks": [{"name": "surface_confirmed_stable", "kwargs": {}}]}}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ActionSchemaRegistry.from_dict(raw)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) and "destination_confirmed_stable_and_clear" in str(x.message) for x in w))

    def test_example_config_uses_no_deprecated_check(self):
        cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "example_action_schema.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ActionSchemaRegistry.from_yaml(cfg)


class DestinationExcludesPlacedObjectTest(unittest.TestCase):
    def test_placed_object_is_not_destination_clutter(self):
        surface = replace(fixtures.confirmed_object("cube_1", position=(0.5, 0.0, 0.05)), supported_stably=True)
        # the object being placed, held right over the destination and marked risky -- still excluded
        held = replace(fixtures.confirmed_object("cube_2", position=(0.5, 0.0, 0.10)), cleared_for_interaction=False,
                       fall_consequence=FallConsequence.HAZARDOUS_RELEASE)
        state = WorldState(objects=(surface, held), robot=fixtures.robot_state())
        action = Action("place", {"object_id": "cube_2", "target_surface_id": "cube_1", "target_position": (0.5, 0.0, 0.1)})
        self.assertTrue(pc.destination_confirmed_stable_and_clear(state, action, None).satisfied)

    def test_a_different_risky_object_near_destination_still_blocks(self):
        surface = replace(fixtures.confirmed_object("cube_1", position=(0.5, 0.0, 0.05)), supported_stably=True)
        other = replace(fixtures.confirmed_object("cube_3", position=(0.55, 0.0, 0.05)), cleared_for_interaction=False)
        state = WorldState(objects=(surface, other), robot=fixtures.robot_state())
        action = Action("place", {"object_id": "cube_2", "target_surface_id": "cube_1", "target_position": (0.5, 0.0, 0.1)})
        self.assertFalse(pc.destination_confirmed_stable_and_clear(state, action, None).satisfied)


if __name__ == "__main__":
    unittest.main()
