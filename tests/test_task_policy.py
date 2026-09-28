"""Tests for safety_harness.task_policy: retryable-vs-non-retryable classification, and the
run_task_queue reference orchestrator ("proceed with whatever the next task is, as long as moving
the blocked task is not a prerequisite" -- see the module docstring for the full rationale).

Run with: python3 -m unittest discover -s tests -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict, PreconditionResult  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402
from safety_harness.schema import Decision  # noqa: E402
from safety_harness.task_policy import Task, is_retryable, run_task_queue  # noqa: E402

SCHEMA_DICT = {
    "action_types": {
        "grasp": {
            "checks": [
                {"name": "robot_state_confirmed", "kwargs": {}},
                {"name": "current_position_confirmed_stable", "kwargs": {"object_id_param": "object_id"}},
                {"name": "mass_within_force_budget", "kwargs": {"object_id_param": "object_id", "force_budget_kg": 3.0}},
            ]
        },
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


def make_gate(state):
    return ActuatorGate(
        perception=_StubPerception(state),
        dynamics=_StubDynamics(fixtures.straight_line_trajectory()),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=ActionSchemaRegistry.from_dict(SCHEMA_DICT),
    )


def _decision(verdict, *results):
    return Decision(verdict=verdict, action=fixtures.grasp_action(), precondition_results=tuple(results))


class IsRetryableTests(unittest.TestCase):
    def test_permit_is_trivially_retryable(self):
        d = _decision(DecisionVerdict.PERMIT, PreconditionResult("mass_within_force_budget", True, "ok"))
        self.assertTrue(is_retryable(d))

    def test_block_on_only_retryable_checks_is_retryable(self):
        d = _decision(
            DecisionVerdict.BLOCK,
            PreconditionResult("current_position_confirmed_stable", False, "not confirmed yet"),
            PreconditionResult("mass_within_force_budget", True, "ok"),
        )
        self.assertTrue(is_retryable(d))

    def test_block_on_a_non_retryable_check_is_not_retryable(self):
        d = _decision(
            DecisionVerdict.BLOCK,
            PreconditionResult("mass_within_force_budget", False, "5.00kg exceeds budget 3.0kg"),
        )
        self.assertFalse(is_retryable(d))

    def test_mixed_retryable_and_non_retryable_is_not_retryable(self):
        # clearing the retryable one wouldn't be enough on its own -- the whole decision stays non-retryable.
        d = _decision(
            DecisionVerdict.BLOCK,
            PreconditionResult("current_position_confirmed_stable", False, "not confirmed yet"),
            PreconditionResult("mass_within_force_budget", False, "5.00kg exceeds budget 3.0kg"),
        )
        self.assertFalse(is_retryable(d))

    def test_unrecognized_check_name_defaults_to_non_retryable(self):
        # default-deny for the classifier itself: an unknown check name is not assumed retryable.
        d = _decision(DecisionVerdict.BLOCK, PreconditionResult("some_future_check_this_module_has_never_heard_of", False, "?"))
        self.assertFalse(is_retryable(d))

    def test_override_retryable_checks(self):
        d = _decision(DecisionVerdict.BLOCK, PreconditionResult("mass_within_force_budget", False, "over budget"))
        self.assertTrue(is_retryable(d, retryable_checks=frozenset({"mass_within_force_budget"})))


class RunTaskQueueTests(unittest.TestCase):
    def setUp(self):
        heavy = fixtures.confirmed_object(object_id="block_a", mass_kg=5.0, position=(0.5, 0.0, 0.05))
        light = fixtures.confirmed_object(object_id="block_b", mass_kg=0.05, position=(0.6, 0.0, 0.05))
        unstable = fixtures.unstable_object(object_id="block_c", position=(0.7, 0.0, 0.05))
        self.state = fixtures.base_world_state(objects=(heavy, light, unstable))
        self.gate = make_gate(self.state)

    def test_permit_marks_task_done(self):
        tasks = [Task("move_b", lambda s: fixtures.grasp_action(object_id="block_b", target_position=(0.6, 0.0, 0.05)))]
        outcomes = run_task_queue(tasks, self.state, self.gate)
        self.assertEqual(outcomes[0].status, "done")
        self.assertEqual(outcomes[0].decision.verdict, DecisionVerdict.PERMIT)

    def test_non_retryable_block_is_skipped_not_frozen(self):
        # This is the user's exact scenario: an over-budget object should not just sit there
        # unresolved forever -- it should come back as a clean "skipped", with the real reason.
        tasks = [Task("move_a", lambda s: fixtures.grasp_action(object_id="block_a", target_position=(0.5, 0.0, 0.05)))]
        outcomes = run_task_queue(tasks, self.state, self.gate)
        self.assertEqual(outcomes[0].status, "skipped")
        self.assertIn("mass_within_force_budget", outcomes[0].reason)

    def test_retryable_block_is_pending_not_skipped(self):
        tasks = [Task("move_c", lambda s: fixtures.grasp_action(object_id="block_c", target_position=(0.7, 0.0, 0.05)))]
        outcomes = run_task_queue(tasks, self.state, self.gate)
        self.assertEqual(outcomes[0].status, "pending")

    def test_independent_task_proceeds_after_an_earlier_task_is_skipped(self):
        # The core ask: a permanently-blocked task must not stop unrelated work from proceeding.
        tasks = [
            Task("move_a", lambda s: fixtures.grasp_action(object_id="block_a", target_position=(0.5, 0.0, 0.05))),
            Task("move_b", lambda s: fixtures.grasp_action(object_id="block_b", target_position=(0.6, 0.0, 0.05))),
        ]
        outcomes = run_task_queue(tasks, self.state, self.gate)
        self.assertEqual(outcomes[0].status, "skipped")
        self.assertEqual(outcomes[1].status, "done")

    def test_dependent_task_is_skipped_transitively_without_being_proposed(self):
        calls = []

        def propose_dependent(s):
            calls.append("move_a_again")
            return fixtures.grasp_action(object_id="block_a", target_position=(0.5, 0.0, 0.05))

        tasks = [
            Task("move_a", lambda s: fixtures.grasp_action(object_id="block_a", target_position=(0.5, 0.0, 0.05))),
            Task("stack_on_a", propose_dependent, depends_on=("move_a",)),
        ]
        outcomes = run_task_queue(tasks, self.state, self.gate)
        self.assertEqual(outcomes[0].status, "skipped")
        self.assertEqual(outcomes[1].status, "skipped")
        self.assertIn("prerequisite 'move_a' was skipped", outcomes[1].reason)
        self.assertEqual(calls, [])  # never proposed, let alone gated -- not even attempted

    def test_independent_task_is_unaffected_by_a_skipped_dependency_elsewhere(self):
        tasks = [
            Task("move_a", lambda s: fixtures.grasp_action(object_id="block_a", target_position=(0.5, 0.0, 0.05))),
            Task("stack_on_a", lambda s: fixtures.grasp_action(object_id="block_a", target_position=(0.5, 0.0, 0.05)), depends_on=("move_a",)),
            Task("move_b", lambda s: fixtures.grasp_action(object_id="block_b", target_position=(0.6, 0.0, 0.05))),
        ]
        outcomes = run_task_queue(tasks, self.state, self.gate)
        self.assertEqual([o.status for o in outcomes], ["skipped", "skipped", "done"])


if __name__ == "__main__":
    unittest.main()
