"""A too-heavy object should not freeze the whole robot forever.

Pure Python -- no Isaac Lab, no GPU, no robot. Run it directly:

    python3 examples/task_reroute/reroute_example.py

The scene: two blocks on a table. `block_a` is 5kg -- over the 3kg force budget in the action
schema below. `block_b` is a normal, liftable 0.4kg block. There's also a `stack_on_a` task that
can only be attempted once `move_a` has actually happened.

Every other example/demo script in this repo (closed_loop_gate.py, gate_policy_g1_stack.py,
render_anymal_demo.py) responds to a BLOCK by freezing the arm in place -- a deliberate
simplification for a legible demo clip, not a claim about how a production robot should behave.
Taken literally, "freeze on any block" means an object that will never get lighter stops the robot
from doing anything else, forever, even work that has nothing to do with that object. This example
is the reference counter-pattern: `safety_harness.task_policy.run_task_queue` classifies WHY a task
was blocked and, for a block that retrying the identical action cannot fix (block_a's mass is a
static fact -- it does not change by proposing the same grasp again), skips just that task and its
dependents, and proceeds with whatever independent task is next. See task_policy.py's module
docstring for the full reasoning and its RETRYABLE_CHECKS default classification.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safety_harness import (  # noqa: E402
    ActionSchemaRegistry,
    ActuatorGate,
    Action,
    DecisionVerdict,
    EnvironmentSignals,
    FallConsequence,
    HazardTag,
    Pose,
    PredictedTrajectory,
    RobotProprioception,
    Task,
    TrackedObject,
    TrajectoryPoint,
    WorldState,
    run_task_queue,
)
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter

ACTION_SCHEMA = ActionSchemaRegistry.from_dict({
    "action_types": {
        "grasp": {
            "checks": [
                {"name": "robot_state_confirmed", "kwargs": {}},
                {"name": "object_cleared_for_interaction", "kwargs": {}},
                {"name": "current_position_confirmed_stable", "kwargs": {}},
                {"name": "mass_within_force_budget", "kwargs": {"force_budget_kg": 3.0}},
            ]
        },
    }
})


def robot_state() -> RobotProprioception:
    return RobotProprioception(
        joint_positions=(0.0,) * 7,
        joint_velocities=(0.0,) * 7,
        end_effector_pose=Pose(position=(0.4, 0.0, 0.3)),
        gripper_state=1.0,
    )


def block(object_id: str, mass_kg: float, position: tuple) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        object_class="cube",
        pose=Pose(position=position),
        estimated_mass_kg=mass_kg,
        hazard_tags=frozenset({HazardTag.FRAGILE}),
        pose_confidence=1.0,
        class_confidence=1.0,
        cleared_for_interaction=True,
        supported_stably=True,
        fall_consequence=FallConsequence.NONE,
    )


def world_state() -> WorldState:
    return WorldState(
        objects=(
            block("block_a", mass_kg=5.0, position=(0.5, 0.0, 0.05)),  # over budget -- non-retryable
            block("block_b", mass_kg=0.4, position=(0.6, 0.0, 0.05)),  # ordinary, liftable
        ),
        agents=(),
        robot=robot_state(),
        environment=EnvironmentSignals(visibility_confidence=1.0),
        sensor_timestamp=time.time(),
    )


class _Perception(PerceptionAdapter):
    def __init__(self, state):
        self._state = state

    def get_world_state(self):
        return self._state


class _Dynamics(DynamicsAdapter):
    def predict_trajectory(self, state, action, horizon_s):
        p = state.robot.end_effector_pose.position
        return PredictedTrajectory(
            points=(TrajectoryPoint(t=0.0, robot=state.robot, swept_volume_center=p, swept_volume_radius_m=0.08),),
            horizon_s=horizon_s,
        )


def grasp(object_id: str, position: tuple) -> Action:
    return Action("grasp", {"object_id": object_id, "target_position": position})


def main():
    state = world_state()
    gate = ActuatorGate(
        perception=_Perception(state),
        dynamics=_Dynamics(),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=ACTION_SCHEMA,
    )

    tasks = [
        Task("move_a", lambda s: grasp("block_a", (0.5, 0.0, 0.05))),
        Task("stack_on_a", lambda s: grasp("block_a", (0.5, 0.0, 0.05)), depends_on=("move_a",)),
        Task("move_b", lambda s: grasp("block_b", (0.6, 0.0, 0.05))),
    ]

    print("Tasks: move_a (5kg, over the 3kg budget), stack_on_a (needs move_a done first), move_b (0.4kg, fine)\n")
    outcomes = run_task_queue(tasks, state, gate)
    for o in outcomes:
        verdict = o.decision.verdict.value if o.decision is not None else "-"
        print(f"  {o.task_id:12s} -> {o.status:8s} (gate verdict: {verdict:7s}) {o.reason}")

    done = [o.task_id for o in outcomes if o.status == "done"]
    skipped = [o.task_id for o in outcomes if o.status == "skipped"]
    print(f"\ndone: {done}")
    print(f"skipped: {skipped}")
    assert "move_b" in done, "an independent task must still get done"
    assert "move_a" in skipped and "stack_on_a" in skipped, "the over-budget task and its dependent must be skipped, not frozen"
    print("\nNo freeze. No infinite retry on block_a. move_b happened anyway.")


if __name__ == "__main__":
    main()
