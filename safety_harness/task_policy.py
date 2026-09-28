"""Task-level policy: what to do when ``gate()`` returns BLOCK.

The harness itself is a per-decision gate (design doc: "Design Principle: Permission, Not
Detection") -- it has no opinion on what a caller does with a BLOCK, and no memory between calls.
This module is an optional, overridable reference for one common caller-side policy question:
should this specific task be retried, or abandoned in favor of independent work?

That question has a real wrong answer in both directions:

* Retrying a block that will never clear on its own -- an object's mass exceeds the robot's rated
  payload, a commanded speed exceeds the actuator's own rating, a config's digest doesn't match its
  pin -- wastes time at best. Nothing in the world is going to change to make it pass, so the robot
  sits there, and every *other* task that doesn't depend on this one sits there with it, for no
  safety reason at all.
* Treating every block as permanent and giving up too eagerly is its own failure mode: a person who
  is momentarily in the swept path, a sensor reading that is one frame stale, an object whose
  current pose perception hasn't confirmed yet -- these are exactly the transient conditions the
  gate exists to catch instant-by-instant, and they routinely clear within the next perception
  cycle. Abandoning the task over one of these throws away work that a moment's wait would have
  let through.

The design doc's "Fallback Control and Recovery" already establishes that freezing in place is a
*safety* response to an in-flight motion that must stop right now -- it is not, by itself, a
task-completion policy, and it says nothing about what to do next. A precondition failure caught
before motion even starts needs no freeze at all: the robot was never moving into that action. What
it needs is a decision about what to attempt next, which is what ``run_task_queue`` below makes:
"proceed with whatever the next independent task is, skipping only the tasks that actually depend
on the one that failed."

The classification below is a default, not a law: whether a given check's failure is retryable is
domain knowledge this module cannot have from a check's name alone. A ``fall_consequence_acceptable``
failure on an object that just needs a clearer camera angle *is* retryable; on one that's a
genuinely sealed hazardous container, it never will be -- by name alone this module cannot tell
those two apart, so it defaults to the conservative side (see below). Override ``RETRYABLE_CHECKS``
for your own deployment if you have that domain knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .engine import ActuatorGate
from .schema import Action, Decision, DecisionVerdict, WorldState

# Checks whose failure reason is about environment or perception state that can plausibly resolve
# on its own before the next attempt: a person moves, a sensor reading refreshes, an object's pose
# gets confirmed, a motor cools, a deadline jitter clears. Retrying the SAME action again shortly is
# a reasonable default for all of these.
#
# Deliberately NOT here (and so treated as non-retryable by default -- see is_retryable): anything
# that is a static fact about the object, the robot's own rated limits, or its configuration, which
# retrying the identical action cannot change without an external correction (re-weighing the
# object, fixing the config, commanding a slower motion instead of the same one again):
# mass_within_force_budget, payload_and_grip_force_within_limits, joint_position_limits_respected,
# joint_velocity_within_limits, joint_effort_within_limits, cartesian_speed_within_limits,
# self_collision_clear, config_integrity_verified, fall_consequence_acceptable (conservative choice,
# see the module docstring).
#
# Unlisted check names -- including any check this module doesn't know about yet -- default to
# non-retryable too, consistent with this project's default-deny stance everywhere else: an
# unrecognized block is treated as "stop and let a human decide", not "keep trying forever".
RETRYABLE_CHECKS = frozenset({
    "object_hazard_confirmed",
    "swept_path_clear_of_agents",
    "stability_margin_maintained",
    "balance_margin_maintained",
    "visibility_above_threshold",
    "surface_confirmed_stable",
    "object_pose_confirmed",
    "robot_state_confirmed",
    "object_cleared_for_interaction",
    "swept_path_clear_of_risky_objects",
    "current_position_confirmed_stable",
    "destination_confirmed_stable_and_clear",
    "motor_temperature_within_limits",
    "battery_charge_sufficient",
    "reduced_speed_near_human",
    "environment_hazard_clear",
    "decision_within_deadline",
    "sensor_data_fresh",
    "swept_path_observed",
    "iso15066_separation_distance_maintained",
    "iso15066_power_force_limiting",
    "vulnerable_bystander_protected",
    "command_integrity_verified",  # the tampered DECISION is dead, but a fresh gate() call isn't
})


def is_retryable(decision: Decision, *, retryable_checks: frozenset = RETRYABLE_CHECKS) -> bool:
    """True if every failing check on this BLOCK decision is one worth retrying later. A PERMIT is
    trivially "retryable" (there is nothing to retry). A single non-retryable failing check makes
    the whole decision non-retryable: clearing the retryable ones too wouldn't be enough anyway.
    """
    if decision.verdict == DecisionVerdict.PERMIT:
        return True
    failing = [r.name for r in decision.precondition_results if not r.satisfied]
    return bool(failing) and all(name in retryable_checks for name in failing)


@dataclass(frozen=True)
class Task:
    """One candidate unit of work for ``run_task_queue``.

    ``propose`` builds the ``Action`` to gate, given the current ``WorldState`` -- called fresh each
    attempt, so it can reflect updated perception rather than a stale target computed earlier.

    ``depends_on`` names other ``Task.task_id`` values that must have been PERMITted before this one
    is even attempted. If a dependency was skipped, this task is skipped too, transitively, without
    ever calling ``propose`` or ``gate`` on it -- this is the "as long as moving the block is not a
    prerequisite" half of the policy: independent tasks proceed, dependent ones don't pretend to.
    """

    task_id: str
    propose: Callable[[WorldState], Action]
    depends_on: tuple = ()


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    status: str  # "done" (PERMIT), "skipped" (non-retryable block, or a skipped dependency), "pending" (retryable block)
    decision: Optional[Decision] = None
    reason: str = ""


def run_task_queue(
    tasks,
    state: WorldState,
    gate: ActuatorGate,
    *,
    retryable_checks: frozenset = RETRYABLE_CHECKS,
) -> list:
    """Gate each task in ``tasks``, in order, against the same ``state``.

    * PERMIT -> "done".
    * A retryable BLOCK -> "pending": worth trying again later (this function does not loop, sleep,
      or retry by itself -- the caller decides when "later" is, typically the next perception cycle).
    * A non-retryable BLOCK -> "skipped", logged with which check(s) made it so, and every task that
      (transitively) depends on it is also skipped without ever being proposed or gated -- exactly
      "raise an error and proceed with whatever the next task is, as long as moving the block is not
      a prerequisite."

    This does not execute anything: actuation stays the caller's responsibility everywhere else in
    this project, including here. It only decides which tasks to attempt and which to skip, and
    returns one ``TaskOutcome`` per task, in the order given.
    """
    outcomes = []
    skip_reason = {}
    for task in tasks:
        blocked_dep = next((d for d in task.depends_on if d in skip_reason), None)
        if blocked_dep is not None:
            reason = f"prerequisite {blocked_dep!r} was skipped: {skip_reason[blocked_dep]}"
            skip_reason[task.task_id] = reason
            outcomes.append(TaskOutcome(task.task_id, "skipped", None, reason))
            continue

        action = task.propose(state)
        decision = gate.gate(action)

        if decision.verdict == DecisionVerdict.PERMIT:
            outcomes.append(TaskOutcome(task.task_id, "done", decision, "permitted"))
        elif is_retryable(decision, retryable_checks=retryable_checks):
            outcomes.append(TaskOutcome(task.task_id, "pending", decision, "blocked, retryable"))
        else:
            failing = ", ".join(r.name for r in decision.precondition_results if not r.satisfied)
            reason = f"non-retryable block: {failing}"
            skip_reason[task.task_id] = reason
            outcomes.append(TaskOutcome(task.task_id, "skipped", decision, reason))
    return outcomes
