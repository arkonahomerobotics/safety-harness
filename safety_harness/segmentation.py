"""Segmenting a continuous, grip-based policy's control stream into the discrete Action proposals
`ActuatorGate` expects.

**The problem this exists to fix.** A discrete-decision gate with object-stability checks wired to
`grasp` cannot be driven by proposing "grasp" on every control step of a continuous gripper-closing
motion. The moment the fingers make contact, the object starts moving IN THE HAND -- the same motion
the policy is trying to produce -- and a check like `current_position_confirmed_stable` correctly
reads that as "not confirmed stable" and BLOCKs. The policy retries "grasp" next step; contact is
still happening; it BLOCKs again. The policy can never finish closing its grip, because closing its
grip is what trips the block. Found live on a Unitree G1 humanoid stacking task: gating "grasp" on
every step while the fingers closed produced 3,838 blocks in one run, and the grasp never completed.

**The fix, already validated live.** The Franka closed-loop measurement (`examples/closed_loop`,
95.3% task success gated vs. 96.1% ungated, on 128 real episodes) proposes "grasp" or "place" only
on the single control step where the commanded grip transitions -- open to closing, or holding to
opening -- and proposes a plain "reach" for every other step, including every step still in the
middle of physically closing or opening the gripper. "reach" works here specifically because its own
checklist (see `configs/example_action_schema.yaml`) never asks whether an object is stable, only
whether the swept path is safe -- so the check that deadlocked "grasp" simply never applies to the
steps after the single edge-triggered "grasp" decision runs.

`GripActionSegmenter` is that same edge-triggering logic, extracted once into this package instead
of being reimplemented -- and, on a first attempt, gotten wrong -- by every new integration.
Deliberately NOT a new "carry" action type distinct from "reach": the validated Franka result shows
plain "reach" already covers safe in-hand transport (human-proximity, swept-path and command/config
integrity checks are wired to `reach` already; only object-self-stability checks are correctly
absent from it). If a deployment later wants object-specific in-transit checks, that is a genuinely
separate, smaller follow-up -- not required to fix the deadlock this module targets.

This is a thin, stateful helper around ONE decision -- which `Action` to propose this step -- for
ONE manipulator. It does not call `ActuatorGate.gate()` itself and does not replace it; you still
gate whatever it proposes and act on the `Decision`, exactly as before. Scalar by design, matching
every adapter interface method and `ActuatorGate.gate()` itself (one call per env, per manipulator);
existing scripts that gate many parallel envs already do so by wrapping N calls in a Python loop,
and one `GripActionSegmenter` instance per env/manipulator follows that same, already-established
pattern rather than inventing a vectorized API this package doesn't otherwise have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .schema import Action

DEFAULT_CLOSE_THRESHOLD = 0.0  # matches the reference scripts' [-1, 1] grip convention: <0 = closing/closed


@dataclass
class GripActionSegmenter:
    """One instance per manipulator (one gripper/hand) being gated. Call :meth:`propose` once per
    control step, in order; it returns the ``Action`` to pass to ``gate.gate()`` for that step.

    Args (constructor):
        is_closing: classifies a raw grip command as "commanding closed" or "commanding open".
            Defaults to ``command < 0`` (the reference scripts' -1..1 convention, closed negative).
            Override for a different convention -- e.g. a 0..1 "closed fraction" would use
            ``lambda g: g > 0.5``.

    Args (:meth:`propose`, every call):
        grip_command: this step's raw commanded grip value, in whatever units ``is_closing``
            expects.
        object_id: the object currently being reached for or held. May change between calls (a
            multi-object task, e.g. pick-then-place-then-pick-again) -- the segmenter does not
            track object identity itself, only the grip command's edges.
        target_position: this step's reach/grasp/place target, forwarded into the proposed
            ``Action``'s ``target_position`` param unchanged.
        is_holding: the CALLER's own ground-truth signal for "is the object actually held right
            now" (e.g. measured contact/force, or distance-to-object plus finger closure) -- NOT
            derived from ``grip_command`` alone, and NOT anything this class infers. This is what
            lets a "place" edge fire only once something was actually grasped, and what stops a
            "grasp" edge from re-firing while genuinely already holding (a defensive guard beyond
            what the validated Franka reference needed for its own well-behaved phase machine --
            included here because a general-purpose library should not assume a caller's policy is
            equally well-behaved).
        grasp_params / place_params: extra ``Action.params`` merged into a "grasp"/"place"
            proposal only (e.g. ``{"grip_force_n": 5.0}``, ``{"target_surface_id": "cube_1"}``).
        reach_action_type: the action type proposed for every non-edge step. "reach" by default;
            override only if your own action schema uses a different name for the same semantics.

    Returns: an ``Action`` -- never ``None``. There is always something to propose, even on the
    very first call (a fresh instance's internal state starts as "not closing", so a first-step
    grip command that already commands closed is read as the closing edge, not missed).
    """

    is_closing: Callable[[float], bool] = field(default=lambda g: g < DEFAULT_CLOSE_THRESHOLD)
    _prev_closing: Optional[bool] = field(default=None, init=False, repr=False)

    def propose(
        self,
        grip_command: float,
        object_id: str,
        target_position,
        *,
        is_holding: bool,
        grasp_params: Optional[dict] = None,
        place_params: Optional[dict] = None,
        reach_action_type: str = "reach",
    ) -> Action:
        closing_now = bool(self.is_closing(grip_command))
        # A closing edge is: was not commanding closed last step (or this is the first call), IS
        # commanding closed now, and the caller doesn't already consider the object held -- that
        # last condition is the defensive guard described above; it can only SUPPRESS an
        # unnecessary re-proposal of "grasp" while already holding, never reintroduce the deadlock
        # this class exists to prevent, since it adds no dependency on any object-stability check.
        closing_edge = closing_now and self._prev_closing is not True and not is_holding
        opening_edge = (not closing_now) and self._prev_closing is True and is_holding
        self._prev_closing = closing_now

        if closing_edge:
            return Action("grasp", {"object_id": object_id, "target_position": target_position, **(grasp_params or {})})
        if opening_edge:
            return Action("place", {"object_id": object_id, "target_position": target_position, **(place_params or {})})
        return Action(reach_action_type, {"object_id": object_id, "target_position": target_position})

    def reset(self) -> None:
        """Forget the last observed grip command -- call between episodes so a fresh episode's
        first step is evaluated as if nothing had happened before it, not as a continuation of
        whatever the previous episode's grip command last was."""
        self._prev_closing = None
