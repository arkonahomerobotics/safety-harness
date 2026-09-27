"""The decision engine every robot integration shares unmodified.

See the design doc's "Design Principle: Permission, Not Detection" and "Architecture Overview".
Robot-specific behavior lives entirely in the four adapters passed to ``ActuatorGate``; this class
is the one piece of the module a robot team configures (via the action schema) but never forks.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .action_schema import ActionSchemaRegistry
from .adapters.base import DynamicsAdapter, FallbackController, Logger, PerceptionAdapter
from .integrity import try_action_digest
from .preconditions import CheckContext
from .schema import Action, Decision, DecisionVerdict, PreconditionResult


class PerceptionFailure(Exception):
    """Perception itself raised, so there is no WorldState to reason about at all -- not even
    enough to command a confident software fallback (a "freeze in place" needs to know the current
    joint positions; there is nothing to freeze *at* without them). This is deliberately not turned
    into a normal BLOCK decision: it must be routed to the hardware safety layer (E-stop, watchdog
    timer) discussed in the design doc's Prior Art and Open Questions, not handled in software. A
    robot's integration should catch this distinctly from ActuatorGate.gate()'s normal return and
    trigger that hardware path directly.
    """


class ActuatorGate:
    def __init__(
        self,
        perception: PerceptionAdapter,
        dynamics: DynamicsAdapter,
        fallback: FallbackController,
        logger: Logger,
        action_schema: ActionSchemaRegistry,
        horizon_s: float = 1.0,
        watchdog=None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """``watchdog`` (optional, a watchdog.DecisionWatchdog) is fed every decision and revoked
        on PerceptionFailure; the actuator side then executes only what ``watchdog.command()``
        returns. ``clock`` is the monotonic clock decisions are timed on -- the watchdog's own clock
        when one is given (they must agree), time.monotonic otherwise."""
        self._perception = perception
        self._dynamics = dynamics
        self._fallback = fallback
        self._logger = logger
        self._schema = action_schema
        self._horizon_s = horizon_s
        self._watchdog = watchdog
        if clock is None:
            clock = watchdog.now if watchdog is not None else time.monotonic
        self._clock = clock

    def gate(self, proposed_action: Action) -> Decision:
        """The one call a robot's control loop inserts before any action reaches actuators.

        Raises ``PerceptionFailure`` if perception itself is broken -- see that class's docstring
        for why this is not just another BLOCK decision. Any other adapter failure (dynamics,
        precondition checks) happens with valid state in hand, so it degrades to a normal, safely
        logged BLOCK instead: default-deny extends to "the harness couldn't evaluate this," not only
        to "the harness evaluated this and it failed."

        Before anything else runs, gate() stamps the decision's start time and hashes the proposed
        action. Both go to the checks through a CheckContext (decision_within_deadline,
        command_integrity_verified), and the hash is bound into the returned Decision as
        ``action_digest``, so the executor can prove the action it's about to send is the one that
        was checked (integrity.verify_decision_action).
        """
        started_at = self._clock()
        checked_digest = try_action_digest(proposed_action)
        try:
            state = self._perception.get_world_state()
        except Exception as exc:
            if self._watchdog is not None:
                self._watchdog.revoke(f"perception failure: {exc}")
            raise PerceptionFailure(str(exc)) from exc

        try:
            trajectory = self._dynamics.predict_trajectory(state, proposed_action, self._horizon_s)
            if not trajectory.points:
                # Found by fuzzing, not by design: every swept-path check loops over
                # trajectory.points, and a loop over zero points returns "no violation found"
                # rather than "no basis for a decision" -- an empty prediction was silently
                # satisfying every check that depends on one. Treat "no predicted points" as its
                # own explicit failure instead of leaving it to fall out of an empty loop.
                results = (PreconditionResult(
                    name="trajectory_has_points", satisfied=False,
                    reason="dynamics adapter returned an empty trajectory",
                ),)
            else:
                context = CheckContext(
                    decision_started_at=started_at, clock=self._clock,
                    checked_action_digest=checked_digest, schema_registry=self._schema,
                )
                results = self._schema.run_checks(proposed_action, state, trajectory, context=context)
        except Exception as exc:
            results = (PreconditionResult(name="adapter_error", satisfied=False, reason=str(exc)),)

        if results and all(r.satisfied for r in results):
            decision = Decision(
                verdict=DecisionVerdict.PERMIT,
                action=proposed_action,
                precondition_results=results,
                triggered_fallback=False,
                # The digest from BEFORE any check ran, not a fresh one: if the action was rewritten
                # during checking, execution-time verification must fail, not bless the rewrite.
                action_digest=checked_digest,
            )
        else:
            fallback_action = self._fallback.execute(state)
            decision = Decision(
                verdict=DecisionVerdict.BLOCK,
                action=fallback_action,
                precondition_results=results,
                triggered_fallback=True,
                action_digest=try_action_digest(fallback_action),
            )

        self._logger.record(decision, state)
        if self._watchdog is not None:
            self._watchdog.feed(decision, state, decided_at=started_at)
        return decision
