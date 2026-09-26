"""The decision engine every robot integration shares unmodified.

See the design doc's "Design Principle: Permission, Not Detection" and "Architecture Overview".
Robot-specific behavior lives entirely in the four adapters passed to ``ActuatorGate``; this class
is the one piece of the module a robot team configures (via the action schema) but never forks.
"""

from __future__ import annotations

from .action_schema import ActionSchemaRegistry
from .adapters.base import DynamicsAdapter, FallbackController, Logger, PerceptionAdapter
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
    ):
        self._perception = perception
        self._dynamics = dynamics
        self._fallback = fallback
        self._logger = logger
        self._schema = action_schema
        self._horizon_s = horizon_s

    def gate(self, proposed_action: Action) -> Decision:
        """The one call a robot's control loop inserts before any action reaches actuators.

        Raises ``PerceptionFailure`` if perception itself is broken -- see that class's docstring
        for why this is not just another BLOCK decision. Any other adapter failure (dynamics,
        precondition checks) happens with valid state in hand, so it degrades to a normal, safely
        logged BLOCK instead: default-deny extends to "the harness couldn't evaluate this," not only
        to "the harness evaluated this and it failed."
        """
        try:
            state = self._perception.get_world_state()
        except Exception as exc:
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
                results = self._schema.run_checks(proposed_action, state, trajectory)
        except Exception as exc:
            results = (PreconditionResult(name="adapter_error", satisfied=False, reason=str(exc)),)

        if results and all(r.satisfied for r in results):
            decision = Decision(
                verdict=DecisionVerdict.PERMIT,
                action=proposed_action,
                precondition_results=results,
                triggered_fallback=False,
            )
        else:
            fallback_action = self._fallback.execute(state)
            decision = Decision(
                verdict=DecisionVerdict.BLOCK,
                action=fallback_action,
                precondition_results=results,
                triggered_fallback=True,
            )

        self._logger.record(decision, state)
        return decision
