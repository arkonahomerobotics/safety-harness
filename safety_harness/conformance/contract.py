"""The same adapter-interface-generic properties ``tests/test_blackbox.py`` checks, run against the
adapter's own live ``get_world_state()``/``predict_trajectory()`` instead of a synthetic stand-in.

``test_blackbox.py`` can afford to fabricate a brand-new ``WorldState`` every iteration because it
is testing the engine, not any particular adapter. Here the adapter *is* the thing under test, so
every iteration calls the real adapter -- the only control exercised is which ``Action`` gets
proposed to it.
"""

from __future__ import annotations

import random
import time

from ..action_schema import ActionSchemaRegistry
from ..adapters.base import DynamicsAdapter, FallbackController, Logger, PerceptionAdapter
from ..adapters.simple import FreezeInPlaceFallback, InMemoryLogger
from ..engine import ActuatorGate, PerceptionFailure
from ..schema import Action, DecisionVerdict
from .report import Violation


def _random_params(rng, object_ids):
    params = {}
    if rng.random() < 0.8 and (object_ids or rng.random() < 0.3):
        params["object_id"] = rng.choice(object_ids) if (object_ids and rng.random() < 0.7) else f"nonexistent_{rng.randint(0, 10**6)}"
    if rng.random() < 0.7:
        params["target_position"] = tuple(rng.uniform(-2, 2) for _ in range(3))
    if rng.random() < 0.3:
        params["target_surface_id"] = rng.choice(object_ids) if object_ids else f"nonexistent_surface_{rng.randint(0, 10**6)}"
    if rng.random() < 0.3:
        params["grip_force_n"] = rng.choice([0.0, 5.0, 60.0, -1.0, 1e6])
    return params


def _random_action(rng, registered_types, object_ids):
    if registered_types and rng.random() < 0.7:
        action_type = rng.choice(registered_types)
    else:
        action_type = f"never_registered_action_type_{rng.randint(0, 10**9)}"
    return Action(action_type=action_type, params=_random_params(rng, object_ids))


def run_contract_fuzz(
    perception: PerceptionAdapter,
    dynamics: DynamicsAdapter,
    action_schema: ActionSchemaRegistry,
    *,
    fallback: FallbackController = None,
    logger: Logger = None,
    n_iterations: int = 200,
    seed: int = 1,
    max_decision_s: float = 1.0,
) -> tuple:
    """Returns ``(violations, histogram)``. ``histogram`` maps action_type -> verdict -> count, for
    the report's ``detail``."""
    rng = random.Random(seed)
    fallback = fallback or FreezeInPlaceFallback()
    logger = logger or InMemoryLogger()
    gate = ActuatorGate(perception=perception, dynamics=dynamics, fallback=fallback, logger=logger, action_schema=action_schema)

    registered_types = sorted(action_schema.effective_config().get("action_types", {}))
    try:
        seed_state = perception.get_world_state()
        object_ids = [o.object_id for o in seed_state.objects]
    except Exception:  # noqa: BLE001 -- a broken get_world_state() is caught properly below instead
        object_ids = []

    violations: list = []
    histogram: dict = {}
    for i in range(n_iterations):
        action = _random_action(rng, registered_types, object_ids)
        t0 = time.perf_counter()
        try:
            decision = gate.gate(action)
        except PerceptionFailure:
            continue  # documented, expected exception type -- not a violation
        except Exception as exc:  # noqa: BLE001
            violations.append(Violation(
                "undocumented_exception",
                f"gate() raised {type(exc).__name__} for action {action.action_type!r}: {exc}",
                {"iteration": i, "action_type": action.action_type, "exception_type": type(exc).__name__},
            ))
            continue
        dt = time.perf_counter() - t0
        if dt > max_decision_s:
            violations.append(Violation(
                "slow_decision", f"gate() took {dt:.3f}s (> {max_decision_s}s) for action {action.action_type!r}",
                {"iteration": i, "action_type": action.action_type, "seconds": dt},
            ))

        histogram.setdefault(action.action_type, {}).setdefault(decision.verdict.value, 0)
        histogram[action.action_type][decision.verdict.value] += 1

        if action.action_type not in registered_types and decision.verdict == DecisionVerdict.PERMIT:
            violations.append(Violation(
                "unregistered_action_permitted",
                f"action type {action.action_type!r}, which the schema never registered, was PERMITted",
                {"iteration": i, "action_type": action.action_type},
            ))
        if decision.verdict == DecisionVerdict.PERMIT and decision.action != action:
            violations.append(Violation(
                "permit_mutated_action", f"PERMIT returned a different action than proposed for {action.action_type!r}",
                {"iteration": i, "proposed": repr(action), "returned": repr(decision.action)},
            ))
        if decision.verdict == DecisionVerdict.BLOCK and decision.action is None:
            violations.append(Violation(
                "block_without_fallback", f"BLOCK returned no fallback action at all for {action.action_type!r}",
                {"iteration": i, "action_type": action.action_type},
            ))

    return tuple(violations), histogram
