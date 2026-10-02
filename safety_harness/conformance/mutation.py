"""The same curated single-field-mutation battery ``tests/test_fuzz.py``'s
``test_all_single_field_mutations_block`` runs against a hand-built golden fixture, run instead
against one real snapshot captured from the adapter under test.

The third party supplies one ``Action`` they believe is currently safe (``baseline_action``) --
there is no way to invent a plausible golden ``WorldState`` for a robot this package has never seen,
so unlike ``tests/fixtures.py``'s hand-built ``confirmed_object()``, the "known-good" starting point
has to come from the adapter itself. The snapshot is held fixed for the whole battery (wrapped in
``_FrozenPerception`` below) so that each mutation is a controlled, single-variable experiment, the
same guarantee ``tests/fixtures.py``'s hand-built fixtures give for free.
"""

from __future__ import annotations

from dataclasses import replace

from ..action_schema import ActionSchemaRegistry
from ..adapters.base import DynamicsAdapter, FallbackController, Logger, PerceptionAdapter
from ..adapters.simple import FreezeInPlaceFallback, InMemoryLogger
from ..engine import ActuatorGate, PerceptionFailure
from ..schema import Action, DecisionVerdict, EnvironmentSignals, FallConsequence, HazardTag, WorldState
from .report import Violation


class _FrozenPerception(PerceptionAdapter):
    """Returns one fixed WorldState forever -- the live adapter's own output, snapshotted once so
    repeated gate() calls during the battery see exactly the one field that was deliberately
    changed, not whatever a live sensor or simulator did between calls."""

    def __init__(self, state: WorldState):
        self._state = state

    def get_world_state(self) -> WorldState:
        return self._state


def _object_mutations(obj):
    return {
        "class_confidence -> 0": replace(obj, class_confidence=0.0),
        "hazard_tags -> unknown": replace(obj, hazard_tags=frozenset({HazardTag.UNKNOWN})),
        "cleared_for_interaction -> False": replace(obj, cleared_for_interaction=False),
        "supported_stably -> False": replace(obj, supported_stably=False),
        "supported_stably -> None": replace(obj, supported_stably=None),
        "fall_consequence -> UNKNOWN": replace(obj, fall_consequence=FallConsequence.UNKNOWN),
        "fall_consequence -> HAZARDOUS_RELEASE, tolerance 0": replace(
            obj, fall_consequence=FallConsequence.HAZARDOUS_RELEASE, drop_tolerance_m=0.0
        ),
        "estimated_mass_kg -> None": replace(obj, estimated_mass_kg=None),
        "pose_confidence -> 0": replace(obj, pose_confidence=0.0),
    }


def run_mutation_battery(
    perception: PerceptionAdapter,
    dynamics: DynamicsAdapter,
    action_schema: ActionSchemaRegistry,
    baseline_action: Action,
    *,
    fallback: FallbackController = None,
    logger: Logger = None,
    horizon_s: float = 1.0,
) -> dict:
    """Returns a dict: ``baseline_permitted``, ``violations`` (tuple of Violation), ``skip_reason``
    (non-empty only when ``baseline_permitted`` is False), and ``detail``."""
    fallback = fallback or FreezeInPlaceFallback()
    logger = logger or InMemoryLogger()

    try:
        snapshot = perception.get_world_state()
    except Exception as exc:  # noqa: BLE001
        return {
            "baseline_permitted": False,
            "violations": (),
            "skip_reason": f"perception.get_world_state() raised {type(exc).__name__}: {exc}",
            "detail": {},
        }

    def _gate_for(state):
        return ActuatorGate(
            perception=_FrozenPerception(state), dynamics=dynamics, fallback=fallback, logger=logger,
            action_schema=action_schema, horizon_s=horizon_s,
        )

    try:
        baseline_decision = _gate_for(snapshot).gate(baseline_action)
    except PerceptionFailure as exc:
        return {
            "baseline_permitted": False, "violations": (), "detail": {},
            "skip_reason": f"baseline run raised PerceptionFailure: {exc}",
        }

    if baseline_decision.verdict != DecisionVerdict.PERMIT:
        fired = [r.name for r in baseline_decision.precondition_results if not r.satisfied]
        return {
            "baseline_permitted": False, "violations": (),
            "skip_reason": (
                f"baseline_action did not PERMIT against a live snapshot from this adapter, so there "
                f"is no known-good state to mutate from; checks that blocked it: {fired}"
            ),
            "detail": {"fired_checks": fired},
        }

    violations: list = []
    object_id = baseline_action.params.get("object_id")
    target = None
    if object_id is not None:
        target = next((o for o in snapshot.objects if o.object_id == object_id), None)
    if target is None and snapshot.objects:
        target = snapshot.objects[0]

    if target is not None:
        for label, mutated_obj in _object_mutations(target).items():
            mutated_objects = tuple(mutated_obj if o is target else o for o in snapshot.objects)
            mutated_state = replace(snapshot, objects=mutated_objects)
            try:
                decision = _gate_for(mutated_state).gate(baseline_action)
            except PerceptionFailure:
                continue
            if decision.verdict != DecisionVerdict.BLOCK:
                violations.append(Violation(
                    "mutation_not_blocked", f"mutating the target object ({label}) still PERMITted",
                    {"mutation": label},
                ))

    mutated_env_state = replace(snapshot, environment=EnvironmentSignals(visibility_confidence=0.0))
    try:
        env_decision = _gate_for(mutated_env_state).gate(baseline_action)
        if env_decision.verdict != DecisionVerdict.BLOCK:
            violations.append(Violation(
                "mutation_not_blocked", "zeroing environment.visibility_confidence still PERMITted", {},
            ))
    except PerceptionFailure:
        pass

    return {
        "baseline_permitted": True, "violations": tuple(violations), "skip_reason": "",
        "detail": {"target_object_id": getattr(target, "object_id", None)},
    }
