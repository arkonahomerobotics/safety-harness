"""Does the adapter's own output actually satisfy the ``schema.py`` contract?

Generic over the schema the same way ``tests/test_blackbox.py``'s generator is: dispatch is on the
*shape* of each field's type hint (Optional, tuple, frozenset, Enum, nested dataclass), never on a
field name the author happened to pick -- except for the finiteness and confidence-range rules
below, which apply to every float found anywhere in the tree, by the same reasoning the design doc's
"NaN-Sensor Stress Test" section gives: a NaN defeats a ``<``/``>`` default-deny comparison silently,
with no Python exception to catch it, so a type-correct-looking value can still be a live hazard.

One deliberate exception to "never dispatch on a field name": ``WorldState.objects``/``.agents`` and
``PredictedTrajectory.points`` are declared as a bare ``tuple`` in ``schema.py`` (not
``tuple[TrackedObject, ...]``), so reflection alone can't recover what belongs inside them. Those
three containers are validated by their known element type explicitly; everything else stays generic.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import typing

from ..schema import (
    KnownSolidRegion,
    ObservedRegion,
    PredictedTrajectory,
    TrackedAgent,
    TrackedObject,
    TrajectoryPoint,
    WorldState,
)
from .report import Violation

_KNOWN_LOOSE_CONTAINERS = {
    # (dataclass type, field name) -> element type, for fields schema.py declares as bare `tuple`.
    (WorldState, "objects"): TrackedObject,
    (WorldState, "agents"): TrackedAgent,
    (WorldState, "observed_regions"): ObservedRegion,
    (WorldState, "solid_regions"): KnownSolidRegion,
    (PredictedTrajectory, "points"): TrajectoryPoint,
}


def _finite_and_range_check(value, path: str, violations: list) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(value):
        violations.append(Violation(
            "structural", f"{path} is not a finite number: {value!r}", {"path": path, "value": repr(value)},
        ))
        return
    if path.endswith("_confidence") and not (0.0 <= value <= 1.0):
        violations.append(Violation(
            "structural", f"{path} = {value!r} is outside the documented [0, 1] confidence range",
            {"path": path, "value": repr(value)},
        ))


def _validate_value(value, hint, path: str, violations: list) -> None:
    if hint is not None:
        origin = typing.get_origin(hint)

        if origin is typing.Union:  # Optional[X]
            args = [a for a in typing.get_args(hint) if a is not type(None)]
            if value is None:
                return
            _validate_value(value, args[0] if args else None, path, violations)
            return

        if hint in (float, int):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                violations.append(Violation(
                    "structural", f"{path} should be a {hint.__name__} but is {type(value).__name__}",
                    {"path": path, "expected": hint.__name__, "actual": type(value).__name__},
                ))
                return
            _finite_and_range_check(value, path, violations)
            return

        if hint is bool:
            if not isinstance(value, bool):
                violations.append(Violation(
                    "structural", f"{path} should be bool but is {type(value).__name__}",
                    {"path": path, "expected": "bool", "actual": type(value).__name__},
                ))
            return

        if hint is str:
            if not isinstance(value, str):
                violations.append(Violation(
                    "structural", f"{path} should be str but is {type(value).__name__}",
                    {"path": path, "expected": "str", "actual": type(value).__name__},
                ))
            return

        if hint is dict:
            if not isinstance(value, dict):
                violations.append(Violation(
                    "structural", f"{path} should be a dict but is {type(value).__name__}",
                    {"path": path, "expected": "dict", "actual": type(value).__name__},
                ))
            return

        if origin is tuple or hint is tuple:
            if not isinstance(value, tuple):
                violations.append(Violation(
                    "structural", f"{path} should be a tuple but is {type(value).__name__}",
                    {"path": path, "expected": "tuple", "actual": type(value).__name__},
                ))
                return
            args = typing.get_args(hint)
            if args and len(args) == 2 and args[1] is Ellipsis:
                for i, item in enumerate(value):
                    _validate_value(item, args[0], f"{path}[{i}]", violations)
            elif args and Ellipsis not in args:
                if len(value) != len(args):
                    violations.append(Violation(
                        "structural", f"{path} should have {len(args)} element(s), has {len(value)}",
                        {"path": path, "expected_length": len(args), "actual_length": len(value)},
                    ))
                else:
                    for i, (item, a) in enumerate(zip(value, args)):
                        _validate_value(item, a, f"{path}[{i}]", violations)
            # bare `tuple` (no args): nothing more specific declared, nothing more to check here.
            return

        if origin is frozenset or hint is frozenset:
            if not isinstance(value, frozenset):
                violations.append(Violation(
                    "structural", f"{path} should be a frozenset but is {type(value).__name__}",
                    {"path": path, "expected": "frozenset", "actual": type(value).__name__},
                ))
                return
            args = typing.get_args(hint)
            if args:
                for item in value:
                    _validate_value(item, args[0], f"{path}{{member}}", violations)
            return

        if isinstance(hint, type) and issubclass(hint, enum.Enum):
            if not isinstance(value, hint):
                violations.append(Violation(
                    "structural", f"{path} should be a member of {hint.__name__}, got {value!r}",
                    {"path": path, "expected": hint.__name__, "actual": repr(value)},
                ))
            return

        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            if not isinstance(value, hint):
                violations.append(Violation(
                    "structural", f"{path} should be a {hint.__name__} but is {type(value).__name__}",
                    {"path": path, "expected": hint.__name__, "actual": type(value).__name__},
                ))
                return
            _validate_dataclass(value, path, violations)
            return

    # No usable hint (or fell through): fall back to a purely structural walk of what's there.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _validate_dataclass(value, path, violations)
    elif isinstance(value, (int, float)):
        _finite_and_range_check(value, path, violations)
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            _validate_value(item, None, f"{path}[{i}]", violations)
    elif isinstance(value, frozenset):
        for item in value:
            _validate_value(item, None, f"{path}{{member}}", violations)
    elif isinstance(value, dict):
        for k, v in value.items():
            _validate_value(v, None, f"{path}[{k!r}]", violations)


def _validate_dataclass(instance, path: str, violations: list) -> None:
    hints = typing.get_type_hints(type(instance))
    for f in dataclasses.fields(instance):
        value = getattr(instance, f.name)
        field_path = f"{path}.{f.name}"
        _validate_value(value, hints.get(f.name), field_path, violations)
        element_type = _KNOWN_LOOSE_CONTAINERS.get((type(instance), f.name))
        if element_type is not None and isinstance(value, tuple):
            for i, item in enumerate(value):
                _validate_value(item, element_type, f"{field_path}[{i}]", violations)


def validate_world_state(state: WorldState) -> list:
    """Structural violations in one ``PerceptionAdapter.get_world_state()`` return value."""
    violations: list = []
    if not isinstance(state, WorldState):
        return [Violation(
            "structural", f"get_world_state() returned a {type(state).__name__}, not a WorldState",
            {"actual": type(state).__name__},
        )]
    _validate_dataclass(state, "world_state", violations)
    return violations


def validate_trajectory(trajectory: PredictedTrajectory) -> list:
    """Structural violations in one ``DynamicsAdapter.predict_trajectory()`` return value."""
    violations: list = []
    if not isinstance(trajectory, PredictedTrajectory):
        return [Violation(
            "structural", f"predict_trajectory() returned a {type(trajectory).__name__}, not a PredictedTrajectory",
            {"actual": type(trajectory).__name__},
        )]
    _validate_dataclass(trajectory, "trajectory", violations)
    return violations
