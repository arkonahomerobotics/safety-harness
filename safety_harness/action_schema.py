"""Declarative action-type -> required-precondition mapping.

A robot team edits the YAML config, not this file (design doc: "Action Precondition Schemas",
"Reference Module Design"). An action type with no registered schema is unsafe by default -- see
``run_checks`` -- which enforces "unsafe unless confirmed" even at the registration level, not just
inside individual checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from .preconditions import REGISTRY
from .schema import Action, PredictedTrajectory, PreconditionResult, WorldState


@dataclass(frozen=True)
class ActionTypeSchema:
    action_type: str
    checks: tuple  # each element: {"name": <registry key>, "kwargs": {...}}


class ActionSchemaRegistry:
    def __init__(self, schemas: dict):
        self._schemas = schemas

    @classmethod
    def from_yaml(cls, path: str) -> "ActionSchemaRegistry":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ActionSchemaRegistry":
        schemas = {}
        for action_type, spec in (raw.get("action_types") or {}).items():
            checks = tuple(spec.get("checks", []))
            for check in checks:
                if check["name"] not in REGISTRY:
                    raise ValueError(f"unknown precondition check {check['name']!r} for action type {action_type!r}")
            schemas[action_type] = ActionTypeSchema(action_type=action_type, checks=checks)
        return cls(schemas)

    def run_checks(
        self, action: Action, state: WorldState, trajectory: PredictedTrajectory,
    ) -> tuple:
        schema = self._schemas.get(action.action_type)
        if schema is None:
            # No declared schema -> no confirmed preconditions -> never permitted. Register the
            # action type explicitly (even with an empty check list) to allow it through.
            return (
                PreconditionResult(
                    name="action_type_registered",
                    satisfied=False,
                    reason=f"action type {action.action_type!r} has no registered schema",
                ),
            )
        results = []
        for check in schema.checks:
            fn = REGISTRY[check["name"]]
            kwargs = check.get("kwargs", {}) or {}
            results.append(fn(state, action, trajectory, **kwargs))
        return tuple(results)
