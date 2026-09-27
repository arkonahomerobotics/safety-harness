"""Declarative action-type -> required-precondition mapping.

A robot team edits the YAML config, not this file (design doc: "Action Precondition Schemas",
"Reference Module Design"). An action type with no registered schema is unsafe by default -- see
``run_checks`` -- which enforces "unsafe unless confirmed" even at the registration level, not just
inside individual checks.

The loaded configuration can be pinned to a known-good digest (``expected_digest``, optionally
HMAC-keyed) -- see ``config_digest`` for exactly what it covers and integrity.py for why. A pinned
registry refuses to construct at all around a config that doesn't match, and the registered
``config_integrity_verified`` check re-verifies the *live* configuration on every decision, so a
tamper after load is caught too, not just one before it.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, replace
from typing import Optional

import yaml

from .integrity import ConfigIntegrityError, config_digest, digests_match, try_action_digest
from .preconditions import DEPRECATED_CHECKS, REGISTRY, CheckContext
from .schema import SCHEMA_VERSION, Action, PredictedTrajectory, PreconditionResult, WorldState


@dataclass(frozen=True)
class ActionTypeSchema:
    action_type: str
    checks: tuple  # each element: {"name": <registry key>, "kwargs": {...}}


class ActionSchemaRegistry:
    def __init__(self, schemas: dict, *, expected_digest: Optional[str] = None, hmac_key: Optional[bytes] = None):
        self._schemas = schemas
        self._expected_digest = expected_digest
        self._hmac_key = hmac_key
        if expected_digest is not None and not self.verify_config():
            raise ConfigIntegrityError(
                "action schema does not match its pinned known-good digest -- refusing to load it "
                f"(expected {expected_digest!r}, running config hashes to {self._safe_digest()!r})"
            )

    @classmethod
    def from_yaml(
        cls, path: str, *, expected_digest: Optional[str] = None, hmac_key: Optional[bytes] = None,
    ) -> "ActionSchemaRegistry":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, expected_digest=expected_digest, hmac_key=hmac_key)

    @classmethod
    def from_dict(
        cls, raw: dict, *, expected_digest: Optional[str] = None, hmac_key: Optional[bytes] = None,
    ) -> "ActionSchemaRegistry":
        schemas = {}
        for action_type, spec in (raw.get("action_types") or {}).items():
            # Deep-copied: the caller's dict is not the live config. Before this, anyone still
            # holding the dict passed in here could rewrite a threshold of an already-running gate.
            checks = tuple(copy.deepcopy(list(spec.get("checks", []))))
            for check in checks:
                if check["name"] not in REGISTRY:
                    raise ValueError(f"unknown precondition check {check['name']!r} for action type {action_type!r}")
                if check["name"] in DEPRECATED_CHECKS:
                    warnings.warn(
                        f"precondition check {check['name']!r} (action type {action_type!r}) is deprecated; "
                        f"use {DEPRECATED_CHECKS[check['name']]!r}",
                        DeprecationWarning, stacklevel=2,
                    )
            schemas[action_type] = ActionTypeSchema(action_type=action_type, checks=checks)
        return cls(schemas, expected_digest=expected_digest, hmac_key=hmac_key)

    @property
    def expected_digest(self) -> Optional[str]:
        return self._expected_digest

    def effective_config(self) -> dict:
        """The configuration as it will actually execute: for every action type, each check's name,
        which function REGISTRY currently binds that name to (module + qualified name), and its
        *effective* parameters -- the function's own keyword defaults overlaid with the YAML kwargs.
        Defaults are included on purpose: a threshold changed in code is as much a change to
        validated behavior as one changed in YAML. The ``context`` parameter is excluded -- it is
        supplied per decision by the engine, not configured."""
        action_types = {}
        for action_type, schema in self._schemas.items():
            entries = []
            for check in schema.checks:
                name = check.get("name")
                fn = REGISTRY.get(name)
                binding = f"{fn.__module__}.{fn.__qualname__}" if fn is not None else "<unregistered>"
                params = dict(getattr(fn, "__kwdefaults__", None) or {})
                params.update(check.get("kwargs") or {})
                params.pop("context", None)
                entries.append({"name": name, "binding": binding, "params": params})
            action_types[action_type] = entries
        return {"schema_version": SCHEMA_VERSION, "action_types": action_types}

    def config_digest(self) -> str:
        """Digest of ``effective_config()``, HMAC-keyed if this registry was given a key. Raises
        UnverifiableError if a configured value has no canonical encoding."""
        return config_digest(self.effective_config(), key=self._hmac_key)

    def _safe_digest(self) -> Optional[str]:
        try:
            return self.config_digest()
        except Exception:  # noqa: BLE001
            return None

    def verify_config(self) -> bool:
        """True only if a known-good digest is pinned AND the live configuration still matches it.
        No pin is not "nothing to verify" -- it is an unvalidated config, so False."""
        return digests_match(self._expected_digest, self._safe_digest())

    def run_checks(
        self, action: Action, state: WorldState, trajectory: PredictedTrajectory,
        context: Optional[CheckContext] = None,
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
        if context is None:
            # Called directly rather than through ActuatorGate.gate(): the best available "decision
            # start" is right now, before any check runs -- later than gate() would record (it misses
            # perception and dynamics time), never earlier.
            context = CheckContext(
                decision_started_at=time.monotonic(), clock=time.monotonic,
                checked_action_digest=try_action_digest(action),
            )
        if context.schema_registry is None:
            context = replace(context, schema_registry=self)
        results = []
        for check in schema.checks:
            fn = REGISTRY[check["name"]]
            kwargs = dict(check.get("kwargs", {}) or {})
            if getattr(fn, "wants_check_context", False):
                # Supplied by the engine, never by YAML: a config that tried to pass its own
                # "context" (e.g. a fabricated decision start time) is overridden here.
                kwargs["context"] = context
            results.append(fn(state, action, trajectory, **kwargs))
        return tuple(results)
