"""Bit-exact digests for command integrity and configuration integrity -- stdlib only (hashlib,
hmac, struct), same dependency-free rule as the rest of this package.

Two failure classes this module exists for, neither of which any sensor-facing check can see:

* **Command integrity** (roadmap item 6). ``ActuatorGate.gate()`` checks one action and returns a
  Decision; something else then sends an action to the actuators. Nothing previously proved those
  were the same action. ``Action`` is a frozen dataclass, but its ``params`` dict is not -- anyone
  holding a reference (the planner that proposed it, a concurrent thread, a buggy adapter) can
  rewrite a target or a grip force after, or *while*, it is being checked, and the checked-and-
  PERMITted Decision would carry the rewritten action to the actuators with its PERMIT intact. The
  fix binds the Decision to a digest of exactly what was checked (``Decision.action_digest``, taken
  when gate() begins) and re-verifies it at execution time (``verify_decision_action``,
  ``DecisionWatchdog.command()``), plus a registered check (``command_integrity_verified``) that
  catches a rewrite that happens *during* checking.

* **Configuration integrity** (roadmap item 7). The action schema and every check's thresholds are
  what was validated; a silently edited YAML (a force budget raised from 3kg to 300kg, a check
  deleted), an in-process rebinding of a REGISTRY entry, or a changed default in code all change
  validated behavior without any check failing. ``ActionSchemaRegistry.config_digest()`` hashes the
  *effective* running configuration and ``config_integrity_verified`` compares it, on every
  decision, to a known-good digest pinned from outside the config itself.

Digests are plain SHA-256 by default, or HMAC-SHA256 when a key is supplied. The difference matters
and is stated plainly rather than implied: an unkeyed digest detects accidental corruption,
misconfiguration and any tamper by someone who doesn't also update the pinned digest, but anyone
who can rewrite both the config and its pinned digest can forge it. Only a keyed HMAC, with the key
held somewhere the config's editor can't read, makes the pin a signature rather than a checksum.
Each digest string carries its scheme as a prefix (``sha256:`` / ``hmac-sha256:``) so an unkeyed
digest can never be accepted where a keyed one was pinned, or vice versa.

Canonical encoding is deliberately strict: every value is type-tagged and length-prefixed (so
``(1, 2)`` and ``[1, 2]`` and ``"1,2"`` all differ), floats are encoded by their exact IEEE-754 bits
(so ``-0.0`` differs from ``0.0`` -- "bit-identical" means bit-identical), and anything that isn't a
plain, fully-inspectable value (a custom object, a lazily-evaluated sequence, a self-referencing
list) raises ``UnverifiableError`` instead of being encoded by ``repr()`` or skipped. A value whose
content can't be pinned can't be proven unchanged, so callers treat it as a failure, not a pass.

Run ``python -m safety_harness.pin path/to/action_schema.yaml`` to print a config's digest
for pinning (set ``SAFETY_HARNESS_CONFIG_KEY`` in the environment to print its HMAC instead).
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import os
import struct
from enum import Enum
from types import MappingProxyType
from typing import Optional

from .schema import Action

COMMAND_SEAL_PARAM = "command_digest"
CONFIG_KEY_ENV = "SAFETY_HARNESS_CONFIG_KEY"

_ACTION_DOMAIN = b"safety-harness/action/v1"
_CONFIG_DOMAIN = b"safety-harness/config/v1"
_MAX_DEPTH = 64


class UnverifiableError(ValueError):
    """A value has no canonical encoding, so it cannot be bound to a digest -- and therefore can't
    be proven unchanged between two points in time. Always treated as an integrity failure."""


class ConfigIntegrityError(RuntimeError):
    """The loaded configuration does not match the known-good digest it was pinned to. Raised at
    load time so a gate is never constructed around an unvalidated config at all."""


def _frame(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack(">Q", len(body)) + body


def canonical_bytes(value, _depth: int = 0) -> bytes:
    """Deterministic, type-tagged, length-prefixed byte encoding of ``value``. Raises
    ``UnverifiableError`` for anything it can't encode unambiguously -- see the module docstring."""
    if _depth > _MAX_DEPTH:
        raise UnverifiableError("value nested too deeply to encode (self-referencing container?)")
    d = _depth + 1
    if value is None:
        return _frame(b"N", b"")
    if isinstance(value, bool):  # before int: bool is an int subclass
        return _frame(b"B", b"\x01" if value else b"\x00")
    if isinstance(value, Enum):  # before str/int: str/int-valued enums subclass those too
        cls = type(value)
        return _frame(b"E", _frame(b"s", f"{cls.__module__}.{cls.__qualname__}".encode()) + canonical_bytes(value.value, d))
    if isinstance(value, int):
        return _frame(b"I", int.__repr__(value).encode())
    if isinstance(value, float):
        return _frame(b"F", struct.pack(">d", value))  # exact bits: -0.0 != 0.0, every NaN payload distinct
    if isinstance(value, str):
        return _frame(b"S", str.encode(value, "utf-8", "surrogatepass"))
    if isinstance(value, bytes):
        return _frame(b"Y", bytes(value))
    # Containers must be *exactly* these types, not subclasses: a subclass can override iteration
    # or item access and present different contents to the checker and to the executor.
    if type(value) in (dict, MappingProxyType):
        items = sorted((canonical_bytes(k, d), canonical_bytes(v, d)) for k, v in value.items())
        return _frame(b"D", b"".join(k + v for k, v in items))
    if type(value) is tuple:
        return _frame(b"T", b"".join(canonical_bytes(v, d) for v in value))
    if type(value) is list:
        return _frame(b"L", b"".join(canonical_bytes(v, d) for v in value))
    if type(value) in (frozenset, set):
        return _frame(b"Z", b"".join(sorted(canonical_bytes(v, d) for v in value)))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        fields = b"".join(
            _frame(b"s", f.name.encode()) + canonical_bytes(getattr(value, f.name), d)
            for f in dataclasses.fields(value)
        )
        return _frame(b"C", _frame(b"s", f"{cls.__module__}.{cls.__qualname__}".encode()) + fields)
    raise UnverifiableError(f"no canonical encoding for a value of type {type(value).__name__!r}")


def keyed_digest(domain: bytes, payload: bytes, key: Optional[bytes] = None) -> str:
    """``sha256:<hex>`` with no key, ``hmac-sha256:<hex>`` with one. The domain separates action
    digests from config digests, so one can never be replayed as the other."""
    message = _frame(b"domain", domain) + payload
    if key:
        return "hmac-sha256:" + hmac.new(bytes(key), message, hashlib.sha256).hexdigest()
    return "sha256:" + hashlib.sha256(message).hexdigest()


def digests_match(expected, actual) -> bool:
    """Constant-time comparison that is False -- never an exception, never True -- for anything
    that isn't two non-empty strings."""
    if not isinstance(expected, str) or not isinstance(actual, str) or not expected or not actual:
        return False
    return hmac.compare_digest(expected.encode("utf-8", "surrogatepass"), actual.encode("utf-8", "surrogatepass"))


def _action_payload(action: Action, drop_param: Optional[str] = None) -> bytes:
    if not isinstance(action, Action):
        raise UnverifiableError(f"not an Action: {type(action).__name__!r}")
    if type(action.params) not in (dict, MappingProxyType):
        raise UnverifiableError(f"action params is a {type(action.params).__name__!r}, not a plain dict")
    params = {k: v for k, v in action.params.items() if k != drop_param}
    return canonical_bytes(action.action_type) + canonical_bytes(params)


def action_digest(action: Action, *, key: Optional[bytes] = None) -> str:
    """Digest of the WHOLE action -- type and every param, a seal included if one is present.
    This is what ``Decision.action_digest`` binds."""
    return keyed_digest(_ACTION_DOMAIN, _action_payload(action), key)


def try_action_digest(action) -> Optional[str]:
    """``action_digest`` that returns None instead of raising, for callers (the engine) that must
    never let an unencodable action crash the control loop -- None is itself the failure signal."""
    try:
        return action_digest(action)
    except Exception:  # noqa: BLE001 -- UnverifiableError, RecursionError, a hostile __eq__/__hash__ ...
        return None


def seal_action(action: Action, *, key: Optional[bytes] = None, seal_param: str = COMMAND_SEAL_PARAM) -> Action:
    """Return a copy of ``action`` carrying a seal over its own content in ``params[seal_param]`` --
    for a proposer (planner, policy server) that wants the gate to verify the command wasn't altered
    in transit. With a key, the seal also authenticates the proposer. Optional: the downstream
    decision-to-actuator binding doesn't depend on it."""
    seal = keyed_digest(_ACTION_DOMAIN, _action_payload(action, drop_param=seal_param), key)
    params = {k: v for k, v in action.params.items() if k != seal_param}
    params[seal_param] = seal
    return Action(action_type=action.action_type, params=params)


def verify_action_seal(action: Action, *, key: Optional[bytes] = None, seal_param: str = COMMAND_SEAL_PARAM) -> bool:
    """True only if ``action`` carries a seal that matches its current content under ``key``.
    Missing seal, wrong key, wrong scheme, or unencodable content: False."""
    try:
        seal = action.params.get(seal_param)
        expected = keyed_digest(_ACTION_DOMAIN, _action_payload(action, drop_param=seal_param), key)
    except Exception:  # noqa: BLE001
        return False
    return digests_match(expected, seal)


def verify_decision_action(decision) -> bool:
    """Execution-time check: is ``decision.action`` still bit-identical to what was checked? Call
    this immediately before handing the action to actuators (DecisionWatchdog.command() does). A
    Decision with no digest (unencodable action, or constructed by hand) never verifies."""
    expected = getattr(decision, "action_digest", None)
    return digests_match(expected, try_action_digest(getattr(decision, "action", None)))


def config_digest(effective_config, *, key: Optional[bytes] = None) -> str:
    """Digest of an already-assembled effective configuration (see
    ``ActionSchemaRegistry.config_digest``, which builds it). Raises UnverifiableError if any
    configured value has no canonical encoding -- e.g. a YAML timestamp parsed into a datetime."""
    return keyed_digest(_CONFIG_DOMAIN, canonical_bytes(effective_config), key)


def key_from_env(name: Optional[str]) -> Optional[bytes]:
    """The HMAC key stored in environment variable ``name``, or None if unnamed, unset or empty.
    Callers that *require* a key treat None as a failure, never as "use no key"."""
    if not name:
        return None
    value = os.environ.get(name)
    return value.encode("utf-8") if value else None


def read_digest_file(path: str) -> str:
    """Read a pinned digest from a one-line file (e.g. ``example_action_schema.yaml.sha256``).
    Anything after the first whitespace-separated token is ignored, sha256sum-style."""
    with open(path) as f:
        tokens = f.read().split()
    if not tokens:
        raise ConfigIntegrityError(f"digest file {path!r} is empty")
    return tokens[0]
