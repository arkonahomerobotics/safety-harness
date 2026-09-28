"""Persistent, tamper-evident logging -- for Regulation (EU) 2023/1230 Annex III 1.1.9 (the
safety-related software must be identifiable, protected against corruption, and that identification
retained no less than 5 years) and 1.2.1(f) (safety-related decisions logged, retained no less than
1 year) -- and, more generally, for anyone who needs a durable ``Decision`` log they can later prove
was not silently edited or truncated.

Two hash-chained, append-only JSON-lines files, sharing one chain construction:

* :class:`HashChainedDecisionLogger` -- a :class:`~.adapters.base.Logger`, so it drops into
  ``ActuatorGate(..., logger=...)`` unchanged. One entry per ``gate()`` call (1.2.1(f)).
* :class:`SoftwareVersionLog` -- one entry per software-identity snapshot: schema/package version
  and the pinned config digest currently in force. Call ``.record()`` at process start and whenever
  the loaded configuration changes (1.1.9). :func:`report_identity` gives that same snapshot as a
  plain dict, independent of logging, for D1: "can the harness report its own version/hash at
  runtime" -- yes, this is that call.

Entry N's digest covers entry N's own payload AND entry N-1's digest, using the same
``keyed_digest``/``canonical_bytes`` primitives as command and config integrity (``integrity.py``).
With the same HMAC key, forging an entry, deleting one from the middle, or reordering the file is
detectable by re-walking the chain (:func:`verify_log`). **Without a key this is tamper-EVIDENT
(detects corruption or edits made without also recomputing the whole tail) but not tamper-PROOF**
(someone who can rewrite the whole file can recompute a consistent unkeyed chain from scratch) --
the exact same unkeyed-vs-keyed distinction ``integrity.py`` documents for config pinning, for the
same reason: only a key held apart from the log file turns detection into a real guarantee.

**Retention is a deployment/ops policy this module cannot enforce by itself.** Nothing running
in-process can stop someone with disk access from deleting the file after 13 months. What this
module gives that policy something real to check against: an append-only file handle (opened with
``"a"``, this process only ever extends it, never seeks or truncates), a monotonic per-entry
timestamp an ops process can filter on, and a chain that :func:`verify_log` can confirm is still
intact end to end -- so retention can be *audited*, not merely asserted.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .adapters.base import Logger
from .integrity import canonical_bytes, digests_match, keyed_digest
from .schema import SCHEMA_VERSION, Decision, WorldState

_DECISION_DOMAIN = b"safety-harness/audit-log/decision/v1"
_VERSION_DOMAIN = b"safety-harness/audit-log/version/v1"
GENESIS_DIGEST = "0" * 64
_PACKAGE_NAME = "safety-harness"


class LogIntegrityError(RuntimeError):
    """A hash-chained log file does not verify against its own recorded digests, or does not begin
    from :data:`GENESIS_DIGEST`. Raised only by :func:`verify_log` / on resuming an existing log
    file -- appending itself only ever extends a chain forward and can't raise this."""


def _entry_body(seq: int, timestamp: float, prev_digest: str, payload) -> dict:
    return {"seq": seq, "timestamp": timestamp, "prev_digest": prev_digest, "payload": payload}


def _entry_digest(domain: bytes, seq: int, timestamp: float, prev_digest: str, payload, key: Optional[bytes]) -> str:
    return keyed_digest(domain, canonical_bytes(_entry_body(seq, timestamp, prev_digest, payload)), key)


def _jsonable(value, _depth: int = 0):
    """A JSON-round-trippable version of ``value`` that is exactly what gets hashed -- so the bytes
    verified by :func:`verify_log` are always the bytes actually stored, never a reconstruction of
    them. Anything ``canonical_bytes`` couldn't encode unambiguously either (an object with no
    principled JSON form) is stringified via ``repr`` and clearly marked, rather than silently
    dropped -- a payload that can't be logged precisely is logged as "imprecise," never as nothing."""
    if _depth > 64:
        return {"__unrepresentable__": "nested too deeply"}
    d = _depth + 1
    # Enum before bool/str/int: several enums here subclass str (DecisionVerdict, HazardTag, ...),
    # so `isinstance(x, str)` is true for their members too. Checking Enum first and recursing into
    # `.value` matches canonical_bytes's own ordering exactly -- get this backwards and a member
    # slips through as a bare (still-typed) str, which canonical_bytes then encodes differently
    # depending on whether the caller handed it the original Enum object or the plain string it
    # became after a JSON round trip (json.dumps writes a str subclass as a plain string), so the
    # write-time and verify-time digests would silently disagree on every enum-valued field.
    if isinstance(value, Enum):
        return _jsonable(value.value, d)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ("nan" if value != value else ("inf" if value > 0 else "-inf"))
    if isinstance(value, (frozenset, set)):
        return sorted((_jsonable(v, d) for v in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v, d) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, d) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name), d) for f in dataclasses.fields(value)}
    return {"__unrepresentable__": repr(value)}


class _HashChainedFile:
    """Append/verify machinery shared by both log types. Not part of the public interface --
    :class:`HashChainedDecisionLogger` and :class:`SoftwareVersionLog` wrap it."""

    def __init__(self, path: str, domain: bytes, key: Optional[bytes] = None):
        self._path = path
        self._domain = domain
        self._key = key
        self._seq = 0
        self._prev = GENESIS_DIGEST
        if os.path.exists(path) and os.path.getsize(path) > 0:
            # Resume: replay the file to find the true tail. Never trust an externally-supplied
            # "last seq"/"last digest" -- that would let a truncated or edited file be resumed as
            # if nothing had happened. verify_log() checks the WHOLE chain; this only positions us
            # to append correctly, so a single-entry check here is deliberately not the full audit.
            last = None
            for n, entry in enumerate(_iter_entries(path)):
                _check_entry(entry, domain, key, expect_seq=n)
                last = entry
            if last is not None:
                self._seq = last["seq"] + 1
                self._prev = last["digest"]
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    def append(self, payload) -> dict:
        ts = time.time()
        digest = _entry_digest(self._domain, self._seq, ts, self._prev, payload, self._key)
        entry = {"seq": self._seq, "timestamp": ts, "prev_digest": self._prev, "payload": payload, "digest": digest}
        self._fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # best-effort durability; never let a fsync failure crash the control loop
        self._seq += 1
        self._prev = digest
        return entry

    def close(self):
        self._fh.close()


def _iter_entries(path: str):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # A mid-write crash can leave a partial line on disk; that is corruption of the log
                # file, exactly the class of thing this module exists to catch -- fail closed with
                # the same LogIntegrityError every other tamper produces, never an uncaught
                # JSONDecodeError that could crash whatever is walking the log.
                raise LogIntegrityError(f"malformed JSON at line {lineno}: {exc}") from exc


def _check_entry(entry: dict, domain: bytes, key: Optional[bytes], expect_seq: int) -> None:
    required = {"seq", "timestamp", "prev_digest", "payload", "digest"}
    if not required.issubset(entry):
        raise LogIntegrityError(f"entry {expect_seq} is missing required fields: {required - set(entry)}")
    if entry["seq"] != expect_seq:
        raise LogIntegrityError(f"entry at position {expect_seq} claims seq={entry['seq']!r} -- reordered or a gap")
    expected = _entry_digest(domain, entry["seq"], entry["timestamp"], entry["prev_digest"], entry["payload"], key)
    if not digests_match(expected, entry["digest"]):
        raise LogIntegrityError(f"entry {expect_seq} digest does not match its own content -- corrupted or forged")
    if expect_seq == 0 and entry["prev_digest"] != GENESIS_DIGEST:
        raise LogIntegrityError("entry 0 does not chain from GENESIS_DIGEST -- the true first entry is missing")


@dataclass(frozen=True)
class LogVerification:
    ok: bool
    entries: int
    reason: str
    first_bad_seq: Optional[int] = None


def verify_log(path: str, domain: bytes, key: Optional[bytes] = None) -> LogVerification:
    """Walk the whole chain and confirm every entry's digest matches its own content and the
    previous entry's digest, in order, from the genesis value. Use the SAME key the log was written
    with -- a keyed log checked without its key, or with the wrong one, reports every entry as bad
    (a missing/wrong key must never read as "verified"), not as "unverifiable"."""
    if not os.path.exists(path):
        return LogVerification(ok=False, entries=0, reason=f"no such file: {path!r}")
    count = 0
    try:
        for n, entry in enumerate(_iter_entries(path)):
            _check_entry(entry, domain, key, expect_seq=n)
            count = n + 1
        return LogVerification(ok=True, entries=count, reason="chain intact" if count else "empty log")
    except LogIntegrityError as exc:
        return LogVerification(ok=False, entries=count, reason=str(exc), first_bad_seq=count)


class HashChainedDecisionLogger(Logger):
    """A :class:`~.adapters.base.Logger` that appends every ``Decision`` (with its ``WorldState``)
    to a hash-chained JSON-lines file -- Annex III 1.2.1(f)'s "safety-related decisions logged."
    Every field of both objects is recorded (verdict, every check's name/pass-fail/reason, the
    action, the action digest, and the full perceived state), not a summary -- so the log alone,
    without re-running anything, shows exactly what was decided and why.
    """

    def __init__(self, path: str, *, key: Optional[bytes] = None):
        self._chain = _HashChainedFile(path, _DECISION_DOMAIN, key)

    def record(self, decision: Decision, state: WorldState) -> None:
        self._chain.append({"decision": _jsonable(decision), "state": _jsonable(state)})

    def close(self) -> None:
        self._chain.close()

    def verify(self, *, key: Optional[bytes] = None) -> LogVerification:
        """Verify this logger's own file (convenience for the common case of checking what you just
        wrote in the same process). ``key`` defaults to the key this logger was constructed with;
        pass a different one only to deliberately test that the wrong key fails to verify."""
        return verify_log(self._chain._path, _DECISION_DOMAIN, self._chain._key if key is None else key)


class SoftwareVersionLog:
    """A hash-chained record of the running safety-software's identity over time -- Annex III
    1.1.9's "identification of that software... retained for no less than 5 years." Call
    ``.record()`` at process start and again whenever the loaded configuration changes; each call
    is one immutable, chained entry, never an overwrite of the last."""

    def __init__(self, path: str, *, key: Optional[bytes] = None):
        self._chain = _HashChainedFile(path, _VERSION_DOMAIN, key)

    def record(self, config_digest: Optional[str] = None, **extra) -> dict:
        entry = self._chain.append({**report_identity(config_digest=config_digest), **{k: _jsonable(v) for k, v in extra.items()}})
        return entry

    def close(self) -> None:
        self._chain.close()

    def verify(self, *, key: Optional[bytes] = None) -> LogVerification:
        return verify_log(self._chain._path, _VERSION_DOMAIN, self._chain._key if key is None else key)


def report_identity(config_digest: Optional[str] = None) -> dict:
    """The running harness's own version/hash, queryable at any time -- D1: "can the harness report
    its own version/hash at runtime." Independent of logging: call this from an integrator's own
    health check, a support bundle, or before every deployment, not only through
    :class:`SoftwareVersionLog`. ``package_version`` is None if the package isn't installed in the
    normal way (e.g. running from a source checkout without ``pip install``) -- reported as None,
    not guessed."""
    try:
        package_version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version,
        "config_digest": config_digest,
        "pin_scheme": (config_digest.split(":", 1)[0] if config_digest and ":" in config_digest else None),
    }
