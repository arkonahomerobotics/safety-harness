"""Software liveness watchdog: no fresh PERMIT within the deadline -> freeze.

Every precondition check, and ActuatorGate itself, assumes the checker is running. The failure class
this module exists for is the one where it isn't: gate() hung on a blocking adapter call, crashed
out of the control loop with an exception the integration swallowed, or simply stopped being
called. None of that produces a BLOCK -- a check that never runs can't fail -- and the last Decision
the actuator side received still says PERMIT, indefinitely. A control loop that keeps executing its
most recent PERMIT (the natural way to drive a continuous action like a reach from a slower
deliberative gate) keeps moving the robot on a safety decision that is no longer being made.

Design -- a dead-man's switch on the *actuator* side, deliberately not inside the gate:

* The gate (``ActuatorGate(..., watchdog=...)``) feeds every decision in, stamped with the time
  gate() *began* (not when it returned: a slow decision has already used up part of its own
  validity), on the watchdog's monotonic clock.
* The actuator loop never executes a Decision directly. It calls ``command()`` every tick, which
  returns the permitted action only while the latest decision is a PERMIT less than ``deadline_s``
  old, verifies it is still bit-identical to what was checked (integrity.verify_decision_action),
  and otherwise returns a freeze from the robot's own FallbackController. Absence of a decision is
  itself the trigger -- nothing has to *happen* for the robot to stop.
* A BLOCK takes effect immediately (the permit is revoked, not merely left to expire), and so does
  a PerceptionFailure (the gate calls ``revoke()`` before raising).
* ``start_monitor(on_expire)`` optionally runs a daemon thread that calls ``on_expire`` once each
  time a live permit lapses, for integrations that must actively *push* a stop (drive a hardware
  E-stop line, publish a halt) rather than rely on the actuator loop polling ``command()``.

What this cannot do: if the whole process hangs -- including the actuator loop and this monitor
thread -- nothing written in Python can act. That is the hardware watchdog timer's job (design doc,
"Prior Art and Open Questions"); this is the software layer in front of it, not a substitute. The
companion precondition ``decision_within_deadline`` covers the other half: a decision that does
finish, but too late to act on.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional

from .integrity import verify_decision_action
from .preconditions import REACTION_INTERVAL_S
from .schema import Action, DecisionVerdict, WorldState


class DecisionWatchdog:
    def __init__(self, fallback, deadline_s: float = REACTION_INTERVAL_S, clock: Callable[[], float] = time.monotonic):
        if not (isinstance(deadline_s, (int, float)) and math.isfinite(deadline_s) and deadline_s > 0):
            raise ValueError(f"deadline_s must be finite and positive, got {deadline_s!r}")
        self._fallback = fallback
        self._deadline_s = float(deadline_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._decision = None
        self._decided_at: Optional[float] = None
        self._state: Optional[WorldState] = None
        self._revoked_reason: Optional[str] = None
        self.last_reason = "no decision received yet"
        self._monitor: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_monitor_error: Optional[BaseException] = None

    @property
    def deadline_s(self) -> float:
        return self._deadline_s

    def now(self) -> float:
        return self._clock()

    def feed(self, decision, state: Optional[WorldState], decided_at: Optional[float] = None) -> None:
        """Record a decision, PERMIT or BLOCK. ``decided_at`` should be when the decision *began*
        (ActuatorGate passes its own start stamp); omitted, the feed time is used, which overstates
        the decision's freshness by however long it took -- pass it."""
        with self._lock:
            self._decision = decision
            self._decided_at = self._clock() if decided_at is None else decided_at
            if state is not None:
                self._state = state
            self._revoked_reason = None

    def revoke(self, reason: str) -> None:
        """Invalidate any live permit now, without waiting for it to expire."""
        with self._lock:
            self._revoked_reason = reason

    def _age_locked(self) -> Optional[float]:
        if self._decided_at is None:
            return None
        return self._clock() - self._decided_at

    def _fresh_permit_locked(self) -> bool:
        if self._decision is None or self._revoked_reason is not None:
            return False
        if getattr(self._decision, "verdict", None) != DecisionVerdict.PERMIT:
            return False
        age = self._age_locked()
        # Negative (clock ran backwards / stamp from the future) or non-finite: not fresh.
        return age is not None and math.isfinite(age) and 0.0 <= age <= self._deadline_s

    def is_fresh(self) -> bool:
        """Is there a PERMIT, not revoked, no older than ``deadline_s``? (Integrity is checked
        separately, in ``command()``.)"""
        with self._lock:
            return self._fresh_permit_locked()

    def _freeze_locked(self, reason: str) -> Action:
        self.last_reason = reason
        try:
            return self._fallback.execute(self._state if self._state is not None else WorldState())
        except Exception:  # noqa: BLE001 -- a broken fallback must not leave the actuators with nothing
            return Action(action_type="freeze", params={"hold_joint_positions": (), "reason": reason})

    def command(self) -> Action:
        """What the actuators may execute *right now*. Call every control tick; never execute a
        Decision's action by any other path."""
        with self._lock:
            decision = self._decision
            if decision is None:
                return self._freeze_locked("no decision received yet")
            if self._revoked_reason is not None:
                return self._freeze_locked(f"permit revoked: {self._revoked_reason}")
            age = self._age_locked()
            if age is None or not math.isfinite(age) or age < 0.0 or age > self._deadline_s:
                return self._freeze_locked(
                    f"no fresh decision within {self._deadline_s:.3f}s (latest is {age!r}s old) -- checker stalled or stopped"
                )
            if not verify_decision_action(decision):
                return self._freeze_locked("action no longer matches the digest it was checked under -- command integrity violated")
            if decision.verdict == DecisionVerdict.PERMIT:
                self.last_reason = "fresh, verified PERMIT"
            else:
                self.last_reason = "fresh BLOCK: executing its fallback"
            return decision.action

    def start_monitor(self, on_expire: Callable[[Action, str], None], poll_interval_s: Optional[float] = None) -> None:
        """Run a daemon thread that calls ``on_expire(freeze_action, reason)`` once each time a live
        permit lapses (expiry or revocation). Exceptions from ``on_expire`` are kept in
        ``last_monitor_error`` rather than killing the thread."""
        if self._monitor is not None and self._monitor.is_alive():
            raise RuntimeError("monitor already running")
        interval = poll_interval_s if poll_interval_s is not None else self._deadline_s / 4
        self._stop.clear()
        self.last_monitor_error = None

        def run():
            was_live = False
            while not self._stop.wait(interval):
                with self._lock:
                    live = self._fresh_permit_locked()
                    action = None if live or not was_live else self._freeze_locked("permit lapsed")
                    reason = self.last_reason
                if action is not None:
                    try:
                        on_expire(action, reason)
                    except Exception as exc:  # noqa: BLE001
                        self.last_monitor_error = exc
                was_live = live

        self._monitor = threading.Thread(target=run, name="safety-harness-watchdog", daemon=True)
        self._monitor.start()

    def stop_monitor(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout)
            self._monitor = None
