"""The certified-hardware-facing half of the boundary/watchdog split: a signal source deliberately
too simple to be the thing that fails.

Real context (2026-09-30/10-01, the concept-proof work for a formal safety-function rating of the
actuator boundary): ``DecisionWatchdog.command()`` and ``start_monitor()`` both compute staleness
*and* decide what to do about it, in the same Python process that could be the thing hanging,
crashing, or running a logic bug -- a real, found gap, not a hypothetical one (a failure that should
trigger a stop is the same class of failure that could break the code computing whether to trigger
it). Real industrial watchdog inputs solve this by not trusting the software's *decision* at all --
they time an incoming signal on their own independent, certified clock, and force the safe state the
moment that signal goes missing, stops toggling, or gets stuck. This module is that signal source,
and nothing else: it does not compute staleness (that's ``DecisionWatchdog.is_fresh()``, reused
unchanged, not reimplemented), it does not decide what "safe" means for any particular robot (that's
the target's own already-certified SS2/Protective-Stop input, per the concept-proof architecture --
Franka and Universal Robots both already ship this, PLd/Category-3), and it has no internal timer or
background thread of its own. Call ``tick()`` from whatever real-time loop already drives the
actuators at a known, fixed rate -- the same loop ``DecisionWatchdog.command()`` is already called
from -- and nothing more is asked of it.

The entire safety-relevant claim this module makes is: *while ``tick()`` keeps being called at the
integrator's own fixed rate, the output genuinely alternates if and only if a fresh, verified permit
currently holds.* Toggling (not just holding a level) is deliberate -- a rated hardware watchdog
input catches a level stuck high the same way it catches a level that goes missing, so a Python bug
that always wrote "high" would still be caught by the certified side, not silently trusted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .watchdog import DecisionWatchdog


class HeartbeatSink(ABC):
    """What ``HeartbeatEmitter`` drives -- a GPIO pin, a serial line, a mocked recorder for tests.
    Implement this against your own target's actual hardware interface; nothing here is prescribed
    beyond "accept a boolean level." A real integration drives the target robot's own certified
    safety-I/O input (see the module docstring), not generic relay hardware."""

    @abstractmethod
    def set_level(self, high: bool) -> None:
        """Drive the output to this level now. Must not raise for a transient I/O error if at all
        avoidable -- a sink that can fail to assert "low" is a sink that can fail to request the
        safe state, which defeats the entire point of this module."""


class HeartbeatEmitter:
    """Ties a ``DecisionWatchdog`` to a ``HeartbeatSink``. The only state is the last level driven;
    the only decision is the one-line check below -- deliberately small enough to read in full."""

    def __init__(self, watchdog: DecisionWatchdog, sink: HeartbeatSink):
        self._watchdog = watchdog
        self._sink = sink
        self._level = False

    def tick(self) -> bool:
        """Call once per control cycle, from the same real-time loop that already calls
        ``watchdog.command()``. Returns the level just driven, for callers that want to log or
        assert on it -- the sink has already been told, that's the actual effect.

        This is the whole decision: toggle while a fresh permit holds, force low the instant it
        doesn't. No deadline arithmetic, no clock reads, no exception handling that could paper
        over a real failure -- ``is_fresh()`` already is the single source of truth for staleness,
        reused here rather than re-derived, so there is exactly one place in this codebase that
        decides what "fresh" means.
        """
        self._level = (not self._level) if self._watchdog.is_fresh() else False
        self._sink.set_level(self._level)
        return self._level
