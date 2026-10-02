"""HeartbeatEmitter: the signal source a certified hardware watchdog input is meant to time on its
own clock. Tests here only cover the one decision this module makes -- toggle while fresh, force low
the instant it isn't -- not DecisionWatchdog's own freshness logic, which is already covered in
tests/test_adversarial_next_checks.py and reused here unchanged (see the module's own docstring for
why re-deriving staleness here would defeat the point)."""

from __future__ import annotations

import os
import sys
import unittest

from safety_harness.adapters.simple import FreezeInPlaceFallback
from safety_harness.heartbeat import HeartbeatEmitter, HeartbeatSink
from safety_harness.watchdog import DecisionWatchdog
from safety_harness.schema import Action, Decision, DecisionVerdict

sys.path.insert(0, os.path.dirname(__file__))


class FakeClock:
    """A manually-advanced monotonic clock, so timing tests are exact and never flaky -- same
    pattern as tests/test_adversarial_next_checks.py's own FakeClock, not imported from there since
    test modules aren't meant to depend on each other."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class RecordingSink(HeartbeatSink):
    """A real HeartbeatSink -- not a mock standing in for one -- that records every level it was
    ever told to drive, so a test can assert on the exact sequence `tick()` produced."""

    def __init__(self):
        self.levels: list = []
        self.set_level_calls = 0

    def set_level(self, high: bool) -> None:
        self.set_level_calls += 1
        self.levels.append(high)


def _permit(clock) -> Decision:
    return Decision(verdict=DecisionVerdict.PERMIT, action=Action("reach", {"target_position": (0.0, 0.0, 0.0)}))


class HeartbeatEmitterTest(unittest.TestCase):
    def test_no_decision_ever_fed_drives_low(self):
        wd = DecisionWatchdog(FreezeInPlaceFallback())
        sink = RecordingSink()
        emitter = HeartbeatEmitter(wd, sink)
        self.assertFalse(emitter.tick())
        self.assertEqual(sink.levels, [False])

    def test_toggles_once_per_tick_while_fresh(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        wd.feed(_permit(clock), None, decided_at=clock.t)
        sink = RecordingSink()
        emitter = HeartbeatEmitter(wd, sink)

        levels = [emitter.tick() for _ in range(6)]
        self.assertEqual(levels, [True, False, True, False, True, False])
        self.assertEqual(sink.levels, levels)

    def test_stops_toggling_the_instant_the_permit_goes_stale(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=1.0, clock=clock)
        wd.feed(_permit(clock), None, decided_at=clock.t)
        sink = RecordingSink()
        emitter = HeartbeatEmitter(wd, sink)

        emitter.tick()
        emitter.tick()
        emitter.tick()
        self.assertTrue(sink.levels[-1])  # 3 ticks from low: True, False, True -- currently high

        clock.advance(10.0)  # well past deadline_s -- no longer fresh
        self.assertFalse(emitter.tick())
        self.assertFalse(emitter.tick())  # stays low, doesn't resume toggling on its own
        self.assertEqual(sink.levels[-2:], [False, False])

    def test_a_block_revokes_the_signal_immediately_not_after_a_deadline(self):
        """Mirrors DecisionWatchdog's own revoke-is-immediate guarantee: a BLOCK or a perception
        failure takes effect on the very next tick, not after deadline_s elapses -- the heartbeat
        must not keep toggling just because the clock hasn't caught up yet."""

        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        wd.feed(_permit(clock), None, decided_at=clock.t)
        sink = RecordingSink()
        emitter = HeartbeatEmitter(wd, sink)

        self.assertTrue(emitter.tick())
        wd.revoke("gate BLOCKed the next action")
        self.assertFalse(emitter.tick())

    def test_resumes_toggling_from_low_once_a_fresh_permit_returns(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        sink = RecordingSink()
        emitter = HeartbeatEmitter(wd, sink)

        self.assertFalse(emitter.tick())  # nothing fed yet
        wd.feed(_permit(clock), None, decided_at=clock.t)
        self.assertTrue(emitter.tick())  # resumes from low, not from wherever it left off

    def test_tick_never_calls_the_sink_more_than_once(self):
        """A sink that's a real hardware line should see exactly one transition per tick -- not
        zero, not two -- regardless of fresh/stale."""

        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        wd.feed(_permit(clock), None, decided_at=clock.t)
        sink = RecordingSink()
        emitter = HeartbeatEmitter(wd, sink)

        for _ in range(10):
            emitter.tick()
        self.assertEqual(sink.set_level_calls, 10)

    def test_sink_is_abstract(self):
        with self.assertRaises(TypeError):
            HeartbeatSink()


if __name__ == "__main__":
    unittest.main()
