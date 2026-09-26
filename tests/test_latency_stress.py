"""Measures gate() wall-clock latency under injected jitter and real OS contention, and checks it
against the reaction-time assumption baked into iso15066_separation_distance_maintained.

Why this exists: reaction_time_s (default 0.15s) + sampling_interval_s (default 0.05s) in that
check model the *robot's own* hardware stop-response lag -- ISO/TS 15066's definition of reaction
time. They say nothing about how long ActuatorGate.gate() itself takes to decide BLOCK before that
hardware lag even starts. Under real load (GC pauses, OS scheduling, a busy machine), gate()'s own
latency is not zero, and nothing today adds it to the separation-distance budget. This suite
measures that latency for real, under real competing load, rather than assuming it away -- see the
design doc's "Latency: Measured, Not Assumed", which measured baseline/idle latency; this measures
worst-case latency under stress, which is the number that actually matters for a reaction-time
budget.

Run with: python3 -m unittest discover -s tests -p "test_latency_stress.py" -v
"""

from __future__ import annotations

import multiprocessing
import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness import ActionSchemaRegistry, ActuatorGate  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402
from safety_harness.preconditions import REACTION_INTERVAL_S  # noqa: E402

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "example_action_schema.yaml")
N_ITERATIONS = 300


def _busy_loop(stop_flag):
    """Burns one CPU core flat out until stop_flag is set -- real OS-scheduling contention, not a
    simulated delay, so this measures whether the *host* starves the test process, not just
    whether we remembered to inject a sleep somewhere."""
    while not stop_flag.value:
        sum(i * i for i in range(2000))


class _JitteryPerception(PerceptionAdapter):
    """Wraps a real perception call with injected latency drawn from a distribution shaped like
    real system jitter: mostly a few ms, occasionally a GC-pause-sized spike -- not a single fixed
    sleep, which would only ever test one point on the curve."""

    def __init__(self, state, rng):
        self._state = state
        self._rng = rng

    def get_world_state(self):
        if self._rng.random() < 0.05:
            time.sleep(self._rng.uniform(0.05, 0.25))  # the long tail: a stall, not typical jitter
        else:
            time.sleep(self._rng.uniform(0.0, 0.01))
        return self._state


class _JitteryDynamics(DynamicsAdapter):
    def __init__(self, trajectory, rng):
        self._trajectory = trajectory
        self._rng = rng

    def predict_trajectory(self, state, action, horizon_s):
        if self._rng.random() < 0.05:
            time.sleep(self._rng.uniform(0.05, 0.25))
        else:
            time.sleep(self._rng.uniform(0.0, 0.01))
        return self._trajectory


class _FastPerception(PerceptionAdapter):
    def __init__(self, state):
        self._state = state

    def get_world_state(self):
        return self._state


class _FastDynamics(DynamicsAdapter):
    def __init__(self, trajectory):
        self._trajectory = trajectory

    def predict_trajectory(self, state, action, horizon_s):
        return self._trajectory


def _percentiles(samples):
    s = sorted(samples)
    n = len(s)

    def pct(p):
        return s[min(n - 1, int(n * p))]

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": s[-1]}


def _report(label, samples, budget_s):
    stats = _percentiles(samples)
    print(f"\n  {label}: n={len(samples)} p50={stats['p50']*1000:.1f}ms p95={stats['p95']*1000:.1f}ms "
          f"p99={stats['p99']*1000:.1f}ms max={stats['max']*1000:.1f}ms  (reaction-interval budget={budget_s*1000:.0f}ms)")
    over_budget = sum(1 for s in samples if s > budget_s)
    if over_budget:
        print(f"  FINDING: {over_budget}/{len(samples)} gate() calls exceeded the {budget_s*1000:.0f}ms reaction-interval "
              f"budget that iso15066_separation_distance_maintained assumes is available in addition to gate()'s own "
              f"decision time -- that budget currently models only the robot's hardware stop lag, not this.")
    return stats


class LatencyStressTests(unittest.TestCase):
    """Reports real numbers rather than asserting a specific latency contract -- what the 'right'
    threshold is remains an open design question (see the design doc's Regulatory Mapping). The one
    thing asserted outright is that nothing here ever hangs outright."""

    ABSOLUTE_HANG_CEILING_S = 5.0

    def _run_batch(self, perception_factory, dynamics_factory, n=N_ITERATIONS):
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=(fixtures.close_agent(),))
        trajectory = fixtures.straight_line_trajectory()
        samples = []
        for _ in range(n):
            gate = ActuatorGate(
                perception=perception_factory(state), dynamics=dynamics_factory(trajectory),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
            )
            t0 = time.perf_counter()
            gate.gate(fixtures.grasp_action())
            dt = time.perf_counter() - t0
            self.assertLess(dt, self.ABSOLUTE_HANG_CEILING_S, f"a single gate() call took {dt:.2f}s -- looks like a hang, not jitter")
            samples.append(dt)
        return samples

    def test_baseline_latency_idle_machine(self):
        samples = self._run_batch(_FastPerception, _FastDynamics)
        _report("baseline (idle, no injected jitter)", samples, REACTION_INTERVAL_S)

    def test_latency_under_injected_adapter_jitter(self):
        rng = random.Random(20260927)
        samples = self._run_batch(
            lambda state: _JitteryPerception(state, rng),
            lambda traj: _JitteryDynamics(traj, rng),
        )
        stats = _report("injected adapter jitter (simulated stalls/GC pauses)", samples, REACTION_INTERVAL_S)
        # A stall in the 50-250ms range was injected 5% of the time by construction, so p95/p99
        # should visibly reflect it -- if they don't, the injection itself isn't reaching gate()'s
        # measured wall-clock time, which would make this whole test meaningless.
        self.assertGreater(stats["p99"], 0.03, "p99 didn't reflect the injected long-tail stalls at all -- check the injection wiring")

    def test_latency_under_real_os_contention(self):
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        stop_flag = multiprocessing.Value("b", False)
        workers = [multiprocessing.Process(target=_busy_loop, args=(stop_flag,)) for _ in range(n_workers)]
        for w in workers:
            w.start()
        try:
            time.sleep(0.2)  # let contention actually ramp up before measuring
            samples = self._run_batch(_FastPerception, _FastDynamics)
        finally:
            stop_flag.value = True
            for w in workers:
                w.join(timeout=5)
        _report(f"real OS contention ({n_workers} CPU-saturated worker processes)", samples, REACTION_INTERVAL_S)


if __name__ == "__main__":
    unittest.main()
