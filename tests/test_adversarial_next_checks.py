"""Adversarial stress tests for the next-safety-checks roadmap: software liveness (watchdog +
decision_within_deadline), payload/grip force, stability margin, sensor staleness, sensor coverage,
command integrity, config integrity, and vulnerable bystanders.

Every class follows the same shape, so a BLOCK can only be explained by the new check:

1. **Gets past today.** A concrete unsafe scenario is run through a real ``ActuatorGate.gate()``
   with the pre-existing checks only (test_engine.SCHEMA_DICT's grasp list, or test_anymal_adapter's
   navigate list, or the specific old check a unit-level scenario targets) and is shown to PERMIT.
2. **Blocked now.** The same scenario, with the new check appended to that same list, BLOCKs --
   and the new check is among the failures.
3. **Fails closed.** NaN / None / missing / malformed inputs to the new check never pass.
4. **Boundaries.** Exactly-at-limit values pass and just-past-limit values fail, using exactly
   representable floats so the boundary is the check's, not IEEE rounding's.

Finally, an end-to-end class runs configs/example_action_schema.yaml -- with every new check wired
and pinned to its committed digest -- and confirms the golden scenario still PERMITs while each
scenario above BLOCKs on the right check.

Run with: python3 -m unittest discover -s tests -p "test_*.py" -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
import threading
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402
from test_anymal_adapter import NAVIGATE_SCHEMA_DICT, _base_footprint_trajectory  # noqa: E402
from test_engine import SCHEMA_DICT, _StubDynamics, _StubPerception  # noqa: E402

from safety_harness import (  # noqa: E402
    ActionSchemaRegistry,
    ActuatorGate,
    AgentCategory,
    ConfigIntegrityError,
    DecisionVerdict,
    DecisionWatchdog,
    HazardTag,
    ObservedRegion,
    PerceptionFailure,
    seal_action,
    verify_action_seal,
    verify_decision_action,
)
from safety_harness import integrity  # noqa: E402
from safety_harness import preconditions as pc  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402
from safety_harness.schema import Action, Pose, PredictedTrajectory, TrackedAgent, TrajectoryPoint, WorldState  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_SCHEMA_PATH = os.path.join(ROOT, "configs", "example_action_schema.yaml")
EXAMPLE_DIGEST_PATH = EXAMPLE_SCHEMA_PATH + ".sha256"


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def _failed_names(decision):
    return [r.name for r in decision.precondition_results if not r.satisfied]


def _with_checks(base_dict, action_type, *extra_checks):
    """A copy of ``base_dict`` with ``extra_checks`` appended to ``action_type``'s list -- the
    pre-existing checks, unchanged, plus only the new one(s) under test."""
    d = copy.deepcopy(base_dict)
    d["action_types"][action_type]["checks"].extend(copy.deepcopy(list(extra_checks)))
    return d


def _check(name, **kwargs):
    return {"name": name, "kwargs": kwargs}


def _gate(state, schema_dict=SCHEMA_DICT, *, trajectory=None, dynamics=None, clock=None, watchdog=None,
          registry=None, perception=None):
    trajectory = trajectory or fixtures.straight_line_trajectory()
    return ActuatorGate(
        perception=perception or _StubPerception(state),
        dynamics=dynamics or _StubDynamics(trajectory),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=registry or ActionSchemaRegistry.from_dict(schema_dict),
        clock=clock,
        watchdog=watchdog,
    )


def _golden_state(**kwargs):
    return fixtures.base_world_state(objects=(fixtures.confirmed_object(),), **kwargs)


def _ctx(started_at=0.0, now=0.0, digest=None, registry=None):
    return pc.CheckContext(decision_started_at=started_at, clock=lambda: now, checked_action_digest=digest,
                           schema_registry=registry)


class FakeClock:
    """A manually-advanced monotonic clock, so timing tests are exact and never flaky."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class _StallingDynamics(DynamicsAdapter):
    """Returns a perfectly good trajectory -- after the clock has moved on by ``stall_s``: a
    blocking call that eventually returned, a GC pause, a slow forward model."""

    def __init__(self, trajectory, clock, stall_s):
        self._trajectory, self._clock, self._stall_s = trajectory, clock, stall_s

    def predict_trajectory(self, state, action, horizon_s):
        self._clock.advance(self._stall_s)
        return self._trajectory


class _BrokenPerception(PerceptionAdapter):
    def get_world_state(self):
        raise RuntimeError("camera driver crashed")


# ----------------------------------------------------------------------------------------------
# 1. Software liveness: decision_within_deadline + DecisionWatchdog
# ----------------------------------------------------------------------------------------------

class DecisionDeadlineStallsPastOldChecks(unittest.TestCase):
    """A decision that stalls half a second inside dynamics prediction and then returns normally.
    Every pre-existing check evaluates the world as it was when the decision began -- so all of
    them pass, and the PERMIT arrives 0.5s late for a world that has moved on."""

    DEADLINE = _check("decision_within_deadline", max_decision_latency_s=0.125)

    def _stalled_gate(self, schema_dict, stall_s):
        clock = FakeClock()
        dyn = _StallingDynamics(fixtures.straight_line_trajectory(), clock, stall_s)
        return _gate(_golden_state(), schema_dict, dynamics=dyn, clock=clock)

    def test_old_checks_permit_a_half_second_stall(self):
        decision = self._stalled_gate(SCHEMA_DICT, 0.5).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)

    def test_new_check_blocks_the_half_second_stall(self):
        decision = self._stalled_gate(_with_checks(SCHEMA_DICT, "grasp", self.DEADLINE), 0.5).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertEqual(_failed_names(decision), ["decision_within_deadline"])

    def test_boundary_exactly_at_deadline_permits(self):
        decision = self._stalled_gate(_with_checks(SCHEMA_DICT, "grasp", self.DEADLINE), 0.125).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)

    def test_boundary_just_past_deadline_blocks(self):
        decision = self._stalled_gate(_with_checks(SCHEMA_DICT, "grasp", self.DEADLINE), 0.125 + 2 ** -20).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_no_context_fails_closed(self):
        r = pc.decision_within_deadline(None, None, None)
        self.assertFalse(r.satisfied)

    def test_nan_or_none_start_time_fails_closed(self):
        for started in (math.nan, None, "0"):
            ctx = pc.CheckContext(decision_started_at=started, clock=lambda: 0.0)
            self.assertFalse(pc.decision_within_deadline(None, None, None, context=ctx).satisfied, started)

    def test_nan_clock_reading_fails_closed(self):
        self.assertFalse(pc.decision_within_deadline(None, None, None, context=_ctx(0.0, math.nan)).satisfied)

    def test_clock_running_backwards_fails_closed(self):
        self.assertFalse(pc.decision_within_deadline(None, None, None, context=_ctx(10.0, 9.0)).satisfied)

    def test_nan_or_nonpositive_deadline_fails_closed(self):
        for deadline in (math.nan, -1.0, math.inf):
            r = pc.decision_within_deadline(None, None, None, max_decision_latency_s=deadline, context=_ctx(0.0, 0.01))
            self.assertFalse(r.satisfied, deadline)

    def test_yaml_cannot_forge_the_context(self):
        """A config that tries to pass its own 'context' (a fabricated start time) is overridden
        by the engine's."""
        forged = _check("decision_within_deadline", max_decision_latency_s=0.125,
                        context=pc.CheckContext(decision_started_at=10 ** 9, clock=lambda: 10 ** 9))
        clock = FakeClock()
        gate = _gate(_golden_state(), _with_checks(SCHEMA_DICT, "grasp", forged),
                     dynamics=_StallingDynamics(fixtures.straight_line_trajectory(), clock, 0.5), clock=clock)
        self.assertEqual(gate.gate(fixtures.grasp_action()).verdict, DecisionVerdict.BLOCK)


class WatchdogFreezesWhenTheCheckerStopsRunning(unittest.TestCase):
    """The failure class no check can catch: the checker hung or crashed, so nothing BLOCKs --
    nothing runs at all. Without a watchdog, the actuator side's only artifact is the last Decision,
    which says PERMIT and never expires."""

    def _permitting_gate(self, clock, watchdog=None):
        return _gate(_golden_state(), clock=clock, watchdog=watchdog)

    def test_old_engine_last_permit_never_expires(self):
        clock = FakeClock()
        decision = self._permitting_gate(clock).gate(fixtures.grasp_action())
        clock.advance(3600.0)  # an hour with no further gate() call: hung, crashed, or just not called
        # Nothing in the pre-watchdog API invalidates it -- a loop executing "the latest decision"
        # keeps moving the robot on it.
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)
        self.assertEqual(decision.action, fixtures.grasp_action())

    def test_watchdog_releases_fresh_permit_then_freezes_after_deadline(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.25, clock=clock)
        self._permitting_gate(clock, wd).gate(fixtures.grasp_action())
        self.assertEqual(wd.command().action_type, "grasp")
        clock.advance(0.25)  # boundary: exactly at the deadline is still fresh
        self.assertEqual(wd.command().action_type, "grasp")
        clock.advance(2 ** -20)
        self.assertEqual(wd.command().action_type, "freeze")
        self.assertFalse(wd.is_fresh())

    def test_gate_uses_the_watchdogs_clock_by_default(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.25, clock=clock)
        gate = _gate(_golden_state(), watchdog=wd)  # no explicit clock
        gate.gate(fixtures.grasp_action())
        clock.advance(1.0)
        self.assertEqual(wd.command().action_type, "freeze")

    def test_perception_crash_revokes_immediately_not_after_deadline(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        self._permitting_gate(clock, wd).gate(fixtures.grasp_action())
        with self.assertRaises(PerceptionFailure):
            _gate(None, perception=_BrokenPerception(), clock=clock, watchdog=wd).gate(fixtures.grasp_action())
        self.assertEqual(wd.command().action_type, "freeze")
        self.assertIn("revoked", wd.last_reason)

    def test_slow_permit_arrives_already_expired(self):
        """Stamped with when the decision *began*: a PERMIT that took longer than the watchdog's
        deadline is never released, even with no deadline check in the schema."""
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.2, clock=clock)
        gate = _gate(_golden_state(), dynamics=_StallingDynamics(fixtures.straight_line_trajectory(), clock, 0.3),
                     clock=clock, watchdog=wd)
        self.assertEqual(gate.gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)
        self.assertEqual(wd.command().action_type, "freeze")

    def test_block_takes_effect_immediately(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        self._permitting_gate(clock, wd).gate(fixtures.grasp_action())
        _gate(fixtures.base_world_state(objects=()), clock=clock, watchdog=wd).gate(fixtures.grasp_action())
        self.assertEqual(wd.command().action_type, "freeze")

    def test_no_decision_yet_freezes(self):
        self.assertEqual(DecisionWatchdog(FreezeInPlaceFallback()).command().action_type, "freeze")

    def test_clock_running_backwards_freezes(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        self._permitting_gate(clock, wd).gate(fixtures.grasp_action())
        clock.advance(-1.0)
        self.assertEqual(wd.command().action_type, "freeze")

    def test_nan_decision_time_freezes(self):
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=10.0, clock=clock)
        decision = self._permitting_gate(clock).gate(fixtures.grasp_action())
        wd.feed(decision, _golden_state(), decided_at=math.nan)
        self.assertEqual(wd.command().action_type, "freeze")

    def test_invalid_deadline_is_refused_at_construction(self):
        for bad in (math.nan, math.inf, 0.0, -1.0, "0.2", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=bad)

    def test_broken_fallback_still_yields_a_freeze(self):
        class _Broken(FreezeInPlaceFallback):
            def execute(self, state):
                raise RuntimeError("fallback controller down")

        self.assertEqual(DecisionWatchdog(_Broken()).command().action_type, "freeze")

    def test_monitor_thread_pushes_a_stop_while_gate_is_hung(self):
        """Real threads, real time: gate() blocks forever inside dynamics on its second call. The
        monitor thread must fire on_expire on its own -- nothing is polling command()."""
        release, fired = threading.Event(), threading.Event()
        calls = []

        class _HangsOnSecondCall(DynamicsAdapter):
            def predict_trajectory(self, state, action, horizon_s):
                calls.append(1)
                if len(calls) > 1:
                    release.wait(5.0)
                return fixtures.straight_line_trajectory()

        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.05)
        gate = _gate(_golden_state(), dynamics=_HangsOnSecondCall(), watchdog=wd)
        self.assertEqual(gate.gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)
        received = []
        wd.start_monitor(lambda action, reason: (received.append(action), fired.set()), poll_interval_s=0.005)
        hung = threading.Thread(target=gate.gate, args=(fixtures.grasp_action(),), daemon=True)
        hung.start()
        try:
            self.assertTrue(fired.wait(2.0), "monitor never fired while gate() was hung")
            self.assertEqual(received[0].action_type, "freeze")
            self.assertEqual(wd.command().action_type, "freeze")
        finally:
            release.set()
            hung.join(5.0)
            wd.stop_monitor()


# ----------------------------------------------------------------------------------------------
# 2. Payload / grip force
# ----------------------------------------------------------------------------------------------

class PayloadAndGripForce(unittest.TestCase):
    CHECK = _check("payload_and_grip_force_within_limits", rated_payload_kg=3.0)
    NEW = _with_checks(SCHEMA_DICT, "grasp", CHECK)

    def _plain_object(self, mass_kg):
        """Confirmed, cleared, stable, harmless -- and NOT fragile (no crush cap)."""
        return replace(fixtures.confirmed_object(mass_kg=mass_kg), hazard_tags=frozenset())

    def _run(self, schema, obj, grip_force_n, robot=None):
        state = fixtures.base_world_state(objects=(obj,))
        if robot is not None:
            state = replace(state, robot=robot)
        return _gate(state, schema).gate(fixtures.grasp_action(grip_force_n=grip_force_n))

    def _direct(self, obj, grip, robot=None, **kwargs):
        state = fixtures.base_world_state(objects=(obj,))
        if robot is not None:
            state = replace(state, robot=robot)
        params = {"object_id": obj.object_id}
        if grip is not ...:
            params["grip_force_n"] = grip
        kwargs.setdefault("rated_payload_kg", 3.0)
        return pc.payload_and_grip_force_within_limits(state, Action("grasp", params), fixtures.straight_line_trajectory(), **kwargs)

    # (a) -> (b): three concrete scenarios every old check permits
    def test_crushing_grip_on_fragile_object_gets_past_old_checks_and_is_blocked(self):
        obj = fixtures.confirmed_object()  # fragile 0.05kg cube
        self.assertEqual(self._run(SCHEMA_DICT, obj, 60.0).verdict, DecisionVerdict.PERMIT)
        blocked = self._run(self.NEW, obj, 60.0)
        self.assertEqual(blocked.verdict, DecisionVerdict.BLOCK)
        self.assertEqual(_failed_names(blocked), ["payload_and_grip_force_within_limits"])

    def test_over_robots_own_rated_payload_gets_past_old_budget_and_is_blocked(self):
        """1.5kg is under the config's flat 3kg budget -- but this robot is rated for 1kg."""
        obj, robot = self._plain_object(1.5), replace(fixtures.robot_state(), rated_payload_kg=1.0)
        self.assertEqual(self._run(SCHEMA_DICT, obj, 55.0, robot).verdict, DecisionVerdict.PERMIT)
        blocked = self._run(self.NEW, obj, 55.0, robot)
        self.assertEqual(_failed_names(blocked), ["payload_and_grip_force_within_limits"])
        self.assertIn("rated payload", blocked.precondition_results[-1].reason)

    def test_grip_too_weak_to_hold_gets_past_old_checks_and_is_blocked(self):
        obj = self._plain_object(1.5)  # needs ~49N at mu=0.3, 2 contacts, safety factor 2
        self.assertEqual(self._run(SCHEMA_DICT, obj, 10.0).verdict, DecisionVerdict.PERMIT)
        blocked = self._run(self.NEW, obj, 10.0)
        self.assertEqual(_failed_names(blocked), ["payload_and_grip_force_within_limits"])
        self.assertIn("slip", blocked.precondition_results[-1].reason)

    def test_control_safe_grip_on_fragile_cube_permits(self):
        self.assertEqual(self._run(self.NEW, fixtures.confirmed_object(), 5.0).verdict, DecisionVerdict.PERMIT)

    def test_object_specific_crush_limit_lowers_the_cap(self):
        egg = replace(fixtures.confirmed_object(), max_safe_grip_force_n=3.0)
        self.assertFalse(self._direct(egg, 5.0).satisfied)
        self.assertTrue(self._direct(replace(egg, max_safe_grip_force_n=5.0), 5.0).satisfied)

    # fail closed
    def test_missing_or_malformed_grip_force_fails_closed(self):
        obj = fixtures.confirmed_object()
        for grip in (..., None, math.nan, math.inf, -math.inf, -5.0, 0.0, True, "5", (5.0,)):
            self.assertFalse(self._direct(obj, grip).satisfied, repr(grip))

    def test_missing_or_malformed_mass_fails_closed(self):
        for mass in (None, math.nan, math.inf, -0.1, "0.05"):
            obj = replace(fixtures.confirmed_object(), estimated_mass_kg=mass)
            self.assertFalse(self._direct(obj, 5.0).satisfied, repr(mass))

    def test_no_payload_rating_anywhere_fails_closed(self):
        self.assertFalse(self._direct(fixtures.confirmed_object(), 5.0, rated_payload_kg=None).satisfied)

    def test_nan_payload_rating_fails_closed_from_config_or_robot(self):
        obj = fixtures.confirmed_object()
        self.assertFalse(self._direct(obj, 5.0, rated_payload_kg=math.nan).satisfied)
        nan_robot = replace(fixtures.robot_state(), rated_payload_kg=math.nan)
        self.assertFalse(self._direct(obj, 5.0, robot=nan_robot).satisfied)

    def test_unknown_hazard_class_fails_closed(self):
        obj = replace(fixtures.confirmed_object(), hazard_tags=frozenset({HazardTag.UNKNOWN}))
        self.assertFalse(self._direct(obj, 5.0).satisfied)

    def test_nan_caps_and_model_parameters_fail_closed(self):
        obj = fixtures.confirmed_object()
        for kwargs in ({"max_grip_force_n": math.nan}, {"max_grip_force_fragile_n": math.nan},
                       {"friction_coefficient": math.nan}, {"friction_coefficient": 0.0}, {"contact_count": 0}):
            self.assertFalse(self._direct(obj, 5.0, **kwargs).satisfied, kwargs)
        self.assertFalse(self._direct(replace(obj, max_safe_grip_force_n=math.nan), 5.0).satisfied)

    def test_absent_object_fails_closed(self):
        state = fixtures.base_world_state(objects=())
        r = pc.payload_and_grip_force_within_limits(state, fixtures.grasp_action(), fixtures.straight_line_trajectory(), rated_payload_kg=3.0)
        self.assertFalse(r.satisfied)

    # boundaries
    def test_boundary_grip_exactly_at_fragile_cap_permits_and_just_over_blocks(self):
        obj = fixtures.confirmed_object()
        self.assertTrue(self._direct(obj, 15.0).satisfied)
        self.assertFalse(self._direct(obj, math.nextafter(15.0, math.inf)).satisfied)

    def test_boundary_mass_exactly_at_rated_payload_permits(self):
        obj = self._plain_object(1.0)
        robot = replace(fixtures.robot_state(), rated_payload_kg=1.0)
        self.assertTrue(self._direct(obj, 60.0, robot=robot).satisfied)
        self.assertFalse(self._direct(replace(obj, estimated_mass_kg=1.0 + 2 ** -20), 60.0, robot=robot).satisfied)

    def test_boundary_grip_exactly_at_minimum_hold_permits(self):
        obj = self._plain_object(1.0)
        min_hold = 1.0 * pc.GRAVITY_MPS2 * 2.0 / (0.3 * 2)
        self.assertTrue(self._direct(obj, min_hold).satisfied)
        self.assertFalse(self._direct(obj, min_hold * (1 - 1e-9)).satisfied)


# ----------------------------------------------------------------------------------------------
# 3. Stability / tip-over margin
# ----------------------------------------------------------------------------------------------

class StabilityMargin(unittest.TestCase):
    """A quadruped whose center of mass is statically 8cm inside the front edge of its support
    polygon -- comfortably past balance_margin_maintained's 3cm -- but moving toward that edge at
    0.6m/s. Its capture point is ~7cm OUTSIDE the polygon: it cannot stop without stepping."""

    NEW = _with_checks(NAVIGATE_SCHEMA_DICT, "navigate", _check("stability_margin_maintained", min_margin_m=0.03))

    def _lunging_robot(self, velocity=(0.6, 0.0, 0.0)):
        return fixtures.quadruped_robot_state(base_position=(0.22, 0.0, 0.6), com_velocity=velocity)

    def _nav(self, schema, robot):
        state = fixtures.quadruped_world_state(robot=robot)
        return _gate(state, schema, trajectory=_base_footprint_trajectory(robot)).gate(fixtures.navigate_action())

    def _direct(self, robot, fn=pc.stability_margin_maintained, **kwargs):
        traj = PredictedTrajectory(points=(TrajectoryPoint(t=0.0, robot=robot, swept_volume_center=(0, 0, 0.6), swept_volume_radius_m=0.35),))
        return fn(None, None, traj, **kwargs)

    def test_lunge_toward_edge_gets_past_old_balance_check_and_is_blocked(self):
        self.assertEqual(self._nav(NAVIGATE_SCHEMA_DICT, self._lunging_robot()).verdict, DecisionVerdict.PERMIT)
        blocked = self._nav(self.NEW, self._lunging_robot())
        self.assertEqual(blocked.verdict, DecisionVerdict.BLOCK)
        self.assertEqual(_failed_names(blocked), ["stability_margin_maintained"])
        self.assertIn("capture point", blocked.precondition_results[-1].reason)

    def test_control_same_robot_standing_still_permits(self):
        self.assertEqual(self._nav(self.NEW, self._lunging_robot(velocity=(0.0, 0.0, 0.0))).verdict, DecisionVerdict.PERMIT)

    def test_legacy_name_still_registered_and_behaves_exactly_as_before(self):
        self.assertIs(pc.REGISTRY["balance_margin_maintained"], pc.balance_margin_maintained)
        self.assertTrue(self._direct(self._lunging_robot(), pc.balance_margin_maintained).satisfied)  # static-only, as before
        tipping = self._direct(fixtures.tipping_quadruped_robot_state(), pc.balance_margin_maintained)
        self.assertFalse(tipping.satisfied)
        self.assertEqual(tipping.name, "balance_margin_maintained")
        # and it never required a velocity
        self.assertTrue(self._direct(replace(self._lunging_robot(), center_of_mass_velocity=None), pc.balance_margin_maintained).satisfied)

    def test_one_step_allowance_lets_a_stepping_platform_through(self):
        self.assertTrue(self._direct(self._lunging_robot(), capture_step_allowance_m=0.1).satisfied)
        self.assertFalse(self._direct(self._lunging_robot(), capture_step_allowance_m=0.05).satisfied)

    def test_missing_velocity_fails_closed_unless_explicitly_static(self):
        still = replace(fixtures.quadruped_robot_state(), center_of_mass_velocity=None)
        self.assertFalse(self._direct(still).satisfied)
        self.assertTrue(self._direct(still, require_com_velocity=False).satisfied)

    def test_nan_inputs_fail_closed(self):
        base = fixtures.quadruped_robot_state()
        for robot in (replace(base, center_of_mass_velocity=(math.nan, 0.0, 0.0)),
                      replace(base, center_of_mass=(0.0, 0.0, math.nan)),
                      replace(base, center_of_mass=(math.nan, 0.0, 0.6)),
                      replace(base, center_of_mass_velocity=(0.1,))):
            self.assertFalse(self._direct(robot).satisfied, robot)
        for kwargs in ({"min_margin_m": math.nan}, {"capture_step_allowance_m": math.nan},
                       {"gravity_mps2": 0.0}, {"gravity_mps2": math.nan}, {"ground_height_m": 0.6}):
            self.assertFalse(self._direct(base, **kwargs).satisfied, kwargs)

    def test_missing_balance_state_or_trajectory_fails_closed(self):
        self.assertFalse(self._direct(fixtures.robot_state()).satisfied)
        self.assertFalse(pc.stability_margin_maintained(None, None, PredictedTrajectory()).satisfied)
        point = TrajectoryPoint(t=0.0, robot=None, swept_volume_center=(0, 0, 0), swept_volume_radius_m=0.1)
        self.assertFalse(pc.stability_margin_maintained(None, None, PredictedTrajectory(points=(point,))).satisfied)

    def test_boundary_static_margin_exactly_at_minimum_permits(self):
        square = ((0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5))
        robot = fixtures.quadruped_robot_state(base_position=(0.25, 0.0, 0.6), support_polygon=square)
        self.assertTrue(self._direct(robot, min_margin_m=0.25).satisfied)
        self.assertFalse(self._direct(robot, min_margin_m=0.25 + 2 ** -20).satisfied)


# ----------------------------------------------------------------------------------------------
# 4. Sensor staleness
# ----------------------------------------------------------------------------------------------

class SensorStaleness(unittest.TestCase):
    NEW = _with_checks(SCHEMA_DICT, "grasp", _check("sensor_data_fresh", max_sensor_age_s=0.2))

    def test_two_second_old_frame_gets_past_old_checks_and_is_blocked(self):
        state = _golden_state(sensor_age_s=2.0)
        # WorldState.timestamp is fresh -- the state was *assembled* just now -- which is exactly
        # why it can't be what catches this.
        self.assertLess(abs(state.timestamp - state.sensor_timestamp - 2.0), 0.5)
        self.assertEqual(_gate(state).gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)
        blocked = _gate(state, self.NEW).gate(fixtures.grasp_action())
        self.assertEqual(_failed_names(blocked), ["sensor_data_fresh"])

    def test_control_fresh_frame_permits(self):
        self.assertEqual(_gate(_golden_state(), self.NEW).gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)

    def _direct(self, ts, **kwargs):
        return pc.sensor_data_fresh(WorldState(sensor_timestamp=ts), None, None, **kwargs)

    def test_unreported_or_malformed_timestamp_fails_closed(self):
        for ts in (None, math.nan, math.inf, -math.inf, "100.0", True):
            self.assertFalse(self._direct(ts, now=100.0).satisfied, repr(ts))

    def test_default_worldstate_has_no_sensor_timestamp(self):
        """Nothing defaults to looking fresh on an adapter's behalf."""
        self.assertFalse(pc.sensor_data_fresh(WorldState(), None, None).satisfied)

    def test_future_timestamp_fails_closed_unless_skew_tolerated(self):
        self.assertFalse(self._direct(101.0, now=100.0).satisfied)
        self.assertFalse(self._direct(100.125, now=100.0, max_future_skew_s=0.0625).satisfied)
        self.assertTrue(self._direct(100.0625, now=100.0, max_future_skew_s=0.0625).satisfied)

    def test_nan_limits_or_clock_fail_closed(self):
        self.assertFalse(self._direct(100.0, now=100.0, max_sensor_age_s=math.nan).satisfied)
        self.assertFalse(self._direct(100.0, now=math.nan).satisfied)
        self.assertFalse(self._direct(100.0, now=100.0, max_future_skew_s=math.nan).satisfied)

    def test_boundary_exactly_at_max_age_permits_just_past_blocks(self):
        self.assertTrue(self._direct(100.0, now=100.25, max_sensor_age_s=0.25).satisfied)
        self.assertFalse(self._direct(100.0, now=100.25 + 2 ** -20, max_sensor_age_s=0.25).satisfied)


# ----------------------------------------------------------------------------------------------
# 5. Sensor coverage / occlusion
# ----------------------------------------------------------------------------------------------

class SweptPathCoverage(unittest.TestCase):
    """The camera sees its own half of the table perfectly (visibility 1.0, no agents detected)
    and nothing of the half the arm reaches into -- a cabinet occludes it."""

    NEW = _with_checks(SCHEMA_DICT, "grasp", _check("swept_path_observed", margin_m=0.15))
    LEFT_HALF_ONLY = (ObservedRegion(min_corner=(-1.0, -1.0, -0.5), max_corner=(0.3, 1.0, 1.5)),)

    def test_occluded_path_gets_past_old_checks_and_is_blocked(self):
        state = _golden_state(observed_regions=self.LEFT_HALF_ONLY)
        self.assertEqual(_gate(state).gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)
        blocked = _gate(state, self.NEW).gate(fixtures.grasp_action())
        self.assertEqual(_failed_names(blocked), ["swept_path_observed"])

    def test_control_observed_path_permits(self):
        self.assertEqual(_gate(_golden_state(), self.NEW).gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)

    def test_unobserved_is_distinct_from_observed_but_uncertain(self):
        both = _with_checks(self.NEW, "grasp")
        observed_foggy = _gate(_golden_state(visibility=0.2), both).gate(fixtures.grasp_action())
        self.assertEqual(_failed_names(observed_foggy), ["visibility_above_threshold"])
        unobserved_clear = _gate(_golden_state(observed_regions=self.LEFT_HALF_ONLY), both).gate(fixtures.grasp_action())
        self.assertEqual(_failed_names(unobserved_clear), ["swept_path_observed"])

    def _direct(self, regions, trajectory=None, **kwargs):
        state = WorldState(observed_regions=regions)
        return pc.swept_path_observed(state, None, trajectory or fixtures.straight_line_trajectory(), **kwargs)

    def test_unreported_empty_or_malformed_coverage_fails_closed(self):
        good = fixtures.WORKSPACE_OBSERVED[0]
        for regions in (None, (), [], "everywhere", ((-1.0, -1.0, -0.5, 2.0, 1.0, 1.5),),
                        (ObservedRegion(min_corner=(math.nan, -1.0, -0.5), max_corner=(2.0, 1.0, 1.5)),),
                        (ObservedRegion(min_corner=(-math.inf, -1.0, -0.5), max_corner=(math.inf, 1.0, 1.5)),),
                        (ObservedRegion(min_corner=good.max_corner, max_corner=good.min_corner),),  # inverted
                        (ObservedRegion(min_corner=(-1.0, -1.0), max_corner=(2.0, 1.0)),)):  # 2-D
            self.assertFalse(self._direct(regions).satisfied, repr(regions))

    def test_nan_swept_geometry_or_margin_fails_closed(self):
        point = TrajectoryPoint(t=0.0, robot=None, swept_volume_center=(math.nan, 0.0, 0.3), swept_volume_radius_m=0.05)
        self.assertFalse(self._direct(fixtures.WORKSPACE_OBSERVED, PredictedTrajectory(points=(point,))).satisfied)
        self.assertFalse(self._direct(fixtures.WORKSPACE_OBSERVED, margin_m=math.nan).satisfied)
        self.assertFalse(self._direct(fixtures.WORKSPACE_OBSERVED, PredictedTrajectory()).satisfied)

    def test_straddling_two_boxes_counts_as_unobserved(self):
        """Conservative by design: the union of two boxes is not assumed to be seamless."""
        split = (ObservedRegion(min_corner=(-1.0, -1.0, -0.5), max_corner=(0.5, 1.0, 1.5)),
                 ObservedRegion(min_corner=(0.5, -1.0, -0.5), max_corner=(2.0, 1.0, 1.5)))
        self.assertFalse(self._direct(split).satisfied)

    def test_boundary_sphere_exactly_touching_the_box_face_permits(self):
        point = TrajectoryPoint(t=0.0, robot=None, swept_volume_center=(0.5, 0.0, 0.25), swept_volume_radius_m=0.125)
        traj = PredictedTrajectory(points=(point,))
        touching = (ObservedRegion(min_corner=(-1.0, -1.0, -1.0), max_corner=(1.0, 1.0, 0.5)),)
        self.assertTrue(self._direct(touching, traj, margin_m=0.125).satisfied)
        self.assertFalse(self._direct(touching, traj, margin_m=0.125 + 2 ** -20).satisfied)


# ----------------------------------------------------------------------------------------------
# 6. Command integrity
# ----------------------------------------------------------------------------------------------

class _RewritesActionAfterPredicting(DynamicsAdapter):
    """Predicts the trajectory for the action as proposed, then -- simulating a planner thread that
    still holds the params dict, or a compromised adapter -- rewrites it: a crushing grip force and
    a different target. Every check after this point sees a trajectory for the safe command."""

    def predict_trajectory(self, state, action, horizon_s):
        trajectory = fixtures.straight_line_trajectory()
        action.params["grip_force_n"] = 200.0
        action.params["target_position"] = (0.55, 0.0, 0.05)  # onto the person standing there
        return trajectory


class CommandIntegrity(unittest.TestCase):
    NEW = _with_checks(SCHEMA_DICT, "grasp", _check("command_integrity_verified"))

    def test_rewrite_after_permit_gets_past_old_engine_and_is_refused_at_execution(self):
        decision = _gate(_golden_state()).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)
        self.assertTrue(verify_decision_action(decision))
        decision.action.params["grip_force_n"] = 200.0  # Action is frozen; its params dict is not
        # Old: the Decision still says PERMIT and now carries the rewritten command.
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)
        self.assertEqual(decision.action.params["grip_force_n"], 200.0)
        # New: execution-time verification refuses it, and so does the watchdog.
        self.assertFalse(verify_decision_action(decision))
        clock = FakeClock()
        wd = DecisionWatchdog(FreezeInPlaceFallback(), clock=clock)
        wd.feed(decision, _golden_state(), decided_at=clock())
        self.assertEqual(wd.command().action_type, "freeze")
        self.assertIn("integrity", wd.last_reason)

    def test_rewrite_during_checking_gets_past_old_checks_and_is_blocked(self):
        permitted = _gate(_golden_state(), dynamics=_RewritesActionAfterPredicting()).gate(fixtures.grasp_action())
        self.assertEqual(permitted.verdict, DecisionVerdict.PERMIT)
        self.assertEqual(permitted.action.params["grip_force_n"], 200.0)  # what the actuators would get
        # Even with the check unwired, the digest bound at gate() start refuses it at execution:
        self.assertFalse(verify_decision_action(permitted))
        blocked = _gate(_golden_state(), self.NEW, dynamics=_RewritesActionAfterPredicting()).gate(fixtures.grasp_action())
        self.assertEqual(blocked.verdict, DecisionVerdict.BLOCK)
        self.assertEqual(_failed_names(blocked), ["command_integrity_verified"])

    def test_unencodable_param_gets_past_old_checks_and_is_blocked(self):
        class LazyTarget:
            """Evaluates to a different position every time something reads it."""

            def __iter__(self):
                return iter((0.5, 0.0, 0.05))

        action = Action("grasp", {**fixtures.grasp_action().params, "target_position": LazyTarget()})
        permitted = _gate(_golden_state()).gate(action)
        self.assertEqual(permitted.verdict, DecisionVerdict.PERMIT)
        self.assertIsNone(permitted.action_digest)
        self.assertFalse(verify_decision_action(permitted))
        self.assertEqual(_failed_names(_gate(_golden_state(), self.NEW).gate(action)), ["command_integrity_verified"])

    def test_control_untouched_command_permits_and_verifies(self):
        decision = _gate(_golden_state(), self.NEW).gate(fixtures.grasp_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT)
        self.assertTrue(verify_decision_action(decision))

    def test_no_context_or_no_start_digest_fails_closed(self):
        action = fixtures.grasp_action()
        self.assertFalse(pc.command_integrity_verified(None, action, None).satisfied)
        self.assertFalse(pc.command_integrity_verified(None, action, None, context=_ctx(digest=None)).satisfied)

    def test_hostile_encodings_fail_closed_not_crash(self):
        loop = []
        loop.append(loop)

        class TupleSubclass(tuple):
            pass

        for bad in (loop, TupleSubclass((0.5, 0.0, 0.05)), bytearray(b"x"), object(), lambda: 0):
            action = Action("grasp", {"object_id": "cube_2", "target_position": bad})
            self.assertIsNone(integrity.try_action_digest(action), repr(bad))
            self.assertFalse(pc.command_integrity_verified(None, action, None, context=_ctx(digest="sha256:0")).satisfied)

    def test_digest_is_bit_exact(self):
        d = lambda params: integrity.action_digest(Action("grasp", params))  # noqa: E731
        self.assertNotEqual(d({"x": 0.0}), d({"x": -0.0}))
        self.assertNotEqual(d({"x": 1}), d({"x": 1.0}))
        self.assertNotEqual(d({"x": (1, 2)}), d({"x": [1, 2]}))
        self.assertNotEqual(d({"x": 5.0}), d({"x": math.nextafter(5.0, 6.0)}))
        self.assertEqual(d({"a": 1, "b": math.nan}), d({"b": math.nan, "a": 1}))  # key order irrelevant, NaN stable

    def test_seal_verified_whenever_present(self):
        sealed = seal_action(fixtures.grasp_action())
        ctx = lambda a: _ctx(digest=integrity.action_digest(a))  # noqa: E731
        self.assertTrue(pc.command_integrity_verified(None, sealed, None, context=ctx(sealed)).satisfied)
        tampered = Action("grasp", {**sealed.params, "grip_force_n": 200.0})
        r = pc.command_integrity_verified(None, tampered, None, context=ctx(tampered))
        self.assertFalse(r.satisfied)
        self.assertIn("seal", r.reason)

    def test_required_seal_missing_fails_closed(self):
        action = fixtures.grasp_action()
        ctx = _ctx(digest=integrity.action_digest(action))
        self.assertFalse(pc.command_integrity_verified(None, action, None, require_seal=True, context=ctx).satisfied)

    def test_hmac_seal_needs_the_right_key(self):
        env = "SAFETY_HARNESS_TEST_COMMAND_KEY"
        sealed = seal_action(fixtures.grasp_action(), key=b"proposer-secret")
        ctx = _ctx(digest=integrity.action_digest(sealed))
        old = os.environ.pop(env, None)
        try:
            unset = pc.command_integrity_verified(None, sealed, None, require_seal=True, hmac_key_env=env, context=ctx)
            self.assertFalse(unset.satisfied, "an unset key must fail closed, not verify unkeyed")
            os.environ[env] = "someone-else"
            self.assertFalse(pc.command_integrity_verified(None, sealed, None, hmac_key_env=env, context=ctx).satisfied)
            os.environ[env] = "proposer-secret"
            self.assertTrue(pc.command_integrity_verified(None, sealed, None, require_seal=True, hmac_key_env=env, context=ctx).satisfied)
            unkeyed = seal_action(fixtures.grasp_action())
            self.assertFalse(pc.command_integrity_verified(
                None, unkeyed, None, hmac_key_env=env, context=_ctx(digest=integrity.action_digest(unkeyed))).satisfied)
        finally:
            os.environ.pop(env, None)
            if old is not None:
                os.environ[env] = old

    def test_verify_helpers_never_raise_on_garbage(self):
        self.assertFalse(verify_action_seal(Action("grasp", {"command_digest": 7})))
        self.assertFalse(verify_decision_action(object()))
        self.assertFalse(integrity.digests_match(None, None))
        self.assertFalse(integrity.digests_match("", ""))


# ----------------------------------------------------------------------------------------------
# 7. Config integrity
# ----------------------------------------------------------------------------------------------

def _grasp_check_index(schema_dict, name):
    return next(i for i, c in enumerate(schema_dict["action_types"]["grasp"]["checks"]) if c["name"] == name)


class ConfigIntegrity(unittest.TestCase):
    """A validated config with a 3kg force budget -- and a 5kg object."""

    NEW = _with_checks(SCHEMA_DICT, "grasp", _check("config_integrity_verified"))
    HEAVY = fixtures.confirmed_object(mass_kg=5.0)

    def _tampered(self, schema_dict, budget=300.0):
        d = copy.deepcopy(schema_dict)
        d["action_types"]["grasp"]["checks"][_grasp_check_index(d, "mass_within_force_budget")]["kwargs"]["force_budget_kg"] = budget
        return d

    def _pinned(self, schema_dict, **kwargs):
        good = ActionSchemaRegistry.from_dict(schema_dict, hmac_key=kwargs.get("hmac_key")).config_digest()
        return ActionSchemaRegistry.from_dict(schema_dict, expected_digest=good, **kwargs), good

    def _heavy_state(self):
        return fixtures.base_world_state(objects=(self.HEAVY,))

    def test_control_validated_config_blocks_the_heavy_object(self):
        self.assertEqual(_gate(self._heavy_state()).gate(fixtures.grasp_action()).verdict, DecisionVerdict.BLOCK)

    def test_tampered_config_file_gets_past_old_checks_and_is_refused_at_load(self):
        self.assertEqual(_gate(self._heavy_state(), self._tampered(SCHEMA_DICT)).gate(fixtures.grasp_action()).verdict,
                         DecisionVerdict.PERMIT)
        _, good = self._pinned(self.NEW)
        with self.assertRaises(ConfigIntegrityError):
            ActionSchemaRegistry.from_dict(self._tampered(self.NEW), expected_digest=good)

    def test_tampered_yaml_on_disk_is_refused_at_load(self):
        with open(EXAMPLE_SCHEMA_PATH) as f:
            text = f.read()
        self.assertIn("force_budget_kg: 3.0", text)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "schema.yaml")
            with open(path, "w") as f:
                f.write(text.replace("force_budget_kg: 3.0", "force_budget_kg: 300.0"))
            with self.assertRaises(ConfigIntegrityError):
                ActionSchemaRegistry.from_yaml(path, expected_digest=integrity.read_digest_file(EXAMPLE_DIGEST_PATH))

    def test_runtime_tamper_after_load_gets_past_old_checks_and_is_blocked(self):
        old_registry = ActionSchemaRegistry.from_dict(SCHEMA_DICT)
        new_registry, _ = self._pinned(self.NEW)
        for registry in (old_registry, new_registry):
            checks = registry._schemas["grasp"].checks  # in-process tamper: a plugin, a debugger, memory corruption
            checks[_grasp_check_index(SCHEMA_DICT, "mass_within_force_budget")]["kwargs"]["force_budget_kg"] = 300.0
        self.assertEqual(_gate(self._heavy_state(), registry=old_registry).gate(fixtures.grasp_action()).verdict,
                         DecisionVerdict.PERMIT)
        blocked = _gate(self._heavy_state(), registry=new_registry).gate(fixtures.grasp_action())
        self.assertEqual(_failed_names(blocked), ["config_integrity_verified"])

    def test_registry_rebinding_gets_past_old_checks_and_is_blocked(self):
        new_registry, _ = self._pinned(self.NEW)
        original = pc.REGISTRY["mass_within_force_budget"]

        def mass_within_force_budget(state, action, trajectory, **kwargs):  # same name, always OK
            return pc.PreconditionResult("mass_within_force_budget", True, "rebound")

        pc.REGISTRY["mass_within_force_budget"] = mass_within_force_budget
        try:
            self.assertEqual(_gate(self._heavy_state()).gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)
            blocked = _gate(self._heavy_state(), registry=new_registry).gate(fixtures.grasp_action())
            self.assertEqual(_failed_names(blocked), ["config_integrity_verified"])
        finally:
            pc.REGISTRY["mass_within_force_budget"] = original

    def test_changed_code_default_is_detected(self):
        """iso15066_power_force_limiting runs with kwargs {} -- its thresholds live only in code."""
        new_registry, _ = self._pinned(self.NEW)
        fn = pc.iso15066_power_force_limiting
        saved = dict(fn.__kwdefaults__)
        fn.__kwdefaults__ = {**saved, "max_transient_force_n": 1e9}
        try:
            self.assertFalse(new_registry.verify_config())
        finally:
            fn.__kwdefaults__ = saved
        self.assertTrue(new_registry.verify_config())

    def test_caller_holding_the_raw_dict_can_no_longer_edit_a_live_config(self):
        raw = copy.deepcopy(SCHEMA_DICT)
        registry = ActionSchemaRegistry.from_dict(raw)
        before = registry.config_digest()
        raw["action_types"]["grasp"]["checks"][0]["kwargs"]["injected"] = True
        self.assertEqual(registry.config_digest(), before)

    def test_unpinned_config_fails_closed(self):
        blocked = _gate(_golden_state(), self.NEW).gate(fixtures.grasp_action())
        self.assertEqual(_failed_names(blocked), ["config_integrity_verified"])
        self.assertFalse(pc.config_integrity_verified(None, None, None).satisfied)
        self.assertFalse(pc.config_integrity_verified(None, None, None, context=_ctx()).satisfied)

    def test_control_pinned_config_permits(self):
        registry, _ = self._pinned(self.NEW)
        self.assertEqual(_gate(_golden_state(), registry=registry).gate(fixtures.grasp_action()).verdict, DecisionVerdict.PERMIT)

    def test_hmac_pin_requires_the_same_key_and_scheme(self):
        keyed, keyed_digest = self._pinned(self.NEW, hmac_key=b"release-signing-key")
        self.assertTrue(keyed_digest.startswith("hmac-sha256:"))
        self.assertTrue(keyed.verify_config())
        for kwargs in ({"hmac_key": b"wrong-key"}, {}):
            with self.assertRaises(ConfigIntegrityError, msg=kwargs):
                ActionSchemaRegistry.from_dict(self.NEW, expected_digest=keyed_digest, **kwargs)
        unkeyed_digest = ActionSchemaRegistry.from_dict(self.NEW).config_digest()
        with self.assertRaises(ConfigIntegrityError):
            ActionSchemaRegistry.from_dict(self.NEW, expected_digest=unkeyed_digest, hmac_key=b"release-signing-key")

    def test_garbage_pins_are_refused(self):
        for pin in ("", "sha256:", "not-a-digest", 12345):
            with self.assertRaises(ConfigIntegrityError, msg=repr(pin)):
                ActionSchemaRegistry.from_dict(self.NEW, expected_digest=pin)

    def test_unencodable_config_value_fails_closed(self):
        import datetime

        d = _with_checks(SCHEMA_DICT, "grasp", _check("config_integrity_verified"), _check("robot_state_confirmed", note=datetime.date(2026, 9, 27)))
        registry = ActionSchemaRegistry.from_dict(d)
        registry._expected_digest = "sha256:" + "0" * 64
        self.assertFalse(registry.verify_config())

    def test_boundary_digest_covers_parsed_config_not_bytes(self):
        """Comments, whitespace and key order don't change the digest; any change of *value* or
        *type* does -- even 3 -> 3.0, the same number: the pin is exact, so re-pin after any edit."""
        with open(EXAMPLE_SCHEMA_PATH) as f:
            text = f.read()
        base = ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH).config_digest()
        with tempfile.TemporaryDirectory() as tmp:
            def digest_of(t):
                path = os.path.join(tmp, "s.yaml")
                with open(path, "w") as f:
                    f.write(t)
                return ActionSchemaRegistry.from_yaml(path).config_digest()

            self.assertEqual(digest_of("# reviewed 2026-09-27\n" + text + "\n\n"), base)
            self.assertNotEqual(digest_of(text.replace("force_budget_kg: 3.0", "force_budget_kg: 3")), base)
            self.assertNotEqual(digest_of(text.replace("min_visibility: 0.5", "min_visibility: 0.49", 1)), base)

    def test_committed_example_digest_matches_the_example_config(self):
        """Config-drift tripwire: fails whenever example_action_schema.yaml -- or the code defaults
        of a check it wires -- changes without the pinned digest being regenerated after review:
            python -m safety_harness.pin configs/example_action_schema.yaml > configs/example_action_schema.yaml.sha256"""
        pinned = integrity.read_digest_file(EXAMPLE_DIGEST_PATH)
        self.assertEqual(ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH).config_digest(), pinned)
        self.assertTrue(ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH, expected_digest=pinned).verify_config())

    def test_pin_cli_prints_the_digest(self):
        import contextlib
        import io

        from safety_harness import pin

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(pin.main([EXAMPLE_SCHEMA_PATH]), 0)
        self.assertEqual(out.getvalue().strip(), ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH).config_digest())


# ----------------------------------------------------------------------------------------------
# 8. Vulnerable bystanders (child / crowd / ISO/TS 15066 body regions)
# ----------------------------------------------------------------------------------------------

def _agent(position, category=AgentCategory.ADULT, speed=1.5, stature=None, agent_id="person_1"):
    return TrackedAgent(agent_id=agent_id, pose=Pose(position=position), tracking_confidence=0.95,
                        time_since_confirmed_s=0.0, worst_case_speed_mps=speed, category=category, stature_m=stature)


class VulnerableBystanders(unittest.TestCase):
    NEW = _with_checks(SCHEMA_DICT, "grasp", _check("vulnerable_bystander_protected"))

    def _run(self, schema, *agents):
        return _gate(_golden_state(agents=agents), schema).gate(fixtures.grasp_action())

    def test_child_two_meters_away_gets_past_old_checks_and_is_blocked(self):
        child = _agent((0.5, 2.0, 0.05), AgentCategory.CHILD)
        self.assertEqual(self._run(SCHEMA_DICT, child).verdict, DecisionVerdict.PERMIT)
        blocked = self._run(self.NEW, child)
        self.assertEqual(_failed_names(blocked), ["vulnerable_bystander_protected"])

    def test_control_adult_at_same_spot_permits_baseline_unchanged(self):
        self.assertEqual(self._run(self.NEW, _agent((0.5, 2.0, 0.05))).verdict, DecisionVerdict.PERMIT)

    def test_unclassified_agent_is_treated_as_a_child(self):
        self.assertEqual(_failed_names(self._run(self.NEW, _agent((0.5, 2.0, 0.05), AgentCategory.UNKNOWN))),
                         ["vulnerable_bystander_protected"])
        self.assertEqual(TrackedAgent(agent_id="x", pose=Pose(position=(0, 0, 0))).category, AgentCategory.UNKNOWN)

    def test_animal_is_vulnerable(self):
        self.assertEqual(self._run(self.NEW, _agent((0.5, 2.0, 0.05), AgentCategory.ANIMAL)).verdict, DecisionVerdict.BLOCK)

    def test_two_adults_get_past_old_checks_and_are_blocked_as_a_crowd(self):
        a, b = _agent((0.5, 1.8, 0.05), agent_id="a"), _agent((0.5, -1.8, 0.05), agent_id="b")
        self.assertEqual(self._run(SCHEMA_DICT, a, b).verdict, DecisionVerdict.PERMIT)
        blocked = self._run(self.NEW, a, b)
        self.assertEqual(_failed_names(blocked), ["vulnerable_bystander_protected"])
        self.assertIn("crowd", blocked.precondition_results[-1].reason)
        self.assertEqual(self._run(self.NEW, a).verdict, DecisionVerdict.PERMIT, "one adult at 1.8m is fine")

    # ISO/TS 15066 body regions: an overhead reach at an adult's face height, in a PFL (contact-
    # permitted) cell. Unit-level against the old PFL and flat-margin checks.
    def _overhead_reach(self, z=1.6):
        points = (
            TrajectoryPoint(t=0.0, robot=None, swept_volume_center=(0.0, 0.0, z), swept_volume_radius_m=0.05),
            TrajectoryPoint(t=0.2, robot=None, swept_volume_center=(0.05, 0.0, z), swept_volume_radius_m=0.05),
        )
        return PredictedTrajectory(points=points, horizon_s=0.2)

    def _face_height_state(self, stature=1.75, z=1.6, category=AgentCategory.ADULT):
        return WorldState(agents=(_agent((0.0, 0.25, z), category, speed=0.15, stature=stature),))

    def test_face_height_contact_gets_past_old_pfl_and_is_blocked(self):
        state, traj = self._face_height_state(), self._overhead_reach()
        self.assertTrue(pc.iso15066_power_force_limiting(state, None, traj).satisfied)  # ~133N < flat 150N
        self.assertTrue(pc.swept_path_clear_of_agents(state, None, traj).satisfied)
        r = pc.vulnerable_bystander_protected(state, None, traj)
        self.assertFalse(r.satisfied)
        self.assertIn("face", r.reason)

    def test_same_contact_at_thigh_height_permits(self):
        r = pc.vulnerable_bystander_protected(self._face_height_state(z=0.6), None, self._overhead_reach(z=0.6))
        self.assertTrue(r.satisfied, r.reason)

    def test_unknown_or_nan_stature_assumes_the_face_is_in_reach(self):
        traj = self._overhead_reach(z=0.6)
        for stature in (None, math.nan, -1.0, 0.0, "tall"):
            r = pc.vulnerable_bystander_protected(self._face_height_state(stature=stature, z=0.6), None, traj)
            self.assertFalse(r.satisfied, repr(stature))

    def test_no_contact_permitted_with_a_child_even_where_an_adult_may_be_touched(self):
        state = self._face_height_state(z=0.6, category=AgentCategory.CHILD)
        r = pc.vulnerable_bystander_protected(state, None, self._overhead_reach(z=0.6), vulnerable_min_clearance_m=0.0)
        self.assertFalse(r.satisfied)
        self.assertIn("no contact", r.reason)

    def test_nan_positions_speeds_and_parameters_fail_closed(self):
        traj = fixtures.straight_line_trajectory()
        for agent in (_agent((math.nan, 5.0, 0.0)), _agent((5.0, 5.0, 0.0), speed=math.nan)):
            self.assertFalse(pc.vulnerable_bystander_protected(WorldState(agents=(agent,)), None, traj).satisfied, agent)
        far = WorldState(agents=(_agent((5.0, 5.0, 0.0)),))
        self.assertTrue(pc.vulnerable_bystander_protected(far, None, traj).satisfied)
        for kwargs in ({"transient_multiplier": math.nan}, {"effective_mass_kg": math.nan}, {"assumed_contact_time_s": 0.0},
                       {"adult_min_clearance_m": math.nan}, {"crowd_force_scale": math.nan}):
            self.assertFalse(pc.vulnerable_bystander_protected(self._face_height_state(z=0.6), None, self._overhead_reach(z=0.6), **kwargs).satisfied, kwargs)

    def test_nan_crowd_size_is_treated_as_a_crowd(self):
        a = WorldState(agents=(_agent((0.5, 1.8, 0.05)),))
        self.assertTrue(pc.vulnerable_bystander_protected(a, None, fixtures.straight_line_trajectory()).satisfied)
        self.assertFalse(pc.vulnerable_bystander_protected(a, None, fixtures.straight_line_trajectory(), crowd_size=math.nan).satisfied)

    def test_empty_trajectory_fails_closed(self):
        self.assertFalse(pc.vulnerable_bystander_protected(WorldState(), None, PredictedTrajectory()).satisfied)

    def test_boundary_exactly_at_required_clearance_permits(self):
        point = TrajectoryPoint(t=0.0, robot=None, swept_volume_center=(0.0, 0.0, 5.0), swept_volume_radius_m=0.25)
        traj = PredictedTrajectory(points=(point,))
        at = WorldState(agents=(_agent((0.5, 0.0, 5.0), speed=0.0, stature=1.75),))
        self.assertTrue(pc.vulnerable_bystander_protected(at, None, traj, adult_min_clearance_m=0.25).satisfied)
        self.assertFalse(pc.vulnerable_bystander_protected(at, None, traj, adult_min_clearance_m=0.25 + 2 ** -20).satisfied)

    def test_boundary_force_exactly_at_body_region_limit_permits(self):
        # Face (65N, no transient multiplier) at 1.6m on a 1.75m adult; closing speed 1.0m/s, contact
        # time 1.0s -> force == effective_mass_kg exactly.
        state = WorldState(agents=(_agent((0.0, 0.25, 1.6), speed=1.0, stature=1.75),))
        point = TrajectoryPoint(t=0.0, robot=None, swept_volume_center=(0.0, 0.0, 1.6), swept_volume_radius_m=0.05)
        traj = PredictedTrajectory(points=(point,))
        kw = dict(assumed_contact_time_s=1.0)
        self.assertTrue(pc.vulnerable_bystander_protected(state, None, traj, effective_mass_kg=65.0, **kw).satisfied)
        self.assertFalse(pc.vulnerable_bystander_protected(state, None, traj, effective_mass_kg=65.0 + 2 ** -20, **kw).satisfied)


# ----------------------------------------------------------------------------------------------
# End to end: the shipped example config, every new check wired and pinned
# ----------------------------------------------------------------------------------------------

class ExampleConfigEndToEnd(unittest.TestCase):
    PIN = integrity.read_digest_file(EXAMPLE_DIGEST_PATH)

    def _gate(self, state, *, trajectory=None, dynamics=None, clock=None, pin=PIN):
        registry = ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH, expected_digest=pin) if pin else ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH)
        return _gate(state, registry=registry, trajectory=trajectory, dynamics=dynamics, clock=clock)

    def _assert_blocked_by(self, decision, name):
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        self.assertIn(name, _failed_names(decision))

    def test_registry_has_all_32_names(self):
        self.assertEqual(len(pc.REGISTRY), 32)
        for name in ("decision_within_deadline", "payload_and_grip_force_within_limits", "stability_margin_maintained",
                     "sensor_data_fresh", "swept_path_observed", "command_integrity_verified",
                     "config_integrity_verified", "vulnerable_bystander_protected", "balance_margin_maintained"):
            self.assertIn(name, pc.REGISTRY)

    def test_golden_grasp_place_and_reach_permit(self):
        surface = fixtures.confirmed_object(object_id="cube_1", position=(0.5, 0.0, 0.02))
        state = fixtures.base_world_state(objects=(fixtures.confirmed_object(), surface))
        for action in (fixtures.grasp_action(), fixtures.place_action(), Action("reach", {"target_position": (0.5, 0.0, 0.05)})):
            decision = self._gate(state).gate(action)
            self.assertEqual(decision.verdict, DecisionVerdict.PERMIT, (action.action_type, _failed_names(decision)))
            self.assertTrue(verify_decision_action(decision))

    def test_golden_navigate_permits_and_lunge_blocks(self):
        robot = fixtures.quadruped_robot_state()
        state = fixtures.quadruped_world_state(robot=robot)
        decision = self._gate(state, trajectory=_base_footprint_trajectory(robot)).gate(fixtures.navigate_action())
        self.assertEqual(decision.verdict, DecisionVerdict.PERMIT, _failed_names(decision))
        lunging = fixtures.quadruped_robot_state(base_position=(0.22, 0.0, 0.6), com_velocity=(0.6, 0.0, 0.0))
        blocked = self._gate(fixtures.quadruped_world_state(robot=lunging), trajectory=_base_footprint_trajectory(lunging)).gate(fixtures.navigate_action())
        self._assert_blocked_by(blocked, "stability_margin_maintained")

    def test_unpinned_example_config_blocks_everything(self):
        self._assert_blocked_by(self._gate(_golden_state(), pin=None).gate(fixtures.grasp_action()), "config_integrity_verified")

    def test_each_new_check_fires_through_the_example_config(self):
        cases = {
            "sensor_data_fresh": (_golden_state(sensor_age_s=2.0), fixtures.grasp_action(), {}),
            "swept_path_observed": (_golden_state(observed_regions=SweptPathCoverage.LEFT_HALF_ONLY), fixtures.grasp_action(), {}),
            "payload_and_grip_force_within_limits": (_golden_state(), fixtures.grasp_action(grip_force_n=60.0), {}),
            "vulnerable_bystander_protected": (_golden_state(agents=(_agent((0.5, 2.0, 0.05), AgentCategory.CHILD),)), fixtures.grasp_action(), {}),
            "command_integrity_verified": (_golden_state(), fixtures.grasp_action(), {"dynamics": _RewritesActionAfterPredicting()}),
        }
        for name, (state, action, kw) in cases.items():
            self._assert_blocked_by(self._gate(state, **kw).gate(action), name)
        clock = FakeClock()
        stalled = self._gate(_golden_state(), dynamics=_StallingDynamics(fixtures.straight_line_trajectory(), clock, 0.5), clock=clock)
        self._assert_blocked_by(stalled.gate(fixtures.grasp_action()), "decision_within_deadline")
        tampered = ActionSchemaRegistry.from_yaml(EXAMPLE_SCHEMA_PATH, expected_digest=self.PIN)
        sensor_check = next(c for c in tampered._schemas["grasp"].checks if c["name"] == "sensor_data_fresh")
        sensor_check["kwargs"]["max_sensor_age_s"] = 3600.0  # silently accept hour-old sensor data
        self._assert_blocked_by(_gate(_golden_state(), registry=tampered).gate(fixtures.grasp_action()), "config_integrity_verified")


if __name__ == "__main__":
    unittest.main()
