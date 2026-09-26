"""Stress tests that don't presuppose which check will fire.

Every other test file constructs a fixture knowing exactly which precondition it will trip. These
tests instead check properties that must hold across *any* input: mutate a known-good scenario
field by field and confirm something blocks it (without saying which), generate random/garbage
world states and check structural invariants, and feed pathological input nobody specifically wrote
a check for. See the design doc's "Testing and Validation Strategy".

Run with: python3 -m unittest tests.test_fuzz -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures  # noqa: E402

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict, PerceptionFailure  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402
from safety_harness.schema import (  # noqa: E402
    Action,
    EnvironmentSignals,
    FallConsequence,
    HazardTag,
    Pose,
    PredictedTrajectory,
    PreconditionResult,
    RobotProprioception,
    TrackedAgent,
    TrackedObject,
    TrajectoryPoint,
    WorldState,
)

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "example_action_schema.yaml")


class _StubPerception(PerceptionAdapter):
    def __init__(self, state):
        self._state = state

    def get_world_state(self):
        return self._state


class _StubDynamics(DynamicsAdapter):
    def __init__(self, trajectory):
        self._trajectory = trajectory

    def predict_trajectory(self, state, action, horizon_s):
        return self._trajectory


def gate_for(state, trajectory=None):
    trajectory = trajectory or fixtures.straight_line_trajectory()
    return ActuatorGate(
        perception=_StubPerception(state),
        dynamics=_StubDynamics(trajectory),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
    )


def golden_state():
    """The one scenario currently confirmed to PERMIT: a fully cleared, stable, harmless cube,
    no agents, good visibility. Every mutation test starts here."""
    return fixtures.base_world_state(objects=(fixtures.confirmed_object(),), agents=())


class MutationTests(unittest.TestCase):
    """Corrupt one field of a known-good scenario at a time. The assertion is only ever "this must
    now block" -- never which check catches it. That's the point: this finds gaps even if a check
    for that specific field doesn't exist yet, because the absence of a passing result blocks by
    construction."""

    def _assert_blocks(self, state=None, trajectory=None, label=""):
        state = state if state is not None else golden_state()
        gate = gate_for(state, trajectory)
        decision = gate.gate(fixtures.grasp_action())
        fired = [r.name for r in decision.precondition_results if not r.satisfied]
        return decision.verdict, fired

    def test_golden_scenario_actually_permits(self):
        # Sanity check the baseline itself, or every mutation test below is meaningless.
        verdict, fired = self._assert_blocks(golden_state())
        self.assertEqual(verdict, DecisionVerdict.PERMIT, f"golden scenario didn't permit; fired={fired}")

    def test_all_single_field_mutations_block(self):
        base = golden_state()
        obj = base.objects[0]
        mutations = {
            "class_confidence -> 0": replace(obj, class_confidence=0.0),
            "hazard_tags -> unknown": replace(obj, hazard_tags=frozenset({HazardTag.UNKNOWN})),
            "cleared_for_interaction -> False": replace(obj, cleared_for_interaction=False),
            "supported_stably -> False": replace(obj, supported_stably=False),
            "supported_stably -> None": replace(obj, supported_stably=None),
            "fall_consequence -> UNKNOWN": replace(obj, fall_consequence=FallConsequence.UNKNOWN),
            "fall_consequence -> HAZARDOUS_RELEASE, tolerance 0": replace(
                obj, fall_consequence=FallConsequence.HAZARDOUS_RELEASE, drop_tolerance_m=0.0
            ),
            "estimated_mass_kg -> None": replace(obj, estimated_mass_kg=None),
            "estimated_mass_kg -> 100kg": replace(obj, estimated_mass_kg=100.0),
            "pose_confidence -> 0": replace(obj, pose_confidence=0.0),
        }
        failures = []
        for label, mutated_obj in mutations.items():
            state = replace(base, objects=(mutated_obj,))
            verdict, fired = self._assert_blocks(state)
            if verdict != DecisionVerdict.BLOCK:
                failures.append(f"{label}: expected BLOCK, got {verdict.value} (fired={fired})")
        self.assertEqual(failures, [], "mutations that FAILED to block:\n" + "\n".join(failures))

    def test_agent_mutations_all_block(self):
        base = golden_state()
        agents = {
            "agent right on top of the target": (fixtures.close_agent(),),
            "agent far but zero tracking confidence": (
                TrackedAgent(agent_id="p", pose=Pose(position=(9.0, 9.0, 0.0)), tracking_confidence=0.0),
            ),
            "agent stale enough that worst-case radius reaches the path": (fixtures.stale_agent(),),
        }
        failures = []
        for label, agent_tuple in agents.items():
            state = replace(base, agents=agent_tuple)
            verdict, fired = self._assert_blocks(state)
            if verdict != DecisionVerdict.BLOCK:
                failures.append(f"{label}: expected BLOCK, got {verdict.value} (fired={fired})")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_bystander_object_mutations_all_block(self):
        base = golden_state()
        bystanders = {
            "uncleared hazardous bystander near the path": fixtures.uncleared_bystander_object(object_id="cube_3"),
            "cleared but sharp/hot bystander near the path": replace(
                fixtures.confirmed_object(object_id="cube_3"), hazard_tags=frozenset({HazardTag.SHARP})
            ),
        }
        failures = []
        for label, bystander in bystanders.items():
            state = replace(base, objects=(base.objects[0], bystander))
            verdict, fired = self._assert_blocks(state)
            if verdict != DecisionVerdict.BLOCK:
                failures.append(f"{label}: expected BLOCK, got {verdict.value} (fired={fired})")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_environment_mutation_blocks(self):
        base = golden_state()
        state = replace(base, environment=EnvironmentSignals(visibility_confidence=0.0))
        verdict, fired = self._assert_blocks(state)
        self.assertEqual(verdict, DecisionVerdict.BLOCK)

    def test_surface_hazard_mutation_blocks(self):
        # EnvironmentSignals.surface_hazards existed in the schema with nothing ever reading it --
        # found while scoping a broader environmental sweep, fixed with environment_hazard_clear.
        base = golden_state()
        state = replace(base, environment=EnvironmentSignals(visibility_confidence=1.0, surface_hazards=frozenset({"spill"})))
        verdict, fired = self._assert_blocks(state)
        self.assertEqual(verdict, DecisionVerdict.BLOCK, f"a detected surface hazard didn't block; fired={fired}")


class RandomFuzzTests(unittest.TestCase):
    """Generate many random -- often nonsensical -- world states and check invariants that must
    hold regardless of the specific scenario, rather than any single expected outcome."""

    N_ITERATIONS = 3000
    SEED = 20260926

    def _random_object(self, rng, object_id):
        conf_choices = [0.0, rng.random(), 1.0]
        tag_choices = [
            frozenset({HazardTag.UNKNOWN}), frozenset({HazardTag.FRAGILE}),
            frozenset({HazardTag.SHARP}), frozenset({HazardTag.HOT}), frozenset(),
        ]
        return TrackedObject(
            object_id=object_id,
            object_class="cube",
            pose=Pose(position=tuple(rng.uniform(-2, 2) for _ in range(3))),
            estimated_mass_kg=rng.choice([None, 0.0, rng.uniform(0, 10), -1.0, 1e6]),
            hazard_tags=rng.choice(tag_choices),
            pose_confidence=rng.choice(conf_choices),
            class_confidence=rng.choice(conf_choices),
            cleared_for_interaction=rng.choice([True, False]),
            supported_stably=rng.choice([True, False, None]),
            fall_consequence=rng.choice(list(FallConsequence)),
            drop_tolerance_m=rng.choice([None, 0.0, rng.uniform(0, 2)]),
        )

    # The fixed straight-line trajectory used everywhere below (fixtures.straight_line_trajectory)
    # sweeps (0.5, 0, 0.3) -> (0.5, 0, 0.05). An agent uniformly random over +-5m in every dimension
    # almost never lands within the two proximity-gated checks' tight radii (contact_plausible_range_m
    # =0.3 on iso15066_power_force_limiting, collaborative_zone_radius_m=1.5 on reduced_speed_near_human)
    # -- confirmed by a real local run where both showed fail=0 across 3000 iterations. Bias some
    # fraction of agents to actually land near the swept path instead of only ever generating "far".
    _PATH_MIDPOINT = (0.5, 0.0, 0.175)

    def _random_agent_position(self, rng):
        if rng.random() < 0.4:
            return tuple(self._PATH_MIDPOINT[i] + rng.uniform(-2.0, 2.0) for i in range(3))
        return tuple(rng.uniform(-5, 5) for _ in range(3))

    def _random_state(self, rng):
        n_objects = rng.randint(1, 4)
        objects = tuple(self._random_object(rng, f"obj_{k}") for k in range(n_objects))
        n_agents = rng.randint(0, 3)
        agents = tuple(
            TrackedAgent(
                agent_id=f"agent_{k}",
                pose=Pose(position=self._random_agent_position(rng)),
                tracking_confidence=rng.choice([0.0, rng.random(), 1.0]),
                time_since_confirmed_s=rng.uniform(0, 10),
                worst_case_speed_mps=rng.uniform(0.1, 3.0),
            )
            for k in range(n_agents)
        )
        # robot=None and a populated surface_hazards are real states the harness must handle (see
        # robot_state_confirmed / environment_hazard_clear) but this generator never produced either
        # before -- both checks showed fail=0 across 3000 iterations in a real run as a direct result.
        robot = fixtures.robot_state() if rng.random() < 0.85 else None
        surface_hazards = (
            frozenset(rng.sample(["spill", "smoke", "debris", "ice", "loose_cable"], k=rng.randint(1, 2)))
            if rng.random() < 0.3
            else frozenset()
        )
        return WorldState(
            objects=objects, agents=agents, robot=robot,
            environment=EnvironmentSignals(
                visibility_confidence=rng.choice([0.0, rng.random(), 1.0]),
                surface_hazards=surface_hazards,
            ),
        )

    def _random_trajectory(self, rng):
        # The fixed fixtures.straight_line_trajectory() moves at exactly 0.25m/s -- which sits
        # exactly ON reduced_speed_near_human's default max_speed_in_zone_mps=0.25 boundary (a
        # strict ">" check), so that check could never fail no matter how close an agent got.
        # Confirmed by a real run: fail=0 even after agents were biased onto the path. Vary the
        # horizon to vary commanded speed over the same physical path instead of only ever moving
        # at exactly the threshold speed.
        horizon_s = rng.choice([1.0, 1.0, 0.5, 0.1, 0.05])
        return fixtures.straight_line_trajectory(horizon_s=horizon_s)

    def test_permit_implies_every_result_satisfied(self):
        rng = random.Random(self.SEED)
        permits, blocks, crashes = 0, 0, []
        fired_histogram = {}
        for i in range(self.N_ITERATIONS):
            state = self._random_state(rng)
            target_id = state.objects[0].object_id
            gate = gate_for(state, self._random_trajectory(rng))
            try:
                decision = gate.gate(Action(action_type="grasp", params={"object_id": target_id, "target_position": (0.5, 0, 0.05)}))
            except Exception as exc:  # noqa: BLE001
                crashes.append((i, repr(exc)))
                continue
            if decision.verdict == DecisionVerdict.PERMIT:
                permits += 1
                unsatisfied = [r for r in decision.precondition_results if not r.satisfied]
                self.assertEqual(unsatisfied, [], f"iteration {i}: PERMIT with unsatisfied results {unsatisfied}")
            else:
                blocks += 1
                for r in decision.precondition_results:
                    if not r.satisfied:
                        fired_histogram[r.name] = fired_histogram.get(r.name, 0) + 1

        print(f"\n  fuzz: {self.N_ITERATIONS} iterations -> {permits} permit, {blocks} block, {len(crashes)} crash")
        print(f"  fuzz: block reasons fired: {dict(sorted(fired_histogram.items(), key=lambda kv: -kv[1]))}")
        self.assertEqual(crashes, [], f"unhandled exceptions during fuzzing: {crashes[:5]}")

    def test_no_check_is_dead_code(self):
        # A check that never fires across 3000 randomized world states either has a bug, or is
        # unreachable given how the fixtures are generated -- worth knowing either way. This used to
        # only print a NOTE; two checks (robot_state_confirmed, environment_hazard_clear) sat at
        # fail=0 and two more (iso15066_power_force_limiting, reduced_speed_near_human) sat at
        # fail=0 too, unnoticed, until a real local run surfaced it. Asserting on it turns that back
        # into something CI catches on its own instead of a fact someone has to remember to check.
        rng = random.Random(self.SEED + 1)
        fired_histogram = {}
        for _ in range(self.N_ITERATIONS):
            state = self._random_state(rng)
            target_id = state.objects[0].object_id
            gate = gate_for(state, self._random_trajectory(rng))
            decision = gate.gate(Action(action_type="grasp", params={"object_id": target_id, "target_position": (0.5, 0, 0.05)}))
            for r in decision.precondition_results:
                fired_histogram.setdefault(r.name, {"pass": 0, "fail": 0})
                fired_histogram[r.name]["pass" if r.satisfied else "fail"] += 1
        print("\n  coverage per check (pass/fail counts across all iterations):")
        for name, counts in fired_histogram.items():
            print(f"    {name}: pass={counts['pass']} fail={counts['fail']}")
        never_fails = [name for name, c in fired_histogram.items() if c["fail"] == 0]
        never_passes = [name for name, c in fired_histogram.items() if c["pass"] == 0]
        self.assertEqual(
            never_fails, [],
            f"checks with zero failures across {self.N_ITERATIONS} random states -- either dead "
            f"code, an always-true check, or (as happened before) the generator never produces the "
            f"input shape that would make them fail: {never_fails}",
        )
        self.assertEqual(
            never_passes, [],
            f"checks that never passed across {self.N_ITERATIONS} random states -- either always-"
            f"false, or the generator never produces the input shape that satisfies them: {never_passes}",
        )


class PathologicalInputTests(unittest.TestCase):
    """Malformed input nobody wrote a targeted check for. The bar is just: fail safe, don't crash
    the control loop, and don't silently permit."""

    def test_nan_position_does_not_crash_and_does_not_permit(self):
        obj = replace(fixtures.confirmed_object(), pose=Pose(position=(math.nan, 0.0, 0.05)))
        state = golden_state()
        state = replace(state, objects=(obj,))
        gate = gate_for(state)
        try:
            decision = gate.gate(fixtures.grasp_action())
        except Exception as exc:  # a crash here is itself the finding
            self.fail(f"NaN position crashed the engine instead of failing safe: {exc!r}")
        self.assertEqual(
            decision.verdict, DecisionVerdict.BLOCK,
            "a NaN position with a high confidence score permitted -- NaN comparisons are always "
            "False in Python, so a naive `<`/`>` threshold check silently passes; "
            "object_pose_confirmed must explicitly reject non-finite coordinates.",
        )

    def test_empty_trajectory_does_not_crash(self):
        state = golden_state()
        gate = gate_for(state, trajectory=PredictedTrajectory(points=(), horizon_s=1.0))
        try:
            decision = gate.gate(fixtures.grasp_action())
        except Exception as exc:
            self.fail(f"empty trajectory crashed instead of failing safe: {exc!r}")
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK, "empty trajectory (no predicted points) should not permit")

    def test_robot_none_does_not_crash(self):
        state = replace(golden_state(), robot=None)
        gate = gate_for(state)
        try:
            decision = gate.gate(fixtures.grasp_action())
        except Exception as exc:
            self.fail(f"robot=None crashed instead of failing safe: {exc!r}")
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_target_object_id_not_in_params_does_not_crash(self):
        state = golden_state()
        gate = gate_for(state)
        decision = gate.gate(Action(action_type="grasp", params={}))  # no object_id at all
        self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)

    def test_buggy_custom_check_that_raises_still_blocks_not_crashes(self):
        from safety_harness import preconditions as pc

        def _buggy_check(state, action, trajectory, **kwargs):
            return 1 / 0  # a bug in someone's custom precondition function

        pc.REGISTRY["buggy_check_for_testing"] = _buggy_check
        try:
            schema = ActionSchemaRegistry.from_dict(
                {"action_types": {"grasp": {"checks": [{"name": "buggy_check_for_testing", "kwargs": {}}]}}}
            )
            gate = ActuatorGate(
                perception=_StubPerception(golden_state()),
                dynamics=_StubDynamics(fixtures.straight_line_trajectory()),
                fallback=FreezeInPlaceFallback(),
                logger=InMemoryLogger(),
                action_schema=schema,
            )
            try:
                decision = gate.gate(fixtures.grasp_action())
            except Exception as exc:
                self.fail(f"a buggy precondition check crashed the control loop instead of blocking: {exc!r}")
            self.assertEqual(decision.verdict, DecisionVerdict.BLOCK)
        finally:
            del pc.REGISTRY["buggy_check_for_testing"]


if __name__ == "__main__":
    unittest.main()
