"""Black-box stress test: treats safety_harness.preconditions and the loaded action schema as
completely opaque. Never imports preconditions.py, never references a check by name, never
hand-picks "the field that matters" for a scenario. Inputs are generated mechanically by reflecting
over the public dataclasses in schema.py -- so the test author's own assumptions about which fields
are interesting can't bias what gets covered. Only externally observable contract properties are
asserted: verdict, determinism, exception type, and the shape of the returned action.

Run with: python3 -m unittest tests.test_blackbox -v   (from the safety_harness/ package root)
"""

from __future__ import annotations

import dataclasses
import enum
import os
import random
import sys
import time
import typing
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict, PerceptionFailure  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.base import DynamicsAdapter, PerceptionAdapter  # noqa: E402
from safety_harness import schema as sch  # noqa: E402

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "example_action_schema.yaml")

# The only names ever referenced from the schema module: the public data TYPES, for reflection.
# Never a precondition function, never a field the test author "knows" matters.
_ENUM_TYPES = (sch.HazardTag, sch.FallConsequence)


def _random_scalar(rng):
    return rng.choice([
        0.0, -1.0, 1.0, 1e12, -1e12, 1e-12, float("nan"), float("inf"), float("-inf"),
        rng.uniform(-100, 100), rng.uniform(0, 1),
    ])


def _random_str(rng):
    return rng.choice(["", "x" * 200, "weird!@#$%^&*()_+", "\x00\x01", "grasp", "PLACE", "totally_unknown_action_qwzx", "🤖"])


def gen_value(type_hint, rng, depth=0):
    """Generates a value for an arbitrary type hint by reflection, generically -- dispatch is on
    the *shape* of the type (Optional, tuple, frozenset, Enum, nested dataclass), never on which
    field it happens to be. This is what keeps the generator from encoding the author's own guess
    about which fields matter."""
    origin = typing.get_origin(type_hint)

    if origin is typing.Union:  # covers Optional[X]
        args = [a for a in typing.get_args(type_hint) if a is not type(None)]
        if rng.random() < 0.35:
            return None
        return gen_value(args[0], rng, depth)

    if type_hint is float:
        return _random_scalar(rng)
    if type_hint is bool:
        return rng.choice([True, False])
    if type_hint is str:
        return _random_str(rng)
    if type_hint is int:
        return rng.choice([0, -1, 2**31, rng.randint(-1000, 1000)])
    if type_hint is dict:
        return {} if rng.random() < 0.5 else {"object_id": _random_str(rng), "target_position": gen_value(tuple, rng, depth)}

    if origin in (tuple,) or type_hint is tuple:
        args = typing.get_args(type_hint)
        if args and len(args) == 2 and args[1] is Ellipsis:
            n = rng.randint(0, 6)
            return tuple(gen_value(args[0], rng, depth + 1) for _ in range(n))
        if args and Ellipsis not in args:
            return tuple(gen_value(a, rng, depth + 1) for a in args)
        n = rng.randint(0, 4)
        return tuple(_random_scalar(rng) for _ in range(n))

    if origin is frozenset or type_hint is frozenset:
        if depth > 3 or rng.random() < 0.4:
            return frozenset()
        enum_cls = rng.choice(_ENUM_TYPES)
        return frozenset(rng.sample(list(enum_cls), k=rng.randint(0, len(list(enum_cls)))))

    if isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return rng.choice(list(type_hint))

    if isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint) and depth < 5:
        return gen_dataclass(type_hint, rng, depth + 1)

    return None  # anything not recognized -- forces the harness to cope with an unexpected None


def gen_dataclass(cls, rng, depth=0):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        hint = hints.get(f.name, str)
        # Occasionally skip a field with a default, to exercise the "field not provided" path too.
        if f.default is not dataclasses.MISSING and rng.random() < 0.2:
            continue
        if f.default_factory is not dataclasses.MISSING and rng.random() < 0.2:  # type: ignore[misc]
            continue
        kwargs[f.name] = gen_value(hint, rng, depth)
    try:
        return cls(**kwargs)
    except TypeError:
        # A required field with no default got skipped by the "occasionally omit" logic above, or
        # a generated value violated a structural requirement (e.g. object_id must be str-shaped
        # for equality checks elsewhere) -- retry once with everything filled in.
        kwargs = {f.name: gen_value(hints.get(f.name, str), rng, depth) for f in dataclasses.fields(cls)}
        return cls(**kwargs)


def gen_random_world_state(rng):
    n_objects = rng.randint(0, 4)
    objects = tuple(gen_dataclass(sch.TrackedObject, rng) for _ in range(n_objects))
    # Force object ids to be discoverable by a random action targeting one of them, some of the
    # time -- otherwise almost every generated action would target nothing and everything would
    # trivially block on "object not found," which would tell us nothing.
    fixed_objects = []
    for i, o in enumerate(objects):
        if rng.random() < 0.6:
            o = dataclasses.replace(o, object_id=f"obj_{i}")
        fixed_objects.append(o)
    n_agents = rng.randint(0, 3)
    agents = tuple(gen_dataclass(sch.TrackedAgent, rng) for _ in range(n_agents))
    robot = gen_dataclass(sch.RobotProprioception, rng) if rng.random() < 0.85 else None
    env = gen_dataclass(sch.EnvironmentSignals, rng)
    return sch.WorldState(objects=tuple(fixed_objects), agents=agents, robot=robot, environment=env)


def gen_random_action(rng):
    action_type = rng.choice(["grasp", "place", "reach", _random_str(rng)])
    params = {}
    if rng.random() < 0.7:
        params["object_id"] = rng.choice(["obj_0", "obj_1", "obj_2", _random_str(rng)])
    if rng.random() < 0.7:
        params["target_position"] = tuple(_random_scalar(rng) for _ in range(3))
    if rng.random() < 0.4:
        params["target_surface_id"] = rng.choice(["obj_0", "obj_1", _random_str(rng)])
    return sch.Action(action_type=action_type, params=params)


def gen_random_trajectory(rng, robot):
    n = rng.randint(0, 8)
    points = tuple(
        sch.TrajectoryPoint(
            t=_random_scalar(rng), robot=robot,
            swept_volume_center=tuple(_random_scalar(rng) for _ in range(3)),
            swept_volume_radius_m=_random_scalar(rng),
            self_collision_margin_m=rng.choice([None, _random_scalar(rng)]),
        )
        for _ in range(n)
    )
    return sch.PredictedTrajectory(points=points, horizon_s=_random_scalar(rng))


class _Perception(PerceptionAdapter):
    def __init__(self, state):
        self._state = state

    def get_world_state(self):
        return self._state


class _Dynamics(DynamicsAdapter):
    def __init__(self, traj):
        self._traj = traj

    def predict_trajectory(self, state, action, horizon_s):
        return self._traj


N_ITERATIONS = 5000
SEED = 424242


class BlackBoxContractTests(unittest.TestCase):
    """Only checks properties statable from outside: what the public types and the design doc's
    stated contract promise, never which internal check produced the result."""

    def test_never_raises_an_undocumented_exception_type(self):
        rng = random.Random(SEED)
        undocumented = []
        for i in range(N_ITERATIONS):
            state = gen_random_world_state(rng)
            action = gen_random_action(rng)
            traj = gen_random_trajectory(rng, state.robot)
            gate = ActuatorGate(
                perception=_Perception(state), dynamics=_Dynamics(traj),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
            )
            try:
                gate.gate(action)
            except PerceptionFailure:
                pass  # documented, expected exception type
            except Exception as exc:  # noqa: BLE001
                undocumented.append((i, type(exc).__name__, str(exc)[:120]))
        self.assertEqual(undocumented, [], f"undocumented exceptions escaped: {undocumented[:8]}")

    def test_never_hangs(self):
        rng = random.Random(SEED + 1)
        slow = []
        for i in range(500):
            state = gen_random_world_state(rng)
            action = gen_random_action(rng)
            traj = gen_random_trajectory(rng, state.robot)
            gate = ActuatorGate(
                perception=_Perception(state), dynamics=_Dynamics(traj),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
            )
            t0 = time.perf_counter()
            try:
                gate.gate(action)
            except PerceptionFailure:
                pass
            except Exception:
                pass
            dt = time.perf_counter() - t0
            if dt > 1.0:
                slow.append((i, dt))
        self.assertEqual(slow, [], f"calls that took >1s (possible hang/blowup): {slow}")

    def test_identical_input_gives_identical_decision(self):
        """A safety decision that isn't reproducible for the exact same input is itself a hazard --
        this is checkable from outside with zero knowledge of internals.

        Needs a fixed clock injected into both runs. ``decision_within_deadline`` deliberately
        measures real elapsed wall-clock time between when ActuatorGate.gate() started and when
        that check runs -- that is its entire job (catching a decision that took too long for a
        reason no other check can see), not a bug. Two independently-constructed gates processing
        the literal same input a few microseconds apart will, correctly, measure a *different*
        elapsed time each time if left to the engine's default real clock, which makes
        ``PreconditionResult.reason``'s embedded millisecond figure differ, and -- rarely, only
        under real system load, if the elapsed time happens to straddle the deadline -- can flip
        ``satisfied`` itself. Found by an independent third-party review: this flipped once at
        trial 434 of 1000 in a loop. That is real, desired behavior for one long-lived
        ActuatorGate in an actual deployment, where elapsed-time measurements naturally differ
        decision to decision; it has nothing to do with whether *this test's* two reconstructed
        gates were given the same input. The fix belongs here, in the test, not in
        decision_within_deadline: inject the same deterministic clock into both constructions so
        the property this test actually means to check -- same input, same engine logic, same
        decision -- isn't confounded by real time having moved between the two calls.
        """
        rng = random.Random(SEED + 2)
        frozen_clock = lambda: 0.0  # noqa: E731 -- elapsed is always 0.0 - 0.0, identically, every call
        nondeterministic = []
        for i in range(1000):
            state = gen_random_world_state(rng)
            action = gen_random_action(rng)
            traj = gen_random_trajectory(rng, state.robot)

            def run_once():
                gate = ActuatorGate(
                    perception=_Perception(state), dynamics=_Dynamics(traj),
                    fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                    action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
                    clock=frozen_clock,
                )
                try:
                    d = gate.gate(action)
                    return (d.verdict, d.action, tuple(r.satisfied for r in d.precondition_results))
                except PerceptionFailure:
                    return "PerceptionFailure"

            r1, r2 = run_once(), run_once()
            if r1 != r2:
                nondeterministic.append((i, r1, r2))
        self.assertEqual(nondeterministic, [], f"non-deterministic decisions for identical input: {nondeterministic[:5]}")

    def test_frozen_clock_makes_the_full_decision_byte_identical(self):
        """Regression test for the exact bug class above, pinned down directly rather than relying
        on randomly hitting it again: one hand-built state/action/trajectory, two independently
        constructed gates, the same frozen clock given to both. Asserts full ``Decision`` equality
        -- not just the (verdict, action, satisfied-tuple) the test above checks -- so this also
        catches a ``PreconditionResult.reason`` string differing even when every ``satisfied`` value
        happens to agree, which is exactly the form this bug actually took (see
        ``decision_within_deadline``'s docstring): the embedded elapsed-time figure in the reason
        text varied between two otherwise-identical runs under the engine's default real clock. If
        clock injection into ``ActuatorGate``/``CheckContext`` is ever broken, this fails
        immediately, independent of the random generator and of real-world timing variance.
        """
        rng = random.Random(SEED + 99)
        state = gen_random_world_state(rng)
        action = gen_random_action(rng)
        traj = gen_random_trajectory(rng, state.robot)

        def decide():
            gate = ActuatorGate(
                perception=_Perception(state), dynamics=_Dynamics(traj),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
                clock=lambda: 42.0,
            )
            return gate.gate(action)

        d1, d2 = decide(), decide()
        self.assertEqual(
            d1, d2,
            "a frozen clock should make the full Decision -- including every PreconditionResult's "
            "reason string -- byte-identical across two separate gate() calls on the same input",
        )

    def test_unrecognizable_action_type_never_permits(self):
        """Without knowing what the loaded schema declares, a sufficiently novel action_type string
        must never be recognized as anything -- and per the stated contract, unrecognized means
        blocked, not permitted."""
        rng = random.Random(SEED + 3)
        wrong_permits = []
        for i in range(300):
            novel = f"never_a_real_action_type_{rng.randint(0, 10**9)}"
            state = gen_random_world_state(rng)
            gate = ActuatorGate(
                perception=_Perception(state), dynamics=_Dynamics(gen_random_trajectory(rng, state.robot)),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
            )
            decision = gate.gate(sch.Action(action_type=novel, params={}))
            if decision.verdict == DecisionVerdict.PERMIT:
                wrong_permits.append((i, novel))
        self.assertEqual(wrong_permits, [], f"a never-registered action type was permitted: {wrong_permits}")

    def test_permit_returns_the_original_action_unmodified(self):
        rng = random.Random(SEED + 4)
        mismatches = []
        for i in range(N_ITERATIONS):
            state = gen_random_world_state(rng)
            action = gen_random_action(rng)
            traj = gen_random_trajectory(rng, state.robot)
            gate = ActuatorGate(
                perception=_Perception(state), dynamics=_Dynamics(traj),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
            )
            try:
                decision = gate.gate(action)
            except PerceptionFailure:
                continue
            if decision.verdict == DecisionVerdict.PERMIT and decision.action != action:
                mismatches.append((i, action, decision.action))
        self.assertEqual(mismatches, [], f"PERMIT returned a different action than proposed: {mismatches[:5]}")

    def test_block_always_returns_some_action_never_none(self):
        rng = random.Random(SEED + 5)
        missing = []
        for i in range(N_ITERATIONS):
            state = gen_random_world_state(rng)
            action = gen_random_action(rng)
            traj = gen_random_trajectory(rng, state.robot)
            gate = ActuatorGate(
                perception=_Perception(state), dynamics=_Dynamics(traj),
                fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
                action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
            )
            try:
                decision = gate.gate(action)
            except PerceptionFailure:
                continue
            if decision.verdict == DecisionVerdict.BLOCK and decision.action is None:
                missing.append(i)
        self.assertEqual(missing, [], f"BLOCK with no fallback action at all: iterations {missing[:5]}")

    def test_returned_objects_are_still_immutable(self):
        """A purely structural, public-API-only check: the documented types are frozen dataclasses,
        so attempting to mutate anything handed back must raise, not silently succeed."""
        rng = random.Random(SEED + 6)
        state = gen_random_world_state(rng)
        gate = ActuatorGate(
            perception=_Perception(state), dynamics=_Dynamics(gen_random_trajectory(rng, state.robot)),
            fallback=FreezeInPlaceFallback(), logger=InMemoryLogger(),
            action_schema=ActionSchemaRegistry.from_yaml(SCHEMA_PATH),
        )
        try:
            decision = gate.gate(gen_random_action(rng))
        except PerceptionFailure:
            return
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decision.verdict = DecisionVerdict.PERMIT  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
