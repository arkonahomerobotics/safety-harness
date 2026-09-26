# Physical AI Safety Harness

[![tests](https://github.com/naganumakr/safety-harness/actions/workflows/tests.yml/badge.svg)](https://github.com/naganumakr/safety-harness/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A default-deny precondition gate for physical actuators. Every action a robot proposes — grasp,
place, reach, anything else you register — must earn a `PERMIT` from measured, structured
evidence before it executes. No evidence, or a failed check, or an adapter that raises: `BLOCK`.
Permission, not detection.

Full design spec, architecture, worked examples, test/validation record, and roadmap:
**[Physical AI Safety Harness — design doc](https://claude.ai/artifact/V1cBKhjD94CZBj4D8kQjdU)**.

## Status

Reference implementation, not yet independently assessed. Validated against one robot (a Franka
Panda arm) in one simulator (Isaac Lab), with 75 automated unit/fuzz/mutation/black-box tests and
7 live hazard-scenario recordings. It has **not** been reviewed by a functional-safety assessor
against IEC 61508, ISO 13849, or ISO 10218/TS 15066 — see the design doc's Scope & Non-Goals
section for what's out of scope today (joint-space kinematic checks, certified numeric
thresholds, data-protection handling of logged human-position data). Treat this as engineering
evidence, not a certification.

## What's here

- `safety_harness/schema.py` — the data contract (`WorldState`, `Action`, `Decision`, and friends).
  `SCHEMA_VERSION` tracks this contract specifically; bump it deliberately (see the design doc's
  Version History).
- `safety_harness/engine.py` — `ActuatorGate.gate()`, the single decision entry point.
- `safety_harness/preconditions.py` — the registered precondition checks (24 at last count:
  object/target safety, placement, agent proximity incl. ISO/TS 15066, robot self-limits,
  environmental signals).
- `safety_harness/action_schema.py` — the YAML-driven registry mapping action types to the checks
  they must pass (`configs/example_action_schema.yaml` is the reference wiring).
- `tests/` — unit tests, mutation/random-fuzz tests, and reflection-driven black-box contract
  tests.

Robot-specific behavior lives entirely behind four adapter interfaces
(`PerceptionAdapter`, `DynamicsAdapter`, `FallbackController`, `Logger`) so the engine and checks
are robot-agnostic. Only an Isaac Lab / Franka adapter exists today.

## Install & test

Not on PyPI yet — install straight from GitHub:

```bash
pip install git+https://github.com/naganumakr/safety-harness.git
```

Or for local development (editable, so edits to `safety_harness/` take effect immediately):

```bash
git clone https://github.com/naganumakr/safety-harness.git
cd safety-harness
pip install -e .
python -m unittest discover -s tests -p "test_*.py"
```

## Usage

```python
from safety_harness import ActionSchemaRegistry, ActuatorGate
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger
from safety_harness.adapters.isaac_lab import (
    IsaacLabCubeStackPerceptionAdapter,
    IsaacLabCubeStackDynamicsAdapter,
)

schema = ActionSchemaRegistry.from_yaml("configs/example_action_schema.yaml")
gate = ActuatorGate(
    perception=IsaacLabCubeStackPerceptionAdapter(...),   # swap for your own stack's adapter
    dynamics=IsaacLabCubeStackDynamicsAdapter(...),
    fallback=FreezeInPlaceFallback(),
    logger=InMemoryLogger(),
    schema=schema,
)

decision = gate.gate(proposed_action)  # -> Decision(verdict=PERMIT|BLOCK, ...)
if decision.verdict.name == "PERMIT":
    robot.execute(decision.action)
else:
    robot.execute(decision.action)  # the fallback action FreezeInPlaceFallback produced
```

Nothing here is Isaac-Lab-specific except the two adapter classes — swap those for adapters
targeting your own robot stack and the engine, checks, and tests are unchanged. See "Contributing
an adapter" below.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The engine, the adapter
interfaces, and the conformance test suite are all open under this license: the goal is for any
robot maker to implement the four adapters for their own stack and run the same conformance suite
against it, not to license the core code per-implementation. See the design doc's Release &
Distribution section for the reasoning and the certification model this enables.

## Contributing an adapter

1. Implement `PerceptionAdapter`, `DynamicsAdapter`, `FallbackController`, and `Logger` for your
   stack (see `safety_harness/adapters/isaac_lab.py` for the reference shape).
2. Run the black-box and fuzz test suites against your adapter's `WorldState`/`PredictedTrajectory`
   output — they're written against the interfaces, not the Isaac Lab implementation, so they
   should run unmodified.
3. Open an issue or PR with your results. Passing the conformance suite is what "compliant
   adapter" means here — there's no separate certification process yet, but that's the intent
   (see the design doc's Roadmap).
