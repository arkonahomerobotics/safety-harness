# Safety Harness for Physical AI

[![tests](https://github.com/naganumakr/safety-harness/actions/workflows/tests.yml/badge.svg)](https://github.com/naganumakr/safety-harness/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/safety-harness.svg)](https://pypi.org/project/safety-harness/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A perception-grounded safety harness for physical AI: a default-deny precondition gate for
physical actuators. Every action a robot proposes — grasp, place, reach, anything else you
register — must earn a `PERMIT` from measured, structured evidence before it executes. No
evidence, or a failed check, or an adapter that raises: `BLOCK`. Permission, not detection.

**"Perception-grounded" describes the architecture's contract, not today's sensing maturity.**
`ActuatorGate` only ever decides from a structured `WorldState`, never from a raw sensor stream or a
policy's own claim — that boundary is real and enforced. But every adapter shipped today reads
either privileged simulator ground truth (Isaac Lab) or a hand-populated/mocked bridge object (ROS 2
hooks, TurtleBot3) — there is no real camera/lidar perception pipeline (object detection, tracking)
behind any of them yet. Read "perception-grounded" as "built so a real perception pipeline plugs
straight into the same contract with zero engine changes," not as "already running one."

Full design spec, architecture, worked examples, test/validation record, and roadmap:
**[Perception-Grounded Safety Harness for Physical AI — design doc](docs/design.md)**.

## Status

Reference implementation, not yet independently assessed. Of its 31 precondition checks, **24 are
confirmed blocking live** in one simulator (Isaac Lab) on two robots, a Franka Panda arm and an
ANYmal-C quadruped — but that number is a mix of two different kinds of evidence, and the headline
figure alone doesn't say which is which, so here it is split out rather than left blended:
- **6 blocked on a genuinely real, un-modified simulator condition** — 4 on a physical state change
  (e.g. a joint driven to its real limit) and 2 on the action's own commanded parameters (e.g. an
  over-speed reach or an over-force grip). These are the strongest evidence: nothing about the
  world state was altered to produce them.
- **18 blocked because a test wrapper overrode one field of an otherwise-real world state** (e.g.
  setting a tracked agent's category to CHILD, or a sensor timestamp to stale) — real code, real
  `ActuatorGate` decision path, but the hazardous condition itself was injected by the test, not
  independently produced by the simulator or the commanded action. This is standard fault-injection
  testing and is how most of these checks *can* currently be exercised at all (the simulator has no
  built-in way to, say, spawn a child near the robot on its own) — but it's a materially weaker
  claim than the physical/commanded 6, and shouldn't be read as equivalent to it.
- 1 more check was evaluated live but never the reason for a block; 2 more (`joint_velocity_within_limits`,
  `joint_effort_within_limits`) are wired into the example config as of 0.3.4 but not yet
  re-validated live; 4 are registered but not wired into the example config;
- see the design doc's "Live Re-validation of the Wired Set" for the per-check record, including
  exactly which check is in which tier.

Plus 352 automated unit/fuzz/mutation/black-box/stress tests. A third simulated adapter (a
Unitree G1 humanoid with a dexterous hand) ships with a trained block-stacking policy to gate — see
[`examples/isaac_lab_g1_stack`](examples/isaac_lab_g1_stack/); gating that policy under injected
hazards is in progress, not yet a validation result. A fourth adapter targets a real ROS 2 graph
instead of a simulator — see [`examples/ros2_hooks`](examples/ros2_hooks/); it demonstrates the
software genuinely integrates into a real ROS 2 control loop, not a physical-hardware validation
result either. A separate, standalone proof-of-concept — [`examples/nav2_hazard_scan`](examples/nav2_hazard_scan/)
— applies the same named-check pattern to a real Nav2 map at deployment time instead of live
actions; it is not wired through `ActuatorGate` and not a certified risk assessment. It has **not**
been reviewed by a functional-safety assessor
against IEC 61508, ISO 13849, or ISO 10218/TS 15066 — see the design doc's Scope & Non-Goals
section for what's out of scope today (joint-space kinematic checks, certified numeric
thresholds, data-protection handling of logged human-position data). Treat this as engineering
evidence, not a certification.

## What's here

- `safety_harness/schema.py` — the data contract (`WorldState`, `Action`, `Decision`, and friends).
  `SCHEMA_VERSION` tracks this contract specifically; bump it deliberately (see the design doc's
  Version History).
- `safety_harness/engine.py` — `ActuatorGate.gate()`, the single decision entry point.
- `safety_harness/preconditions.py` — the registered precondition checks (31 distinct checks,
  one of which, `surface_confirmed_stable`, is deprecated in favor of
  `destination_confirmed_stable_and_clear`; the registry also keeps `balance_margin_maintained`
  as a legacy alias of `stability_margin_maintained`): object/target safety, placement, agent proximity incl. ISO/TS
  15066, vulnerable bystanders, robot self-limits, payload/grip force, stability, perception
  integrity (sensor freshness, swept-path coverage), decision deadline, command and configuration
  integrity, environmental signals.
- `safety_harness/watchdog.py` — `DecisionWatchdog`, an actuator-side dead-man's switch: execute
  only a PERMIT that is still fresh and bit-identical to what was checked, otherwise freeze.
- `safety_harness/integrity.py` / `safety_harness/pin.py` — action digests and configuration
  pinning (`python -m safety_harness.pin configs/example_action_schema.yaml >
  configs/example_action_schema.yaml.sha256` regenerates the pin; the command prints the digest).
- `safety_harness/audit_log.py` — persistent, hash-chained (optionally HMAC-keyed) logging:
  `HashChainedDecisionLogger` (a drop-in `Logger`) and `SoftwareVersionLog` for durable,
  tamper-evident records; `report_identity()` reports the running harness's own schema/package
  version and pinned config digest at any time. `verify_log()` re-walks a log file end to end.
- `safety_harness/action_schema.py` — the YAML-driven registry mapping action types to the checks
  they must pass (`configs/example_action_schema.yaml` is the reference wiring).
- `safety_harness/sbom.py` — CycloneDX Software Bill of Materials generation (`python -m
  safety_harness.sbom > examples/sbom/results/safety-harness.cyclonedx.json` regenerates it; see
  [`examples/sbom/`](examples/sbom/) for the checked-in reference output and why CycloneDX, not
  SPDX). Reports exactly one runtime third-party dependency (`pyyaml`) — see that module's
  docstring for why that's the honest answer, not "stdlib only, zero dependencies."
- `safety_harness/segmentation.py` — `GripActionSegmenter`, turning a continuous grip-based
  policy's control stream into the discrete `grasp`/`place`/`reach` proposals `gate()` expects, by
  watching for edges in the commanded grip rather than gating every step — fixes the deadlock a
  naive per-step "grasp" proposal causes once the fingers make contact (found live on a G1 humanoid
  stacking task; the fix this extracts is what the closed-loop Franka measurement already validates).
- `tests/` — unit tests, mutation/random-fuzz tests, and reflection-driven black-box contract
  tests.

Robot-specific behavior lives entirely behind four adapter interfaces
(`PerceptionAdapter`, `DynamicsAdapter`, `FallbackController`, `Logger`) so the engine and checks
are robot-agnostic. Reference adapters exist for three Isaac Lab robots (Franka Panda,
`adapters/isaac_lab.py`; ANYmal-C, `adapters/isaac_lab_anymal.py`; Unitree G1,
`adapters/isaac_lab_g1.py`) and, since 0.3.8, a real ROS 2 graph instead of a simulator
(`adapters/ros2.py` — see [`examples/ros2_hooks`](examples/ros2_hooks/) for a runnable, self-checking
demonstration against real `rclpy` nodes, verified in a `ros:humble-ros-base` container, and
[`examples/turtlebot3_gazebo_hooks`](examples/turtlebot3_gazebo_hooks/) for the same adapter gating
a real TurtleBot3 physically simulated in a real headless Gazebo, not a mock publisher).

**Not yet independently verified** (deliberately flagged, not buried): the ISO/TS 15066 Table A.2
body-region force figures, the child clearance and the crowd factors in
`vulnerable_bystander_protected` are placeholder values that need sign-off from a qualified safety
engineer before being described as standards-aligned; and `config_integrity_verified` is a
checksum unless you supply an HMAC key — it catches corruption and uncoordinated edits, not an
attacker who can rewrite both the config and its pinned hash.

## Install & test

```bash
pip install safety-harness
```

Or straight from GitHub (e.g. for an unreleased fix):

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

**`pip install safety-harness` gives you the importable `safety_harness` package only** —
`configs/` (the example action schema and its digest pin) is a reference file in this *repository*,
not packaged into the wheel/sdist (`[tool.setuptools.packages.find]` in `pyproject.toml` includes
only `safety_harness*`). That's deliberate, not an oversight: the schema is meant to be *yours*,
written and digest-pinned for your own robot's real checks, the same way `configs/g1_action_schema.yaml`
and `configs/turtlebot3_action_schema.yaml` are each specific to their own robot rather than shared.
`configs/example_action_schema.yaml` below is the reference shape to copy from — either clone this
repo, or fetch just that file and its `.sha256` from
[`configs/` on GitHub](https://github.com/naganumakr/safety-harness/tree/main/configs).

```python
from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionWatchdog
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger
from safety_harness.adapters.isaac_lab import (
    IsaacLabCubeStackPerceptionAdapter,
    IsaacLabCubeStackDynamicsAdapter,
)
from safety_harness.integrity import read_digest_file

cfg = "configs/example_action_schema.yaml"  # path to your own copy -- see note above
# Pin the config to its known-good digest: a registry loaded without one blocks every action
# (config_integrity_verified), and one whose file doesn't match refuses to construct.
schema = ActionSchemaRegistry.from_yaml(cfg, expected_digest=read_digest_file(cfg + ".sha256"))
fallback = FreezeInPlaceFallback()
watchdog = DecisionWatchdog(fallback, deadline_s=0.2)  # set from your real control cycle
gate = ActuatorGate(
    perception=IsaacLabCubeStackPerceptionAdapter(...),   # swap for your own stack's adapter
    dynamics=IsaacLabCubeStackDynamicsAdapter(...),
    fallback=fallback,
    logger=InMemoryLogger(),
    action_schema=schema,
    watchdog=watchdog,
)

gate.gate(proposed_action)       # -> Decision(verdict=PERMIT|BLOCK, ...), also fed to the watchdog
robot.execute(watchdog.command())  # the permitted action only while fresh and bit-identical; else freeze
```

Nothing here is Isaac-Lab-specific except the two adapter classes — swap those for adapters
targeting your own robot stack and the engine, checks, and tests are unchanged. See "Contributing
an adapter" below.

## Security

Reporting a vulnerability (including a default-deny bypass — see the design doc's "NaN-Sensor
Stress Test" for a real example of the kind of finding this covers), supported versions, and the
coordinated-disclosure policy: [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The engine, the adapter
interfaces, and the existing black-box/fuzz test suites are all open under this license: the goal
is for any robot maker to implement the four adapters for their own stack and run the same tests
against it, not to license the core code per-implementation. See the design doc's Release &
Distribution section for the reasoning and the certification model this enables (aspirational —
a real, separately-packaged third-party conformance fixture doesn't exist yet, see the Roadmap).

## Contributing an adapter

1. Implement `PerceptionAdapter`, `DynamicsAdapter`, `FallbackController`, and `Logger` for your
   stack (see `safety_harness/adapters/isaac_lab.py` for the reference shape).
2. Run the black-box and fuzz test suites against your adapter's `WorldState`/`PredictedTrajectory`
   output — they're written against the interfaces, not the Isaac Lab implementation, so they
   should run unmodified.
3. Open an issue or PR with your results. Passing those tests is what "compliant adapter" means
   here today — there's no separate conformance fixture, submission process, or certification mark
   yet, just these adapter-generic tests; a real third-party-facing conformance process is intent,
   not current state (see the design doc's Roadmap).
