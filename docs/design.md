# Perception-Grounded Safety Harness for Physical AI

Sep 26, 2026 · Kaoru Naganuma

Every proposed robot action must be affirmatively confirmed safe by perception before it executes — absence of a detected hazard is not enough. A bounded set of preconditions per action type is checked against a structured world model and a forward-simulated trajectory, with any unconfirmed condition defaulting to blocked. This inverts the usual framing: instead of enumerating and detecting every possible hazard in an open-ended environment, a problem that never converges, the harness enumerates what must be confirmed true for a small, fixed set of action types, and blocks by default whenever that confirmation is missing.

## Part 1 — Overview

What this is, the principle it's built on, and what it deliberately does not cover.

## Design Principle: Permission, Not Detection

Every proposed action is unsafe by default and must be affirmatively confirmed safe before it executes. This inverts the usual approach: instead of trying to detect every possible hazard in an open-ended environment — enumerating dangers, which never converges — the harness enumerates the preconditions each action type requires, a small and fixed list, and blocks whenever a precondition lacks confirming evidence.

- **Block-if-detected** (the common approach): safe by default, blocked only when a known hazard pattern is recognized. Fails open on anything the hazard list didn't anticipate.
- **Permit-if-confirmed** (this design): unsafe by default, permitted only when required preconditions are affirmatively confirmed. Fails closed on anything perception hasn't verified.

The corollary that makes this work: missing, stale, or low-confidence evidence counts as a failed precondition, not as "no violation detected." An unrecognized object defaults to the most restrictive hazard assumption. Degraded perception — low light, occlusion, a sensor fault — lowers confidence on every downstream check rather than being ignored. This is how the harness catches a hazard it was never specifically told about: it isn't recognizing the hazard, it's failing closed on the absence of a confirmed "this is fine."

## Part 2 — Architecture & Interfaces

The decision pipeline, the data contract every adapter targets, and two worked examples of the whole path end to end.

## Architecture Overview

&#91;embedded content: pipeline · perceive, check, decide, execute or fall back, log\]

Every control cycle runs this pipeline in order. A blocked action never reaches the actuators; it triggers the fallback and logs why, and that log is what tightens the precondition rules or perception confidence over time.

## Perception → Structured World State

Perception's output is a structured, legible description of the world — never a safe/unsafe verdict on its own. Collapsing straight to a verdict would just move the coverage problem into a single opaque classifier, one level up from the policy it's meant to check.

What it produces, every cycle:

- Detected objects: class, estimated pose, velocity, hazard attributes (fragile, hot, sharp, human, animal, unknown), a confidence score per attribute.
- Tracked humans and animals: position, velocity, and an uncertainty bound that grows with time since last confirmed sighting.
- The robot's own state: joint positions and velocities; for a legged or humanoid platform, center of mass and current support polygon or contact state.
- Environmental signals: a visibility/confidence quality metric, floor or surface condition where sensed, any detected hazard such as smoke or a spill.

This layer can be learned — object detection and tracking are appropriately statistical. What must not be learned end to end is the step after it: turning this structured state into a permit-or-deny decision. That step is explicit and rule-based, so it can be tested the way a perception model cannot be.

## Worked Examples

Two real decisions, shown end to end through the same `WorldState` -> `ActuatorGate.gate()` -> `Decision` path every action takes, using the actual checks and thresholds documented in this spec.

### Example A — permit (grasp, everything clears)

```
WorldState                          ActuatorGate.gate(grasp cube_2)
-----------                          ---------------------------------
robot: proprioception OK        -->  robot_state_confirmed         PASS
cube_2: pose confirmed 0.82     -->  object_pose_confirmed         PASS
cube_2: hazard_confidence 0.9   -->  object_hazard_confirmed       PASS
cube_2: cleared_for_interaction -->  object_cleared_for_interaction PASS
cube_2: supported_stably        -->  current_position_confirmed    PASS
cube_2: fall_consequence=LOW    -->  fall_consequence_acceptable   PASS
cube_2: mass 0.05kg / 3kg budget-->  mass_within_force_budget      PASS
agent: none within 7m           -->  swept_path_clear_of_agents    PASS
                                 -->  iso15066_separation_distance  PASS
                                 -->  iso15066_power_force_limiting PASS
                                 -->  reduced_speed_near_human      PASS
risky objects: none in path     -->  swept_path_clear_of_risky     PASS
visibility: 0.83                -->  visibility_above_threshold    PASS
surface_hazards: none           -->  environment_hazard_clear      PASS
                                      ============================
                                      DECISION: PERMIT
```

All 14 `grasp` checks pass against real measured values — a bystander shaped like the `far_agent()` test fixture, roughly 7m away, is functionally absent from every proximity check. The arm executes the action unmodified.

### Example B — block (agent inside the force-limiting radius)

Same action, one field changed: a tracked person is now roughly 0.2m from the grasp point. `ActuatorGate.gate()` does not short-circuit — every registered check for the action type runs (see `action_schema.py: run_checks`), and the resulting `Decision` carries every result, so a block is logged with every reason that fired, not just one:

```
agent: distance 0.20m           -->  iso15066_power_force_limiting BLOCK
agent: distance 0.20m           -->  iso15066_separation_distance  BLOCK
agent: distance 0.20m           -->  swept_path_clear_of_agents    BLOCK
(11 other checks)               -->  ...                            PASS
                                      ============================
                                      DECISION: BLOCK (3 of 14 checks failed)
```

`iso15066_power_force_limiting` firing here is itself the fix for a real bug found during this build: the check originally had no distance gate at all, so a tracked agent anywhere in the scene — 7m away, in one regression test — failed it as though contact were imminent. The `contact_plausible_range_m=0.3` gate (Functional & Behavior Specification, below) is what makes this example correctly tell '0.2m, must block' apart from Example A's \~7m bystander, correctly ignored.

## Action Precondition Schemas

Each action type carries an explicit, hand-authored list of what must be confirmed — not learned, so it can be unit-tested against constructed world states.

| Action type | Required preconditions | Blocks if | Confidence gate |
| --- | --- | --- | --- |
| grasp(object) | Object hazard class confirmed (not "unknown"); estimated mass within gripper force budget; swept path clear of any tracked human's worst-case region for the motion's duration | Any precondition unconfirmed, or object class unknown | Object pose confidence above threshold |
| place(object, location) | Destination surface confirmed stable and clear; if object tagged liquid-containing, orientation constraint holds throughout | Surface unconfirmed, or orientation constraint predicted to be violated | Surface classification confidence above threshold |
| step(foot, target) — legged/humanoid | Target surface confirmed solid and traversable; balance margin maintained throughout the swing, not just at landing | Surface unconfirmed, or balance margin predicted to cross threshold at any point | Terrain classification confidence above threshold |
| reach near a tracked human | Human's worst-case reachable region — given tracking latency and a velocity bound — does not intersect the swept path for the action's full duration | Worst-case region intersects the swept path at any point | Human tracking confidence above threshold, else assume the closest plausible position |
| any action — general gate | Perception visibility/confidence quality above minimum operating threshold | Visibility metric below threshold | Blocks all actions, not just one type |

**Extended since the table above.** The original checks only asked whether the target's hazard class was known. That misses several real questions: is the target itself cleared to approach at all, is its current position confirmed stable, what happens if it's dropped, and can an *uncleared or hazardous bystander object* — never the action's own target — block an action purely by being near the swept path.

`object_cleared_for_interaction`, `current_position_confirmed_stable`, and `fall_consequence_acceptable` extend `grasp`; `destination_confirmed_stable_and_clear` (superseding `surface_confirmed_stable`) extends `place`; and `swept_path_clear_of_risky_objects` is a new general gate, alongside visibility, that blocks proximity to any uncleared or hazardous-release object regardless of whether it's the current target — this is what lets a bystander object be too dangerous to work near even when nothing is being done to it directly.

`TrackedObject` gained matching fields — `cleared_for_interaction` (default false), `supported_stably`, `fall_consequence`, `drop_tolerance_m` — all defaulting to the most restrictive assumption, consistent with the Design Principle section above.

`fall_consequence_acceptable` reasons about lift height by comparing the object's resting height against the highest point in whatever trajectory the dynamics adapter predicts. That's only meaningful if the adapter is actually predicting the post-grasp carry, not just the pre-grasp approach — worth checking specifically when validating a new `DynamicsAdapter`, since the two look similar but answer different questions.

## Part 3 — Functional & Behavior Specification

What the harness checks, in what order, including regulatory-mapped checks and the robot's own kinematic, electrical, and balance limits.

## Trajectory Forward-Check

Checking only the current instant misses hazards that appear partway through a motion — a swept path that's clear now but crosses a tracked human's position half a second later. The harness forward-simulates the proposed action over a short horizon using the robot's own dynamics model, then checks the predicted trajectory — not just its endpoint — against every precondition in the table above, at every point along it.

This is the established control-theoretic pattern of a Control Barrier Function: a safety function that must stay non-negative throughout the predicted rollout, with any action whose forward simulation would drive it negative rejected before execution. For a platform already built on a physics simulator, the same simulator can run this forward check in a lightweight prediction mode — it doesn't need to be a separate system.

For a humanoid, this check also covers whole-body balance: an arm reach that's fine at the target pose can still be unsafe mid-motion if it predicts the center of mass leaving the support polygon at any intermediate point.

## Fallback Control and Recovery

When a precondition fails, the harness needs a trusted fallback — simple enough that it doesn't inherit the coverage problem of the policy it's replacing.

- Freeze in place: safe by default for a fixed-base platform, or a legged one currently in a statically stable stance.
- Retract to a known safe pose: use once the retract trajectory itself has been verified collision-free.
- A simpler, better-characterized closed-loop controller substituting for the primary policy on a specific sub-task, where one exists and is well-validated.

Freezing is not automatically safe for a walking humanoid. Halting mid-single-support-phase can cause a fall, which is itself a serious hazard — a heavy platform falling is arguably worse than whatever the harness was blocking. The fallback needs its own state machine: if currently in a statically stable double-support stance, halt immediately; if mid-step, complete the current step to a stable stance first, then halt.

**Freezing is a motion-safety response, not a task-completion policy.** It answers "what does the robot do right now, physically, while a proposed action is unsafe" — it says nothing about what to do about the *task* that action was serving. Conflating the two has a real failure mode: an object whose mass exceeds the rated payload does not get lighter because the same grasp is proposed again, so "hold the freeze until this task succeeds" means the robot does nothing else, forever, even work that has nothing to do with that object. A precondition failure caught before motion even starts needs no freeze at all — the robot was never moving into that action — only a decision about what to attempt next. `safety_harness/task_policy.py` (v0.3.7) is a reference answer to that decision: classify whether a BLOCK's failing check(s) could plausibly clear on their own before the next attempt (`RETRYABLE_CHECKS` — an agent walks away, a sensor refreshes, an object's pose gets confirmed) versus represent a static fact about the object, the robot's own rated limits, or its configuration that retrying the identical action cannot change (unlisted checks default here, same default-deny spirit as everywhere else in this project). A non-retryable block marks that task — and anything that depends on it — skipped, with the real reason logged, and lets independent work proceed. See `examples/task_reroute/` for a runnable, no-GPU demonstration. This is caller-side policy, not a new field on `PreconditionResult`: the harness stays a stateless per-decision gate (see "Design Principle: Permission, Not Detection" above); `run_task_queue` is one reasonable reference orchestrator, meant to be read and adapted, not the only correct one.

## Logging, Escalation and the Correction Loop

Every block is logged with full context: which precondition failed, and the perception state that produced the rejection. Repeated blocks on the same action escalate to a human rather than retrying indefinitely — "failed three times, paused for review" — with a reachable kill switch at every stage.

**Persistent, tamper-evident logging (v0.3.1, `safety_harness/audit_log.py`).** The `Logger` interface itself is retention-agnostic by design — the reference `InMemoryLogger` is exactly what its name says, a process-lifetime list — so nothing before this shipped a durable record, and nothing detected a log entry edited or deleted after the fact. `HashChainedDecisionLogger` (a drop-in `Logger`) and `SoftwareVersionLog` write append-only, hash-chained JSON-lines files: each entry's digest covers its own content and the previous entry's digest, so editing, deleting, or reordering any entry breaks the chain from that point forward, detectable with `verify_log()`. Unkeyed, this is tamper-*evident* (corruption or an edit made without also recomputing the whole tail is caught); with an HMAC key held apart from the log file, it is closer to tamper-*proof* (a forger without the key cannot produce a chain that verifies at all) — the same unkeyed-vs-keyed distinction `config_integrity_verified`'s pin already relies on, applied to the log instead of the config. `report_identity()` answers "can the harness report its own version/hash at runtime" directly: schema version, package version, and the pinned config digest currently in force, queryable at any time, independent of whether logging is even configured. What this module explicitly does *not* do is enforce retention — nothing running in-process can stop deletion of the file by someone with disk access — it gives a retention *policy* something to check against (`verify_log`), not a substitute for one.

This log is not just an incident record. A human reviews it and either improves perception confidence where it was needlessly conservative, or tightens a precondition rule where it should have caught something it didn't. The correction happens deliberately, by inspection of an explicit rule — not by retraining on more data and hoping the next model generalizes better, which would reintroduce the same coverage problem this design exists to avoid.

## Robot Self-Limits: Kinematic, Electrical, and Balance

Everything above checks the world around the robot -- objects, agents, surfaces. None of it checks whether the robot's own commanded motion stays within what it can physically and electrically do. That's a distinct category, and until now only balance (`balance_margin_maintained`, already in the table above) was covered -- and even that only against hand-built fixtures. Updated (2026-09-27): the ANYmal-C navigation adapter (Development Roadmap, stage 4) is the first anywhere in this project to feed it a real support polygon -- four real foot contact positions -- and doing so immediately found a real bug in \_distance\_to\_polygon\_edge's own documented simplification: nearest-vertex distance keeps growing, not shrinking, the further a point moves outside the polygon, so a center of mass 10m past every foot read as a \~9m margin -- comfortably balanced, on a robot that has clearly already fallen over. Fixed with a proper signed distance to the polygon's convex hull (positive inside, negative outside), at the same call site its own docstring had already invited a real computational-geometry library to replace. See tests/test\_anymal\_adapter.py and safety\_harness/adapters/isaac\_lab\_anymal.py.

Six new checks, all following the same default-deny convention as everything else -- missing data blocks, it never permits:

| Check | Verifies | Needs from the adapter |
| --- | --- | --- |
| `joint_position_limits_respected` | Every joint stays within its range, with margin, throughout the motion | Per-joint (min, max) limits, reported every point |
| `joint_velocity_within_limits` | Every joint's speed stays within a fraction of its rated limit -- this is also where an unmodeled kinematic singularity shows up, since required joint speeds spike near one even for modest commanded Cartesian speed | Per-joint velocity limits |
| `joint_effort_within_limits` | Torque/current stays within each joint's rated limit -- the electrical/mechanical load side | Per-joint effort limits and an effort estimate |
| `motor_temperature_within_limits` | Every motor currently has thermal headroom before taking on more sustained load | Current temperature and limit, per joint |
| `cartesian_speed_within_limits` | The swept end-effector path stays within the platform's rated speed | A configured speed limit; computed directly from predicted trajectory points |
| `self_collision_clear` | The robot's own links stay clear of each other throughout the motion | A collision margin per point, from the adapter's own collision geometry -- not something a generic check can recompute from joint angles alone |

**Not yet wired into the reference Isaac Lab adapter.** `RobotProprioception` carries fields for all of the above, but `IsaacLabCubeStackPerceptionAdapter`/`DynamicsAdapter` don't populate them yet, and the reference dynamics adapter only predicts the swept Cartesian path -- it doesn't do joint-space or differential-IK prediction at all. That means `cartesian_speed_within_limits` could be made meaningful today; the joint-space checks (position, velocity, effort) and self-collision cannot be, until the adapter predicts joint-level motion rather than just interpolating an end-effector position. Wiring hardcoded joint limits into the adapter now, without reading them from the robot's own articulation data, would be worse than leaving the gap explicit -- a wrong number creates false confidence, which is exactly what this design exists to avoid. *Updated (v0.3.1):* the joint position limits and the Cartesian speed rating are now reported by all three reference adapters and wired into the example schema (not for ANYmal-C's joint limits, which its simulation asset doesn't define), with the reference dynamics adapters predicting from the commanded speed. The rest of this paragraph still holds for the other self-limits. See "Live Re-validation of the Wired Set".

This is deliberately not added to `configs/example_action_schema.yaml`'s `grasp`/`place` checks yet, for the same reason: requiring a check the adapter can't yet answer would either block everything or silently misrepresent what's actually being verified.

## Regulatory Mapping

The Prior Art section cited ISO/TS 15066 and ISO 10218 by name without implementing either faithfully. This section closes that gap for the one clause that maps directly onto what's already built, and is explicit about what still doesn't.

**Implemented: ISO/TS 15066:2016 Annex A, Speed and Separation Monitoring.** `iso15066_separation_distance_maintained` computes the standard's actual protective separation distance, S(t0) = Sh + Sr + Ss + C + Zd + Zr — the human's own reach during the system's reaction interval, the robot's travel during that same interval, the robot's stopping distance once decelerating, a fixed intrusion allowance, and position-uncertainty terms for both the human and the robot. This supersedes `swept_path_clear_of_agents`' flat margin for human/animal proximity specifically, the same way `destination_confirmed_stable_and_clear` superseded `surface_confirmed_stable` earlier — both stay registered and wired; either can block.

**What this is not:** the numeric defaults (reaction time, sampling interval, deceleration, intrusion distance) are representative literature values for the *structure* of the calculation, not a certified figure for any real deployment. ISO/TS 15066 compliance means a qualified safety engineer characterizes the actual system's reaction time and sets these from the standard's current edition — this module makes that a small set of named, documented parameters instead of a hand-picked margin, but it does not make the module "ISO/TS 15066 certified" on its own, any more than the Commercialization section's certification-packaging idea is itself a certificate.

**Not yet implemented, named honestly rather than skipped:**

- ISO/TS 15066's Power and Force Limiting clause (biomechanical contact-force and pressure limits per body region) — would need an estimate of collision force from robot effective mass and approach speed, a real but more speculative addition than the separation-distance formula above.
- ISO 10218-1/2's broader industrial robot safety requirements — most of what applies is hardware-level (E-stop, safety-rated monitored stop), already scoped out of this software layer in Prior Art and Open Questions above.
- ISO 12100's risk-assessment methodology (severity × exposure × avoidance possibility) as a formal rating of each hazard this module addresses — the hazard checklist and stress-testing sections above are the informal version of this; a real deployment's safety case should redo it formally, per hazard, with a qualified reviewer.

## Part 4 — Performance Specification

Measured latency, not assumed.

## Latency: Measured, Not Assumed

The Reference Module Design section flagged control-loop rate as a constraint the module can't paper over, but hadn't been measured against the real wired schema. It now has:

| Scenario | mean | p50 | p99 |
| --- | --- | --- | --- |
| Normal scene (1 object, no agents) | 20.9μs | 16.0μs | 27.5μs |
| Stress: 50 bystander objects + 10 tracked agents + 50-point trajectory | 506μs | 495μs | 605μs |

The median cost is not a concern at any realistic control-loop rate; the harness runs once per proposed action, not once per joint-servo tick. The two swept-path checks (`swept_path_clear_of_agents`, `swept_path_clear_of_risky_objects`) are the hot path -- each is O(tracked entities × trajectory points) -- so scene *population* (many tracked people or objects), not trajectory resolution, is what would actually scale this up.

**The one real finding: occasional multi-millisecond tail latency, confirmed to be garbage-collection pauses, not the harness's own logic.** With Python's GC on, worst observed was \~28ms across 20,000 calls; disabling it dropped that to \~1.5ms with no change to the median. That's integration guidance, not something to bake into `ActuatorGate` itself: `gc.freeze()` after process startup, or a scheduled manual `gc.collect()` during a known-idle window, rather than leaving automatic collection to fire mid-cycle. Disabling GC globally is the host application's call, not this module's to impose.

For the live Isaac Lab deployment specifically: perception's own `get_world_state()` call (0.7–1.1ms per env, measured in the live validation run above) costs more than the harness's own decision logic does in a normal scene. **Today, perception is the bottleneck, not the harness.**

## Part 5 — Integration Guide

How to wire this into a robot stack that isn't the Isaac Lab reference implementation.

## Integration Points

The harness sits at one place in any robot's control stack: between whatever produces a proposed action (a learned policy, a planner, a teleoperation input) and whatever executes it (the motor or actuator command interface). It does not need to know how the action was produced — only what it proposes to do next, and enough current world state to check it against.

Concrete integration points, in the order a control loop reaches them:

- **Action interface boundary.** Wrap the single function or message where a proposed action leaves the policy or planner and enters actuation — the `step()` call, the joint-command publisher, the action-server callback. The harness intercepts here and reads the proposed action; it does not touch the policy's internals.
- **Perception feed.** Subscribe to whatever the robot already publishes as perceived state — object detections, human tracking, proprioception, camera or depth topics. The world-state layer consumes existing sensor and perception output; it doesn't add new sensors.
- **Robot dynamics or kinematics model.** The trajectory forward-check needs the same forward-kinematics or physics model the robot's own motion planner already has — reuse it, don't duplicate it.
- **Actuator command path.** The harness's permit, deny, or fallback decision is what actually reaches the motors — this is the one place its authority must be absolute: even if every upstream check were somehow bypassed, this final clip is the last line before physical motion.
- **Logging and telemetry sink.** Write every decision — permitted, blocked, fallback triggered, why — to whatever the robot already logs to. This needs no new infrastructure, just a new event type.

Making this portable across different robots depends on keeping three things robot-specific and swappable behind one fixed interface:

1. **The world-state schema is fixed; the perception filling it is not.** Every robot implements the same structured output — objects and attributes, tracked agents and uncertainty, robot proprioceptive state, environment signals — regardless of what sensors or models produce it. A wheeled arm and a bipedal humanoid both produce this same shape; only the humanoid additionally fills the balance and support-polygon fields.
2. **The precondition table is per-robot; the checking engine is not.** `grasp`, `step`, and `reach-near-human` are declared per platform — a wheeled robot has no `step`, a fixed-base arm has no balance constraint — but the code evaluating "are this action's declared preconditions confirmed" is the same engine for every robot, reading a config rather than running different code per platform.
3. **The fallback behavior is per-robot; the trigger logic is not.** What "safe fallback" means differs by embodiment — freeze, complete-current-step-then-halt, retract — but the decision of when to invoke it is the same shared monitor logic.

That separation — a fixed interface contract (world-state schema, decision API, logging format) with robot-specific content behind it — is what lets one harness codebase sit in front of many different robots, the same way a fixed hardware safety-rated-monitored-stop interface sits in front of many different industrial arms today: the interface is standard, the sensors and stopping behavior behind it are not.

## Reference Module Design

Package this as a small library with a narrow public surface: one shared data schema, four adapter interfaces a robot team implements, and one decision engine they configure but never fork.

**Interfaces a robot team implements** — the only integration work required:

| Interface | Method | What it wraps |
| --- | --- | --- |
| `PerceptionAdapter` | `get_world_state() -> WorldState` | The robot's existing detection, tracking, and proprioception, mapped into the fixed schema |
| `DynamicsAdapter` | `predict_trajectory(action, horizon) -> PredictedTrajectory` | The robot's existing forward-kinematics or physics model |
| `FallbackController` | `execute(state) -> Action` | The robot-specific safe fallback — freeze, retract, complete-step-then-halt |
| `Logger` | `record(decision)` | Whatever telemetry sink the robot already writes to |

**What ships built in, and is configured rather than rewritten per robot:**

- The `WorldState` schema itself — versioned, not reimplemented per robot.
- The decision engine (`ActuatorGate`): runs the precondition checks and the trajectory forward-check, returns permit or block-plus-fallback. This is the one piece every integration shares byte for byte — letting each robot team fork it defeats the point of keeping it small enough to test exhaustively.
- A library of generic precondition checks written only against `WorldState` and `PredictedTrajectory` — swept-path-clear-of-humans, mass-within-force-budget, balance-margin-maintained, surface-confidence-above-threshold — reusable across robots because they're written against the shared schema, never a robot's native representation.
- A declarative action-schema config (YAML or JSON): a robot team lists its action types and which checks apply, by name, from the built-in library. A genuinely new precondition is a registered function, not a change to the engine.

**The integration itself, in an existing control loop:**

```
# since 0.3.0 the config is pinned to a known-good digest kept outside it; an unpinned registry
# blocks every action (config_integrity_verified)
schema = ActionSchemaRegistry.from_yaml("config.yaml", expected_digest=read_digest_file("config.yaml.sha256"))
watchdog = DecisionWatchdog(fallback_controller, deadline_s=control_cycle_s)
harness = ActuatorGate(perception_adapter, dynamics_adapter,
                        fallback_controller, logger,
                        action_schema=schema, watchdog=watchdog)

# the one line inserted before any action reaches actuators
harness.gate(proposed_action)
execute(watchdog.command())  # the permitted action while fresh and bit-identical to what was checked; else freeze
```

**Distribution choices that keep it portable across stacks, not just one robot:**

- No hard dependency on any specific ML framework or simulator in the core engine — an adapter can use whatever it needs internally to satisfy its interface, but the engine itself stays framework-agnostic and small.
- A thin ROS2 wrapper package alongside the core library, subscribing to standard topic types and calling `gate()` — for most teams already on ROS2, that wrapper is the actual integration point, not the raw library.
- One worked reference adapter (against a simulator) shipped as a copyable starting template, not just the interface spec — teams integrate faster from a working example than from an abstract contract.
- The black-box and fuzz test suites from Testing and Validation (`tests/test_blackbox.py`, `tests/test_fuzz.py`) are genuinely adapter-interface-generic today — they reflect over the public `schema.py` dataclasses and never reference a specific adapter or check by name, so a team can clone the repo and run them against their own `PerceptionAdapter`/`DynamicsAdapter` output to self-check does its `WorldState` satisfy the schema, does its dynamics prediction look sane — before ever trusting the harness with their robot. **What this is not, yet:** a separately packaged/installable fixture, a formal third-party submission or acceptance process, or anything that issues a compliance mark — those are roadmap, not current state (see Commercialization).
- The `WorldState` schema and the action-schema config format are versioned with an explicit migration path. "Fixed interface" has to mean stable and versioned, not frozen forever, or the module can't evolve without breaking every existing integration.

**A constraint the module can't paper over:** control-loop rate. The gate typically runs once per proposed action, well below joint-servo rate, so a high-level-language implementation is fine on most platforms — but a fast-moving humanoid with a tight action cadence may need the engine's hot path compiled or otherwise low-latency. Each team makes that call for its own timing budget; the module doesn't hide it.

## Stack-by-Stack Integration

The four adapter interfaces are the same everywhere; what changes per stack is only what each adapter wraps.

| Stack | `PerceptionAdapter` wraps | `DynamicsAdapter` wraps | `FallbackController` wraps | `Logger` wraps |
| --- | --- | --- | --- | --- |
| ROS2 | Standard perception topics (detection/pose arrays, TF for tracked frames) and joint-state topics | The kinematics library the existing motion planner already uses, against the robot's URDF | The robot's own action-server trajectory interface, commanding a hold or retract | The existing diagnostics topic |
| ROS1 | The same shape, over topics and services instead of DDS — the adapter code differs, the `WorldState` it produces does not | The existing FK/IK service | The existing trajectory action interface | `rosout` or an existing bag-recorded topic |
| Proprietary industrial-arm SDK, no ROS | The vendor SDK's own state-query calls | The vendor SDK's forward-kinematics call, or a URDF-based library shipped alongside it | The vendor SDK's own stop or hold command | Whatever the vendor SDK already logs to, or a local file |
| Simulation-first stack (Isaac Lab, MuJoCo, PyBullet) | Privileged simulator state read directly — useful for validating the engine before any real perception is involved | The same simulator, run in a lightweight non-training rollout mode | A scripted hold or retract policy in the same simulator | Whatever the simulation harness already logs |
| Bare or embedded custom stack, no framework | Whatever sensor-reading calls already exist — the most integration work of any row, since there's no framework to lean on | A hand-rolled or library forward-kinematics function | A hand-written safe-stop routine | A local log file or serial diagnostic output |

In every row, the adapter's job ends at producing a `WorldState`, a `PredictedTrajectory`, or executing a `FallbackController.execute()` call — it never reimplements the decision engine itself.

## Getting Started

1. Install the core library and, if applicable, the stack-specific wrapper package — for example the ROS2 wrapper — alongside your existing robot code. No change to existing code is required at this step.
2. Implement `PerceptionAdapter.get_world_state()` against whatever perception your robot already runs. Start narrow: a stub returning only proprioceptive state, with no object or human tracking yet, is enough to bring the engine up.
3. Implement `DynamicsAdapter.predict_trajectory()` against your existing kinematics or physics model.
4. Write an `action_schema.yaml` covering just one action type to start — the one you consider lowest-risk — referencing the built-in precondition checks by name. Pin it: `python -m safety_harness.pin action_schema.yaml > action_schema.yaml.sha256` writes the digest you load it with (`expected_digest=`); since 0.3.0 an unpinned configuration blocks every action. Re-pin only after a reviewed change.
5. Implement `FallbackController.execute()`. For a first integration, "freeze in place" is enough if your platform is fixed-base or currently in a stable stance.
6. Run the black-box and fuzz test suites (`tests/test_blackbox.py`, `tests/test_fuzz.py`) against your adapters before wiring anything to actuators. Fix whatever they flag first.
7. Wire `harness.gate()` in front of that one action type only — executing through `DecisionWatchdog.command()` (or `verify_decision_action()`), never the proposal directly — with logging on, and watch the decision log before trusting it — in simulation first, then on hardware with a supervised kill switch, before adding more action types.

Never wire all action types at once on a first integration. One type, watched closely, first.

## Part 6 — Validation & Test Record

Every test suite and every live-simulation hazard run, with the actual pass/fail criteria and results.

## Testing and Validation Strategy

Because the precondition and trajectory-check layers are explicit and rule-based rather than learned end to end, they can be validated before deployment the way any critical software is: construct a world state by hand, propose an action, and assert the harness blocks or permits it correctly. Build this as a real test suite, covering at minimum:

- Every action type's positive case (all preconditions confirmed, correctly permitted) and at least one negative case per precondition (that precondition unconfirmed, correctly blocked).
- Degraded-perception cases: low confidence on each input a precondition depends on, confirming the harness fails closed rather than defaulting to permit.
- Adversarial and boundary cases: a human at the edge of the exclusion margin, an object just above the mass threshold, a foot target just below the terrain-confidence gate.
- Replayed real sensor logs from prior operation, to catch cases the hand-constructed test states missed.

This is the payoff of keeping the gating logic explicit instead of learned: a neural safety classifier can't be exhaustively tested this way, because there's no way to enumerate what it has and hasn't actually learned to recognize.

## Live Validation Results (Isaac Lab, 2026-09-26)

Development Roadmap stage 3, actually run against a live environment (`Isaac-Stack-Cube-Franka-IK-Rel-v0`, 8 parallel envs): perception schema conformance, multi-env isolation (a perturbation to one env's cube was confirmed invisible to another), cube height against this project's own established ground truth (0.0203m, matched exactly), the dynamics adapter's side-effect-freedom and trajectory sanity, and the missing-target-position fix from before -- all passed on the first or second try.

**Two real fixes landed during this run, not just checked:**

- The gripper-aperture read now resolves finger joints by name (`find_joints("panda_finger_joint.*")`) instead of assuming the last two joint indices -- the gap flagged in the conformance checklist earlier, now fixed and confirmed (`finger_ids=[7, 8]`).
- The first end-to-end run correctly **blocked** a grasp of a harmless training cube, because the adapter never asserted `cleared_for_interaction`, `supported_stably`, or `fall_consequence` for its own cubes -- default-deny working exactly as designed, just not usefully. Fixed by having the adapter assert what's actually known about these specific, controlled props (cleared, no fall consequence) and by computing `supported_stably` from a real measured signal -- vertical velocity near zero -- rather than assuming it. A second run then correctly permitted the grasp, all 8 wired preconditions satisfied.

That second fix is worth normalizing: it's an example of an adapter needing to *earn* a permissive field from real evidence about its specific deployment, not inherit a permissive default -- exactly the distinction the Design Principle section draws, now confirmed live rather than only in fixtures.

## Stress Testing Without Presupposing the Failure Mode

Every test up to this point was written knowing exactly which precondition it would trip -- useful for confirming each check works, but it can't tell you whether the harness catches something nobody specifically coded a check for. Three techniques address that directly, none of which require deciding in advance what should fail:

- **Mutation testing**: take one confirmed-good scenario, corrupt one field at a time, assert only that *something* blocks it -- never which check by name.
- **Random fuzzing against structural invariants**: generate thousands of random, often nonsensical world states and check properties that must hold regardless of the specific scenario (chiefly: permit implies every wired result was satisfied; nothing raises an unhandled exception).
- **Pathological input**: malformed data nobody wrote a targeted check for (NaN coordinates, an empty predicted trajectory, `robot=None`, a target id that doesn't exist, a third party's precondition function that itself has a bug and raises) -- the bar is fail safe, never silently permit or crash the control loop.

**Three real gaps found this way, all fixed:**

1. Nothing checked the target's *pose* confidence, only its *class* confidence -- `object_pose_confirmed` added, and extended to reject a non-finite (NaN/Inf) position outright, independent of whatever confidence score came with it.
2. An empty predicted trajectory silently permitted, because every swept-path check loops over `trajectory.points`, and a loop over zero points returns "no violation found" rather than "no basis for a decision." Fixed at the engine level: zero predicted points is now its own explicit failure, checked before any precondition runs.
3. Nothing checked that `state.robot` even exists before permitting a grasp -- `robot_state_confirmed` added as a general gate, alongside visibility.

**One honest limitation of the fuzzer itself, not the harness:** the random generator always produces *some* robot state, so `robot_state_confirmed` never failed across 3,000 random iterations -- it's exercised only by the dedicated pathological test, not the random sweep. Worth strengthening the generator later rather than treating the 0% fuzz-failure rate as suspicious on the harness's part.

All 61 tests (49 targeted + 12 stress) pass. The stress suite lives in `tests/test_fuzz.py`.

## NaN-Sensor Stress Test (2026-09-27): A Comparison-Based Default-Deny Bypass

Every numeric precondition in `preconditions.py` enforces default-deny through a `<`/`>` comparison against a sensor-derived float, and IEEE-754 makes every such comparison against NaN `False` -- so a NaN in the one field a check exists to guard never trips that check's failure branch. `object_pose_confirmed`'s existing NaN guard (Stress Testing above) fixed this for one field on one check; this pass checked whether it generalized. It mostly didn't. Full reproduction against the real `ActuatorGate.gate()` (not the precondition functions in isolation): `tests/test_adversarial_stress.py`.

- **`mass_within_force_budget` -- confirmed, standalone.** `obj.estimated_mass_kg = nan` makes `gate()` PERMIT a grasp of an object whose mass was never actually measured. Nothing else registered for `grasp` looks at mass at all, so this is its only guard and there is no second check to catch what it misses.
- **`object_hazard_confirmed` / `object_pose_confirmed` -- the confidence field itself, not just position.** A NaN `class_confidence` or NaN `pose_confidence` both bypass their respective confidence gates (`< min_confidence` is `False` for NaN) the same way a NaN *position* did before the existing fix -- that fix cross-checked position against `math.isfinite`, but nothing cross-checks the confidence score itself the same way.
- **Compound: a single NaN agent-position coordinate defeats `swept_path_clear_of_agents` and `iso15066_separation_distance_maintained` simultaneously** -- both individually confirmed reporting `satisfied=True` on it. For a standard-speed human (1.5 m/s) the overall decision still blocks, but only because `iso15066_power_force_limiting`'s force estimate exceeds 150N at that speed regardless of true distance, not because that check is immune to the same bug: its own "too far to matter, skip" gate (`d > 0.3m`) is defeated by the same NaN in the *opposite* direction (it stops skipping instead of stops blocking). Swap in a slower-moving tracked agent (0.15 m/s -- a mobile-base teammate, not a pedestrian) and the coincidence disappears: `gate()` fully PERMITs a grasp passing 0.19m from that agent -- inside the ISO/TS 15066 separation distance -- with every registered check reporting satisfied.
- **`balance_margin_maintained` -- same pattern, latent.** Not wired into the example `grasp`/`place`/`reach` schema at the time (no legged platform in that reference config; the later `navigate` action type wires it, now as `stability_margin_maintained`), so it couldn't be reached through `gate()` then, but a NaN center-of-mass component defeats it directly at the function level the same way -- worth fixing before any humanoid/legged adapter registers it, not after.

**The root cause is one line, repeated across most of the file.** Each check above writes its guard as `if measured_value < threshold: fail()` and trusts that a corrupted or never-actually-measured value lands on the failing side. NaN doesn't -- it fails every comparison, in both directions, which is exactly backwards from what a default-deny gate needs. The fix isn't per-check; it's one shared primitive (treat any comparison operand that fails `math.isfinite` as an automatic, explicit fail, before the numeric comparison runs at all) applied everywhere a sensor-derived float currently flows straight into `<`/`>`. Not applied yet -- this pass found and reproduced the gap, it didn't patch it.

**Fixed, validated, and released (2026-09-27):** all four findings above are fixed in `preconditions.py` via three shared comparison helpers (`_below`/`_exceeds`/`_at_or_within`) plus a `_safe_max` helper for two places Python's own `max()`/`min()` could silently drop a NaN candidate depending on iteration order, and a direction-aware fix to `iso15066_power_force_limiting`'s own "too far, skip" gate (which needed the opposite NaN handling from every other comparison in the file). Validated three ways: the full suite plus 14 new regression tests in `tests/test_adversarial_stress.py` passing locally; the same checkout's tests passing against the Isaac Lab GPU box's own bundled Python; and 5 live fault-injection scenarios (one field, or one injected tracked agent, corrupted to NaN on top of real, live cube/robot state) run through the real `IsaacLabCubeStackPerceptionAdapter` in a running Isaac Sim environment on `isaac-launchable-1aebd7` -- all pass. Released as [v0.2.2](https://github.com/naganumakr/safety-harness/releases/tag/v0.2.2) and published to [PyPI](https://pypi.org/project/safety-harness/0.2.2/).

## Hazards Generated Live in Isaac Sim (2026-09-26)

A different exercise from the black-box data-level fuzzing above: real simulated conditions -- actual object velocities, actual poses, actual robot joint state -- driven through the real `IsaacLabCubeStackPerceptionAdapter`/`DynamicsAdapter`/`ActuatorGate`, not hand-built fixtures.

| Hazard | What was actually done in the simulator | Result |
| --- | --- | --- |
| Baseline | Nothing disturbed | Permit |
| Real falling cube | Cube given an actual -0.5 m/s vertical velocity | Block (`current_position_confirmed_stable`) |
| Real sideways knock | Cube given an actual 1.5/0.8 m/s lateral velocity | **First run: permitted -- a real bug** |
| Bystander proximity sweep, unflagged | A second cube moved to 2–50cm from the target | Permitted at every distance (expected -- see below) |
| Bystander proximity sweep, hazard-flagged | Same sweep, with the bystander's classification overridden to uncleared/hazardous-release via a wrapper adapter | Blocks at ≤20cm, permits at ≥30cm -- matches the configured math exactly (0.05m swept-volume radius + 0.15m margin = 0.20m) |
| Robot driven to within 0.01 rad of a joint limit | Real joint positions written via the simulator | Permits -- the known, already-documented gap (joint checks aren't wired into this schema), not a surprise. *Since v0.3.1: wired, and this exact condition BLOCKs live -- see Live Re-validation.* |
| Commanded motion implying \~40 m/s | An unreachable target with a 0.05s horizon | Permits -- same known gap, Cartesian speed checks aren't wired either. *Since v0.3.1: wired, adapters predict the commanded speed, and a ~40 m/s command BLOCKs live.* |
| Compound: falling cube + close hazardous bystander | Both at once | Blocks on both reasons simultaneously |

**One real bug found, fixed, and reconfirmed live:** the sideways-knock case initially permitted. `supported_stably` checked only vertical velocity (`abs(vel[2]) < 0.05`) -- a cube sliding across the table at 1.5 m/s isn't falling, so it read as stable. Fixed to check total speed instead of just the vertical component; rerun against the same live disturbance now blocks correctly.

**One apparent gap that turned out to be a test-design gap, not a harness bug:** the first bystander sweep permitted at every distance down to 2cm. The reference adapter marks every cube it sees as cleared and harmless by default (a deliberate, documented choice for this specific controlled task -- see Live Validation Results above), so moving an already-harmless object closer was never going to trigger anything. Re-run with a wrapper adapter that overrides one object's classification to genuinely hazardous, the same sweep produces a clean, sharp threshold that matches the configured margin exactly. Worth remembering generally: a black-box result that looks like a miss is sometimes the test's setup, not the system under test -- check which one before reporting a finding.

## Environmental and Regulatory Hazards, Live in Isaac Sim (2026-09-26)

A broader sweep than the earlier hazard videos: real light dimming, two real physical stand-ins (a human-sized and a pet-sized capsule, spawned into the scene config, tracked from their actual measured position/velocity), simulated sensor dropout, and dense multi-hazard clutter — run against the newly added ISO/TS 15066 checks, not just the original flat-margin ones. Every scenario below states its pass/fail criterion explicitly and was re-verified step by step, not just checked at a final frame.

| Hazard | Criterion | Result |
| --- | --- | --- |
| 1. Lighting degradation | Block only once visibility drops below the configured 0.5 threshold | **Pass** — crossed to block at visibility=0.492, just below threshold |
| 2. Human approaching | Permit while far, block once close | **Pass** — permit at step 0, block at step 25/70 |
| 3. Pet approaching (faster, smaller) | Same, accounting for the pet's higher worst-case speed requiring a larger safe starting distance | **Pass on the second attempt** — see below |
| 4. Stale tracking, proxy stationary | Permit at zero staleness, block from staleness alone, object never moves | **Pass** — permit at step 0, blocked at step 13 from staleness alone (measured worst-case radius 3.45m against a 3.5m starting distance) |
| 5. Dense multi-hazard clutter | Multiple simultaneous hazard sources each contribute a reason | **Pass** — 4 distinct checks fired at once; no explicit no-clutter control was run alongside it, worth noting |

**Hazard 3 failed its own criterion on the first attempt, and it was a test-design bug, not a harness bug.** The pet was given `worst_case_speed_mps=2.5` (versus 1.5 for the human, on the reasoning that pets dart faster) but started at the same \~2.5m distance used for the human. At 2.5 m/s, the swept-path check's own 1-second prediction horizon alone requires \~2.7m of clearance — the pet was inside that radius before it ever moved, so the scenario blocked from step 0 and never demonstrated an approach at all. Moved the start to \~4.6m; second run passed cleanly. The lesson generalizes: a faster worst-case speed rating pushes the *always-blocks* radius out proportionally, and a hazard scenario's starting distance has to be chosen relative to that, not copied from a slower agent's scenario.

### New regulatory checks added this round

- **`iso15066_separation_distance_maintained`** — replaces the flat margin with ISO/TS 15066 Annex A's actual protective separation formula, S(t0) = Sh + Sr + Ss + C + Zd + Zr. Registered alongside the original flat-margin check; either can block. At this parameterization the original flat margin is actually the *more* conservative of the two (governs first) — not a bug, just worth knowing before assuming the newer formula is always the binding one.
- **`iso15066_power_force_limiting`** — a simplified proxy for the standard's biomechanical force-limiting intent (estimated transient contact force from effective mass and relative speed), gated to only apply when an agent is within a plausible contact range. Found and fixed a real bug the moment it was wired in: the first version had no proximity gate at all, so a tracked agent 7m away still failed the check as if a collision were imminent.
- **`reduced_speed_near_human`** — an ISO 10218 / ANSI-RIA R15.06 style collaborative speed cap: commanded speed must drop below a configured limit whenever any tracked agent is within a defined zone, independent of the separation-distance calculation.

All three carry the same caveat as the rest of this section: representative literature defaults for the *structure* of each calculation, not certified figures — see Regulatory Mapping above.

**A regulatory category this harness doesn't address at all, flagged rather than solved:** tracking real human positions and logging every decision means this system processes personal data. Data protection law (GDPR, CCPA, and similar) is a genuinely different regulatory axis from safety standards — retention limits, anonymization, consent — and nothing here handles it. A real deployment tracking real people needs this addressed separately from everything in this design doc.

## Next Safety Checks (2026-09-27): Liveness, Perception Integrity, Cyber Integrity, Vulnerable Bystanders

Eight checks from the next-safety-checks roadmap, each written against a concrete unsafe scenario that every existing check PERMITs -- replayed against the unmodified v0.2.x release code before the new check existed, then shown to BLOCK once it's wired. See tests/test\_adversarial\_next\_checks.py (104 tests: gets-past-today, blocked-now, NaN/None/missing fail-closed, and exact boundaries for each).

| Check | Unsafe scenario every existing check permits |
| --- | --- |
| `decision_within_deadline` + `DecisionWatchdog` | A decision stalls 0.5s in dynamics and still PERMITs; or gate() hangs/crashes and the last PERMIT never expires. |
| `payload_and_grip_force_within_limits` | 60N grip on a fragile cube; 1.5kg on a robot rated for 1kg (under the config's flat 3kg budget); 1.5kg held with 10N (slips). |
| `stability_margin_maintained` | Quadruped CoM statically 8cm inside its support polygon but moving 0.6m/s toward the edge: capture point outside the polygon. |
| `sensor_data_fresh` | A WorldState assembled just now from a 2-second-old sensor frame. |
| `swept_path_observed` | Visibility 1.0 and no agents detected -- but the path runs through space no sensor observed. |
| `command_integrity_verified` | The action's params rewritten during checking (or after PERMIT): the checked command isn't the executed one. |
| `config_integrity_verified` | The force budget edited 3kg -> 300kg on disk or in memory, or a REGISTRY entry rebound to an always-OK function. |
| `vulnerable_bystander_protected` | A child 2m from the path; two adults at 1.8m; \~133N contact at an adult's face height (flat 150N limit, ISO/TS 15066 face limit 65N). |

Design choices worth knowing before relying on them:

- **Liveness is two halves.** `decision_within_deadline` catches a decision that finishes late (measured from gate() start on the gate's monotonic clock, not from `WorldState.timestamp`). `DecisionWatchdog` catches one that never finishes: a dead-man's switch on the actuator side that releases a PERMIT only while it is fresh and bit-identical to what was checked, and otherwise returns a freeze -- with an optional monitor thread to actively push a stop. Neither replaces a hardware watchdog timer if the whole process hangs.
- **Checks can now receive a `CheckContext`** (decision start time, clock, start-of-decision action digest, running registry) -- only checks that declare `wants_check_context`, and never from YAML. It's how the timing and integrity checks see facts no WorldState can carry.
- **Command integrity is bound, not assumed.** gate() hashes the proposed action before anything runs and binds that digest into `Decision.action_digest`; `integrity.verify_decision_action` (and the watchdog) re-verify at execution. Optional proposer seals (HMAC-keyed) cover the leg before the gate.
- **Config integrity hashes the *effective* configuration** -- each check's name, the function REGISTRY binds it to, and its code defaults overlaid with YAML kwargs -- so a changed default in code is caught too. The pin lives outside the config (`configs/example_action_schema.yaml.sha256` for the example; regenerate with `python -m safety_harness.pin configs/example_action_schema.yaml > configs/example_action_schema.yaml.sha256` after a reviewed change). Unkeyed SHA-256 detects corruption and uncoordinated edits; only an HMAC key held apart from the config makes the pin a signature.
- **Stability**: `balance_margin_maintained` stays registered as the static-only half of `stability_margin_maintained`, unchanged in behavior. 32 registry names; 31 distinct checks.
- **Vulnerable bystanders**: ISO/TS 15066's Table A.2 body-region limits apply to adults only; children, animals and unclassified agents get no-contact and a larger clearance. The clearance/crowd numbers are structural placeholders a qualified engineer must set -- the standard has no child or crowd provisions to take them from.

Validated in unit/adversarial/fuzz/black-box tests. The Franka and ANYmal-C reference adapters now report `sensor_timestamp`, `observed_regions`, rated payload and (ANYmal) base velocity as CoM velocity, but those lines haven't been executed against a live environment. The first live contact is the G1 example below. The new checks ran inside a live Isaac Lab control loop and fired, both `swept_path_observed` and `decision_within_deadline`. That was nominal smoke only, not a hazard campaign.

## Unitree G1 Learned-Policy Example (2026-09-27): Gating a Trained Policy, First Findings

A third robot, and the first **learned** policy put behind the gate: a Unitree G1 (fixed base, left arm, 3-finger Dex3 hand) that picks a blue block and stacks it on a red one. Grasping is real contact physics, with no kinematic attach.

The policy is behavior-cloned from a scripted expert and fine-tuned with PPO. It stacks in 82.7% of 1024 randomized episodes when stopped at a strict, sustained, released success. If left running it knocks its own tower down (1/1024 still standing at t = 10 s), so it is not yet a robust policy. Full results, reproduction steps and the task/scripts/checkpoints are in `examples/isaac_lab_g1_stack/`.

The harness-side findings from the first nominal gating runs are the part that matters here. Each is a property of running this harness around *any* continuous policy, not a G1 quirk:

- **Decisions must be segmented, not sampled.** Gating `grasp` on every control step while the grip closes deadlocked the policy (3,838 blocks in one run). Mid-grasp, the object moves in the fingers and is never "confirmed stable". Edge-triggered segmentation fixes the deadlock: `grasp` on the closing edge, `place` on the opening edge, `reach` otherwise. *Addressed (v0.3.3):* `safety_harness/segmentation.GripActionSegmenter` extracts exactly this logic into the package, with a test suite that reproduces the validated Franka closed-loop measurement's own action-type sequence step for step -- integrators no longer have to discover or hand-roll edge-triggering themselves. Not yet re-applied to the G1 example's own driver script (that script lives in `examples/isaac_lab_g1_stack/`, not re-run this pass -- see that example's README for its own current status).
- **`swept_path_observed` needs known-solid space.** With the observed region set to the tabletop volume a camera would realistically see, it blocked ~98% of nominal near-table reaches. The margin sphere around any near-table path dips under the tabletop, which is unobservable but also physically unoccupiable. The G1 adapter declares the whole privileged-sim volume observed, which is honest for simulation only. Real perception needs an "occupied/solid" region type the check can subtract. *Addressed in 0.3.1 (code, unit-tested; not yet re-run live):* `WorldState.solid_regions` takes `KnownSolidRegion` boxes of static solid geometry (the tabletop) from the cell's known layout. Coverage is now the exact union of observed and solid boxes over the swept sphere's bounding box. Solid space adds coverage only where no agent can be, and never substitutes for observation. Two observed boxes that share a face also now count as covered; a real gap between them still blocks.
- **Nominal false blocks remain and must reach ~0 before any hazard result means anything.**
  - Grasp onsets are still sometimes blocked by `current_position_confirmed_stable`.
  - Nominal places are blocked by `destination_confirmed_stable_and_clear`. *Correction (0.3.1):* an earlier version of this line guessed it counted the held object as destination clutter; it doesn't (it has always excluded the action's own `object_id`, now pinned by a regression test). The remaining causes are the check's other two conditions: the red block not confirmed stable (it can be nudged as the hand lowers onto it) or low pose confidence. Those runs recorded only check names, not reasons, so which one is still open; the next G1 run must log reasons.
- **One `decision_within_deadline` block per run** came from the first gate() call's Python warm-up. That is a real reason to measure and budget first-call latency on a live robot.

The hazard campaign is wired but not yet run: adult hand, child bystander, 5 kg object, SHARP tag, NaN pose, low visibility, stale sensor data, occluded or unstable destination, command tamper and config tamper, each gated vs ungated. It is next, after the nominal false-block issues above.

## Live Re-validation of the Wired Set (2026-09-27, v0.3.1)

Everything above that ran live did so on earlier code, and an audit found that "validated live" had been claimed more broadly than it was earned. Of the original 24 checks, 13 had blocked something in a live run, 2 had only ever passed live, 1 had never run live, and 8 were never wired into any live config. So this pass re-ran every earlier live confirmation on the v0.3.1 code, wired two self-limit checks that had been live-demonstrated as missing, and gave every wired check at least one live BLOCK attempt. Each attempt had a PERMIT control where one applies. It ran on a live `Isaac-Stack-Cube-Franka-IK-Rel-v0` and a live `Isaac-Navigation-Flat-Anymal-C-v0`, through the real reference adapters and `ActuatorGate`, with the pinned example config.

- **Franka: 33 of 34 scenarios behaved as expected. ANYmal-C: 10 of 10.**
- Each scenario states the check it must fire; the recorded result lists exactly which checks fired.
- Scripts and raw results: `examples/live_validation/`.

**How the hazards were created.**
- **Physical:** the simulator state itself was changed. Examples: cube velocities, a cube moved next to the target, a joint written to its limit.
- **Commanded:** the proposed action itself asked for it, such as a 40 m/s reach or a 60 N grip.
- **Injected:** a wrapper overrode one field of the otherwise-real WorldState, such as an agent, a NaN, visibility, a sensor timestamp or observed coverage. The earlier live runs used the same technique.

**Result, over the 31 distinct checks.**

| Tier | Count | Checks |
| --- | --- | --- |
| Confirmed BLOCKING live | **24** | **Physical:** `current_position_confirmed_stable`, `swept_path_clear_of_risky_objects`, `destination_confirmed_stable_and_clear`, `joint_position_limits_respected` (Franka).<br>**Commanded:** `cartesian_speed_within_limits` (Franka and ANYmal-C), `payload_and_grip_force_within_limits`.<br>**Injected:** `object_cleared_for_interaction`, `fall_consequence_acceptable`, `visibility_above_threshold`, `mass_within_force_budget`, `object_hazard_confirmed`, `object_pose_confirmed`, `swept_path_clear_of_agents`, `iso15066_separation_distance_maintained`, `iso15066_power_force_limiting`, `reduced_speed_near_human`, `environment_hazard_clear`, `stability_margin_maintained` (static and capture-point, ANYmal-C), `sensor_data_fresh`, `swept_path_observed`, `vulnerable_bystander_protected`, `command_integrity_verified`, `config_integrity_verified`, `decision_within_deadline` |
| Evaluated live, not the reason for any live block | 1 | `robot_state_confirmed` (see findings) |
| Wired (Franka: grasp/place/reach), not yet re-validated live | 2 | `joint_velocity_within_limits`, `joint_effort_within_limits` -- *added in 0.3.4 (2026-09-28).* Same scope caveat as `joint_position_limits_respected`: reads the dynamics adapter's per-point robot state, which the reference adapters carry as the *current* joint velocity/effort, not a forward prediction. Unit-tested against a mocked Isaac Lab `robot.data` object (`tests/test_wired_self_limits.py`); the real attribute names it depends on (`soft_joint_vel_limits`, `joint_effort_limits`, `applied_torque`) were confirmed against Isaac Lab's public source, not this project's installed 3.0.0 build -- a live smoke test is the remaining step. NOT wired for `navigate`: the ANYmal-C adapter doesn't report these fields yet. |
| Registered, not wired | 4 | `motor_temperature_within_limits`, `self_collision_clear`, `battery_charge_sufficient`, `surface_confirmed_stable` (deprecated in 0.3.1 in favor of `destination_confirmed_stable_and_clear`; still registered so old configs load, with a DeprecationWarning) |

Controls that PERMIT live, so these checks do not simply block everything:
- a nominal grasp and a nominal place;
- a normal-speed reach;
- ANYmal-C nominal navigation, with a human 8 m to the side, and at a commanded 0.8 m/s.

`DecisionWatchdog` was also verified live: a fresh PERMIT executes, and the same PERMIT 0.3 s later freezes.

**What wiring the two checks surfaced:**

- **The speed model was not conservative. This mattered more than either check.**
  - Both reference dynamics adapters extrapolated every motion at a fixed, capped speed (Franka 0.5 m/s). `cartesian_speed_within_limits` could therefore only ever see the cap, so a ~40 m/s command PERMITted.
  - Worse, every swept-path check under-predicted how far a faster controller actually moves within the horizon.
  - The adapters now predict from the action's `commanded_speed_mps` or `duration_s`. The cap remains only as a fallback when no speed is stated.
  - Live, on real state: a reach past a slow teammate 2.8 m ahead PERMITs under the old capped model, and BLOCKs on `swept_path_clear_of_agents` once the real 1.6 m/s command is predicted. That speed is within the Franka's 1.7 m/s rating.
- **ANYmal-C has no joint limits to check against.** The Isaac Lab ANYmal-C asset defines no position limits on any of its 12 leg joints (all −inf to +inf, confirmed in its USD). Reporting those infinities made `joint_position_limits_respected` fail closed on every step. The adapter now reports them as unreported, and `navigate` does not wire the check until real ANYmal-C datasheet ranges are supplied. The joint-limit result above is Franka-only.
- **`joint_position_limits_respected` checks the current pose only.** The reference adapters carry the current joint state on every predicted point, so the check catches "at or near a limit now", not "this command will drive a joint past its limit". It also needed an explicit per-joint exemption (a `None` entry): the Franka's gripper fingers rest on their stops by design and would otherwise have blocked every action. A limits tuple shorter than the joint state now fails closed; before, the unmatched joints were silently unchecked.
- **`robot_state_confirmed` can't be the reason for a live block with these adapters.** It fires only when robot state is absent entirely. In that case the dynamics adapter can't predict a path and raises first, so `gate()` BLOCKs on `adapter_error` before any check runs. The action is still blocked, just not by this check. It also didn't examine the state's contents, so NaN joint readings passed it; `joint_position_limits_respected` caught them instead, and BLOCKed on them live. *Fixed in 0.3.1 after this run:* it now requires finite joint readings of matching length and a finite end-effector position. That is unit-tested, not yet re-run live, so the count above still stands.
- **`iso15066_power_force_limiting` only applies within 0.3 m.** Its contact-plausibility range means an adult exactly 0.3 m from a 0.5 m/s grasp sits on the boundary. It fired in one run of that scenario and not in its identical repeat. The separation-distance, swept-path and reduced-speed checks blocked both, and it fired in the faster-reach scenario. That is expected behavior at the boundary; don't read it as a miss.
- **Still unwired:** `self_collision_clear` needs a dynamics adapter that predicts link geometry, which neither reference adapter does. Motor temperature and battery aren't modelled by these simulator tasks. *Correction (0.3.4):* an earlier version of this line also listed `joint_velocity_within_limits` and `joint_effort_within_limits` here, on the assumption that both needed motion prediction too, the same as `self_collision_clear` -- they don't. Like `joint_position_limits_respected`, both only ever needed the robot's *current* joint state plus its rated limits, which Isaac Lab's articulation data reports directly (`soft_joint_vel_limits`, `joint_effort_limits`, `applied_torque`); the real gap was that the Franka adapter simply wasn't reading them yet. Now wired -- see the "Wired, not yet re-validated live" row above.

## Closed-Loop Measurement (2026-09-27): Does the Gate Get in the Way of a Working Robot?

Every result above gates single, hand-picked decisions. This one runs a working controller in closed loop, with the gate deciding every one of its 20 Hz control steps, and compares it against the same controller ungated on the same seeds.

**Setup.**
- **Controller:** the scripted privileged-state Franka cube-stacking expert, 92–94% success on its own.
- **Why not the Franka RL policy:** it never became competent (full-stack success 0%), so it couldn't separate the gate's effect from policy failure.
- **Gating:** each step is turned into its decision (grasp on the gripper-closing edge, place on the opening edge while holding, reach otherwise, with the real commanded speed). It goes through the real adapters, `ActuatorGate` and `DecisionWatchdog`, with the pinned config.
- **Human proxy:** a physical kinematic capsule walks to the far table edge (about 0.45 m from the cubes), stays about 5 s, and leaves. It is tracked from its simulated pose and velocity as an adult with a 1.5 m/s worst-case speed.
- **Scale:** 128 episodes per condition, 60 s episodes.
- Script and raw results: `examples/closed_loop/`.

| Condition | Task success | Mean steps to success | Decisions blocked | Arm-moving steps within 1 m of the person |
| --- | --- | --- | --- | --- |
| Ungated, nominal | 123/128 (96.1%) | 444 | — | — |
| Gated, nominal | 122/128 (95.3%) | 443 | 5.1% (3,128 / 61,208) | — |
| Ungated, person approaches | 123/128 (96.1%) | 444 | — | 14,987 |
| Gated, person approaches | 121/128 (94.5%) | 661 | 28.7% | **0** |

**What it shows.**
- **Task success unaffected:** 95.3% gated vs 96.1% ungated, a one-episode difference, well inside the ±~4% interval at n = 128. Time to success is also unchanged.
- **The robot stops for the person and resumes:**
  - The ungated arm kept moving within 1 m of the person for 14,987 env-steps.
  - The gated arm didn't move within 1 m at all. It froze while the person was near and finished afterwards, about 11 s later on average.
  - The two ungated runs are identical, as they should be: the proxy has no collision.
- **Gate latency:** mean 7.7 ms per decision, p99 24 ms.

**Where the nominal 5.1% comes from.** Nominal blocks come from exactly two checks, plus 7 deadline blocks:
- **`joint_position_limits_respected` (2,305):** every one is Franka joint index 5 (`panda_joint6`) within the 0.02 rad margin of a limit during normal stacking. *Root cause confirmed (2026-09-28), not yet acted on:* this isn't a simulator quirk or an arbitrary controller choice. Franka's own published joint limits (`franka_ros`'s `joint_limits.yaml`) give joint 6 a range of **[-0.0175, 3.7525] rad** -- asymmetric and unusually tight at the lower end (about 1 degree of margin), unlike the other six joints' roughly symmetric ~±2.9 rad ranges. A wrist orientation near "0" -- an entirely ordinary orientation for a downward grasp -- sits right against that hardware boundary by construction, which is why the scripted expert (or any controller reaching this way) spends real time within 0.02 rad of it during nothing more hazardous than routine stacking. A flat, absolute-radian margin applied identically to all seven joints is therefore disproportionately strict on this one specific joint, not evidence of a defect in the check, the controller, or the margin's general design.
  Three real options, in order of how much they change vs. how much evidence backs them, deliberately left as a decision rather than made here: (a) a per-joint margin scaled to each joint's own range (e.g. a fixed *fraction* of `hi - lo` rather than one absolute radian value for every joint) -- keeps the same relative safety cushion everywhere, fixable without touching the controller, but is itself a safety-parameter change that needs sign-off and live re-validation before trusting it; (b) bias the scripted expert's (or any future controller's) inverse-kinematics null-space preference away from joint 6's limit when the arm's redundancy allows an equivalent end-effector pose -- addresses the actual root cause rather than the margin, but needs real IK work and a live re-run to confirm it doesn't just move the problem to a different joint; (c) leave it as-is: these blocks are transient, none of them cost task success in the measurement above, and a controller legitimately operating near a real hardware limit getting flagged for it is arguably exactly what a default-deny margin is for. **Not recommended: a blanket per-joint exemption** for joint 6 (the existing warning above stands) -- unlike the gripper fingers' mechanical rest-stops, joint 6's limit is a real, load-bearing hardware boundary, not a designed resting position.
- **`destination_confirmed_stable_and_clear` (816, all at the release moment):** the check blocks while the destination cube is still moving as the held cube is lowered onto it, then permits once it settles. *Root cause confirmed by code review, not yet re-verified live:* `TrackedObject.supported_stably` is computed from real linear velocity in the reference adapter (`speed = |velocity|; supported_stably = speed < 0.05 m/s` -- `safety_harness/adapters/isaac_lab.py`), so this reflects the destination cube's genuine, physically real velocity spike from being nudged as the held cube lands on or near it -- not a logic bug, and not the already-ruled-out "held object counted as its own clutter" hypothesis (that regression is pinned by a dedicated test since 0.3.1). This is the check correctly refusing to confirm stability while the destination is, in fact, still moving. Same conclusion as the joint-6 case: transient, doesn't cost task success, and arguably correct default-deny behavior rather than something to fix.
- **`decision_within_deadline` (7):** latency spikes up to about 1.5 s while four simulation processes and a training run shared the machine. A real deployment needs a dedicated core for the gate.

None of these cost task success here, because blocks are transient and the controller retries. *Revised (2026-09-28):* the root-cause analysis above suggests neither of the two nominal-block sources is actually a defect -- the joint-6 blocks reflect a real, tight hardware limit the controller legitimately operates near, and the destination-release blocks reflect a real, physically accurate momentary instability. A "false-block rate is ~0" claim may be the wrong target for this measurement, then: the honest framing is closer to "the gate correctly declines to certify a destination or joint state that briefly, genuinely isn't stable/within-margin yet, and the controller retries and succeeds anyway" -- a demonstration of the gate working as designed under normal operation, not a gap to close. Whether to change the margin or the controller regardless (for reasons other than "the check is wrong") is the deliberate decision named above.

**Not shown here:**
- a learned policy under the gate (the G1 policy is the next candidate, after its decision-segmentation fix);
- real perception;
- a person who actually collides (the proxy is non-colliding and tracked perfectly);
- hardware.

## Part 7 — Roadmap, Compliance & Change Log

What's next, the commercialization case, prior art, and the version history.

## Development Roadmap

Build the contract first, prove the abstraction holds with a second implementation before scaling to many, and don't touch real hardware until everything below it is already validated. Each stage gates the next.

1. **Freeze the data contract.** `WorldState`, `PredictedTrajectory`, `Action`, and `Decision` — before writing the engine. Changing this later is expensive once adapters exist against it; get it reviewed and versioned from the start.
2. **Build the decision engine in isolation.** The precondition-check library and the trajectory forward-check, tested entirely against hand-constructed `WorldState` fixtures — no robot, no adapter, no perception noise. This is where the testing strategy earns its cost, before there's anything real to break.
3. **Build one reference adapter against a simulator already in hand.** Privileged state, no perception noise yet — this proves the engine's logic is right before perception uncertainty is even in the picture.
4. **Build a second adapter against a genuinely different stack** — ROS2, or a second simulator — specifically to test the abstraction, not to ship an integration. If the second adapter forces a change to the core engine or the `WorldState` schema, the boundary was wrong; better to find that with two adapters than with ten. Done (2026-09-27): ANYmal-C running Isaac Lab's navigation task, via safety\_harness/adapters/isaac\_lab\_anymal.py -- see the Robot Self-Limits and Stress Testing sections for what it found.
5. **Harden.** A real, separately-packaged third-party conformance fixture (today there's only the adapter-generic black-box/fuzz tests in `tests/`, not a standalone installable one), schema versioning with a migration path, packaging, and the documentation above — all before a third integration, so every integration after this one follows a stable contract.
6. **Pilot against real perception, no real actuation yet.** Feed the engine a live camera and tracker on a stationary or simulated robot, and confirm the default-deny behavior degrades gracefully under real sensor noise — confidence gating is the piece most likely to surprise you once it meets real data.
7. **Pilot on real hardware, one action type, supervised.** A human at a reachable kill switch, the lowest-risk action type only, watching the decision log — exactly as in Getting Started — before adding a second action type or removing supervision.

## Commercialization

A per-implementation license on the core engine alone is a weak model: the engine is built from published control theory and planning concepts (Prior Art and Open Questions), so a well-resourced customer can read this document and reimplement it. The defensible value sits elsewhere.

What's actually monetizable, in roughly increasing order of moat strength:

- **Adapter breadth.** Pre-built, tested adapters across robot stacks (Stack-by-Stack Integration) are real, ongoing engineering cost — cheap to amortize across many customers, expensive for any one customer to justify building alone.
- **Certification packaging.** Mapping the black-box/fuzz tests' and live-validation evidence to ISO 10218/15066 clauses, once a real third-party-facing conformance fixture exists (see Roadmap stage 5) to actually produce that evidence for someone else's adapter, so a customer's path to regulatory or insurance approval is faster through this than built in-house — the TÜV/UL pattern, applied to this product rather than sold as raw software.
- **An aggregated precondition-rule library.** As deployments run through the correction loop (Logging, Escalation and the Correction Loop), anonymized, consented learnings improve the shared rule library for every customer — a network effect, hardest to bootstrap, strongest once it exists.
- **Liability or insurance backing.** If certification carries actual insurance-backed protection, that's real pricing power — but a materially bigger, more regulated business than the module itself.

**Recommended structure: open-core.** Release the data contract and the decision engine openly — this builds adoption and makes it the reference implementation others get compared against — and charge for adapters, certification packaging, the aggregated rule library, and support.

**Timing.** There is currently no regulatory mandate forcing physical AI companies to buy a third-party safety layer. That has historically changed abruptly once a technology category matures — autonomous vehicles are the recent precedent — and compliance tooling that existed before the mandate tends to become the incumbent once it arrives. The bet is adoption now, monetization once (if) the category is regulated, which argues for giving the core away rather than licensing it.

This has not been checked against the market: no review yet of existing competitors or actual customer demand.

## Release & Distribution

**Decision (2026-09-26): open the reference engine, the four adapter interfaces, and the existing black-box/fuzz test suites under Apache License 2.0. This supersedes the per-implementation license-fee model above** as the primary monetization path once this reaches other robot makers, for one reason: this is a safety gate other people's robots will depend on, and a closed, licensed core is a much harder trust story than an inspectable one. Precedent for safety-relevant infrastructure (OpenSSL, Kubernetes, most functional-safety tooling) backs this — credibility that drives adoption comes from being auditable, not from being paid-for.

**What's open:** `safety_harness/` (schema, engine, action-schema registry, precondition checks), the four adapter interfaces, and `tests/` (the mutation, random-fuzz, and reflection-driven black-box suites) — all Apache-2.0, in the repo alongside this doc (`LICENSE`, `NOTICE`, `pyproject.toml`, `README.md`). Apache 2.0 specifically for the explicit patent grant: cross-vendor safety infrastructure is exactly the case where an implicit-patent-license gap (which a plain MIT/BSD license leaves open) matters.

**What's monetized instead:** certification/audit services (running a real third-party-facing conformance fixture against a vendor's adapter and issuing a certification mark — nothing here does that today, not even a fixture to run; it's the Roadmap stage-5 item this decision points toward), enterprise support and SLA-backed response for production deployments, and hosted compliance dashboards aggregating `Decision` logs across a fleet. None of these gate the code itself.

**Publishing mechanics, in order:**

1. Public GitHub repo: github.com/naganumakr/safety-harness (done).
2. PyPI package: safety-harness, through 0.3.0 (done -- published via the repo's Trusted Publisher workflow on each tagged GitHub Release).
3. A published "certified adapters" registry: any robot maker's adapter that passes the black-box + fuzz suites unmodified gets listed — this is the actual "integrated into every robot's code" path, since it doesn't require them to fork or relicense anything.
4. Independent functional-safety assessment (see Scope & Non-Goals) before calling any of this "certified" rather than "reference implementation" — publishing the code doesn't substitute for this, and claiming compliance without it is the fastest way to lose the trust this whole strategy depends on.

**Compliance-track artifacts, added rather than deferred:** a generated CycloneDX SBOM
(`safety_harness/sbom.py`, checked-in reference in `examples/sbom/`) and a vulnerability-disclosure
process (`SECURITY.md`) — the kind of Annex-IV-adjacent evidence a Machinery Regulation "safety
component" technical file eventually needs, added now rather than assembled retroactively once
independent assessment is actually underway. Neither substitutes for item 4 above; both make the
eventual assessment easier to start.

## Prior Art and Open Questions

This design assembles established concepts rather than inventing new ones:

- **Simplex architecture / runtime assurance** (safety-critical control) — pairing an unverified performance controller with a simpler, verified safety controller and a switching monitor.
- **Control Barrier Functions** (Ames et al.) — the formal basis for the trajectory forward-check.
- **Shielding** (safe reinforcement learning) — a module between policy and actuator with authority to override.
- **STRIPS/PDDL action schemas** (classical AI planning) — the precondition/effect structure borrowed for the per-action-type table.
- **Speed-and-separation monitoring** (ISO/TS 15066, collaborative robot safety) — reimagined here as perception-driven software rather than a dedicated hardware sensor.

Open questions this design doesn't resolve:

- **Calibrated confidence.** Current perception and policy models are generally not well-calibrated — a confidence score from an object detector, or an action head's sample variance, is a proxy, not a guarantee. The default-deny principle tolerates a bad proxy by erring toward blocking, but a systematically overconfident model would erode that margin.
- **Precondition coverage is still human-authored, per action type.** Missing a precondition for a new action type is a real gap — the harness only protects against what its schema anticipates for that action, even though it doesn't need to anticipate every hazard.
- **Latency budget.** The forward-check must complete within the control loop's cycle time; on a fast-moving humanoid this constrains how far ahead it can afford to simulate.
- **This is not a substitute for hardware-level safety** — the E-stop, torque-limited joints, and watchdog timers discussed earlier. This is the software layer that sits above that floor, not a replacement for it.

## Version History

| Version | What changed |
| --- | --- |
| 0.3.13 (default-deny bypass on mismatched limits length, 2026-09-29) | **Real safety bug, found by an independent third-party review of the public repo, fixed same day.** `joint_velocity_within_limits`, `joint_effort_within_limits`, and `motor_temperature_within_limits` (`preconditions.py`) each `zip()`'d a reading against its limits sequence with no length check -- the identical flaw `joint_position_limits_respected` already documented fixing once (`zip()` silently truncates to the shorter sequence, so any joint past the end of a too-short limits list is never checked at all), never propagated to these three siblings when they were wired in later (0.3.4/0.3.1). Reproduced exactly as reported: a joint at 10 rad/s against a 2.5 rad/s rated limit, with the limits list one entry short, PERMITted end to end. Fixed with the same length-check-then-fail-closed guard `joint_position_limits_respected` already uses. **Verified, not just fixed:** 3 new regression tests (`tests/test_wired_self_limits.py::MismatchedLimitsLengthFailsClosedTest`) confirmed to genuinely fail against the pre-fix code (checked by temporarily reverting `preconditions.py` and re-running them) and pass against the fix. Full local suite: 355 passed. No adapter or schema changes; every existing adapter already reports matched-length limits, so this closes a latent gap rather than changing behavior for any currently-shipping integration. |
| 0.3.12 (Nav2 hazard scan against a live costmap, 2026-09-29) | `examples/nav2_hazard_scan/live_costmap.py` + `live_scan_demo.py` + `Dockerfile.live`: the same `hazard_rules.py` functions from 0.3.9, unmodified, run against a real *live* `/global_costmap/costmap` topic instead of only a downloaded map file -- a real headless Gazebo, a real spawned TurtleBot3, real `nav2_bringup`, read exactly once (the design boundary from 0.3.10 still holds: one read, one report, no loop, no feedback into navigation). Real bug found and fixed by live-diagnosing the actual running container: `global_costmap` waited forever for a `map` frame that never appeared, because AMCL never publishes `map`→`odom` until it receives an initial pose, which `nav2_bringup`'s default launch doesn't provide automatically -- fixed by publishing one `/initialpose` message at the real known spawn point before waiting for the costmap. Real verified output pasted in the example's own README: 24 real findings against a real live costmap, same rules as the static-file demo. 4 new tests for the live-costmap message conversion. Core `safety_harness` suite unaffected. |
| 0.3.11 (TurtleBot3 + Gazebo + Nav2, 2026-09-28) | `examples/turtlebot3_gazebo_hooks/`: a fuller real-ROS-2-system validation than 0.3.8's mock-publisher demo -- a real TurtleBot3 physically simulated in a real headless Gazebo (Classic), spawned via the real `gazebo_ros` factory service, gated on its actual live `/odom`. `configs/turtlebot3_action_schema.yaml` mirrors the ANYmal-C `navigate` schema, omitting three checks with no real data behind them for this robot (documented in the file itself). Two real bugs found and fixed while building it, live-diagnosed by inspecting the running container, not guessed: `gzserver`'s ROS factory plugin took longer to initialize than `spawn_entity.py`'s own hardcoded 30s wait under emulation (fixed by polling the real service instead of a fixed sleep), and `sensor_data_fresh` always read the bridge as catastrophically stale because TurtleBot3's Gazebo odometry plugin stamps messages with simulation time, not wall-clock time (fixed by stamping with the bridge node's own wall-clock receipt time). `--platform linux/amd64` is deliberate even on Apple Silicon: `ros-humble-gazebo-ros-pkgs`/`ros-humble-turtlebot3-gazebo` ship as prebuilt binaries for amd64 only, checked directly against the apt index. Real verified output pasted in the example's own README, not claimed. |
| 0.3.10 (Nav2 hazard-scan design boundary, 2026-09-28) | Documentation/design-boundary only, no logic change. Internal legal research (not outside counsel) on the Nav2 hazard-scan proof-of-concept (0.3.9): a tool that analyzes a static map once and hands a human a report sits outside "safety component" classification under the EU Machinery Regulation (Art. 3(3)) -- closer to a CAD safety-check plugin than a runtime safety system -- but the moment it re-evaluates live and feeds directly into path-planning decisions, it crosses into the same classification territory `safety_harness` itself sits in. Built that boundary into the product now rather than retrofitting it later: a new "Design boundary" section in `examples/nav2_hazard_scan/README.md`, and the same constraint restated in `hazard_rules.py`'s own module docstring so it's visible to anyone reading the code directly, not just the README. Also fixes naming: "ISO 3691-4-*mapped* hazard flagging," never "certified against" or "compliant with" ISO 3691-4. |
| 0.3.9 (Nav2 hazard-scan proof-of-concept, 2026-09-28) | `examples/nav2_hazard_scan/`: a standalone tool (not part of the `safety_harness` package, not wired through `ActuatorGate` -- a static map scan has no `Action`/live `WorldState`) automating the "someone spends 2-3 days manually going through the Nav2 map flagging hazards" step named in a real ROS Discourse thread on ISO 3691-4 site risk assessments. Real algorithms against a real Nav2 map (the `navigation2` project's own example warehouse map): a two-pass chamfer distance transform, Bresenham ray-casting, a real A*, and three hazard rules (`corridor_width_sufficient`, `blind_corner_absent`, `intersection_flagged_for_review`) each returning a `PreconditionResult`-shaped finding -- the same named-check pattern as the core engine, applied once at deployment time instead of continuously. Found and fixed one real bug while building it (a route's own endpoint computed its forward sightline direction backward). Honest scope stated in the example's own README: ramps/elevation and semantic zones (doors, charging stations) are out of scope for this first version; not reviewed by a safety assessor. 20 new tests (self-contained under `examples/nav2_hazard_scan/tests/`, not part of the main suite -- this tool's own `numpy`/`pillow` dependencies stay out of the core package, same as Isaac Lab examples). |
| 0.3.8 (ROS 2 reference adapter, 2026-09-28) | `safety_harness/adapters/ros2.py`: a second reference integration pattern, this time against a real ROS 2 graph instead of a simulator -- `ROS2PerceptionAdapter`/`ROS2DynamicsAdapter`, importing no `rclpy` (same convention as the Isaac Lab adapters import no `isaaclab`: everything is read off a duck-typed `bridge` object, documented field-by-field against real ROS 2 message shapes -- `sensor_msgs/JointState`, `geometry_msgs/PoseStamped`, URDF-sourced joint limits, this project's own minimal JSON bridge topics for its safety-specific object/agent fields). `configs/ros2_action_schema.yaml` mirrors the Franka schema's grasp/place/reach one-for-one. `examples/ros2_hooks/`: a real `rclpy` bridge node, a mock robot node, and a self-checking demo, verified end to end in a `ros:humble-ros-base` Docker container -- real DDS message passing, no Isaac Lab, no GPU, no physical robot (results in that example's README). Prompted by ROS Discourse moderator feedback (a real OSRA TGC member) that several similar precondition-gating submissions were vibe-coded and unvalidated; this is the concrete "test against a real ROS 2 system" response. Does not substitute for real-hardware fault-condition validation (ISO 13849-2, a separate physical-actuator requirement) -- stated plainly in the example's own README, not glossed over. 16 new tests. |
| 0.3.7 (task-level reroute policy, 2026-09-28) | `safety_harness/task_policy.py`: `is_retryable()` and `run_task_queue()`, a reference caller-side policy for what to do with a BLOCK instead of freezing indefinitely — see "Fallback Control and Recovery" above for the full rationale. Distinguishes checks whose failure can plausibly clear on its own (`RETRYABLE_CHECKS`) from a static fact about the object/robot/config that retrying the identical action cannot change (the default for anything unlisted, including future checks this module doesn't know about — default-deny, same as everywhere else). A non-retryable block skips that task and everything depending on it, transitively, without proposing or gating the dependents at all, and lets independent work proceed. New pure-Python, no-GPU example: `examples/task_reroute/`. Package version only; `SCHEMA_VERSION`/`PreconditionResult` unchanged — this is caller-side policy, not a new gate-level field. Not yet applied to any of the three GPU-dependent demo scripts (Franka, ANYmal-C, G1), which still freeze on every BLOCK — a live-tested follow-up, not done here. 12 new tests. |
| 0.1.0 | Initial data contract and engine: `WorldState`/`Action`/`Decision`, default-deny `ActuatorGate.gate()`, the four adapter interfaces, first precondition checks (object pose/hazard/clearance, swept-path-vs-agents, visibility). |
| 0.1.x (unreleased, built up this session) | Object & placement safety fields and checks (`fall_consequence`, `drop_tolerance_m`, `supported_stably`, destination-clearance). Robot self-limit fields (joint position/velocity/effort limits, motor temperature, cartesian speed cap, battery charge fraction) and the 6 kinematic/electrical/balance checks plus 1 battery check — registered, not yet wired into the example schema (see Scope & Non-Goals). ISO/TS 15066 separation-distance and power-force-limiting checks, plus reduced-speed-near-human. Environmental checks: visibility, surface hazards, multi-agent human + pet scenarios. Mutation, random-fuzz, pathological-input, and reflection-driven black-box contract test suites. Live Isaac Lab validation, 7 hazard-scenario videos, 5 environmental/regulatory hazard videos. |
| 0.2.0 | `SCHEMA_VERSION` bumped from 0.1.0 — the constant had not been incremented despite everything added above. Document reorganized into this product-spec-sheet structure; added Scope & Non-Goals, Worked Examples, and this Version History. |
| 0.2.0 (release model) | Release & Distribution decision: Apache License 2.0 for the engine, adapter interfaces, and conformance suite; monetization moved from per-implementation licensing to certification/audit services and enterprise support. LICENSE, NOTICE, pyproject.toml, and README.md added to the repo. Published as of 2026-09-27: github.com/naganumakr/safety-harness (public, CI green) and PyPI (safety-harness, through 0.2.2). |
| 0.2.0 (adversarial stress-test pass, 2026-09-27) | NaN-sensor stress test found a systemic default-deny bypass: comparison-based checks ("<"/">" against a sensor float) silently pass on NaN. Confirmed standalone in mass\_within\_force\_budget; confirmed on the confidence field in object\_hazard\_confirmed/object\_pose\_confirmed; confirmed compound (full gate() PERMIT next to an under-tracked agent) via swept\_path\_clear\_of\_agents + iso15066\_separation\_distance\_maintained + iso15066\_power\_force\_limiting's own related bug; confirmed latent in balance\_margin\_maintained. Fixed in preconditions.py, validated locally/remote-Python/live-Isaac-Sim, and released as v0.2.2 (PyPI + GitHub). See tests/test\_adversarial\_stress.py and the section above. |
| 0.2.x (second-adapter pass, 2026-09-27) | Development Roadmap stage 4: a second adapter (ANYmal-C, Isaac Lab navigation task) against a genuinely different robot -- a moving base, not a fixed arm. New \`navigate\` action type wires balance\_margin\_maintained for the first time. That activation found and fixed a real bug in \_distance\_to\_polygon\_edge (nearest-vertex distance grew, rather than shrank, once a point moved outside the polygon -- replaced with a proper signed convex-hull distance). Also documented a real naming finding: the base's pose is reported through RobotProprioception.end\_effector\_pose, since no less Franka-specific field exists -- flagged as a follow-up, not fixed here. Verified locally, on the GPU box's own Python, and against a live running Isaac-Navigation-Flat-Anymal-C-v0 environment. See tests/test\_anymal\_adapter.py. |
| 0.3.4 (root-causing the closed-loop nominal false blocks, 2026-09-28) | Documentation only, no code change. Confirmed the source of both nominal false-block categories from the Closed-Loop Measurement above: `panda_joint6`'s real hardware range (Franka's own `franka_ros` datasheet: [-0.0175, 3.7525] rad) is asymmetric and unusually tight at the lower bound, so an ordinary downward-grasp wrist orientation sits close to it by construction -- not a simulator quirk or a controller bug. The `destination_confirmed_stable_and_clear` release-moment blocks were confirmed by code review to come from the destination object's real, physically accurate velocity spike when nudged by the incoming object, per the reference adapter's own `supported_stably` computation -- also not a bug. See "Where the nominal 5.1% comes from" above for the full analysis and the (deliberately undecided) options for the joint-6 margin. |
| 0.3.4 (wire joint velocity/effort limits, 2026-09-28) | `joint_velocity_within_limits` and `joint_effort_within_limits` wired into the example schema for `grasp`/`place`/`reach` (Franka) -- see "Live Re-validation of the Wired Set" above. The Franka adapter now reads real per-joint velocity/effort limits and applied torque from Isaac Lab's articulation data (`soft_joint_vel_limits`, `joint_effort_limits`, `applied_torque`); a design-doc correction: both checks had been assumed to need a joint-motion-predicting dynamics adapter neither reference adapter has, same as `self_collision_clear` -- they don't, they only needed the current joint state and rated limits, which just weren't being read yet. NOT wired for ANYmal-C's `navigate`: that adapter doesn't report these fields. Package version only; `SCHEMA_VERSION` unchanged (the fields already existed on `RobotProprioception`, unpopulated). Unit-tested against a mocked Isaac Lab `robot.data`; real attribute names confirmed against Isaac Lab's public source. *Live-smoke-tested against this project's installed 3.0.0 build in 0.3.5*: `joint_velocity_within_limits` genuinely fired live on the real Franka task (wrist joints exceeding 90% of their rated velocity during nominal reaching) -- confirms the attribute names resolve and both checks evaluate correctly against the real installed build, not just GitHub main's source. 6 new tests. |
| 0.3.6 (G1 example: `--video` demo-clip capture) | Added `--video`/`--fps`/`--max_steps` to `gate_policy_g1_stack.py` (same `overlay.annotate`/`write_mp4` pattern as the Franka and ANYmal-C demo scripts -- box-only dependency, not in this repo). A real camera-framing bug found while building the first clip: the camera sat on the opposite side of the robot from the blocks, so the sight line ran through the robot's own torso -- the clip showed its back the whole time. Root-caused live (queried real block/robot positions) and fixed in two frame-verified passes: same-side placement first, then a steeper/further-back angle once sampling frames across the *whole* render (not just frame 0) showed the arm's own reach motion still swinging across the sightline and hiding the block. See the README's "Gating it with the safety harness" section for the full account and the lesson it's worth keeping for any future camera work. Package version only; `SCHEMA_VERSION` unchanged. |
| 0.3.5 (G1 example: fix the silent 100% gating deadlock, live-validate 0.3.4) | The G1 example's `gate_policy_g1_stack.py` never reported `joint_position_limits` or `max_cartesian_speed_mps`, so once 0.3.1 wired `joint_position_limits_respected`/`cartesian_speed_within_limits` into grasp/place/reach, G1 gating silently default-denied every single decision -- undiscovered because the hazard campaign had never been run since. Fixed (real per-joint position limits from the sim asset, a disclosed placeholder Cartesian speed cap) and ran the full 12-scenario x gated/ungated campaign for the first time: coherent, scenario-appropriate gating throughout, including 0/64 envs lifting an over-payload block gated vs 55/64 ungated, and 175 tampered commands caught with 0 executed. `joint_velocity_within_limits`/`joint_effort_within_limits` deliberately NOT wired for G1 -- introspected live and found the sim asset's velocity limits are 0.0 on every joint and effort limits are sentinel/implausibly-uniform placeholders; G1 uses its own `configs/g1_action_schema.yaml` omitting those two rather than wiring them against untrustworthy data. Also fixed a real `set_masses()` Warp-frontend crash in the `heavy_block` scenario (a version-skew bug, not a logic error) by following the method's own documented `warp.from_numpy` calling convention instead of its buggy torch.Tensor input path. See `examples/isaac_lab_g1_stack/README.md` for the full campaign results table. Package version only; `SCHEMA_VERSION` unchanged. |
| 0.3.3 (decision segmentation, 2026-09-28) | `safety_harness/segmentation.py`: `GripActionSegmenter`, extracting the edge-triggered `grasp`/`place`/`reach` proposal logic the validated Franka closed-loop measurement already relied on into a reusable, tested library -- see the "Decisions must be segmented, not sampled" finding above. Package version only; `SCHEMA_VERSION` unchanged. Not yet re-applied to the G1 example's own driver script. 16 new tests. |
| 0.3.2 (persistent tamper-evident logging, 2026-09-27) | `safety_harness/audit_log.py`: `HashChainedDecisionLogger`, `SoftwareVersionLog`, `verify_log()`, `report_identity()` -- see "Logging, Escalation and the Correction Loop" above. Package version only; `SCHEMA_VERSION` unchanged (no `WorldState`/`Decision` field added). 30 new tests. |
| 0.3.1 (live re-validation, 2026-09-27) | `joint_position_limits_respected` and `cartesian_speed_within_limits` are wired into the example schema; `navigate` gets the speed check only, because the ANYmal-C asset has no joint limits. The reference dynamics adapters predict from the commanded speed (`commanded_speed_mps` / `duration_s`) instead of a fixed cap. Joint-limit reporting supports explicit per-joint exemptions and fails closed on a length mismatch or non-finite limits. The Cartesian speed check allows exactly-at-rating motion (1e-6 relative float slack). Full live re-validation: 24 of 31 checks confirmed blocking live on Franka and ANYmal-C (6 on a genuinely real physical/commanded condition, 18 via field-injection fault testing -- see the tier breakdown in the table above and in the README's Status section); see the section above. Also: `robot_state_confirmed` validates content (finite joint readings and end-effector position); `surface_confirmed_stable` is deprecated (DEPRECATED_CHECKS, with a warning on config load); and a regression test pins that `destination_confirmed_stable_and_clear` excludes the object being placed. |
| 0.3.0 (G1 learned-policy example, 2026-09-27) | Third reference adapter (Unitree G1) brought to schema 0.3.0. All three Isaac Lab adapters now convert quaternions from Isaac Lab 3.x's (x, y, z, w) to the schema's `orientation_wxyz`; they had passed them through unconverted. No check reads orientation yet, so no decision was affected. Added `examples/isaac_lab_g1_stack/`: a GPU-trainable G1 stacking task, scripted experts, BC→PPO training, trained checkpoints and a gating driver. First nominal gating findings are in the section above. |
| 0.3.0 (next safety checks, 2026-09-27) | `SCHEMA_VERSION` 0.2.0 -> 0.3.0 (additive, default-deny fields: `WorldState.sensor_timestamp`/`observed_regions`, `ObservedRegion`, `TrackedAgent.category`/`stature_m`, `AgentCategory`, `TrackedObject.max_safe_grip_force_n`, `RobotProprioception.rated_payload_kg`/`center_of_mass_velocity`, `Decision.action_digest`). Eight new checks, `DecisionWatchdog`, integrity module, config pinning -- see "Next Safety Checks" above. Unit/adversarial-tested only; not yet live-validated. |
| 0.3.x (SBOM and vulnerability-disclosure process, 2026-09-28) | Two CE-certification-track gaps closed: a generated CycloneDX 1.5 SBOM (`safety_harness/sbom.py`, stdlib-only, built from installed-distribution metadata; reference output checked into `examples/sbom/`) and `SECURITY.md` at the repo root. The SBOM work found a real, previously unstated discrepancy: `integrity.py`'s docstring describes the package as following the same dependency-free rule throughout, but `action_schema.py` imports PyYAML unconditionally at runtime and `pyproject.toml` correctly declares it -- the SBOM and its drift-detection test (`tests/test_sbom.py`) record one real runtime dependency, not zero. `SECURITY.md` cites the NaN-Sensor Stress Test above as precedent that a safety-relevant defect found in this project gets fixed, tested harder than it needed to, and disclosed openly rather than quietly patched. |

The code's `SCHEMA_VERSION` constant (`safety_harness/schema.py`) tracks the data contract specifically; this table tracks the whole system. They're expected to move together but aren't the same axis — tightening an existing threshold doesn't need a schema bump, but adding or removing a dataclass field does.
