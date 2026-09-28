# Task reroute: a too-heavy object shouldn't freeze the whole robot

`reroute_example.py` is a pure-Python, no-GPU, no-robot demonstration of `safety_harness.task_policy`.

## The problem it answers

Every demo script elsewhere in this repo (`closed_loop_gate.py`, `gate_policy_g1_stack.py`,
`render_anymal_demo.py`) responds to a BLOCK by freezing the arm in place -- zero the commanded
pose delta, hold whatever grip is current. That's a deliberate simplification for a legible demo
clip: it makes a BLOCK visible on screen as "the arm stopped moving." It is not a claim about how a
production robot should behave, and taken literally it has a real failure mode: an object whose
mass exceeds the rated payload does not get lighter by retrying the same grasp again, so "freeze
until this specific task succeeds" means the robot never does anything else, ever, even work that
has nothing to do with that object.

## What this example does instead

Two blocks on a table: `block_a` is 5kg (over the schema's 3kg force budget), `block_b` is an
ordinary 0.4kg block. Three tasks: `move_a`, `stack_on_a` (depends on `move_a`), `move_b`
(independent). `run_task_queue`:

1. Gates `move_a` -> BLOCK on `mass_within_force_budget`. That check isn't in
   `task_policy.RETRYABLE_CHECKS` (a mass reading is a static fact -- proposing the identical grasp
   again won't change it), so the task is marked `"skipped"`, with the real reason recorded, no
   freeze, no retry loop.
2. `stack_on_a` depends on `move_a`. Since `move_a` was skipped, `stack_on_a` is skipped too,
   *transitively*, without ever calling its own `propose` or `gate` -- it is never even attempted.
3. `move_b` doesn't depend on either -- it's proposed, gated, PERMITted, and marked `"done"`,
   completely unaffected by `block_a`'s mass.

Run it:

```bash
python3 examples/task_reroute/reroute_example.py
```

## Why this lives outside the core check set

`task_policy.py`'s `RETRYABLE_CHECKS` classification is caller-side domain policy, not a property
of the checks themselves -- the harness's `PreconditionResult` doesn't (and shouldn't) carry a
retryability flag, the same way it doesn't carry a severity score or a priority: see the design
doc's "Design Principle: Permission, Not Detection". `ActuatorGate.gate()` stays a stateless,
per-decision function; `run_task_queue` is one reasonable reference policy for what to do with a
sequence of BLOCKs, not the only one, and it's meant to be read, copied, and adapted -- override
`RETRYABLE_CHECKS` for checks whose retryability depends on domain knowledge this module can't have
from a check name alone (see its module docstring for the `fall_consequence_acceptable` example).

Not yet done: none of the three GPU-dependent demo scripts have been changed to use this pattern
instead of freezing. That's a real behavior change to an in-flight control loop and needs live
Isaac Lab testing before it ships, not a blind edit -- see the design doc's Version History and
`examples/isaac_lab_g1_stack/README.md`'s "Next steps" for the same caution applied to the G1
resume-after-clearing work.
