# Closed-loop measurement: a working controller, gated vs ungated

`closed_loop_gate.py` runs a working Franka cube-stacking controller (the scripted privileged-state
expert) in closed loop on `Isaac-Stack-Cube-Franka-IK-Rel-v0`, with and without the safety harness.

**Gating.** Every 20 Hz control step is translated into the decision it represents and passed
through the real reference adapters, `ActuatorGate` and `DecisionWatchdog`, with the pinned example
config:
- `grasp` on the gripper-closing edge;
- `place` on the opening edge while holding;
- `reach` otherwise, carrying the real commanded speed.

**Human proxy.** In the human condition, a physical kinematic capsule walks up to the far table
edge, stays about 5 s, and leaves. It is tracked from its simulated pose and velocity.

**Recorded run.** 128 envs per condition, seed 1, 60 s episodes, 2026-09-27. Raw results are in
`results/`: `g{gated}_h{human}.json`, plus `diag.json`, a 16-env nominal gated run that records
block reasons. Findings are in the design doc's "Closed-Loop Measurement" section.

```bash
./isaaclab.sh -p closed_loop_gate.py --headless --num_envs 128 --gated 1 --human 1 --seed 1
```
