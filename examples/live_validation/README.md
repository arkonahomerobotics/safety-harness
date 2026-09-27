# Live validation scripts

The scripts and raw results behind the design doc's "Live Re-validation of the Wired Set"
(2026-09-27, v0.3.1). Each script runs the real reference adapters and `ActuatorGate`, with the
pinned example config, against a live Isaac Lab environment:

- `live_revalidate_franka.py`: `Isaac-Stack-Cube-Franka-IK-Rel-v0` (34 scenarios)
- `live_revalidate_anymal.py`: `Isaac-Navigation-Flat-Anymal-C-v0` (10 scenarios)

Each scenario states the check it must fire (or `permit` for a control) and records exactly which
checks fired. Results for the recorded run are in `results/`: Franka 33 of 34, ANYmal-C 10 of 10.
The single Franka miss is explained in the design doc: an adult exactly at
`iso15066_power_force_limiting`'s 0.3 m contact range.

To run one, from an Isaac Lab checkout with this repo at `/workspace/safety_harness`:

```bash
./isaaclab.sh -p live_revalidate_franka.py --headless --harness /workspace/safety_harness
```
