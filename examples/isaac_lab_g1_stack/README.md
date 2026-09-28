# Unitree G1: pick a block and stack it on another, as a policy to gate

A trained policy for the Unitree G1 humanoid (fixed base, left arm and 3-finger Dex3 hand) that picks
up a blue 4.5 cm block and stacks it on a red one in Isaac Lab. It exists to give the safety
harness a real learned manipulation policy to gate, not a scripted demo.

**Every grasp is real contact physics.** No kinematic attach, no teleporting the block into the hand.
An earlier scripted demo used an attach; it was removed after the "10/10 grips" it reported turned
out not to be real.

## Results (Isaac Lab 3.x, L40S; all numbers from the simulator)

Evaluation: 1024 fresh episodes, randomized block positions (±2 cm each), deterministic actor.

| Policy | Stacked by the strict success test* | Tower still standing, hand clear, at t = 10 s |
|---|---|---|
| Scripted expert, Pink IK (`stack_expert_g1.py`) | ~84% (104–111/128 across seeds) | not measured |
| Scripted expert, RL action space (`stack_expert_rl.py`) | 66% (680/1024) | not measured |
| Behavior cloning, `checkpoints/g1_stack_bc_v2.pt` | 40.1% (411/1024) | 0% |
| BC + PPO fine-tune, `checkpoints/g1_stack_ppo_ft2_rc_300.pt` | **82.7% (847/1024)** | 0.1% (1/1024) |
| From-scratch PPO (Franka recipe) | 0%; never discovered the grasp | — |

\* Strict success: the blue block is seated on the red one (xy within 1.5 cm, height within 1.2 cm)
and at rest (under 3 cm/s), with the fingers open and the wrist at least 11 cm away, for 15
consecutive steps (0.3 s). The episode then ends.

**Read the last column before using these policies.** Both learned policies place the block and
then, if left running, knock their own tower down: training always ended the episode at success, so
they never saw a post-success state. Run them with an external stop at success (that is what
`eval_policy.py` does by default), or finish the hold-reward training below.

**An earlier "85.5%" was not real.** A one-step success test (seated, fingers below 30% closure,
under 0.2 m/s) fired while the block was still tipping off the red one. PPO exploited it. The strict
test above replaced it.

## What's here

- **`task/`**: the Isaac Lab task. Drop the files into
  `isaaclab_tasks/manager_based/locomanipulation/pick_place/` (`mdp/*` into its `mdp/`,
  `agents/*` into its `agents/`) and append `register_snippet.py` to that package's `__init__.py`.
  - `Isaac-Stack-Blocks-G1-RL-v0`: the task.
  - `-Phase1-v0`: reward curriculum phase 1 (reach/grasp/lift only).
  - `-RL-RC-v0`: plus 50% reverse-curriculum starts from expert snapshots. Needs
    `checkpoints/g1_stack_snapshots.pt` at the `SNAPSHOT_PATH` set in the env cfg.
  - Actions are 7-D: waist + left-arm relative-pose differential IK (6), batched on the GPU,
    plus a scalar Dex3 grip (1).
  - The PPO configs are `G1BlockStackPPORunnerCfg` and `G1BlockStackBCPPORunnerCfg`; select the
    latter with `train.py --agent rsl_rl_bc_cfg_entry_point`.
- **`scripts/`**: each script runs with `./isaaclab.sh -p <script> --headless ...`.
  - `stack_expert_g1.py`: the scripted expert on the stock Pink IK task.
  - `stack_expert_rl.py`: the same expert in the RL action space. `--snapshots` writes the
    reverse-curriculum snapshots; `--record --act_noise 0.3` writes DART-style BC data.
  - `bc_train.py`: behavior cloning, written straight into an RSL-RL checkpoint (normalizer
    matched exactly to `rsl_rl` `EmpiricalNormalization`).
  - `eval_policy.py`: success rate. Add `--full_episode` to judge the end state at t = 10 s.
  - `gate_policy_g1_stack.py`: runs the policy behind `ActuatorGate`, gated or ungated, with an
    injectable hazard scenario.
  - `hand_fk.py`, `reach_map.py`, `grid_pocket.py`: the measurements the grasp was designed from.
- **`checkpoints/`**: the two policies in the table and the expert snapshots.

### Reproduce (≈1 GPU-hour on an L40S)

1. `stack_expert_rl.py --num_envs 1024 --snapshots .../g1_stack_snapshots.pt`
2. Record BC data, 4 processes each:
   - `stack_expert_rl.py --num_envs 2048 --act_noise 0.3 --record bc_data/demo_1X.pt`
   - `stack_expert_rl.py --num_envs 2048 --act_noise 0.1 --record bc_data/demo_2X.pt`
3. `train.py --task Isaac-Stack-Blocks-G1-RL-v0 --agent rsl_rl_bc_cfg_entry_point --max_iterations 1`
   gives a template checkpoint.
4. `bc_train.py --template <that>/model_0.pt --out .../model_bc.pt --init_std 0.05`
5. `train.py --task Isaac-Stack-Blocks-G1-RL-RC-v0 --agent rsl_rl_bc_cfg_entry_point --resume --load_run <dir> --checkpoint model_bc.pt`
   (~6 s/iteration at 4096 envs). The table's checkpoint is iteration 300.

## What was learned (each cost real time; don't relearn them)

1. **Quaternion order.** Isaac Lab 3.x stores quaternions as `(x, y, z, w)`. Rotations built as
   `(cos, 0, sin, 0)` silently became 180° flips, and several thousand-config grasp grids were run
   against impossible wrist orientations.
2. **Pose-error argument order.** `compute_pose_error(source, …, target, …)` takes the source first.
   With the arguments swapped, a differential-IK loop diverges.
3. **Root frame.** Differential-IK deltas are in the robot root frame, and the G1 root is yawed 90°.
4. **Waist joints.** The waist must be included in the IK joint set. Arm-only IK twists the wrist
   into its limits, drifts in orientation and knocks the blocks.
5. **Pink IK speed.** Stock Pink IK solves on the CPU once per env: about 85 env-steps/s. GPU
   differential IK does about 12,600 env-steps/s at 1024 envs.
6. **Dex3 geometry.** Fingers point along wrist +x, the palm faces wrist −y and the thumb sticks
   out sideways. The robust grip found:
   - roll 20° about the finger axis;
   - block at wrist-frame (0.11, −0.05, −0.02) m;
   - approach along the palm normal;
   - close to 80%, thumb yaw −0.4;
   - open fully to release (partial opening wedges the block).
7. **Reachability.** The stock object spawn (−0.35, 0.45) is outside the left arm's reach; the
   blocks sit at (−0.26, 0.36) and (−0.14, 0.36).
8. **Exploration noise.** PPO exploration noise has to be tiny for a precise grasp. At action std
   0.15 the BC policy collapses to 1%; at 0.05 it holds 43%.
9. **Normalizer match.** BC → PPO needs the observation normalizer matched to the byte: `rsl_rl`
   uses `(x − mean)/(sqrt(var) + 0.01)`. A `sqrt(var + 0.01)` mismatch took a 49% policy to 0% at
   the first normalizer update.

## Gating it with the safety harness (in progress)

`scripts/gate_policy_g1_stack.py` segments the continuous policy into harness decisions:
- `grasp` on the grip-closing edge near the blue block;
- `place` on the opening edge while holding;
- `reach` otherwise, with a 0.5 s look-ahead target.

A BLOCK freezes the arm and holds the grip. Execution goes through `verify_decision_action()`.

Scenarios are wired but have **not** yet been run as a campaign:
- **Nominal:** no hazard.
- **Human hand:** an adult's hand reaches in over the red block.
- **Child nearby:** a child stands beside the table.
- **Heavy block:** the blue block's physics mass is set to 5 kg.
- **Sharp object:** the blue block is tagged SHARP.
- **NaN pose:** the blue block's pose goes NaN.
- **Low visibility.**
- **Stale sensor data.**
- **Occluded destination.**
- **Unstable destination.**
- **Command tamper:** the permitted command is rewritten in transit.
- **Config tamper:** the live config is edited after verification.

What the nominal smoke runs found (8–16 envs, `ft2_rc_300`):

1. **Every-step `grasp` gating deadlocks the policy.** Gating `grasp` on every closing step
   blocked 3,838 times. Mid-grasp, the block moves in the fingers, so it is never "confirmed
   stable". The fix is edge-triggered segmentation, and it applies to any continuous policy behind
   this harness.
2. **`swept_path_observed` blocks 98% of near-table reaches when the observed region covers only
   the tabletop.** The margin sphere dips into space under the table, which no camera sees. The
   check needs a notion of known-solid or occupied space before it can run on real perception.
3. **Grasp onsets are still blocked by `current_position_confirmed_stable`.** 13 grasp requests
   were permitted and 1,549 blocked; a blocked onset re-requests on every step, so the block count
   is inflated. Nominal place requests are also blocked by `destination_confirmed_stable_and_clear`
   (55 blocks vs 12 permitted). *Correction:* this earlier guessed the check counted the held
   blue block as clutter; it doesn't, since it has always excluded the placed object. The cause
   is its other conditions (red block not confirmed stable, or low pose confidence). Which one
   is still open; those runs recorded only check names, so the next run must log reasons. Both need diagnosis before the hazard campaign,
   because a harness that blocks nominal operation isn't a usable result.

## Next steps

1. **Robust policy.** Finish the hold-reward fine-tune: no success termination, and +0.05 per
   step while the released tower stands with the hand clear. Resume from `ft2_rc_300`; the target
   metric is the tower standing at t = 10 s. 90 iterations showed no gain yet; budget ~1,500+.
   Also try fine-tuning with normal starts mixed in, since a normal-start-only run collapsed early.
2. **Nominal gating must PERMIT.**
   - Fix the grasp-onset stability test: judge stability before contact, or use a velocity
     threshold in the adapter.
   - Exclude the held object from destination clutter.
   - Re-measure the false-block rate and the task success cost of gating.
3. **Hazard campaign.** Run all scenarios gated vs ungated at 64+ envs each. Report:
   - PERMIT/BLOCK per action type and which checks fired;
   - scenario metrics: minimum hand-to-human distance, whether the heavy block was lifted, motion
     on NaN perception, tampered commands executed.
4. **Harness follow-ups surfaced here.**
   - `swept_path_observed` needs known-solid regions (added in 0.3.1: `WorldState.solid_regions`; not yet re-run live on G1).
   - `DecisionWatchdog` deadline should be set from the real control cycle (20 ms here).
   - Seed-to-seed variance of the success numbers.
5. **Release.** Bump to 0.3.0 and publish to PyPI after review. Update external numbers only once
   the ISO/TS 15066 figures are signed off or explicitly caveated.
