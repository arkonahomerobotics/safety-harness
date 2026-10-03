# Franka cube-stack RL pipeline (recovered from Brev, ported to Isaac Lab 3.0)

A two-stage PPO pipeline for the Franka Panda cube-stacking task: stage 1 picks up the red cube and
stacks it on the blue one; stage 2 picks up the green cube and stacks it on the red-on-blue tower.
It exists to give the safety harness a second real learned manipulation policy to gate, alongside
[`examples/isaac_lab_g1_stack`](../isaac_lab_g1_stack/)'s Unitree G1 policy.

**This is a code port, not a result.** Nothing here has been re-run yet. See "Current status" below.

## Provenance

The original training ran 2026-09-23 to 2026-09-25 on a Brev GPU box (`isaac-launchable-1aebd7`,
Isaac Lab inside a `vscode` docker container). That box no longer exists. The code was recovered
2026-10-03 by deterministically replaying every file write against the Claude Code transcripts from
those sessions — `Write`/`Edit` calls, local patch scripts and `sed` edits actually executed against
sandbox copies, and `docker cp` deploys — then cross-checked against every full-file `Read`,
partial `Read` and read-only command (`grep`/`wc -l`/`tail`) result in the same transcripts. All of
the task-package files below replayed as an exact, fully-verified match (see the recovery's own
`RECOVERY_NOTES.md` for the file-by-file confidence table and the three independent verification
passes); nothing here is a reconstruction from memory or a guess at what the code "probably" looked
like.

**What's genuinely missing, not a hole in a recovered file:** the trained checkpoints and the
snapshot (start-state) data were never themselves part of any transcript — they're binary blobs, not
code — so neither is included here. Regenerating them is the GPU-time-consuming part; see "Snapshot
and training commands" below. `cli_args.py` (an unmodified copy of Isaac Lab's own
`scripts/reinforcement_learning/rsl_rl/cli_args.py`) and a one-line patch to the installed `rsl_rl`
library (`ppo.py`, syncs cached LR with restored optimizer state on resume) are also not recovered —
neither ever appeared in a transcript either, since both are upstream/installed code the original
sessions never touched directly.

## Brev results (original runs, Isaac Lab 2.x-era box — not yet reproduced on this port)

Deterministic evaluation, normal (non-snapshot) starts, from the original Brev training:

| Stage | Checkpoint | Metric | Value |
|---|---|---|---|
| 1 (red onto blue) | `G_gripstd03_sparse_snap_8k/model_4100.pt` | Stacked at episode end, 1024 envs | **98.4%** |
| 2, chained (stage 1 → stage 2) | stage 1 `model_4100` → `K7_stage2skill_handoffheavy_fromK6_8k/model_5500.pt` | Tower standing at end, 1024 envs | **92.3%** |

A recorded video of the stage-1 policy (`record_closeup.py`, 6 episodes, deterministic) showed 6/6
clean stacks; the chained stage-1→stage-2 video (`record_chain.py`, 8 episodes) showed 6/8 clean
towers. Full comparison tables (earlier checkpoints, stochastic vs. deterministic, K6/K8
alternatives) and the complete training/resume lineage are in `RECOVERY_NOTES.md` (kept with the
recovery, not duplicated here) — ask Engineering Development for it if it isn't already with this
PR.

## What's here

- **`task/`**: the Isaac Lab task package.
  - `task/mdp/{rl_actions.py, rl_actions_impl.py, rl_rewards.py, robosuite_rewards.py, rl_events.py}`
    → drop into `isaaclab_tasks/contrib/stack/mdp/`.
  - `task/config/franka/{__init__.py, stack_ik_rel_rl_env_cfg.py}` → drop into
    `isaaclab_tasks/contrib/stack/config/franka/`. **`__init__.py` is the full file, replacing the
    one there, not a snippet to append** — unlike `isaac_lab_g1_stack`'s `register_snippet.py`,
    this file *is* the recovered artifact (upstream registrations plus 12 RL gym ids plus 3 new
    `-Play-v0` variants added during the port), so shipping a diff-friendly snippet would have meant
    re-deriving it instead of recovering it.
  - `task/config/franka/agents/rsl_rl_ppo_cfg.py` → drop into
    `isaaclab_tasks/contrib/stack/config/franka/agents/`.
  - This assumes an Isaac Lab checkout where this task already lives at
    `isaaclab_tasks/contrib/stack/` (see "Port changes" for why that path, not
    `manager_based/manipulation/stack/` where the original Brev code lived).
- **`scripts/`**: run with `./isaaclab.sh -p <script> ...` from the Isaac Lab repo root, same as
  `isaac_lab_g1_stack/scripts/`. Two groups:
  - **Recovered from Brev** (expert, snapshot-building, evaluation, diagnostics — see
    `RECOVERY_NOTES.md` for what each one does and its exact recovery confidence):
    `expert_stack.py`, `capture_expert_snapshots.py`, `capture_expert_snapshots2.py`,
    `capture_handoff.py`, `merge_snapshots.py`, `filter_snapshots.py`, `merge_stage2.py`,
    `merge_stage2_v2.py`, `merge_stage2_v3.py`, `reset_gripper_std.py`, `eval_stacking.py`,
    `eval_chain.py`, `record_chain.py`, `test_snapshot_reset.py`, `smoke_test_rl_env.py`,
    `inspect_prims.py`, `_smoke_suite.sh`.
  - **New, written during the port** (not from Brev — the behavior-cloning warm start the current
    GPU-box run is using): `bc_expert.py` (the scripted expert's phase machine, pure numpy, no Kit
    dependency), `collect_bc_franka.py` (records expert demonstrations inside the RL task's own
    observation/action space), `bc_train_franka.py` (clones those demonstrations into an `rsl_rl`
    checkpoint `train.py --checkpoint` can warm-start PPO from), `diag_bc_franka.py` (compares a
    policy's rollout against a shadow scripted expert, phase by phase).
  - **`accept_20.py`** (also new): the acceptance gate to run once a chained stage-1/stage-2
    checkpoint pair looks good on `eval_chain.py`'s aggregate numbers — 20 sequential, single-env,
    seeded, deterministic episodes judged on the env's own chained-success definition, with optional
    per-episode captioned video. See `ACCEPTANCE.md` for the full protocol and why it's careful about
    exactly when it's safe to read simulation state relative to Isaac Lab's own auto-reset.

## Port changes (Isaac Lab 3.0 / rsl-rl-lib 5.5.1)

The Brev box ran against an older Isaac Lab/rsl-rl pair. Porting the recovered files to this
project's Isaac Lab 3.0 / rsl-rl 5.5.1 target surfaced real API changes, not just a straight copy:

- **Package location.** Moved from `isaaclab_tasks.manager_based.manipulation.stack` to
  `isaaclab_tasks.contrib.stack`. The four upstream base-task gym ids this package also registers
  (`Isaac-Stack-Cube-Franka-v0` and 13 siblings) were renamed to an `IsaacContrib-` prefix with no
  `-v0` suffix, to avoid colliding with the real upstream registrations once this file lives
  alongside them under the new namespace. **The 12 RL-specific gym ids were left unchanged**
  (`Isaac-Stack-Cube-Franka-IK-Rel-RL-*-v0`) — the recorded Brev run names and resume commands still
  refer to them correctly, and three new `-Play-v0` variants were added (small-scene, no observation
  corruption, for `play.py`).
- **Full-reset `env_ids`.** Isaac Lab 3.x managers pass `None` or `slice(None)` for a full reset,
  not always an explicit id tensor; `rl_rewards.py`, `robosuite_rewards.py` and `rl_events.py` each
  needed a small `_as_env_ids`-style guard before indexing with it.
- **Contact sensor data.** PhysX now only reports the normal component of the filtered contact force
  matrix; `force_matrix_w` is a deprecated alias for `normal_force_matrix_w` that warns on every
  access. `robosuite_rewards.py`'s `_in_contact` reads `normal_force_matrix_w` directly — same
  numbers, no warning spam.
- **Franka asset prim paths.** This Isaac Lab 3.x / Isaac Sim 6.x Franka asset nests its links under
  `Robot/Geometry/panda_link0/.../panda_hand`, not directly under `Robot/` as on the old Brev box.
  `ContactSensorCfg`'s path-matching searches the parent prim's subtree for the leaf name, so the
  shallow `Robot/panda_hand`-style path still resolves — confirmed with `inspect_prims.py` and left
  as named path constants (`_HAND`, `_LEFT_FINGER`, `_RIGHT_FINGER`, `_BLUE`/`_RED`/`_GREEN`) in
  `stack_ik_rel_rl_env_cfg.py` rather than inlined per contact sensor, since the same six paths are
  reused across four different env cfg classes.
- **Cube pose quaternion order.** `root_link_pose_w` (and so the recorded snapshot format) is
  `(x, y, z, qx, qy, qz, qw)` in this Isaac Lab version. Snapshot files are written and read by the
  same version, so nothing needs converting at runtime — but a snapshot file carried over from the
  old Brev box would be `wxyz` and must not be reused directly. (None exist yet; see "Current
  status.")
- **Deprecated `Articulation` API.** `set_joint_position_target_index`/
  `set_joint_velocity_target_index` now warn on every call; `rl_events.py`'s snapshot-reset event
  writes through `robot.actuators.target_command.set_position_index`/`set_velocity_index` instead —
  same underlying buffers, no warning.
- **`torch.load` on snapshot files.** Loads with `weights_only=True` (PyTorch's current default
  expectation for a plain tensor-dict checkpoint).
- **RSL-RL 5.5.1's `obs_groups`.** Now set explicitly on `StackCubePPORunnerCfg`
  (`{"actor": ["policy"], "critic": ["policy"]}`) rather than relying on `rsl_rl`'s old fallback
  behavior — resolves to the same thing, but keeps an extra observation group (e.g. `eval_chain.py`'s
  `policy_s2`, used only for stage-2 evaluation bookkeeping) from ever being fed to the networks.

None of these change the reward shaping, the action space, or the training curriculum itself — they
adapt the same recovered logic to APIs that moved underneath it.

## Current status

**Smoke-tested only. No training results from this port yet — none are claimed above; the Brev
numbers above are the original runs, not reproduced here.** `scripts/smoke_test_rl_env.py` and
`scripts/_smoke_suite.sh` confirm the ported task constructs, steps, and resets cleanly under Isaac
Lab 3.0 (including the contact-sensor path resolution above). Training in progress on the GPU box
follows a BC-warm-start-then-PPO recipe that did not exist on the Brev box (see "New, written during
the port" above): collect scripted-expert demonstrations in the RL task's own observation/action
space (`collect_bc_franka.py`), clone them into a PPO actor checkpoint (`bc_train_franka.py`), then
resume PPO training from that checkpoint (`train.py --checkpoint`) instead of from scratch. No
success-rate numbers exist for this yet — do not cite a number for the ported pipeline until a real
evaluation (`eval_stacking.py`/`eval_chain.py`) has actually been run against it.

## Snapshot and training commands (from the original Brev runs — paths not yet re-verified on this port)

Recorded here for provenance and as the starting point for regenerating snapshots/checkpoints on
this port's GPU box; `SNAPSHOT_DIR` in `stack_ik_rel_rl_env_cfg.py` defaults to
`/workspace/isaaclab/snapshots`, matching the paths below. See `RECOVERY_NOTES.md` for the full
command list, every intermediate run name, and the complete resume lineage.

```bash
# stage-1 expert states (red carried/lowered onto blue) -- used by every Robosuite-Snap* task
./isaaclab.sh -p capture_expert_snapshots.py --out $PWD/snapshots/expert_carry_onto_blue.pt --num_envs 1024 --steps 400
# stage-2 expert states (green carried onto red-on-blue)
./isaaclab.sh -p capture_expert_snapshots2.py --out $PWD/snapshots/expert_carry_green_onto_red.pt --num_envs 1024 --steps 700
./isaaclab.sh -p merge_stage2.py        # stage2_allphases_neargoal.pt (Stage2Skill task default)

# stage 1: Robosuite-Snap-Sparse-v0, from scratch or resumed (see RECOVERY_NOTES.md for the real resume chain)
./isaaclab.sh -p train.py --task Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-Snap-Sparse-v0 --headless --num_envs 8192

# stage 2: Stage2Skill-v0, resumed from a stage-1 checkpoint with the gripper-action std reset to 1.0
./_isaac_sim/python.sh reset_gripper_std.py <stage1_checkpoint>.pt <seed_checkpoint>.pt 1.0
./isaaclab.sh -p train.py --task Isaac-Stack-Cube-Franka-IK-Rel-RL-Stage2Skill-v0 --headless --num_envs 8192 \
    --resume --load_run <run_dir> --checkpoint <seed_checkpoint_name>.pt

# evaluation
./isaaclab.sh -p eval_stacking.py --checkpoint <stage1_checkpoint>.pt --num_envs 1024
./isaaclab.sh -p eval_chain.py --stage1 <stage1_checkpoint>.pt --stage2 <stage2_checkpoint>.pt --num_envs 1024
```

The Brev runs used `train_reset_lr.py` (upstream `train.py` plus a one-line patch so a resume does
not restore optimizer state, letting the configured learning rate actually apply) for every resume
in the lineage above. That file is not included in this port — not because it was lost, but because
it's a two-line structural change to the stock `scripts/reinforcement_learning/rsl_rl/train.py`
that's simpler to reapply directly against whatever `train.py` ships with this Isaac Lab version
than to vendor a second full copy of it here.

## Gating it with the safety harness

Not wired up yet. `examples/isaac_lab_g1_stack/scripts/gate_policy_g1_stack.py` is the pattern to
follow once a trained checkpoint exists for this task: segment the continuous policy into
`grasp`/`place`/`reach` decisions and run them through `ActuatorGate`. No equivalent script exists
here yet, and none should be written before there's a real policy to gate.

## Next steps

1. Finish the BC-warm-start → PPO run in progress; get a real stage-1 and stage-2 success rate on
   this port before claiming any number for it.
2. Regenerate the snapshot files (commands above) and check in the ones small enough to version
   (or document where the large ones live, following `isaac_lab_g1_stack/checkpoints/`'s pattern).
3. Once a checkpoint exists: write a `gate_policy_*` script and an action schema for this task,
   mirroring `isaac_lab_g1_stack`'s hazard campaign, instead of assuming the G1 results generalize.
