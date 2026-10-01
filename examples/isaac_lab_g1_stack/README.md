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
- **`configs/g1_action_schema.yaml`** (repo root, alongside `configs/example_action_schema.yaml`):
  `gate_policy_g1_stack.py`'s own action schema, not the shared example one -- see "Gating it with
  the safety harness" above for why G1 needs its own.

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

## Gating it with the safety harness

`scripts/gate_policy_g1_stack.py` segments the continuous policy into harness decisions:
- `grasp` on the grip-closing edge near the blue block;
- `place` on the opening edge while holding;
- `reach` otherwise, with a 0.5 s look-ahead target.

A BLOCK freezes the arm and holds the grip. Execution goes through `verify_decision_action()`.

**`--video <path.mp4>` renders an annotated demo clip** of env 0 (use `--num_envs 1`; `--max_steps`
caps the episode for a smoke test before committing to a full render). Depends on
`overlay.annotate`/`write_mp4`, now `scripts/overlay.py` in this repo (PIL for the caption bars,
imageio or a direct `ffmpeg` subprocess for the mp4 itself). **This file didn't used to be
checked in** -- it only ever existed directly on a GPU box, so `--video` was silently
unreproducible from a fresh clone; reconstructed 2026-10-01 from the exact call signature the
scripts here already use, verified with real rendered preview frames (both PERMIT/green and
BLOCK/red cases) before committing, not just "it imports."

**A real camera-framing bug, found and fixed 2026-09-28 (Kaoru caught it: "I can't even see the
block").** The first rendered clip's camera (`eye=(0.35,-0.75,1.35)`) sat on the *opposite* side of
the robot from the blocks -- robot root at `(0,0,0.75)`, blocks at roughly `(-0.2, 0.35, 0.72)` --
so the sight line to the lookat point ran straight through the robot's own torso: the clip showed
its back the whole time, the actual workspace never visible. Root-caused by querying the real
block/robot positions live (`root_pos_w`), not by re-guessing angles. Took two frame-verified
iterations to actually fix:
1. Move the camera to the same side as the blocks -- fixed the opening frame, but the arm's own
   reach motion swings across the sightline and hides the target block behind the arm for most of
   the clip. Missed on the first pass because only frame 0 was checked.
2. Pull back further and raise to a steeper, more overhead angle, so the tabletop stays visible
   past the arm's swing instead of being seen edge-on. Confirmed this time by sampling frames
   spread across the *entire* render, not just the opening shot.

The working values are the `env_cfg.viewer.eye`/`lookat` currently in the script. **Lesson for any
future camera work here or on another robot:** verify a new camera setup across the whole clip, not
just the first frame or two -- a shot that looks fine with the arm at rest can still get
self-occluded once the arm actually moves into its work envelope.

**A real, silent regression, found and fixed 2026-09-28.** `joint_position_limits_respected` and
`cartesian_speed_within_limits` were wired into the example schema's `grasp`/`place`/`reach` in
0.3.1 -- but this script never reported `RobotProprioception.joint_position_limits` or
`max_cartesian_speed_mps`, so both checks default-denied *every single decision*. Because the
hazard campaign below had never actually been run since 0.3.1, nobody noticed: G1 gating was
100% blocked, silently, the whole time. Fixed by reading real per-joint position limits from the
sim asset (`robot.data.soft_joint_pos_limits`, same pattern as the Franka adapter, Dex3 finger
joints exempted at their own rest stops) and giving the robot a `max_cartesian_speed_mps` --
**not** a verified Unitree G1 datasheet figure (a web search turned up only the whole-body 2 m/s
walking speed, a different number); flagged as a placeholder in the same spirit as this project's
disclosed ISO/TS 15066 placeholders.

`joint_velocity_within_limits` / `joint_effort_within_limits` (wired into the shared schema in
0.3.4) are **deliberately not wired for G1**: introspected live against the real
`Isaac-Stack-Blocks-G1-RL-v0` asset and found `robot.data.soft_joint_vel_limits` is `0.0` on every
one of G1's 43 joints (clearly unconfigured in this asset's actuator model, not a real "no motion"
limit) and `robot.data.joint_effort_limits` is either a `1e9` sentinel or a suspiciously uniform
`300.0` N·m across every arm/hand joint including the fingers -- not trustworthy data. Wiring
either check against it would either always fail or check against numbers that aren't real, the
same failure mode `joint_position_limits_respected` already avoids for ANYmal-C's leg joints (see
`configs/example_action_schema.yaml`'s own comment). G1 uses its own schema,
`configs/g1_action_schema.yaml`, a copy of the shared one's `grasp`/`place`/`reach` minus those two
checks, with its own digest pin.

**Also found and fixed:** the `heavy_block` scenario's `root_physx_view.set_masses()` call crashed
(`TypeError: issubclass() arg 1 must be a class`) on the torch.Tensor input path in this installed
Isaac Sim build -- a real version-skew bug inside Warp's own frontend, not fixed by forcing the
tensor's dtype (tried that first). The method's own docstring example goes through
`warp.from_numpy(..., dtype=warp.float32)` instead of a bare torch.Tensor; switched to that
documented path for both the mass values and the indices and it works cleanly.

### Hazard campaign results (2026-09-28, 64 envs/run, `ft2_rc_300`, seed 11)

All 12 scenarios × gated/ungated, run to completion after the fixes above:

| Scenario | Gated result |
| --- | --- |
| Nominal | Real, mixed decisions (not a deadlock): reach 15421 permit / 12556 block, grasp 48 permit / 3859 block, place 35 permit / 81 block. Grasp-onset blocks (`current_position_confirmed_stable`) and place-onset blocks (`destination_confirmed_stable_and_clear`) reproduce the exact same pattern documented below from before these checks even existed -- consistent, not new, and (per the Franka closed-loop measurement's own root-cause pass, see the main design doc) very likely correct transient-instability behavior rather than a defect, though not separately re-diagnosed for G1 specifically. |
| Human hand | Gated kept the hand farther from the person than ungated: min distance 0.244 m vs 0.151 m. Worked out why it isn't zero-approach the way the Franka closed-loop measurement was: at the robot's *immediate* predicted point (t=0), required clearance is only `swept_radius(0.08) + 0 (zero-horizon worst-case-radius) + margin(0.15)` = 0.23 m, matching the measured 0.244 m almost exactly. The "worker\_hand" hazard here is a *fixed* point near the destination block, not an actively-approaching proxy like Franka's -- the check is working correctly against a different, closer-by-construction scenario geometry, not under-protecting. |
| Child nearby | `vulnerable_bystander_protected` / `iso15066_separation_distance_maintained` fire correctly. |
| Heavy block (5 kg, over the 3 kg budget and 2 kg rated payload) | **0/64 envs ever lifted it, gated. 55/64 lifted it, ungated.** `mass_within_force_budget` and `payload_and_grip_force_within_limits` both fire on every grasp attempt. |
| Sharp object | Gates the same as nominal (the SHARP tag alone doesn't change grasp geometry in this task; it's exercised, not specially reactive here). |
| NaN pose | `object_pose_confirmed` correctly blocks grasp attempts on the corrupted object. General arm motion continues during the NaN window -- checked the code: `reach` actions target the policy's own commanded wrist velocity, not the corrupted object's pose, so continuing to move isn't a gap, it's correct: nothing requires freezing the whole arm because one tracked object's pose is bad, only avoiding *interacting* with that object, which it did. |
| Low visibility | `visibility_above_threshold` fires correctly. |
| Stale sensor | `sensor_data_fresh` fires correctly. |
| Occluded destination | `swept_path_observed` fires correctly. |
| Unstable destination | `destination_confirmed_stable_and_clear` fires correctly. |
| Command tamper | **175 tampered commands caught, 0 executed** (`verify_decision_action`). Real evidence the command-integrity protection works on a learned policy, not just the scripted Franka closed-loop demo. |
| Config tamper | **100% blocked (32000/32000)** on `config_integrity_verified` -- a tampered live config correctly fails every decision closed. |

Raw per-scenario JSON results were generated on the GPU box; not checked into this repo (large,
regenerable via the reproduce command below).

## ROS 2 bridge variant: the same policy, gated over real ROS 2 messages

`scripts/gate_policy_g1_stack_ros2.py` runs the identical trained policy and the identical grasp/
place/reach segmentation as `gate_policy_g1_stack.py`, but every field the harness reads comes off
real `sensor_msgs/JointState`, `geometry_msgs/PoseStamped` and `std_msgs/String` (JSON) ROS 2
messages -- published and consumed via real `rclpy` pub/sub in-process (Isaac Sim's own bundled
ROS 2 libs, `isaacsim.ros2.bridge` extension) -- through `ROS2PerceptionAdapter`/
`ROS2DynamicsAdapter` (`safety_harness/adapters/ros2.py`), reusing `SafetyHarnessBridgeNode` from
`examples/ros2_hooks/` unmodified. It answers the question `examples/ros2_hooks/` and
`examples/turtlebot3_gazebo_hooks/` couldn't on their own: does this same real-ROS-2-wire code path
work against a real GPU-simulated humanoid, not just a hand-written mock publisher or a simpler
wheeled robot.

**Real bug #1, found and fixed: `isaacsim.ros2.bridge` fails to load at all with no env vars set.**
`isaacsim.ros2.core`'s own Ubuntu-version auto-detection (`ros2_common.py`) picks "jazzy" for this
box's Ubuntu 24.04 container, and that bundled distro's `librmw_implementation.so` fails to `dlopen`
its own `libament_index_cpp.so` dependency -- the file exists on disk in the same lib dir, it's just
not on `LD_LIBRARY_PATH`. The extension's own log prints the exact fix once startup fails (easy to
miss if you only see the *second-order* symptom, the user script's own `import rclpy` throwing
`ModuleNotFoundError` moments later, which looks unrelated):
```
export ROS_DISTRO=humble   # matches this project's existing ROS2 Humble work elsewhere
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.core/humble/lib
```
Verified with NVIDIA's own `standalone_examples/api/isaacsim.ros2.bridge/clock.py` smoke test before
touching this project's code at all: 0 messages received with no env vars, 22 real clock callbacks
+ 3 manual-step callbacks received cleanly with them set.

**Real bug #2, found and fixed: enabling the bridge before `gym.make()` breaks env creation.**
Following the ordering NVIDIA's own `clock.py` example uses (enable the extension right after
`AppLauncher`, before anything else) makes `ManagerBasedEnv`'s own one-time seeding step
(`env_cfg.seed` -> `isaaclab`'s `env.seed()` -> `omni.replicator.core.rep.set_global_seed()`) fail
outright:
```
omni.graph.core._impl.errors.OmniGraphError: OmniGraphError: Failed to wrap graph in node given
{'graph_path': '/Replicator/SDGPipeline', 'evaluator_name': 'execution', ...}
```
`gate_policy_g1_stack.py` never hits this because it never enables the ROS 2 bridge at all.
Enabling `isaacsim.ros2.bridge` pulls in enough extra OmniGraph/extension state that Replicator's
own domain-randomization graph creation breaks. Fixed by deferring bridge setup until *after*
`env = gym.make(...)` -- nothing in this test needs it active any earlier; publishing/subscribing
only happens inside the step loop.

**Real bug #3 (in this test's own code, not the adapter or the bridge): invalid `HazardTag` value.**
First real run published `"hazard_tags": ["none"]` on the tracked-block JSON messages.
`safety_harness.schema.HazardTag` has no `"none"` member (valid values: `fragile`/`hot`/`sharp`/
`human`/`animal`/`liquid_containing`/`unknown`) -- `HazardTag("none")` raises inside
`ROS2PerceptionAdapter.get_world_state()`, which `ActuatorGate.gate()` surfaces as
`PerceptionFailure`, so 149/150 steps default-denied with no other explanation. The empty-list case
(no hazard tags at all -- what `gate_policy_g1_stack.py`'s own `frozenset()` equivalent means) is
`[]`, not `["none"]`. Fixed by publishing `[]`.

**Scope decision, made explicitly:** joint position limits are set directly from the same real
per-joint sim data `gate_policy_g1_stack.py` uses (`_real_joint_position_limits`), not round-tripped
through `SafetyHarnessBridgeNode`'s `/robot_description` URDF-parsing path -- that path is real code
already exercised by `examples/ros2_hooks/` and `examples/turtlebot3_gazebo_hooks/` against their
own robots' real URDFs, and re-proving URDF parsing here would add GPU wall-clock without testing
anything new.

**Result, once all three bugs were fixed (200 steps, single env, `ft2_rc_300`, seed 11, real output):**
```
HARNESS_RESULT {"mode": "ros2_bridge", "checkpoint": "examples/isaac_lab_g1_stack/checkpoints/g1_stack_ppo_ft2_rc_300.pt",
 "decisions": {"reach:block": 110, "reach:permit": 13, "grasp:block": 77},
 "blocked_by": {"swept_path_observed": 1, "sensor_data_fresh": 2, "current_position_confirmed_stable": 76,
                "joint_position_limits_respected": 162},
 "stacked": false, "steps_without_fresh_message": 0}
```
Zero crashes, zero stale/never-arrived messages across 200 real rclpy round trips, a real mix of
PERMIT and BLOCK -- not a deadlock or a rubber-stamp. `joint_position_limits_respected` dominates the
blocks; checked what's actually firing rather than assuming a bug:
```
joint 8 at 0.448 within 0.02rad of its [-0.468, 0.468] limit at t=0.00s   (waist_pitch_joint)
```
That's real: this G1 task's reach motion bends the waist forward to get the arm over the table, and
it genuinely runs close to `waist_pitch_joint`'s own soft limit doing it (consistent with "Waist
joints" in the What-was-learned list above -- the waist was already known to matter a lot for this
task's reachability). The harness catching that in real time, over the real ROS 2 wire path, on a
real trained policy, is the check working as designed, not an integration defect. This single-env,
200-step run didn't reach a stack (`"stacked": false`) -- not surprising for one trial at this
episode length; the hazard-campaign table above, from the direct (non-ROS2) script, already has the
real success-rate statistics from 1024 episodes. What this variant adds is proof that the same
decisions come out the same way when everything in between is a real ROS 2 message, not a privileged
in-process Python object.

## Next steps

1. **Robust policy.** Finish the hold-reward fine-tune: no success termination, and +0.05 per
   step while the released tower stands with the hand clear. Resume from `ft2_rc_300`; the target
   metric is the tower standing at t = 10 s. 90 iterations showed no gain yet; budget ~1,500+.
   Also try fine-tuning with normal starts mixed in, since a normal-start-only run collapsed early.
2. **Nominal gating false-block diagnosis for G1 specifically.** The Franka closed-loop measurement's
   equivalent nominal blocks were root-caused (see the main design doc) and found to be correct,
   physically-grounded behavior, not defects -- G1's own grasp-onset and place-onset blocks look
   like the same pattern but haven't been separately confirmed against G1's specific dynamics.
3. **Harness follow-ups surfaced here.**
   - `joint_velocity_within_limits` / `joint_effort_within_limits` need either a real G1 actuator
     model in the sim asset (velocity limits are currently all zero, effort limits are sentinel/
     uniform placeholders) or real datasheet figures, before they can be honestly wired for G1.
   - `max_cartesian_speed_mps` (1.5, set above) is an unverified placeholder -- replace with a real
     Unitree G1 datasheet figure once one is found.
   - `DecisionWatchdog` deadline should be set from the real control cycle (20 ms here).
   - Seed-to-seed variance of the success numbers.
4. **Release.** Bump to 0.3.0 and publish to PyPI after review. Update external numbers only once
   the ISO/TS 15066 figures are signed off or explicitly caveated.
