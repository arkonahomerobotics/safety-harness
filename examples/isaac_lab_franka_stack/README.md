# Franka + GR00T closed-loop cube stacking

A GR00T N1-class policy, fine-tuned on a Franka cube-stacking demonstration set, driven
closed-loop against a live Isaac Lab simulation through a real policy-server round trip (not a
scripted expert, not an open-loop replay).

**2026-10-03: committed to stop losing this script.** The original closed-loop client that
connected this Isaac Lab env to the GR00T policy server only ever existed on a since-terminated
GPU box's disk and was never committed — the same loss pattern as `rl_events.py` and `overlay.py`
in the G1 example. `scripts/franka_gr00t_closed_loop.py` here is the reconstruction, rebuilt from
the checkpoint's own self-describing training config, the env's real action-space split, and
NVIDIA's own published reference client (`Isaac-GR00T examples/rebot-arm-dm/eval_rebot_arm_dm.py`)
for the server protocol shape — see the file's own docstring for exactly which piece of the
observation/action encoding was verified against which source.

## Baseline result (the one to compare against)

The original 10/10 run, before the Brev box that produced it was lost: **10/10 successful
rollouts**, 400-544 steps each (20Hz control, so 20-27s), on
`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0` — the **base visuomotor Mimic task** (DLAA,
default dome lighting), not the Cosmos-rendered variant. Raw log:
`s3://arkona-gr00t-migration-534283254878/gr00t_output/closed_loop_v4/rollout_v4_final.log` (also
`rollout_v4_mid.log`, `server_final.log`, `server_mid.log` in the same `closed_loop_v4/` prefix).
`franka_gr00t_closed_loop.py`'s `TASK` constant defaults to this task for exactly this reason —
`COSMOS_TASK` is kept as a named alternative, selectable with `--task`, not the default.

## Checkpoint

`gr00t_output/franka_stack_run4/checkpoint-5000` in the same S3 bucket
(`s3://arkona-gr00t-migration-534283254878/`) — full `config.json` / `processor_config.json` /
`statistics.json` / safetensors shards, ~12.5 GB total. The training dataset itself
(`./demo_data/franka_stack_lerobot_v4`, referenced in the checkpoint's own
`experiment_cfg/config.yaml`) was never migrated to S3 and is not on any box's disk as of
2026-10-03 — only the trained checkpoint survived. If the exact training-time language
instruction ever matters again, it is not recoverable from this checkpoint or its logs; confirmed
by exhausting S3, the GPU box's filesystem, the GR00T server's own log, and this script itself
(which hardcodes `LANG_INSTRUCTION = "stack the cubes"`, with no alternatives recorded anywhere).

## Running it

Start the policy server first (separate process; the client connects over the network, default
`127.0.0.1:5556`):

```bash
GROOT_PATCH_MISTRAL=1 GROOT_HF_LOCAL_FIRST=1 HF_TOKEN=<your token, never commit this> \
  python gr00t/eval/run_gr00t_server.py \
  --model-path <path to checkpoint-5000> \
  --embodiment-tag new_embodiment \
  --port 5556
```

- `run_gr00t_server.py` takes its CLI from a `tyro`-wrapped dataclass (`ServerConfig`) — field
  names become `--model-path` / `--embodiment-tag` / `--port` automatically; `embodiment-tag`
  already defaults to `new_embodiment`, kept explicit here for clarity.
- `GROOT_HF_LOCAL_FIRST` and `GROOT_PATCH_MISTRAL` are real flags read by `gr00t/__init__.py` at
  import time (normally only set by `conftest.py` under pytest) — they patch `AutoProcessor`/HF
  model loading to prefer the local cache over a network call, and patch the
  `nvidia/Cosmos-Reason2-2B` tokenizer loading path respectively. Set both for a server launched
  outside pytest, as above.
- **Never write an actual `HF_TOKEN` value into this file, a commit, or a log** — export it in the
  shell environment only.

Then the closed-loop client, from `/workspace/isaaclab` (or wherever Isaac Lab is checked out),
via its own Python:

```bash
./isaaclab.sh -p /path/to/safety-harness/examples/isaac_lab_franka_stack/scripts/franka_gr00t_closed_loop.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --num_rollouts 10 --policy_port 5556
```

Pass `--task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos` to run the Cosmos-rendered
variant instead — see the file's own module docstring and the `TASK`/`COSMOS_TASK` constants for
why these are not interchangeable (different lighting/render pipeline, confirmed to matter: the
original 10/10 baseline is on the base task specifically).

## Debugging CLI flags (added 2026-10-03)

`franka_gr00t_closed_loop.py` grew several flags while chasing the gap below — all real, all
exercised, none hypothetical:

- `--action_horizon N` (default 8): how many of the predicted 16-step action chunk to execute
  before re-querying the server. 8 is NVIDIA's own convention everywhere in this repo
  (`gr00t/eval/rollout_policy.py`, `examples/rebot-arm-dm/eval_rebot_arm_dm.py` — both default to
  8), not a deviation worth chasing on its own.
- `--rtx key=val,...` — extra `set_isaac_rtx_global_settings` overrides (e.g.
  `enable_reflections=1,enable_shadows=1,enable_global_illumination=1`), `--light_intensity` —
  override the dome light's default 3000 intensity, `--friction` — explicit static/dynamic
  friction on the gripper fingers and cubes (neither is set anywhere in this task's own Python
  config by default — see "Franka grasp-physics audit" below).
- `--probe path` — dump the first `table_cam` frame and exit, for a fast visual sanity check
  without running a full rollout. `--render_sweep` — measure camera noise under different RTX
  settings and exit.
- `--record_dir dir`, `--env_seed N`, `--no_video` — self-imitation data collection: save each
  successful episode's observations/actions as an `.npz` (schema in `scripts/selfimit/`'s own
  docstrings), vary cube layouts across collection batches, skip the demo-cam mp4 for speed.

## Franka grasp-physics audit (2026-10-03)

Neither the gripper's friction nor the cubes' mass/friction are set anywhere in this task's own
Python config (`isaaclab_assets/robots/franka.py`, `isaaclab_tasks/.../stack_env_cfg.py`) — both
come entirely from the shared NVIDIA USD assets, identical regardless of which script drives the
sim. `--friction` above exists to test that directly rather than guess at it.

## Status, 2026-10-03 — Isaac Sim 6.1/6.2 does not reproduce the original 10/10

Measured results, this checkout, same checkpoint (`checkpoint-5000`), against the baseline above:

| Variant | Result | Notes |
|---|---|---|
| Original run (lost box) | **10/10** | 400-544 steps/rollout — the number everything else is measured against |
| Cosmos env, defaults | 6/10 | first reproduction attempt |
| Base Mimic env, defaults | 6/10 | ruling out "wrong task" as the sole cause — same shortfall on the task the baseline actually used |
| Base Mimic, dome light 2200 (closer to original brightness) | 7/10 | real improvement, not a fix |
| `--action_horizon 4` | 1/3 (stopped early) | worse, not better — 8 stays the default |
| `--friction 1.0` | 9/10 and 7/10 (16/20 combined, two batches) | the best lever found so far, still short of 10/10 and not fully consistent batch-to-batch |
| `--friction 2.0` | pending | running |

**Plain statement of where this stands:** nothing tried so far — render settings, dome light
intensity, action-chunk horizon, or gripper/cube friction — gets this checkout back to the
original 10/10 on Isaac Sim 6.1/6.2's bundled assets and renderer. `--friction 1.0` is the
strongest single lever (16/20 combined) but isn't a clean fix on its own.

**Planned fix: self-imitation fine-tuning.** Collect successful rollouts in *this* sim
(`--record_dir`), convert them to a LeRobot v2.1 dataset, and further fine-tune `checkpoint-5000`
on them so the policy adapts to whatever's actually different about this Isaac Sim version — see
`scripts/selfimit/` below. Treated as the main path if friction tuning doesn't close the gap on
its own; genuinely useful even if friction does, since Sim-version drift of this kind isn't a
one-time problem.

### `scripts/selfimit/` — self-imitation fine-tune pipeline

- **`npz_to_lerobot.py`** — converts `--record_dir`'s `.npz` episodes into a LeRobot v2.1 dataset
  (parquet + h264 mp4 + meta/*). `--max_steps` (default 540) keeps only the cleaner/faster
  successes. Video codec is h264, not av1 — `gr00t/utils/video_utils.py`'s only decoder is
  `torchcodec.decoders.VideoDecoder`, and this repo ships
  `examples/SimplerEnv/convert_av1_to_h264.py` specifically to get *off* av1, so h264 is the
  proven-safe choice here, not a guess. **Also seeds `meta/stats.json` from the original
  checkpoint's own normalization statistics by default** (`--seed_stats_from`, defaults to
  `/home/ubuntu/gr00t_checkpoint/statistics.json`) — without this, GR00T's `generate_stats()`
  would recompute normalization ranges from the self-imitation set, which is narrower than the
  original demonstration spread (it's the policy's own successful outputs), silently shifting the
  space the pretrained action head was trained to decode in. Confirmed working by diffing
  `generate_stats()`'s output before/after seeding: state/action stats come back byte-identical,
  not recomputed.
- **`franka_stack_modality_config.py`** — registers the `new_embodiment` tag's modality config
  (video/state/action/language schema, matching `checkpoint-5000`'s own
  `experiment_cfg/conf.yaml` exactly). Needed because `new_embodiment` isn't pre-registered
  anywhere in `gr00t/configs/data/embodiment_configs.py` — the original training run's modality
  config file was lost the same way as everything else; this is the reconstruction, verified by
  actually loading a converted episode through `gr00t.data.dataset.sharded_single_step_dataset.
  ShardedSingleStepDataset` end to end (not just file-existence-checked).
- **`launch_finetune_selfimit.sh`** — **not run yet.** `--base-model-path` (not
  `--resume-from-checkpoint`, which means something different — continuing an interrupted run in
  the same output dir, not further-tuning a finished checkpoint on new data) points at the
  existing `checkpoint-5000`; `--learning-rate 5e-5` (halved from the original `1e-4` since this
  is a further fine-tune of an already-converged checkpoint on a small new dataset, not a fresh
  run); `--global-batch-size 16`, `--max-steps 1500`, `--save-steps 500`. Flag names verified
  against `launch_finetune.py --help` directly.
