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

## Status, 2026-10-03

Re-running the base Mimic task to confirm 10/10 still reproduces on this (different) IsaacLab
checkout is in progress — the Cosmos variant was getting 6/10, with failures traced to slow or
toppled third-cube placement (the task's own success criterion is a 3-cube tower, all three
stacked, not 2) rather than grasp failures. If that turns out to be a Cosmos-render
distribution-shift issue specifically, this README will be updated with the result once the
retest finishes.
