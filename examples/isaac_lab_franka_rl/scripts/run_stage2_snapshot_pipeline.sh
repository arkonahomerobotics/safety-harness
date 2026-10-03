#!/bin/bash
# One-command Brev-scale (1024-env) stage-2 snapshot pipeline: the K7 recipe from
# RECOVERY_NOTES.md, chained end to end instead of run as five separate manual commands.
#
# Order: capture_expert_snapshots.py (stage-1 expert states) -> capture_expert_snapshots2.py
# (stage-2 expert states, non-all-phases) -> filter_snapshots.py (near-goal mix) ->
# capture_expert_snapshots2.py --all_phases -> merge_stage2.py -> capture_handoff.py --rounds 6
# (from a given Stage-1 checkpoint) -> merge_stage2_v3.py. Each intermediate artifact is skipped
# if it already exists (idempotent/resumable), since which of these box B already has from earlier
# work is exactly the kind of thing this script shouldn't need to assume -- see the usage note below.
#
# Deliberately NOT like _smoke_suite.sh, which logs every step's exit code and keeps going
# regardless: every step below is a required INPUT to the next one, so this fails fast on the
# first real failure instead of burning GPU time running later steps against missing/bad data.
#
# Run from /workspace/isaaclab (same convention as every other script in this directory).
# No IPs, keys, account/bucket IDs, or other infrastructure detail belongs in this file --
# everything it needs is a relative snapshot path, a checkpoint path you pass in, and whatever
# this box's own nvidia-smi/free report.
#
# Usage: ./isaaclab.sh -p franka_rl/run_stage2_snapshot_pipeline.sh --stage1 <checkpoint>.pt [options]
#   (runs under this project's own Python via isaaclab.sh -p like every other script here, even
#   though it's a shell script -- isaaclab.sh execs whatever you hand it)
# Or directly: bash franka_rl/run_stage2_snapshot_pipeline.sh --stage1 <checkpoint>.pt [options]

set -uo pipefail

STAGE1_CHECKPOINT=""
NUM_ENVS=1024
HANDOFF_ROUNDS=6
HANDOFF_STEPS=300
HANDOFF_NUM_ENVS=4096
MIN_RAM_GB=8
MIN_GPU_FREE_GB=4
ASSUME_YES=0
SNAPSHOT_DIR="/workspace/isaaclab/snapshots"
LOG_DIR="/workspace/isaaclab/logs/stage2_snapshot_pipeline/$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<USAGE
Usage: $0 --stage1 <checkpoint>.pt [--num_envs N] [--handoff_rounds N] [--handoff_steps N] [--handoff_num_envs N]
          [--snapshot_dir DIR] [--log_dir DIR] [--min_ram_gb N] [--min_gpu_free_gb N] [-y|--yes]

--stage1              Stage-1 checkpoint to capture handoff states from (required).
--num_envs            Envs for the expert-snapshot captures (default: $NUM_ENVS, Brev scale).
--handoff_rounds      Rounds for capture_handoff.py (default: $HANDOFF_ROUNDS, the K7 recipe).
--handoff_steps       Steps per round (default: $HANDOFF_STEPS; must exceed the stage-1 policy handoff time, ~190 for BC stage 1).
--handoff_num_envs    Envs for capture_handoff.py (default: $HANDOFF_NUM_ENVS).
--snapshot_dir        Where snapshot .pt files live (default: $SNAPSHOT_DIR).
--log_dir             Where per-step logs are written (default: timestamped under
                      /workspace/isaaclab/logs/stage2_snapshot_pipeline/).
--min_ram_gb          Abort if available RAM is below this (default: $MIN_RAM_GB).
--min_gpu_free_gb     Abort if free GPU memory is below this (default: $MIN_GPU_FREE_GB).
-y, --yes             Skip the confirmation prompt (for non-interactive/automated runs).
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage1) STAGE1_CHECKPOINT="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --handoff_rounds) HANDOFF_ROUNDS="$2"; shift 2 ;;
        --handoff_steps) HANDOFF_STEPS="$2"; shift 2 ;;
        --handoff_num_envs) HANDOFF_NUM_ENVS="$2"; shift 2 ;;
        --snapshot_dir) SNAPSHOT_DIR="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --min_ram_gb) MIN_RAM_GB="$2"; shift 2 ;;
        --min_gpu_free_gb) MIN_GPU_FREE_GB="$2"; shift 2 ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$STAGE1_CHECKPOINT" ]]; then
    echo "error: --stage1 <checkpoint>.pt is required" >&2
    usage
    exit 1
fi
if [[ ! -f "$STAGE1_CHECKPOINT" ]]; then
    echo "error: stage-1 checkpoint not found: $STAGE1_CHECKPOINT" >&2
    exit 1
fi

if ! mkdir -p "$SNAPSHOT_DIR" "$LOG_DIR"; then
    echo "error: could not create $SNAPSHOT_DIR and/or $LOG_DIR -- check permissions" >&2
    exit 1
fi

# -- RAM / GPU pre-flight checks -------------------------------------------------------------
avail_ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
if [[ -z "$avail_ram_gb" ]]; then
    echo "warning: could not read available RAM (no 'free' command?) -- skipping RAM check" >&2
elif (( avail_ram_gb < MIN_RAM_GB )); then
    echo "error: only ${avail_ram_gb}GB RAM available, need at least ${MIN_RAM_GB}GB -- aborting before touching the GPU" >&2
    exit 1
else
    echo "RAM check: ${avail_ram_gb}GB available (minimum ${MIN_RAM_GB}GB) -- OK"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found -- this pipeline needs a GPU, aborting" >&2
    exit 1
fi
gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -z "$gpu_free_mib" ]]; then
    echo "error: could not read GPU free memory from nvidia-smi -- aborting" >&2
    exit 1
fi
gpu_free_gb=$(( gpu_free_mib / 1024 ))
if (( gpu_free_gb < MIN_GPU_FREE_GB )); then
    echo "error: only ${gpu_free_gb}GB GPU memory free, need at least ${MIN_GPU_FREE_GB}GB -- is another job already running on this box?" >&2
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader >&2
    exit 1
fi
echo "GPU check: ${gpu_free_gb}GB free (minimum ${MIN_GPU_FREE_GB}GB) -- OK"

# -- Confirm before doing anything that actually uses the GPU --------------------------------
echo
echo "About to run the Brev-scale stage-2 snapshot pipeline on THIS box:"
echo "  stage-1 checkpoint : $STAGE1_CHECKPOINT"
echo "  num_envs           : $NUM_ENVS (expert captures), $HANDOFF_NUM_ENVS (handoff capture)"
echo "  handoff rounds     : $HANDOFF_ROUNDS"
echo "  snapshot dir        : $SNAPSHOT_DIR"
echo "  logs                : $LOG_DIR"
echo "This will take real GPU time and will not stop to ask again once started."
if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -r -p "Proceed? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "aborted, nothing was run."; exit 1 ;;
    esac
fi

cd /workspace/isaaclab || { echo "error: /workspace/isaaclab not found -- run this from the Isaac Lab container"; exit 1; }

run_step() {
    # run_step <description> <log_file> <command...>
    local desc="$1" log="$2"
    shift 2
    echo
    echo "=== $desc ==="
    echo "    log: $log"
    "$@" > "$log" 2>&1
    local code=$?
    echo "    exit=$code" >> "$log"
    if [[ $code -ne 0 ]]; then
        echo "FAILED: $desc (exit $code) -- see $log" >&2
        tail -30 "$log" >&2
        exit "$code"
    fi
    echo "    OK"
}

skip_if_exists() {
    # skip_if_exists <path> <description> -- returns 0 (skip) if the file already exists
    if [[ -f "$1" ]]; then
        echo
        echo "=== $2: skipped, $1 already exists ==="
        return 0
    fi
    return 1
}

BLUE="$SNAPSHOT_DIR/expert_carry_onto_blue.pt"
GREEN_ON_RED="$SNAPSHOT_DIR/expert_carry_green_onto_red.pt"
NEARGOAL_MIX="$SNAPSHOT_DIR/stage2_neargoal_mix.pt"
ALLPHASES="$SNAPSHOT_DIR/expert_stage2_allphases.pt"
ALLPHASES_NEARGOAL="$SNAPSHOT_DIR/stage2_allphases_neargoal.pt"
HANDOFF="$SNAPSHOT_DIR/handoff_$(basename "$STAGE1_CHECKPOINT" .pt).pt"
STAGE2_V3="$SNAPSHOT_DIR/stage2_v3.pt"

# 1. stage-1 expert states (filter_snapshots.py's stage-1 input)
if ! skip_if_exists "$BLUE" "stage-1 expert states"; then
    run_step "stage-1 expert states (capture_expert_snapshots.py)" "$LOG_DIR/01_expert_carry_onto_blue.log" \
        ./isaaclab.sh -p franka_rl/capture_expert_snapshots.py --out "$BLUE" --num_envs "$NUM_ENVS" --steps 400
fi

# 2. stage-2 expert states, non-all-phases (filter_snapshots.py's stage-2 input)
if ! skip_if_exists "$GREEN_ON_RED" "stage-2 expert states (non-all-phases)"; then
    run_step "stage-2 expert states (capture_expert_snapshots2.py)" "$LOG_DIR/02_expert_carry_green_onto_red.log" \
        ./isaaclab.sh -p franka_rl/capture_expert_snapshots2.py --out "$GREEN_ON_RED" --num_envs "$NUM_ENVS" --steps 700
fi

# 3. near-goal stage-2 filter + stage-1 mix-in
if ! skip_if_exists "$NEARGOAL_MIX" "near-goal stage-2 filter"; then
    run_step "near-goal filter (filter_snapshots.py)" "$LOG_DIR/03_stage2_neargoal_mix.log" \
        ./isaaclab.sh -p franka_rl/filter_snapshots.py "$GREEN_ON_RED" "$BLUE" "$NEARGOAL_MIX" 3.0 1.0 4000
fi

# 4. all-phases stage-2 expert capture, full Brev-scale
if ! skip_if_exists "$ALLPHASES" "all-phases stage-2 expert capture"; then
    run_step "all-phases capture (capture_expert_snapshots2.py --all_phases)" "$LOG_DIR/04_expert_stage2_allphases.log" \
        ./isaaclab.sh -p franka_rl/capture_expert_snapshots2.py --all_phases --out "$ALLPHASES" --num_envs "$NUM_ENVS" --steps 900
fi

# 5. merge into the Stage2Skill task default
if ! skip_if_exists "$ALLPHASES_NEARGOAL" "allphases+neargoal merge"; then
    run_step "merge (merge_stage2.py)" "$LOG_DIR/05_stage2_allphases_neargoal.log" \
        ./isaaclab.sh -p franka_rl/merge_stage2.py "$ALLPHASES" "$NEARGOAL_MIX" "$ALLPHASES_NEARGOAL"
fi

# 6. stage-1 handoff capture from the given checkpoint (named for that checkpoint, not a stale
# identity like the scripts' own hardcoded "handoff_G4100.pt" default -- see the KAN-35/Franka
# README's own note about this).
if ! skip_if_exists "$HANDOFF" "stage-1 handoff capture"; then
    run_step "handoff capture (capture_handoff.py, $HANDOFF_ROUNDS rounds)" "$LOG_DIR/06_handoff.log" \
        ./isaaclab.sh -p franka_rl/capture_handoff.py --stage1 "$STAGE1_CHECKPOINT" --out "$HANDOFF" \
            --num_envs "$HANDOFF_NUM_ENVS" --rounds "$HANDOFF_ROUNDS" --steps "$HANDOFF_STEPS"
fi

# 7. K7 recipe: handoff x10 + the step-5 output
if ! skip_if_exists "$STAGE2_V3" "K7 recipe merge"; then
    run_step "K7 merge (merge_stage2_v3.py)" "$LOG_DIR/07_stage2_v3.log" \
        ./isaaclab.sh -p franka_rl/merge_stage2_v3.py "$HANDOFF" "$ALLPHASES_NEARGOAL" "$STAGE2_V3"
fi

echo
echo "DONE. Stage-2 training snapshot: $STAGE2_V3"
echo "Point env.events.reset_from_snapshot.params.snapshot_path at it to use this recipe."
