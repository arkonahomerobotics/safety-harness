#!/usr/bin/env bash
# Self-imitation fine-tune launch for checkpoint-5000, from rejection-sampled successful
# rollouts collected in the current (Isaac Sim 6.1) sim. NOT run automatically -- review the
# flags below (especially LEARNING_RATE and MAX_STEPS) before executing.
#
# Flag names verified directly against `python gr00t/experiment/launch_finetune.py --help`
# (tyro's real CLI form for FinetuneConfig is hyphenated: --base-model-path, --learning-rate,
# etc. -- examples/finetune.sh in this repo actually calls it with underscores instead, e.g.
# --base_model_path; both apparently work since tyro normalizes '-'/'_', but --help's own
# rendering is the form used here to avoid relying on undocumented leniency).
# Single-GPU CUDA_VISIBLE_DEVICES pin copied from examples/finetune.sh's own comment:
# "Restrict to a single GPU so HF Trainer doesn't wrap the model in DataParallel, which
# crashes with a StopIteration error in the model's device property."
#
# BASE_MODEL_PATH, not --resume_from_checkpoint: we're further-tuning an already-finished
# checkpoint on new data, not resuming an interrupted run in the same output_dir (that flag's
# actual meaning -- gr00t/configs/finetune_config.py:188-190 -- is "continue the latest
# checkpoint-* already inside output_dir", wrong semantics here).
#
# tune_llm/tune_visual/tune_projector/tune_diffusion_model are left at FinetuneConfig's own
# defaults (False/False/True/True) rather than passed explicitly -- examples/finetune.sh
# doesn't expose them as flags at all, and those defaults already match checkpoint-5000's own
# recorded experiment_cfg/conf.yaml exactly, so there's nothing to override.
set -euo pipefail

cd /home/ubuntu/Isaac-GR00T
source .venv/bin/activate

BASE_MODEL_PATH="/home/ubuntu/gr00t_checkpoint"
DATASET_PATH="/home/ubuntu/franka_selfimit/lerobot_dataset"
MODALITY_CONFIG_PATH="/home/ubuntu/franka_selfimit/scripts/franka_stack_modality_config.py"
EMBODIMENT_TAG="new_embodiment"
OUTPUT_DIR="/home/ubuntu/gr00t_selfimit_run1"

GLOBAL_BATCH_SIZE=16
MAX_STEPS=1500
SAVE_STEPS=500
# checkpoint-5000 used 1e-4 from the base pretrained model; halved here since this is a further
# fine-tune of an already-converged checkpoint on a small new dataset, not a fresh run --
# revisit if 1500 steps at 5e-5 clearly underfits.
LEARNING_RATE=5e-5
DATALOADER_NUM_WORKERS=14

# Single GPU, plain python -- not torchrun -- same reason examples/finetune.sh pins this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --output-dir "$OUTPUT_DIR" \
  --num-gpus 1 \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 5 \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"

# After this finishes: point run_gr00t_server.py --model-path at
# /home/ubuntu/gr00t_selfimit_run1/checkpoint-<last_save_steps> and re-run the closed-loop eval.
