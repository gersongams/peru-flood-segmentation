#!/usr/bin/env bash
# Run ON the pod, inside tmux/nohup. Trains the Peru-only runs one after another.
#   bash scripts/cloud/run_queue.sh "unet terramind_v1_base prithvi_eo_v2_300" "0 1 2"
#   bash scripts/cloud/run_queue.sh unet 0 --trainer.max_epochs=2      # speed test
# Each run has a fixed directory outputs/runs_cloud/<model>/lightning_logs/seed<s>; if it holds a
# last.ckpt the run resumes from it, so rerunning the queue after an interruption is safe.
set -uo pipefail
MODELS=${1:?models}; SEEDS=${2:?seeds}; shift 2
cd /workspace/peru-flood-segmentation
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/workspace/.cache/uv HF_HOME=/workspace/.cache/huggingface
mkdir -p outputs/logs

for s in $SEEDS; do
  for m in $MODELS; do
    dir=outputs/runs_cloud/$m; [ $# -gt 0 ] && dir=outputs/runs_cloud_test/$m  # overrides = test run, kept apart
    last=$dir/lightning_logs/seed$s/checkpoints/last.ckpt
    [ -f "$dir/lightning_logs/seed$s/done" ] && { echo "$m seed$s already done"; continue; }
    resume=(); [ -f "$last" ] && resume=(--ckpt_path "$last")
    uv run terratorch fit -c configs/$m.yaml -c configs/datasets/peru_unosat.yaml -c configs/trainer/cuda.yaml \
      --trainer.logger.init_args.save_dir "$dir" --trainer.logger.init_args.version "seed$s" \
      --seed_everything "$s" "${resume[@]}" "$@" >> "outputs/logs/${m}_seed$s.log" 2>&1
    rc=$?
    echo "$m seed$s exit=$rc $(date -Is)"
    [ $rc -eq 0 ] && [ $# -eq 0 ] && touch "$dir/lightning_logs/seed$s/done"
  done
done
