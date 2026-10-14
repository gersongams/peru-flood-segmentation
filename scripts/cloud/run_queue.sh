#!/usr/bin/env bash
# Run ON the pod with setsid/nohup. Trains the Peru-only runs one after another.
#   bash scripts/cloud/run_queue.sh "unet terramind_v1_base prithvi_eo_v2_300" "0 1 2"
#   bash scripts/cloud/run_queue.sh unet 0 --trainer.max_epochs=2      # speed test
# Runs write to the container disk ($LOCAL): checkpoint writes to the /workspace
# volume hung once its quota filled. Each finished run, and
# the queue log, is copied to $KEEP on the volume. A run with a last.ckpt in
# $LOCAL resumes from it; a run marked done in $KEEP is skipped.
set -uo pipefail
MODELS=${1:?models}; SEEDS=${2:?seeds}; shift 2
cd /workspace/peru-flood-segmentation
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/workspace/.cache/huggingface
name=runs_cloud; [ $# -gt 0 ] && name=runs_cloud_test  # overrides = test run, kept apart
LOCAL=/root/outputs/$name KEEP=/workspace/outputs/$name
mkdir -p "$LOCAL/logs" "$KEEP"

for s in $SEEDS; do
  for m in $MODELS; do
    run=$m/lightning_logs/seed$s
    [ -f "$KEEP/$run/done" ] && { echo "$m seed$s already done"; continue; }
    last=$LOCAL/$run/checkpoints/last.ckpt
    resume=(); [ -f "$last" ] && resume=(--ckpt_path "$last")
    uv run terratorch fit -c configs/$m.yaml -c configs/datasets/peru_unosat.yaml -c configs/trainer/cuda.yaml \
      --trainer.logger.init_args.save_dir "$LOCAL/$m" --trainer.logger.init_args.version "seed$s" \
      --seed_everything "$s" "${resume[@]}" "$@" >> "$LOCAL/logs/${m}_seed$s.log" 2>&1
    rc=$?
    echo "$m seed$s exit=$rc $(date -Is)" | tee -a "$LOCAL/queue.txt"
    [ $rc -eq 0 ] && touch "$LOCAL/$run/done"
    rsync -rlt "$LOCAL/" "$KEEP/"
    # Prithvi checkpoints are 3.9 GB: keep only best-*.ckpt of a finished run, on the volume.
    if [ $rc -eq 0 ]; then rm -f "$KEEP/$run/checkpoints/last.ckpt"; rm -rf "$LOCAL/$run/checkpoints"; fi
  done
done
