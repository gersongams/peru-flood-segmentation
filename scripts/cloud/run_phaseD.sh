#!/usr/bin/env bash
# Run ON the pod with setsid/nohup. Works through a job file, one job per line:
#   <model> <seed> <variant> <alpha> <q_end>
# For each job: self-training (scripts/self_train.py), then label-free dev predictions of every
# saved checkpoint (scripts/eval_adapted.py --all; packed for intermediate iterations, float32
# logits for the final one), then the intermediate weights are deleted (final teacher + student
# kept). A job with a `done` marker is skipped.
#   bash scripts/cloud/run_phaseD.sh jobs_unet.txt unet.log
set -uo pipefail
JOBS=${1:?jobs file}; LOG=${2:?log file}
cd /workspace/peru-flood-segmentation
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/workspace/.cache/huggingface
while read -r m s v a q; do
  [ -z "$m" ] && continue
  tag=$v; { [ "$a" != 0.99 ] || [ "$q" != 0.5 ]; } && tag="${v}_a${a}_q${q}"
  run=outputs/phaseD/self_train/$tag/$m/seed$s
  [ -f "$run/done" ] && { echo "$tag $m seed$s already done" >> "$LOG"; continue; }
  mkdir -p "$run"
  uv run python scripts/self_train.py --model "$m" --seed "$s" --variant "$v" --alpha "$a" --q_end "$q" > "$run/train.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    uv run python scripts/eval_adapted.py "$run" --all > "$run/eval.log" 2>&1; rc=$?
  fi
  if [ $rc -eq 0 ]; then
    last=$(ls "$run"/checkpoints/iter*.pt | sed 's/.*iter\([0-9]*\).pt/\1/' | sort -n | tail -1)
    find "$run/checkpoints" -name 'iter*.pt' ! -name "iter$last.pt" -delete
    touch "$run/done"
  fi
  echo "$tag $m seed$s exit=$rc $(date -Is)" >> "$LOG"
done < "$JOBS"
echo "QUEUE DONE $(date -Is)" >> "$LOG"
