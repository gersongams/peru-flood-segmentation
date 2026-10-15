#!/usr/bin/env bash
# Run ON the pod with setsid/nohup. Phase-D pod session 2 (pre-registered in
# docs/METHODS_NOTES.md 2026-10-15, "Pod session 2"). Two queues share the GPU:
#   bash scripts/cloud/queue_phaseD2.sh A   # Prithvi Sen1Floods11 source models
#   bash scripts/cloud/queue_phaseD2.sh B   # training-time HM / FDA (U-Net, TerraMind, then Prithvi HM)
# A: LR sweep 3e-5 / 1e-4 / 3e-4, seed 0, 100 epochs; winner = best `valid` water IoU
#    (val/IoU_Flood, max over epochs); if the winner is at a grid edge, one more value further
#    out (1e-5 or 1e-3) and re-select; then seeds 0-2 at the winner, 200-epoch cap. The winner is
#    written to /workspace/prithvi_lr.txt.
# B: U-Net (LR 3e-4) and TerraMind (LR 5e-5) × {HM, FDA} × seeds 0-2, 200-epoch cap; then Prithvi
#    with HM × seeds 0-2 at the LR of A (starts after queue A is done).
set -uo pipefail
Q=${1:?queue A or B}
R=/workspace/peru-flood-segmentation
cd $R
export DATASET=s1f11 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/workspace/queue_phaseD2_$Q.log
q() { echo "$* $(date -Is)" | tee -a "$LOG"; }

best_lr() {  # prints the LR whose sweep run has the highest max val/IoU_Flood
  uv run python - "$@" <<'EOF'
import sys, pandas as pd
best = None
for lr in sys.argv[1:]:
    f = f"/workspace/outputs/s1f11_prithvi_lr{lr}/prithvi_eo_v2_300/lightning_logs/seed0/metrics.csv"
    v = pd.read_csv(f)["val/IoU_Flood"].max()
    print(f"lr {lr}: {v:.4f}", file=sys.stderr)
    if best is None or v > best[1]:
        best = (lr, v)
print(best[0])
EOF
}

if [ "$Q" = A ]; then
  grid="3e-5 1e-4 3e-4"
  for lr in $grid; do
    RUN_NAME=s1f11_prithvi_lr$lr bash scripts/cloud/run_queue.sh prithvi_eo_v2_300 0 --optimizer.init_args.lr=$lr
    q "sweep lr $lr done"
  done
  lr=$(best_lr $grid 2>>"$LOG")
  extra=""; [ "$lr" = 3e-5 ] && extra=1e-5; [ "$lr" = 3e-4 ] && extra=1e-3
  if [ -n "$extra" ]; then
    q "winner $lr at grid edge: adding $extra"
    RUN_NAME=s1f11_prithvi_lr$extra bash scripts/cloud/run_queue.sh prithvi_eo_v2_300 0 --optimizer.init_args.lr=$extra
    grid="$grid $extra"; lr=$(best_lr $grid 2>>"$LOG")
  fi
  echo "$lr" > /workspace/prithvi_lr.txt
  q "chosen Prithvi LR $lr"
  RUN_NAME=s1f11_src_prithvi bash scripts/cloud/run_queue.sh prithvi_eo_v2_300 "0 1 2" \
    --optimizer.init_args.lr=$lr --trainer.max_epochs=200
  q "QUEUE DONE"
else
  for k in hm fda; do
    for m in unet terramind_v1_base; do
      lr=3e-4; [ $m = terramind_v1_base ] && lr=5e-5
      RUN_NAME=s1f11_tr${k}_$m bash scripts/cloud/run_queue.sh $m "0 1 2" \
        --optimizer.init_args.lr=$lr --trainer.max_epochs=200 -c configs/phaseD/train_$k.yaml
      q "train-time $k $m done"
    done
  done
  # wait for queue A to finish completely: two Prithvi runs at once may not fit in 24 GB
  until grep -q "QUEUE DONE" /workspace/queue_phaseD2_A.log 2>/dev/null; do sleep 60; done
  lr=$(cat /workspace/prithvi_lr.txt)
  RUN_NAME=s1f11_trhm_prithvi_eo_v2_300 bash scripts/cloud/run_queue.sh prithvi_eo_v2_300 "0 1 2" \
    --optimizer.init_args.lr=$lr --trainer.max_epochs=200 -c configs/phaseD/train_hm_prithvi.yaml
  q "train-time hm prithvi (lr $lr) done"
  q "QUEUE DONE"
fi
