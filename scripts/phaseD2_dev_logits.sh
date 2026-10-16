#!/usr/bin/env bash
# Peru-dev logits of the pod-session-2 models (METHODS_NOTES 2026-10-15 "Pod session 2"), with the
# procedure of the phase-D source rows (EXPERIMENT_LOG 2026-10-15, tasks 3.1/3.2): terratorch
# validate on `peru-dev` with the model's base config (Peru images unchanged: the training-time
# transform is not applied at test time) and the run's best checkpoint (chosen on Sen1Floods11 `valid`).
#   s1f11_src  ← outputs/s1f11_src_prithvi              (Prithvi source models)
#   trhm       ← outputs/s1f11_trhm_<model>             (training-time histogram matching)
#   trfda      ← outputs/s1f11_trfda_<model>            (training-time FDA)
# → outputs/logits_dev/<regime>/<model>/seed<s>/<stem>.npy; logs outputs/logits_dev/log_<regime>_<model>_seed<s>.txt
#   bash scripts/phaseD2_dev_logits.sh [regime:model ...]   # default: all available
set -euo pipefail
cd "$(dirname "$0")/.."
jobs=${*:-"s1f11_src:prithvi_eo_v2_300 trhm:unet trhm:terramind_v1_base trhm:prithvi_eo_v2_300 trfda:unet trfda:terramind_v1_base"}
for j in $jobs; do
  r=${j%%:*}; m=${j##*:}
  src=outputs/s1f11_${r#s1f11_}_$m; [ $r = s1f11_src ] && src=outputs/s1f11_src_prithvi
  for s in 0 1 2; do
    ck=$(ls $src/$m/lightning_logs/seed$s/checkpoints/best-*.ckpt 2>/dev/null | head -1) || true
    [ -z "$ck" ] && { echo "skip $r $m seed$s (no checkpoint)"; continue; }
    out=outputs/logits_dev/$r/$m/seed$s
    [ "$(ls $out/*.npy 2>/dev/null | wc -l)" -eq 174 ] && { echo "done $r $m seed$s"; continue; }
    uv run terratorch validate -c configs/$m.yaml -c configs/datasets/peru_unosat.yaml -c configs/trainer/mps.yaml \
      --trainer.callbacks+=floodseg.callbacks.SaveValPredictions --trainer.callbacks.out_dir=$out \
      --ckpt_path "$ck" > outputs/logits_dev/log_${r}_${m}_seed$s.txt 2>&1
    echo "$r $m seed$s: $(ls $out/*.npy | wc -l) files, $(grep -o "val/IoU_Flood[^0-9]*[0-9.]*" outputs/logits_dev/log_${r}_${m}_seed$s.txt | tail -1)"
  done
done
