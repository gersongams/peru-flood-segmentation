#!/usr/bin/env bash
# Sen1Floods11 `valid` logits of the training-time alignment models (pod session 2), for the
# label-free temperature of scripts/phaseD_analysis.py. Each checkpoint is validated with the
# same pipeline it was selected on: model config + its train-time overlay (configs/phaseD/
# train_<hm|fda>[_prithvi].yaml: HistMatchToPool, or FDAToPool with the mean pool amplitude, on
# raw DN before normalization) on the official `valid` split. Check: the logged val/IoU_Flood
# of this validate pass vs the run's best logged value (printed).
# → outputs/logits_valid/<trhm|trfda>/<model>/seed<s>/<id>_S2Hand.npy
#   bash scripts/phaseD2_valid_logits.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for j in trhm:unet trhm:terramind_v1_base trhm:prithvi_eo_v2_300 trfda:unet trfda:terramind_v1_base; do
  r=${j%%:*}; m=${j##*:}; k=${r#tr}; ov=configs/phaseD/train_$k.yaml
  [ $m = prithvi_eo_v2_300 ] && ov=configs/phaseD/train_${k}_prithvi.yaml
  for s in 0 1 2; do
    run=outputs/s1f11_${r}_$m/$m/lightning_logs/seed$s
    ck=$(ls $run/checkpoints/best-*.ckpt | head -1)
    out=outputs/logits_valid/$r/$m/seed$s
    log=outputs/logits_valid/log_${r}_${m}_seed$s.txt
    mkdir -p outputs/logits_valid
    [ "$(ls $out/*.npy 2>/dev/null | wc -l)" -eq 89 ] || uv run terratorch validate -c configs/$m.yaml -c $ov -c configs/trainer/mps.yaml \
      --trainer.callbacks+=floodseg.callbacks.SaveValPredictions --trainer.callbacks.out_dir=$out --ckpt_path "$ck" > $log 2>&1
    got=$(grep -o "val/IoU_Flood[^0-9]*[0-9.]*" $log | tail -1 | grep -o "[0-9.]*$")
    want=$(uv run python -c "import pandas as pd; print(pd.read_csv('$run/metrics.csv')['val/IoU_Flood'].max())")
    echo "$r $m seed$s: $(ls $out/*.npy | wc -l) files, validate $got, logged best $want"
  done
done
