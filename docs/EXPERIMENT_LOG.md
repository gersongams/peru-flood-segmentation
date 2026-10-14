# Experiment log

What was run, when, from which commit, and where each number comes from.
This is the trace for the thesis and paper: every number below is read from
the file named next to it (`metrics.csv` best epoch on `val/IoU_Flood`, or the
`terratorch validate` output log). Run directories live under `outputs/`, which
git ignores; each run directory holds `run.json` (seed, commit, dirty flag,
device, precision, versions, hardware, wall clock) and the resolved config.

Add an entry for every run or evaluation, newest at the bottom. Record failed
and discarded runs too, with the reason.

"Water IoU" = `val/IoU_Flood` (IoU of the water class, ignore pixels −1
excluded). "dev" = `peru-dev` (UNOSAT 2023 labels, 174 chips). Sen1Floods11
`valid` (89 chips) and `peru-dev` scores are not comparable with each other.

## 2026-09-29 — MPS feasibility

Commit `d5ecafa`. Details and measured speed/memory in `docs/feasibility.md`.

## 2026-10-11/12 — Sen1Floods11 baselines, seed 0 (Mac, MPS, fp32)

Train on Sen1Floods11 official `train` (252), select on `valid` (89).

| Model | Best valid water IoU | mIoU | Best epoch / epochs run | Wall clock | Commit | Run dir |
|---|---|---|---|---|---|---|
| U-Net (ResNet-34) | 0.8188 | 0.8973 | 69 / 90 | 1.5 h | `3762328` | `outputs/runs/unet/lightning_logs/version_1` |
| TerraMind v1-B | 0.8168 | 0.8958 | 97 / 100 (hit cap) | 3.8 h | `3762328` | `outputs/runs/terramind_v1_base/lightning_logs/version_1` |
| Prithvi-EO-2.0 300M | 0.8327 | 0.9052 | 72 / 93 | 6.5 h | `5961853` | `outputs/runs/prithvi_eo_v2_300/lightning_logs/version_1` |

All `dirty: false`. Only seed 0 exists so far (tasks 3.4–3.6 open).

## 2026-10-12 — Peru dataset built (no manual labelling)

- UNOSAT 2023 chips: `scripts/build_peru_unosat.py` (commit `9505a7b`), 1,291 chips
  (668 for 2023-03-22, 623 for 2023-04-01), manifest `data/peru/unosat2023/manifest.csv`.
- Spatial-block split: `scripts/split_peru_unosat.py` (commit `53190c4`), train 753 /
  dev 174 / test 172 / buffer 192 chips, `data/peru/unosat2023/splits/`.
- `peru-2017-test`: `scripts/build_peru_emsr2017.py` (commit `f06df2c`), 204 chips,
  `data/peru/emsr2017/manifest.csv`.
- Label sources: `docs/peru_label_sources.md`.
- Held-out guard (`check_train_split`) added in commit `93022b1`.

## 2026-10-12 — Peru-only U-Net demo (Mac)

`outputs/runs_peru/unet_demo/lightning_logs/version_0`, commit `f06df2c` (dirty, demo
mode), 3 epochs: dev water IoU 0.6195. Pipeline check only.

## 2026-10-12/13 — Peru-only, seed 0 (Mac, MPS, fp32)

Pretrained backbone → `peru-train`, early stopping (patience 20) on `peru-dev`.
Command: `terratorch fit -c configs/<m>.yaml -c configs/datasets/peru_unosat.yaml -c configs/trainer/mps.yaml --seed_everything 0`.

| Model | Best dev water IoU | mIoU | Best epoch / epochs run | Wall clock | Commit | Run dir |
|---|---|---|---|---|---|---|
| U-Net | 0.9413 | 0.9664 | 51 / 72 | 2.0 h | `93022b1` | `outputs/runs_peru/unet/lightning_logs/version_0` |
| TerraMind v1-B | 0.9270 | 0.9581 | 50 / 71 | 6.2 h | `483e123` | `outputs/runs_peru/terramind_v1_base/lightning_logs/version_0` |
| Prithvi-EO-2.0 300M | 0.9215 | 0.9550 | 53 / 74 | 13.4 h | `483e123` | `outputs/runs_peru/prithvi_eo_v2_300/lightning_logs/version_0` |

Wall clock includes the Mac sleeping or hibernating (lid closed, battery empty), so it
is not a speed measurement. These runs are kept as a cross-device check; the reported
Peru-only results are the cloud runs below.

Discarded: a U-Net seed-1 run started on the Mac and stopped after about 20 minutes when
training moved to RunPod (`outputs/runs_peru/unet/lightning_logs/version_1`, incomplete).

## 2026-10-13 — RunPod speed test (demo gate)

Pod `r62ve4bh8iksdj`, RTX 4090, Secure Cloud, US-NC-1, $0.89/h (RunPod catalog read).
Commit `7fa328b`, 2 epochs per model, 16-mixed (`configs/trainer/cuda.yaml`):
U-Net 54 s, TerraMind 67 s, Prithvi 96 s wall clock (`run.json`). Estimated full job
(3 models × 3 seeds) about 7 h, about $6–7.

Problems hit and fixed during the cloud set-up (scripts in `scripts/cloud/`):
1. rsync failed on the volume (`chown` not permitted) → `-rlt` (commit `7f39102`).
2. The queue log inside the repo made the tree dirty → provenance guard refused to train.
3. The `/workspace` volume quota filled (12 GB venv + 4.8 GB uv cache) → checkpoint
   writes hung after the first save; venv, cache and run outputs moved to container
   disk (commits `f7c1b49`, `55c5e04`).
4. Container disk filled with checkpoints (Prithvi 3.9 GB each) → Prithvi seed-1
   crashed with `OSError: [Errno 28]`; finished runs now keep only `best-*.ckpt`
   (commit `45270f1`). The failed attempt and a just-started U-Net seed 2 were deleted
   and rerun from scratch.

## 2026-10-13/14 — Peru-only, 3 seeds (RunPod RTX 4090, 16-mixed)

Same configs as the Mac Peru-only runs; only device and precision differ.
Queue: `scripts/cloud/run_queue.sh "unet terramind_v1_base prithvi_eo_v2_300" "0 1 2"`.
Run dirs: `outputs/runs_cloud/<model>/lightning_logs/seed<s>`; per-run table:
`outputs/runs_cloud/summary_peru_dev.csv`. All runs `dirty: false`, commit `55c5e04`
or `45270f1` (these differ only in `scripts/cloud/run_queue.sh`).

| Model | seed 0 | seed 1 | seed 2 | Dev water IoU mean ± std | Dev mIoU mean ± std |
|---|---|---|---|---|---|
| U-Net | 0.9397 | 0.9434 | 0.9414 | **0.9415 ± 0.0019** | 0.9665 ± 0.0011 |
| TerraMind v1-B | 0.9275 | 0.9237 | 0.9283 | 0.9265 ± 0.0025 | 0.9579 ± 0.0014 |
| Prithvi-EO-2.0 300M | 0.9212 | 0.9225 | 0.9217 | 0.9218 ± 0.0007 | 0.9552 ± 0.0004 |

std is the sample std over 3 seeds. Best epochs: U-Net 51/98/63, TerraMind 92/93/88
(all three hit the 100-epoch cap, so TerraMind may be slightly under-trained),
Prithvi 48/51/44. Training time per run 22–35 min (`run.json` wall clock).

Cross-device check, seed 0, Mac fp32 vs 4090 16-mixed: U-Net 0.9413 vs 0.9397,
TerraMind 0.9270 vs 0.9275, Prithvi 0.9215 vs 0.9212.

Cost: pod uptime 24,930 s (RunPod `get-pod` read) × $0.89/h ≈ $6.2 GPU time, plus the
stopped-volume fee. Pod stopped 2026-10-14; results copied to the Mac and checked
file by file.

## 2026-10-14 — Sen1Floods11-only models evaluated on `peru-dev` (comparison row)

The seed-0 Sen1Floods11 checkpoints above, without Peru training, evaluated on
`peru-dev` with each model's own normalization. Commit `45270f1` (clean), Mac MPS fp32.
Command: `terratorch validate -c configs/<m>.yaml -c configs/datasets/peru_unosat.yaml -c configs/trainer/mps.yaml --ckpt_path <best ckpt>`.
Logs: `outputs/eval_s1f11_<m>.log`.

| Model | Checkpoint | Dev water IoU | Dev mIoU | Peru-only mean (above) |
|---|---|---|---|---|
| U-Net | `outputs/runs/unet/.../best-epoch=69.ckpt` | 0.8354 | 0.9041 | 0.9415 |
| TerraMind v1-B | `outputs/runs/terramind_v1_base/.../best-epoch=97.ckpt` | 0.7951 | 0.8797 | 0.9265 |
| Prithvi-EO-2.0 300M | `outputs/runs/prithvi_eo_v2_300/.../best-epoch=72.ckpt` | 0.8758 | 0.9281 | 0.9218 |

Single seed only. Without Peru data Prithvi transfers best and TerraMind worst; with
Peru training the order flips and U-Net leads.

## 2026-10-14 — Peru dataset frozen (v1)

`scripts/freeze_peru.py` (commit `f5319e5`): all 2,990 chip files match their manifest
SHA-256; manifests and split lists pinned in `docs/peru_v1_checksums.txt`. Data card:
`docs/peru_data_card.md`.

## 2026-10-14 — Sen1Floods11 → Peru fine-tuning, 3 seeds (RunPod RTX 4090, 16-mixed)

Same Peru-only configs, but the weights start from the seed-0 Sen1Floods11 best
checkpoints above (`floodseg.callbacks.InitFromCheckpoint`, model weights only;
optimizer and epochs start fresh). Commit `0edf781`. Local check first (Mac, 1 epoch,
U-Net): dev water IoU 0.879 after one epoch, so the weights load.
Pod `827muf75pdgnme` (RTX 4090, Secure Cloud, EUR-IS-2, $0.89/h; the first pod could
not restart because its host had no free GPU). Queue:
`INIT_DIR=/workspace/init_s1f11 scripts/cloud/run_queue.sh ... "0 1 2"`, outputs
`runs_cloud_s1f11_ft`. First epoch of U-Net seed 0: dev water IoU 0.910 (from scratch on
the 4090: 0.556).

Results (run dirs `outputs/runs_cloud_s1f11_ft/<model>/lightning_logs/seed<s>`, per-run
table `outputs/runs_cloud_s1f11_ft/summary_peru_dev.csv`; all `dirty: false`, commit
`0edf781`; all 9 runs `exit=0` and marked done; copy verified file by file, 82 files):

| Model | seed 0 | seed 1 | seed 2 | Dev water IoU mean ± std | Dev mIoU mean ± std | Best epochs |
|---|---|---|---|---|---|---|
| U-Net | 0.9400 | 0.9364 | 0.9385 | 0.9383 ± 0.0018 | 0.9647 ± 0.0011 | 52 / 28 / 31 |
| TerraMind v1-B | 0.9279 | 0.9291 | 0.9275 | 0.9282 ± 0.0009 | 0.9588 ± 0.0005 | 23 / 22 / 15 |
| Prithvi-EO-2.0 300M | 0.9229 | 0.9240 | 0.9240 | 0.9236 ± 0.0006 | 0.9562 ± 0.0004 | 38 / 58 / 37 |

All three seeds start from the same seed-0 Sen1Floods11 checkpoint, so the std covers
only the Peru stage. Pod uptime 19,926 s (RunPod `pod-action stop` read) × $0.89/h ≈
$4.9; pod stopped 2026-10-14 after the download was verified.

## 2026-10-14 — Failure analysis on `peru-dev` (phase C)

- Strata (task 6.1): `scripts/build_strata.py dev` → `data/peru/strata/unosat2023/`
  (ESA WorldCover 2021 v200 tiles S06W081, S09W081; Copernicus DEM GLO-30 tiles
  S06/S07 × W080/W081, from the public AWS buckets, in `data/aux/`; UNOSAT existing
  water; thick-cloud distance; NDWI).
- Predictions: `terratorch validate ... --trainer.callbacks+=floodseg.callbacks.SaveValPredictions`
  (commit `7662657`) for the 9 Peru-only cloud checkpoints and the 3 Sen1Floods11-only
  checkpoints, on the Mac (MPS, fp32) → `outputs/preds_dev/`. The re-evaluated dev
  water IoU of every cloud run matches its training log to 4 decimals (e.g. U-Net seed 0
  0.9397, TerraMind seed 0 0.9275, Prithvi seed 0 0.9212).
- Analysis: `scripts/analyze_dev.py`, `scripts/report_dev.py`, `scripts/plot_dev_chips.py`
  → `outputs/analysis_dev/`. Write-up with all tables: `docs/failure_analysis_dev.md`.
- Follow-up checks (same day): NDWI binned over all labelled pixels (precision/recall
  in the low-contrast band), ECE/AUROC restricted to pixels within 10 px of a water
  edge, patch size checked in code (16 px for both foundation models). Fine-tuned
  models' predictions added (`outputs/preds_dev/s1f11_ft/`; re-evaluated dev IoU matches
  each training log to 4 decimals). Results in `docs/failure_analysis_dev.md`.

## 2026-10-14 — Cloud clean-up

Both RunPod pods (`r62ve4bh8iksdj`, `827muf75pdgnme`) terminated on the user's request after
all results were on the Mac; `list-pods` returns none. Nothing on RunPod bills any more.

## 2026-10-14 — Uncertainty on `peru-dev` (tasks 5.1, 5.3, 5.4) and an AUROC correction

- `scripts/uncertainty_dev.py` → `outputs/analysis_dev/uncertainty.csv`, write-up
  `docs/uncertainty_dev.md`: seed ensembles (entropy, MI) and temperature scaling
  (2-fold by chip on dev) for the Peru-only and Sen1Floods11 → Peru models.
- Correction: the error-detection AUROC in the first version of
  `docs/failure_analysis_dev.md` used a 1000-bin histogram that overstated it (U-Net
  0.710 instead of 0.662). `scripts/analyze_dev.py` now uses the exact rank-based AUROC
  (checked against `scipy.stats.rankdata`: 0.6574 on the same pixels); the document was
  corrected.

## 2026-10-14 — TerraMind Peru-only, 200-epoch cap (under-training check)

Pod `n7g3tfutcmy68k` (RTX 4090, Secure Cloud, US-IL-1, $0.89/h). Commit `0c3ee97`.
`RUN_NAME=runs_cloud_tm200 run_queue.sh terramind_v1_base 0 --trainer.max_epochs=200`.
Result (`outputs/runs_cloud_tm200/terramind_v1_base/lightning_logs/seed0`, `dirty: false`,
29 min): early stopping ended it after 98 epochs; best dev water IoU **0.9270** (mIoU
0.9582) at epoch 77, against 0.9275 at epoch 92 with the 100-epoch cap (seed 0). The cap
was not limiting TerraMind. Copy verified (10 files); pod created 11:42 UTC and deleted
right after the copy, about 45 min ≈ $0.7; `list-pods` returns none.

## 2026-10-14 — Sen1Floods11 `valid` predictions; second AUROC correction (float16 saturation); logits re-inference

- Sen1Floods11 `valid` predictions of the three seed-0 Sen1Floods11 checkpoints
  (`terratorch validate -c configs/<m>.yaml -c configs/trainer/mps.yaml --ckpt_path <best ckpt>`
  with `SaveValPredictions`), commit `b48fb97` (clean), Mac MPS fp32 →
  `outputs/preds_valid/` (float16 probabilities; superseded below). Water IoU 0.8188 /
  0.8168 / 0.8327 = training log.
- **Correction.** `SaveValPredictions` stored float16 probabilities. float16 has only two
  values in [0.99951, 1] (0.99951 and 1.0) but resolves down to ~1e-7 near 0, and 99 % of
  dev pixels lie in that saturated range, so the confidence ranking of confident pixels
  depended on the predicted class. This biased every error-detection AUROC and the
  temperature fits (e.g. Prithvi Peru-only seed 0 AUROC 0.607 as stored vs 0.687 with
  symmetric clipping; ensemble 0.862 vs 0.771). IoU, recall/FPR strata and raw ECE are not
  affected (they depend only on the 0.5 threshold and the top confidence bin). The
  "|logit| ≈ 7.6" caveat in the first `docs/uncertainty_dev.md` was also wrong (1.0 mapped to
  +13.8 with the 1e-6 clip).
- Fix: `SaveValPredictions` now saves the float32 water logit z = logit_water − logit_other
  (commit `583c018`); analysis scripts work on logits in float64, ensemble logit from both
  tails in log space. All 24 predictions recomputed from the same checkpoints, Mac MPS fp32,
  commit `583c018` (clean): 9 Peru-only + 9 Sen1Floods11 → Peru + 3 Sen1Floods11-only on
  `peru-dev` → `outputs/logits_dev/`, 3 Sen1Floods11-only on `valid` → `outputs/logits_valid/`
  (logs next to them). Every run's water IoU matches its training log to 4 decimals.
  Size of the old bias: `scripts/float16_diagnostic.py`.
  `outputs/preds_dev/` and `outputs/preds_valid/` are kept only as a record of the old numbers.

## 2026-10-14 — Calibration analysis on the logits (tasks 5.3, 5.4, 6.4)

`scripts/analyze_dev.py`, `scripts/uncertainty_dev.py`, `scripts/calibration_dev.py`,
`scripts/report_dev.py`, `scripts/report_calibration.py` on `outputs/logits_dev/` and
`outputs/logits_valid/` → `outputs/analysis_dev/` (log `run_logits.log`). Temperatures fitted
on Sen1Floods11 `valid` (seed-0 Sen1Floods11 models, `temperatures_valid.json`): U-Net 1.73,
TerraMind 1.01, Prithvi 2.80. Results: `docs/calibration_dev.md`, `docs/uncertainty_dev.md`
(rewritten), AUROC rows of `docs/failure_analysis_dev.md` corrected. Mac CPU, about 1 h 20 min
in total. Method and decisions: `docs/METHODS_NOTES.md`.

## 2026-10-14 — Label QA, band statistics and transfer matrix (tasks 4.4, 6.2)

- Peru-only and Sen1Floods11 → Peru checkpoints (all 18) evaluated on Sen1Floods11 `valid`:
  `terratorch validate -c configs/<m>.yaml -c configs/trainer/mps.yaml --ckpt_path <best ckpt>`,
  commit `71ccc9c` (clean), Mac MPS fp32, logs `outputs/eval_valid/{peru,s1f11_ft}_<m>_seed<s>.log`.
  All 18 finished. Water IoU, mean ± std over seeds: Peru-only U-Net 0.6898 ± 0.0065,
  TerraMind 0.6559 ± 0.0131, Prithvi 0.6527 ± 0.0190; Sen1Floods11 → Peru 0.7019 ± 0.0133,
  0.6911 ± 0.0234, 0.6805 ± 0.0056. Matrix with the existing logs:
  `scripts/transfer_matrix.py` → `outputs/transfer/matrix.{md,csv}`.
- `scripts/label_qa.py [--figs] [--check]` → `outputs/label_qa/` (no model used for the flags;
  `--check` compares with the Peru-only logits afterwards). List: `docs/peru_doubtful_chips.md`.
  Threshold revision (±0.2 → 0.5 / −0.1) recorded in `docs/METHODS_NOTES.md`.
- `scripts/band_stats.py` → `outputs/band_stats/`.
- Write-up: `docs/domain_shift_dev.md`. CPU parts about 2 min each on the Mac.

## 2026-10-14 — Edge-tolerant IoU, index baselines, UNOSAT provenance (no training)

CPU evaluations on existing outputs, Mac; no test split used.
- `scripts/edge_tolerance.py` (commit `266281b`) on `outputs/logits_dev/{peru,s1f11_ft}` →
  `outputs/analysis_dev/edge_tolerance.{md,csv}`.
- `scripts/index_baseline.py` (commit `822a9c4`) on `peru-dev` and Sen1Floods11 `valid` →
  `outputs/analysis_dev/index_baseline.{md,csv}` (MNDWI/NDWI threshold sweep; Otsu on unlabelled
  `peru-train`: t = 0.0391, dev IoU 0.5718).
- `scripts/unosat_provenance.py` (commit `5d769fe`) on `data/unosat2023/raw/` →
  `outputs/label_qa/unosat_geometry.csv`.
Results and interpretation: `docs/METHODS_NOTES.md` (entries of 2026-10-14).
