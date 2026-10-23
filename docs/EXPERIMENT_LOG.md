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

## 2026-10-14/15 — Sen1Floods11 source models for phase D (phase-D tasks 1.1–1.4; plan 3.4–3.6)

RunPod, EU-RO-1, RTX 4090 Secure Cloud ($0.89/h), `16-mixed`, commit `8a8fee4` (clean on the
pod; every `run.json` has `dirty: false`), torch 2.14.0, lightning 2.6.6, terratorch 1.2.13.
Launched with `DATASET=s1f11 RUN_NAME=<name> bash scripts/cloud/run_queue.sh <model> <seeds>
--optimizer.init_args.lr=<lr> [--trainer.max_epochs=200]`. Official Sen1Floods11 splits
(train 252 / valid 89); early stopping and best checkpoint on `val/IoU_Flood` (patience 20),
so the `valid` IoU below is the selection score, not an unbiased estimate.
Copied back with `scripts/cloud/fetch_results.sh` → `outputs/<name>/`; every folder verified
file by file (three transfers dropped with "rsync: unexpected end of file" and were resumed;
all matched on the retry). Queue logs: `outputs/{setup,sweep,extra,final,tm5}.log`,
`outputs/final_choice.txt`. Numbers below recomputed from each run's `metrics.csv`
(max of `val/IoU_Flood`).

Pods:
- `0abbsqm79fyphu`: failed setup. Host CUDA 12.8 (driver 570), but the lockfile's torch
  2.14.0+cu130 needs CUDA 13. Deleted; $0.28 (pod billing read). Fix for later pods: create
  with `gpu.minCudaVersion: "13.0"`.
- `rev6hvyw1g0fdl`: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, driver 580,
  created 2026-10-14 21:50 UTC, deleted 2026-10-15 after verification; $4.28 (pod billing:
  $1.946 on 10-14 + $2.335 on 10-15). Includes ≈ 50 min idle between the last run
  (00:41 UTC) and the start of the copy. **Section 1 total $4.56.**

LR sweep, seed 0, max 100 epochs (best `valid` water IoU, epoch of best, last epoch):

| Model | LR | Folder | IoU | best ep | last ep |
|---|---|---|---|---|---|
| U-Net | 3e-5 | `s1f11_lr3e-5` | 0.7964 | 63 | 83 |
| U-Net | 1e-4 | `s1f11_lr1e-4` | 0.8140 | 39 | 59 |
| U-Net | **3e-4** | `s1f11_lr3e-4` | **0.8325** | 95 | 99 (cap) |
| U-Net | 1e-3 | `s1f11_lr1e-3` | 0.8154 | 39 | 59 |
| TerraMind v1-B | 1e-5 | `s1f11_lr1e-5` | 0.7504 | 38 | 58 |
| TerraMind v1-B | 2e-5 | `s1f11_lr2e-5` | 0.8197 | 99 | 99 (cap) |
| TerraMind v1-B | **5e-5** | `s1f11_lr5e-5` | **0.8354** | 60 | 80 |

Final runs, max 200 epochs (cap raised because two sweep runs peaked at the 100-epoch cap;
authors approved):

| Model, LR | Folder | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| U-Net, 3e-4 | `s1f11_src_unet` | 0.8330 (ep 83) | 0.8300 (ep 74) | 0.8315 (ep 89) | **0.8315 ± 0.0015** |
| TerraMind, 5e-5 | `s1f11_src_terramind_lr5e-5` | 0.8333 (ep 52) | 0.8359 (ep 88) | 0.8355 (ep 63) | **0.8349 ± 0.0014** |
| TerraMind, 2e-5 (superseded) | `s1f11_src_terramind` | 0.8258 (ep 112) | 0.7971 (ep 47) | 0.8194 (ep 86) | 0.8141 ± 0.0151 |

- **Mistake, superseded runs.** The TerraMind final runs were first queued with LR 2e-5,
  hardcoded before the 5e-5 sweep run had finished. 5e-5 was better, so the three seeds were
  rerun at 5e-5. The 2e-5 runs are kept and logged but not used.
- **Reproducibility (plan 3.5).** U-Net seed 0 rerun with identical settings
  (`s1f11_src_unet_repeat`): 0.8161 (best ep 34, stopped ep 54) vs 0.8330 (best ep 83). Run-to-run
  difference on the 4090 = 1.7 IoU points, larger than the seed std (0.15). Not bit-reproducible
  (cuDNN nondeterminism + mixed precision), amplified by early stopping on a noisy curve.
- `run.json`/`hparams.yaml` show `lr: 0.001` (the task's unused `lr` field). The optimizer
  actually used is in `config.yaml` → `optimizer.init_args.lr` (3e-4 U-Net, 5e-5 TerraMind).
- Wall clock per run: U-Net 336–648 s, TerraMind 379–839 s.
- Phase-D source checkpoints: `outputs/s1f11_src_unet/unet/lightning_logs/seed{0,1,2}/checkpoints/`
  and `outputs/s1f11_src_terramind_lr5e-5/terramind_v1_base/lightning_logs/seed{0,1,2}/checkpoints/`.

## 2026-10-15 — Phase-D source-only models and index baselines on `peru-dev` (tasks 3.1, 3.2)

Mac MPS fp32, commit `b846843` (clean). Logits of the phase-D Sen1Floods11 source models on
`peru-dev`: `terratorch validate -c configs/<m>.yaml -c configs/datasets/peru_unosat.yaml
-c configs/trainer/mps.yaml --trainer.callbacks+=floodseg.callbacks.SaveValPredictions
--trainer.callbacks.out_dir=outputs/logits_dev/<regime>/<m>/seed<s> --ckpt_path <best ckpt>`.
- `s1f11_src`: U-Net (`outputs/s1f11_src_unet`) and TerraMind LR 5e-5
  (`outputs/s1f11_src_terramind_lr5e-5`), seeds 0–2. Dev water IoU (TerraTorch log):
  U-Net 0.8433 / 0.8545 / 0.8114; TerraMind 0.7639 / 0.5978 / 0.7464.
- `s1f11_src_lr2e-5`: the superseded TerraMind LR 2e-5 runs (`outputs/s1f11_src_terramind`),
  diagnostic only: 0.7314 / 0.5788 / 0.7660.
- Logs `outputs/logits_dev/log_s1f11_src*_seed<s>.txt`; 174 logit files per run.
Table: `scripts/phaseD_dev_table.py` (new `src/floodseg/metrics.py`) → `outputs/phaseD/dev_table.md`,
`outputs/phaseD/dev_table_runs.csv`. The script's IoU equals the TerraTorch logs to 4 decimals.
Interpretation: `docs/METHODS_NOTES.md` (2026-10-15).

## 2026-10-15 — AdaBN, U-Net, 3 seeds (task 3.3)

`uv run python scripts/adabn.py`, commit `d48fc3f` (clean), Mac MPS. Sources:
`outputs/s1f11_src_unet/unet/lightning_logs/seed{0,1,2}` (best `valid` checkpoints). Pool
`peru-train` (753 images). Output `outputs/logits_dev/adabn/unet/seed<s>/` (+ `run.json`).
Dev water IoU (`scripts/phaseD_dev_table.py`): 0.9052 / 0.9071 / 0.8783, mean 0.8969 ± 0.0161
(source-only 0.8364 ± 0.0224). 43–66 s per seed.

## 2026-10-15 — Test-time alignment, TerraMind decoder AdaBN, self-training demos (tasks 3.4 part 1, 4.4)

Mac MPS fp32. `scripts/adabn.py terramind_v1_base` and `scripts/input_alignment_dev.py`,
commit `4903b5f` (clean) → `outputs/logits_dev/{adabn,hm,fda}/<model>/seed<s>/` (+ run.json).
Dev water IoU (`scripts/phaseD_dev_table.py`): TerraMind decoder AdaBN 0.7333 / 0.5407 / 0.6948;
HM U-Net 0.8748 / 0.8677 / 0.8778, TerraMind 0.8152 / 0.8091 / 0.8024; FDA U-Net
0.0105 / 0.0117 / 0.0125, TerraMind 0.0085 / 0.0114 / 0.0112. Demos of `scripts/self_train.py
--variant mic --demo` (20 iterations) for U-Net and TerraMind seed 0, commit `db9c33a` +
uncommitted runner (dirty; demo only): logs `outputs/phaseD/demo_{unet,terramind}_mic.log`.

## 2026-10-15 — Phase-D self-training on RunPod (tasks 4.5–4.7, 4.9): pod session

Pod `otv36sv0k42lyh` ("peru-flood-phaseD-selftrain"), Secure Cloud RTX 4090, $0.89/h,
EUR-IS-1 (EU-RO-1 had no capacity at create time), CUDA 13.0 host, driver 580.65.06, 16 vCPU
allocated, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 50 GB container disk,
60 GB `/workspace` (network filesystem). Created 12:55 UTC. Code: commit `e8a5841` (clean),
`bash /workspace/setup_pod.sh e8a5841` (torch 2.14.0+cu130).
- **Data on the pod, no Peru label file:** 753 pool + 174 dev images, Peru split lists,
  Sen1Floods11 downloaded on the pod (`scripts/download_sen1floods11.py`, official GCS bucket),
  slim source checkpoints (sha256 checked after transfer). `read_manifest` verified all 753
  pool images against the manifest checksums; no `label`, `emsr2017` or raw UNOSAT folder.
- **Upload problem:** a single SSH stream from the Mac to Iceland ran at ≈ 0.5 MB/s (the Mac's
  uplink measured 475 Mbps), so `sync_phaseD.sh` would have taken > 2 h. Fixed with 16
  parallel rsync streams (≈ 17 MB/s; too many simultaneous connections were reset by sshd and
  were retried) and 50 MB checkpoint chunks sent in parallel. Upload and setup took until
  13:43 UTC (≈ 48 min of pod time). For later sessions: parallel transfer from the start,
  download public data on the pod.
- Queues started 13:43 UTC: `scripts/cloud/run_phaseD.sh scripts/cloud/jobs_phaseD_<model>.txt`,
  U-Net and TerraMind in parallel on the GPU. Early speed (shared GPU): TerraMind ≈ 4 it/s,
  U-Net ≈ 2.7 it/s (input-bound; GPU utilization 40 %).
- **Queues finished** (`outputs/phaseD/pod_logs/queue_*.log`, all 22 runs exit 0): TerraMind
  17:50 UTC, U-Net 19:34 UTC. All runs at commit `e8a5841` (clean), 4,000 iterations,
  16-mixed precision, 0 collapse events (see the METHODS_NOTES caveat: the watch only detects
  a *shrinking* water fraction). Wall clock under shared GPU load: TerraMind `st` 890–1,025 s,
  `mic` 1,112–1,286 s; U-Net `st` 1,391–1,693 s, `mic` 1,871–1,989 s. Peak memory (CUDA, per
  process): TerraMind 6.8 GB (`mic` 8.9 GB), U-Net 4.5 GB (`mic` 6.1 GB).
- **Fetch:** `scripts/cloud/fetch_phaseD.sh 157.157.221.29 30398 10` failed first (BSD
  `xargs -I` limits a substituted argument to 255 bytes: "command line cannot be assembled,
  too long"); fixed by passing the folder as `$1` to `bash -c`. Re-run: "verified: 4048 files
  match" (same file list and sizes on pod and Mac). Local: `outputs/phaseD/self_train/<tag>/<model>/seed<s>/`
  (run.json, metrics.csv, train/eval logs, packed dev predictions iter 1,000–3,000, final
  teacher + student `iter4000.pt`), `outputs/logits_dev/<tag>/<model>/seed<s>/` (final dev logits).
- **Pod deleted** 19:41 UTC (`list-pods` empty afterwards). Billing read at ≈ 19:42 UTC
  (`list-pod-billing`, podId `otv36sv0k42lyh`): $5.34 (GPU $5.249 + disk $0.089). Expected
  ≈ $6.0 from uptime × rate (12:55–19:41 UTC ≈ 6.8 h × $0.89/h); the billing record lags.
  Second read ≈ 20:30 UTC: $5.72 (GPU $5.620 + disk $0.095), still converging to ≈ $6.0.
- **Scoring (Mac):** `uv run python scripts/phaseD_dev_table.py` → `outputs/phaseD/dev_table.md`,
  `dev_table_runs.csv`; `uv run python scripts/phaseD_oracle.py` → `outputs/phaseD/oracle_selection.{csv,md}`;
  paired per-seed deltas → `outputs/phaseD/paired_deltas.csv`; teacher water fraction on pool
  batches from `metrics.csv` → `outputs/phaseD/water_frac_trajectory.csv`. Results:
  METHODS_NOTES 2026-10-15 "Self-training results on `peru-dev`".

## 2026-10-15 — Post-hoc diagnostic: self-training teachers evaluated without AdaBN (Mac)

`uv run python scripts/eval_adapted.py outputs/phaseD/self_train/<st|mic>/<model>/seed<s> --no_adabn`
for both backbones, seeds 0–2 (12 evaluations, MPS, final teacher `iter4000.pt`) →
`outputs/logits_dev/<st|mic>_noadabn/<model>/seed<s>/`; scored with `scripts/phaseD_dev_table.py`
(rows labelled "post-hoc diagnostic"). Code: commit `ef9ec28`.
Results and the correction they cause: METHODS_NOTES 2026-10-15 "Correction: the
evaluation-time AdaBN pass causes most of the self-training drop".

Addendum: the 10 seed-0 oracle-grid runs (5 per backbone) were also evaluated with `--no_adabn`
(final teacher) → `outputs/logits_dev/st_a*_noadabn/<model>/seed0/`; scored inline (pooled
counts with `floodseg.metrics`, 174 files asserted) → `outputs/phaseD/oracle_grid_noadabn.md`.

## 2026-10-15 — Phase-D diagnostics as a script (no new runs)

`uv run python scripts/phaseD_diagnostics.py` (after `scripts/phaseD_dev_table.py`) →
`outputs/phaseD/{paired_deltas.csv, water_frac_trajectory.csv, dev_water_fraction.csv, oracle_grid_noadabn.md}`;
reproduces the inline results of the earlier phase-D entries (METHODS_NOTES 2026-10-15, reproducibility entry).

## 2026-10-15/16 — Pod session 2 (Prithvi source models, training-time HM/FDA): launch

Pre-registration: METHODS_NOTES 2026-10-15 "Pod session 2". Pod `dkck0ayqh3cylh`
("peru-flood-phaseD2-prithvi-align"), Secure RTX 4090, $0.89/h, EUR-IS-2 (availability read:
LOW stock in EU-CZ-1, EU-RO-1, EUR-IS-1, EUR-IS-2), CUDA 13.2 host, 18 vCPU, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 50 GB container disk, 60 GB `/workspace`.
Code commit `e8001a4`; `scripts/cloud/sync_phaseD2.sh` (repo bundle + `train_params.npz`,
sha256 checked; no Peru data on the pod, checked); Sen1Floods11 downloaded on the pod (446 S2Hand chips).
- **Launch problems:** (1) the pod had only the official split CSVs; the `.txt` lists TerraTorch
  reads are written by `floodseg.sen1floods11.write_split_lists`, which neither the setup nor the
  download script runs. The first launch was stopped before any epoch; lists written (252/89/90)
  and both queues relaunched clean at 23:13 UTC (pod clock). For later: run `write_split_lists`
  in setup. (2) `pkill -f queue_phaseD2.sh` inside the SSH command matched the SSH command line
  itself and killed the session; processes are now stopped by PID from a script file.
- Queues: `bash scripts/cloud/queue_phaseD2.sh A` (Prithvi sweep → finals) and `... B`
  (train-time HM/FDA). At start: 13.4 GB of 24 GB GPU memory with Prithvi + U-Net running together.
- **Queue B restarted with 8 DataLoader workers (23:21 UTC).** U-Net + HM ran ~25 s/epoch vs 5.8 s
  without the transform: input-bound with the config's 4 workers (HM 0.11 s/chip, FDA 0.26
  s/chip on CPU). Queue B was stopped by PID at epoch 1 of `s1f11_trhm_unet` seed 0, its outputs and
  log deleted, and restarted from `/workspace/queueB8.sh` (copy in `outputs/phaseD/pod2/`): the same runs
  and settings plus `--data.init_args.num_workers=8` (recorded in each run's `config.yaml`). Queue A untouched.
- **Queue A done (01:54 UTC).** Sweep, max `val/IoU_Flood` (seed 0, 100 epochs): 3e-5 0.8247,
  1e-4 0.8257, 3e-4 0.8170 → LR 1e-4 (interior: no edge run); finals seeds 0–2 at 1e-4.
- **Watcher failure.** The local watcher never worked: in zsh, `$S` holding the ssh command was
  not word-split, so every poll failed ("no such file or directory: ssh -n ...") and it ran
  to its 540-min limit without reporting. Pod checked by hand at 08:20 UTC; replaced by a script-file watcher.
- **Budget ceiling raised; Prithvi HM started in parallel (08:22 UTC).** At 08:20, ≈ 9.2 h × $0.89
  ≈ $8.2 spent vs the pre-registered $9 ceiling, with TerraMind + FDA seed 1 at epoch 101 (FDA is
  CPU-bound: ~1¾ h per TerraMind seed, GPU mostly idle) and seed 2 and Prithvi HM still to run.
  The $9 was a conservative guard set by us; the user had asked to "run all you can" with the
  ≈ $16.4 balance, so the remaining runs are completed (projected total ≈ $11). To save wall time,
  the outer loop of queue B was stopped by PID (its running `run_queue.sh` for TerraMind FDA
  continues) and `/workspace/queueBp.sh` (copy in `outputs/phaseD/pod2/`) started the Prithvi HM runs
  now, on the idle GPU, with the same settings as queue B except `num_workers` left at the
  config's 4 (HM on 6 bands is cheap; the CPUs are busy with FDA). No result had been looked at.
- **Checkpoints.** The three sweep checkpoints stay on the pod (selection uses metrics only). The
  Prithvi final checkpoints are stripped of optimizer/scheduler state (`best-*.ckpt` 3.7 GB →
  1.24 GB; every `state_dict` tensor checked equal before replacing) to fit the Mac's free disk.
- **Done.** Queue B: Prithvi HM finished 09:57 UTC, TerraMind FDA 10:38 UTC; all 21 runs exit 0 (3 sweep + 3 Prithvi finals + 9 trHM + 6 trFDA; the trHM U-Net attempt stopped at epoch 1 before the 8-worker restart is not counted)
  (`outputs/<run>/queue.txt`). Copied with `scripts/cloud/fetch_phaseD2.sh` (verified: 186 files
  match), pod deleted 2026-10-16 ≈ 10:45 UTC (`list-pods` empty). Billing (pod-billing read after
  deletion): $9.68 = GPU $9.51 + disk $0.16. Session 1 (`otv36sv0k42lyh`) billing read $6.19 (2026-10-16); the earlier balance drop implied ≈ $6.98, not reconciled.
- Run folders (best `val/IoU_Flood` @ best epoch / last epoch, seeds 0/1/2):
  `outputs/s1f11_prithvi_lr{3e-5,1e-4,3e-4}` (seed 0, 100-epoch cap; checkpoints left on the pod):
  0.8247@74/94, 0.8257@74/94, 0.8170@64/84;
  `outputs/s1f11_src_prithvi` (LR 1e-4): 0.8330@74/94, 0.8263@42/62, 0.8354@111/131;
  `outputs/s1f11_trhm_unet`: 0.8194@42, 0.8214@107, 0.8129@53; `s1f11_trhm_terramind_v1_base`:
  0.8316@57, 0.8324@88, 0.8292@62; `s1f11_trhm_prithvi_eo_v2_300`: 0.8306@53, 0.8307@79, 0.8343@95;
  `s1f11_trfda_unet`: 0.7270@34, 0.7158@41, 0.7269@57; `s1f11_trfda_terramind_v1_base`: 0.7642@87,
  0.7762@152, 0.7638@71 (train-time rows: on transformed `valid`). Queue logs and scripts in
  `outputs/phaseD/pod2/`.

## 2026-10-16 — Pod session 2 evaluation on `peru-dev` (Mac, no training)

Commits `3eb2092` (scripts) and `7e872ff` (table regimes); MPS fp32.
- Dev logits: `bash scripts/phaseD2_dev_logits.sh` → `outputs/logits_dev/{s1f11_src/prithvi_eo_v2_300,
  trhm/*, trfda/*}` (174 files per run; logs `outputs/logits_dev/log_<regime>_<model>_seed<s>.txt`;
  console output `outputs/phaseD/pod2/dev_logits_run{1,2}.txt`).
- Prithvi local regimes (`outputs/phaseD/pod2/prithvi_local_regimes.txt`):
  `phaseD_valid_logits.py prithvi_eo_v2_300` → `outputs/logits_valid/s1f11_src/prithvi_eo_v2_300`
  (pooled IoU = logged best: 0.8330 / 0.8263 / 0.8354); `adabn.py prithvi_eo_v2_300` (9 decoder BN
  layers, 753 pool images, ≈ 175 s each) → `logits_dev/adabn/prithvi_eo_v2_300`;
  `input_alignment_dev.py prithvi_eo_v2_300` → `logits_dev/{hm,fda}/prithvi_eo_v2_300`. Run records
  say `git_dirty: true`: the only uncommitted change was `scripts/phaseD_dev_table.py` (not used by them).
- `scripts/phaseD_dev_table.py` → `outputs/phaseD/dev_table.md`, `dev_table_runs.csv`. Dev water IoU:
  Prithvi source 0.8691 / 0.8699 / 0.8712, AdaBN 0.8744 / 0.8566 / 0.8709, HM 0.8088 / 0.8295 / 0.8174,
  FDA ≈ 0.008; trHM U-Net 0.9148 / 0.9098 / 0.9108, TerraMind 0.8170 / 0.8415 / 0.8736, Prithvi
  0.8687 / 0.8610 / 0.8493; trFDA U-Net 0.7251 / 0.5009 / 0.6193, TerraMind 0.8353 / 0.8373 / 0.8368.
  Table IoU equals the TerraTorch-logged dev IoU for the 18 `terratorch validate` runs (max |diff| 2.7e-8, compared by a one-off check). Paired deltas:
  `phaseD_diagnostics.py` → `outputs/phaseD/paired_deltas.csv`. Results and interpretation:
  METHODS_NOTES 2026-10-16.

## 2026-10-15 — Phase-D analysis of the best label-free methods (tasks 5.2, 5.3; Mac, no training)

`uv run python scripts/phaseD_valid_logits.py` (Sen1Floods11 `valid` logits of the 6 phase-D
source models → `outputs/logits_valid/s1f11_src/`; pooled valid IoU equals each run's logged best
`val/IoU_Flood`: U-Net 0.8330/0.8300/0.8315, TerraMind 0.8333/0.8359/0.8355), then
`uv run python scripts/phaseD_analysis.py` (log `outputs/phaseD/analysis_run.log`) →
`outputs/phaseD/analysis/`. Commit: see this entry's commit. Results: METHODS_NOTES 2026-10-15
"Where the best label-free methods still fail".

## 2026-10-16 — Freeze (phase-D 6.1, plan 7.1/7.2) and pre-test check on `peru-dev`

`uv run python scripts/build_frozen.py` → `configs/frozen.yaml` (14 groups, 42 checkpoints with
sha256 and training commits; artifacts: pool manifest, test-time HM map, temperatures, label-QA
table). Decision: METHODS_NOTES 2026-10-16 "Phase-D freeze". Check, Mac MPS fp32, uncommitted
tree on `c3eb94b`: `uv run python scripts/evaluate_frozen.py peru-dev` (log
`outputs/frozen_check_dev.log`; 33 runs) then `uv run python scripts/score_frozen.py peru-dev
--logits_root outputs/frozen_check/logits_dev --compare` → `outputs/phaseD/test/peru-dev_check*`
(`_compare.csv`: 31/33 runs bit-identical (first written as 28/30, a miscount, corrected 2026-10-16) to `outputs/logits_dev`; AdaBN U-Net seeds 0 and 2
differ, IoU +0.0004 / +0.0069, cause not identified, see METHODS_NOTES). Check logits deleted
afterwards (5.6 GB; regenerable with the same command). Freeze commit: see this entry's commit.

## 2026-10-16 — Disk cleanup (authors' request)

Deleted `outputs/phaseD/upload/` (1.4 GB; slim upload copies of the phase-D source checkpoints,
regenerable with `scripts/cloud/stage_phaseD.py`; the originals are kept) and the arrays in
`outputs/preds_dev/{peru,s1f11,s1f11_ft}/` (float16 probabilities superseded by the float32 logits
on 2026-10-14; the `log_*.txt` files are kept). Consequence: `scripts/float16_diagnostic.py` can no
longer be rerun; its result stays as recorded in METHODS_NOTES 2026-10-14.

## 2026-10-16 — One-time test evaluation, phase-D pass (task 6.2)

`uv run python scripts/evaluate_frozen.py peru-test` then `peru-2017-test`, commit `f1f61d6`
(clean, frozen), Mac MPS fp32; 33 runs each (11 groups × 3 seeds) → `outputs/logits_test/`,
`outputs/logits_2017/` (float32, 172 / 204 files per run, run.json each); logs
`outputs/phaseD/test/eval_{peru-test,peru-2017-test}.log`. Scoring:
`uv run python scripts/score_frozen.py peru-test` / `peru-2017-test` →
`outputs/phaseD/test/<split>{.md,_groups.csv,_runs.csv,_chip_counts.csv}` (rescored once after the
gap-closed fix, same logits); `uv run python scripts/bootstrap_frozen.py` →
`outputs/phaseD/test/bootstrap_{groups,diffs}.csv`. Results: METHODS_NOTES 2026-10-16 "One-time
test evaluation". Commit: see this entry's commit.

## 2026-10-16 — Feature engineering pixel classifier (course progress, week 5; branch `avance-sem5-feature-engineering`)

`uv run python scripts/feature_engineering.py` (Mac CPU, ~8 min; HistGradientBoosting, 2
training sets × 4 feature sets × 3 seeds + 2 MNDWI thresholds; peru-dev only, 1/4 of the pixels;
no test split read) → `outputs/feature_engineering/{results_runs,results,auroc,importance}.csv`,
`summary.md`, `figs/`, log `run.log`. Notebook: `uv run python notebooks/build_feature_engineering_nb.py`
→ `notebooks/feature_engineering.ipynb` (executed). Results: METHODS_NOTES 2026-10-16 "Feature
engineering for a pixel classifier". Commit: see this entry's commit (script first committed in
the previous commit on the branch, unchanged since).

## 2026-10-16 — Test calibration of the frozen phase-D rows

`uv run python scripts/calibration_frozen.py`, commit `f2ad639` (method fixed in METHODS_NOTES
before running). Reads the logits of the one-time pass, the frozen `valid` temperatures (sha256
checked) and `outputs/logits_dev` for the dev-label reference. Output:
`outputs/phaseD/test/calibration{.md,.csv,_runs.csv,_temperatures_dev.json}`, log
`calibration.log`. Results: METHODS_NOTES 2026-10-16 "Test calibration … results".

## 2026-10-16 — Section-7 pass (one-time test evaluation of the deferred frozen rows)

- `uv run python scripts/evaluate_frozen.py peru-test --pass section7` (commit `d42c861`) → 9 runs
  `s1f11_ft` × 3 models, `outputs/logits_test/s1f11_ft/`.
- Then `peru-2017-test --pass section7` (`78844bb`) → `outputs/logits_2017/s1f11_ft/`.
- Then `test` (`78844bb`) → 9 runs `s1f11_src` × 3 models on Sen1Floods11 `test` (90 chips),
  `outputs/logits_s1f11_test/`.
- Mac MPS fp32; all `git_dirty: false`; logs `outputs/phaseD/test/eval_section7_*.log`,
  `eval_s1f11_test.log`.
- Scoring: `scripts/score_frozen.py {peru-test,peru-2017-test,test}`. The earlier rows were
  reproduced exactly.
- Bootstrap: `scripts/bootstrap_frozen.py` (extended to `test` and the `s1f11_ft` diffs; earlier
  CIs reproduced exactly).
- Results: METHODS_NOTES 2026-10-16 "Section-7 pass: results", `docs/published_comparison.md`.

Commit: see this entry's commit.

## 2026-10-16 — EDA notebook (course progress, week 4)

`uv run python notebooks/build_eda_nb.py` → `notebooks/eda.ipynb` (executed). Reads the
manifests, train/dev and Sen1Floods11 train/valid labels, 60 `peru-train` images, and
`outputs/{band_stats,label_qa}`. No test labels are read. Results: METHODS_NOTES 2026-10-16 "EDA
notebook". Commit: see this entry's commit.

## 2026-10-23 — Data rebuilt on a second machine; Peru v1 verified

On a fresh clone without `data/`:
- `scripts/download_sen1floods11.py`: 1,338 files, 1,022 MB. `scripts/verify_sen1floods11.py`:
  train 252 / valid 89 / test 90 chips, all ok.
- UNOSAT shapefiles from HDX (`FL0230321PER_SHP.zip`, `EQ20230321PER_SHP.zip`) into
  `data/unosat2023/raw/`.
- EMSR199 vectors for the 5 AOIs from the Copernicus EMS S3 bucket
  (`cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR199/<product>_vector.zip`;
  the old `emergency.copernicus.eu` paths return 404) into `data/emsr199/raw/<product>/`.
- `scripts/build_peru_unosat.py`: CDSE credentials in `.env`. A 502 from the Process API stopped
  the build at location 411 of 668; it was resumed from `manifest.csv`.
- `scripts/split_peru_unosat.py`.
- `scripts/build_peru_emsr2017.py`: one 502, then a full re-run.

Check: `scripts/freeze_peru.py --check` reports that all 2,990 chip files match their manifests,
and the manifests and split lists match `docs/peru_v1_checksums.txt`, so the rebuild is
bit-identical to Peru v1. No training.

## 2026-10-23 — Baseline-and-variants notebook (course progress, week 6; no new runs)

`uv run python notebooks/build_baseline_nb.py` → `notebooks/baseline_variantes.ipynb` (executed),
figures and tables in `docs/figs/sprint6/`. Reads only `notebooks/results/sprint6_results.csv`
(values transcribed from the entries of 2026-10-13/14 and 2026-10-14 above and from
`docs/phaseD_dev.md` / `docs/phaseD_test.md`); no training, no evaluation, no file under `data/`.
Results: METHODS_NOTES 2026-10-23 "Baseline-and-variants notebook". Commit: see this entry's commit.
