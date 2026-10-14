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
