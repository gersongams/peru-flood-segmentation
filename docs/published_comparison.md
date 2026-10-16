# Sen1Floods11: our results vs published numbers (plan 3.8 / phase-D 1.5)

2026-10-16. The `valid` part was written first, before our `test` scores existed. The
test-vs-test comparison was added after the frozen section-7 pass (next section). Published
numbers were read from the PDFs; ours come from logged metrics or from the frozen pass.

## Test vs test (frozen section-7 pass, 2026-10-16): closes plan 3.8 / phase-D 1.5

Our runs: the three `s1f11_src` models (3 seeds each), frozen in `configs/frozen.yaml` before any
test file was read. They were evaluated once on the official Sen1Floods11 `test` split (90
chips; `scripts/evaluate_frozen.py test`, `scripts/score_frozen.py test`). Metrics are pooled over
pixels: water IoU, and two-class mIoU = mean of water and not-water IoU. Prediction is water if
z > 0. The 95 % CI is a chip bootstrap of the seed mean (`scripts/bootstrap_frozen.py`). Sources:
`outputs/phaseD/test/test_groups.csv`, `bootstrap_groups.csv`.

| Model | Ours `test` water IoU (95 % CI) | Ours `test` mIoU | Published water IoU / mIoU | Source | Gap water IoU / mIoU |
|---|---|---|---|---|---|
| Prithvi-EO-2.0 300M (6 HLS bands) | 82.03 ± 1.15 (75.1–87.3) | 89.66 ± 0.66 | 82.2 (std 0.6) / 89.7 (0.3) | Prithvi-EO-2.0, Table IV, p. 12 | **−0.17 / −0.04** |
| TerraMind v1-B (S2 L1C, 13 bands) | 81.85 ± 1.36 (75.4–86.9) | 89.54 ± 0.79 | 82.26 / 89.70 | TerraMind, Table 8, p. 8 (split and S2 level not stated) | **−0.41 / −0.16** |
| U-Net, ResNet-34 (13 bands) | 83.31 ± 0.49 (76.0–88.9) | 90.42 ± 0.28 | 84.03 / 90.80 ([19], quoted in Prithvi-CAFE) | Prithvi-CAFE, Table 1, p. 6 (setup unverified) | −0.72 / −0.38 |
| | | | 80.69 / 88.84 ([16], quoted in Prithvi-CAFE) | same | +2.62 / +1.58 |

**Verdict.**
- Both foundation models reproduce their papers' Sen1Floods11 `test` numbers within 0.5 IoU,
  inside the published seed std (Prithvi 0.6) and our own seed std (1.2–1.4).
- Our U-Net lies between the two secondary U-Net quotes. The only gap above 2 IoU is against
  [16], whose setup is unknown and which differs from [19] by 3.3 points. It is not a sign of a
  defect.
- As in Prithvi-CAFE Table 1, U-Net is the best model on `test`. Its lead is not separable
  (U-Net − Prithvi +0.013, 95 % CI −0.022 to 0.062; U-Net − TerraMind +0.015, CI −0.019 to 0.051;
  `bootstrap_diffs.csv`).
- `valid` → `test`, ours: U-Net 83.15 → 83.31; TerraMind 83.49 → 81.85; Prithvi 83.15 → 82.03.
  The selection optimism of the foundation models on `valid` is 1.1–1.6 points.
- Plan 3.8 and phase-D 1.5 are closed.

# Part 1: `valid` vs published `test` (written before the test pass)

Read-only comparison: no model was run and no test data was used for this part.
Every published number below was read from the PDF in `sources/papers/` (page = PDF page).
Ours were read from each run's `metrics.csv` (row of the best `val/IoU_Flood` epoch, which is
the checkpoint used), mean ± sample std (ddof = 1) over seeds 0–2.

**Main mismatch: our numbers are Sen1Floods11 `valid` (89 chips). Every published number
below is Sen1Floods11 `test` (90 chips), unless marked otherwise.** Our official `test` split has
not been scored yet (frozen pass, later). Our `valid` numbers are also optimistic: LR, early
stopping and the checkpoint were all chosen on `valid` (max over epochs). Valid and test also
contain different chips. So this table checks whether we are in the right range. It does not
replicate any published result.

## Our runs (source-only Sen1Floods11 models, `s1f11_src`)

RunPod RTX 4090, `16-mixed`, official splits (train 252 / valid 89). Early stopping on
`val/IoU_Flood` (patience 20). Dice loss, UNetDecoder for both foundation models. Sources:
`docs/EXPERIMENT_LOG.md` (2026-10-14/15 entry and the pod-session-2 entry) and `docs/METHODS_NOTES.md`
(2026-10-15).

| Model (ours) | Inputs | Run dir | Water IoU | mIoU (2 classes) | Pixel acc. |
|---|---|---|---|---|---|
| U-Net, ResNet-34 (ImageNet), LR 3e-4 | S2 L1C, 13 bands | `outputs/s1f11_src_unet` | 83.15 ± 0.15 | 90.47 ± 0.09 | 98.00 ± 0.02 |
| TerraMind v1-B, LR 5e-5 | S2 L1C, 13 bands (no S1) | `outputs/s1f11_src_terramind_lr5e-5` | 83.49 ± 0.14 | 90.64 ± 0.08 | 98.00 ± 0.02 |
| Prithvi-EO-2.0 300M (non-TL), LR 1e-4 | 6 HLS bands of S2 L1C | `outputs/s1f11_src_prithvi` | 83.15 ± 0.47 | 90.45 ± 0.26 | 97.97 ± 0.04 |

Per-seed water IoU: U-Net 83.30 / 83.00 / 83.15; TerraMind 83.33 / 83.59 / 83.55;
Prithvi 83.30 / 82.63 / 83.54. These match EXPERIMENT_LOG. Means are taken from the
full-precision `metrics.csv` values. Averaging the 4-decimal values in EXPERIMENT_LOG gives
Prithvi 83.16 instead of 83.15 (no Prithvi mean is written in METHODS_NOTES yet). Use one figure in
the thesis. Per-seed mIoU: U-Net 90.55 / 90.38 / 90.47;
TerraMind 90.54 / 90.69 / 90.67; Prithvi 90.52 / 90.17 / 90.67.

**How our metrics are defined.** TerraTorch 1.2.13 (`terratorch/tasks/segmentation_tasks.py`)
uses torchmetrics `MulticlassJaccardIndex` with `ignore_index = -1`. The confusion matrix is
accumulated over the whole split, so every metric is **pooled over pixels** (not averaged per
chip):
- `val/IoU_Flood` = IoU of the water class (`average=None`, water entry). Our water IoU.
- `val/mIoU` = macro mean of IoU over the two classes (water and not-water).
- `val/F1_Score` = **macro** F1. `val/Pixel_Accuracy` = micro accuracy, which equals micro-F1
  for single-label multiclass.
- The pooled water IoU recomputed from the saved valid logits (`scripts/phaseD_valid_logits.py`)
  equals the logged best for all 9 runs (EXPERIMENT_LOG 2026-10-15 and 2026-10-16 entries).
  `src/floodseg/metrics.py` uses the same pooled definition.

## Published numbers (all Sen1Floods11 hand-labelled data)

Gap = ours (`valid`) − published (test), in IoU points. "Matched" means same model and similar
inputs. Training recipes still differ (see Comparability).

| Model (published) | Published water IoU | Published mIoU | Split | Inputs | Source | Our model | Gap water IoU | Gap mIoU | Matched? |
|---|---|---|---|---|---|---|---|---|---|
| Prithvi-EO-2.0-300M | 82.2 (std 0.6) | 89.7 (std 0.3) | test | S2, the 6 pretraining (HLS) bands | Szwarcman et al. 2024, **Table IV, p. 12**; setup p. 6 | Prithvi 300M | **+0.95** | **+0.75** | yes (best match) |
| Prithvi-EO-2.0-300M-TL | 82.6 (0.3) | 90.0 (0.2) | test | S2, 6 bands + time/location | same Table IV, p. 12 | — (we use non-TL) | — | — | no |
| Prithvi-EO-2.0-300M-TL | 80.97 | 89.06 | test (best HPO run) | S2 L1C, 13 channels (new patch-embedding weights for unseen bands) | Gomes et al. (TerraTorch) 2025, **Table II, p. 4** | Prithvi 300M | +2.18 | +1.39 | no (TL variant, 13 bands, UPerNet, one run) |
| TerraMind v1-B, full fine-tuning | 82.26 | 89.70 | **not stated** | "S-2" (L1C or L2A not stated) | Jakubik et al. 2025, **Table 8, p. 8** | TerraMind v1-B | **+1.23** | **+0.94** | yes (closest) |
| TerraMind v1-B (PANGAEA protocol, frozen encoder, UperNet) | — | 89.57 | PANGAEA test (not checked in the PANGAEA paper) | S-2 only | Jakubik et al. 2025, **Table 7, p. 8**; protocol p. 18 | TerraMind v1-B | — | +1.07 | partly (we fine-tune the full encoder) |
| TerraMind v1-B (PANGAEA) | — | 90.62 | as above | S-1 + S-2 | Jakubik et al. 2025, **Table 6 / Table 7, p. 8** | TerraMind v1-B | — | +0.02 | no (adds S1) |
| TerraMind v1-B-**single**, fine-tuned | 76.32 | — | test | S-2 L1C | Jakubik et al. 2025, **Table 12, p. 18** and Table 2 (right), p. 6 | — | (+7.17) | — | **no: different model (v1-B-single)** |
| TerraMind v1-B (row with no citation) | 82.90 | 90.60 | test | not stated | Kaushik et al. (Prithvi-CAFE) 2026, **Table 1, p. 6** | TerraMind v1-B | +0.59 | +0.04 | **source unverified** (see below) |
| Prithvi 2.0 300M (quoted from [28] = Prithvi-EO-2.0) | 82.20 | 89.70 | test | S2 | Prithvi-CAFE **Table 1, p. 6** | Prithvi 300M | +0.95 | +0.75 | yes (same as Table IV) |
| U-Net Base, quoted from [19] (Li et al. 2023) | 84.03 | 90.80 | test | **unverified** (from [19]) | Prithvi-CAFE **Table 1, p. 6** | U-Net | −0.88 | −0.33 | setup unverified |
| U-Net, quoted from [16] (U-Prithvi, Kostejn et al.) | 80.69 | 88.84 | test | **unverified** (from [16]) | Prithvi-CAFE **Table 1, p. 6** | U-Net | +2.46 | +1.63 | setup unverified |
| UNet Baseline, trained from scratch (PANGAEA) | — | 91.42 | PANGAEA test | PANGAEA Sen1Floods11 inputs (not checked) | Jakubik et al. 2025, **Table 6, p. 8** | U-Net | — | −0.95 | partly |
| Prithvi-CAFE (method paper's own model) | 83.41 | 90.50 | test | S2: 6 HLS bands to Prithvi + "remaining channels" to a CNN; band list not stated; SAR named as future work (p. 8) | Prithvi-CAFE **Table 1, p. 6** | — | — | — | context |
| FCNN (ResNet-50), trained on hand labels | 0.3125 ("all water", AW) | — | test (10 events, no Bolivia) | apparently S1 (2-channel normalization stats, p. 5; **our inference**) | Bonafilia et al. 2020, **Table 1, p. 6**; metric p. 5 | — | — | — | **not comparable** (per-chip mean IoU, S1) |

Metric definitions as the papers state them:
- **Prithvi-EO-2.0** (p. 5, Eq. 3; p. 6): mIoU = mean over classes of IoU. For Sen1Floods11 they
  report "the mIoU and the IoU for the water class" and a **micro-averaged** F1. Setup (p. 6):
  UPerNet decoder, cross-entropy, images resized 512 → 448, flips, 50 epochs, patience 20.
  LR and weight decay tuned with TerraTorch-iterate (10 trials), then 10 seeds. The 6 pretraining
  bands are used. Pooled vs per-chip is not stated. They used TerraTorch, which pools (implied,
  not stated).
- **TerraTorch paper** (p. 4): "IoU for the classes water ... as well as the mean IoU". Table II
  reports the best run selected on validation loss. UPerNet, 512 input, 100 epochs. Fig. 2 plots
  `valid` mIoU only as a scatter with no table, so we do not read numbers from it.
- **TerraMind**: Table 2 caption (p. 6): "IoU on water class". Table 8 (p. 8) gives IoU_Water and
  mIoU. The PANGAEA tables (6, 7) report mIoU. Under the PANGAEA protocol "all GFMs including
  TerraMind are finetuned with a frozen encoder and an UperNet head" (p. 18). The split for
  Table 8 is not stated.
- **Prithvi-CAFE** (p. 5): IoU for the foreground (flood) class, mIoU, mean Dice ("m-F1").
  Exact official train/valid/test split (p. 5). Results on the test set (p. 6). Pooled vs per-chip
  is not stated.
- **Bonafilia et al.** (p. 5): "we report the mean IOU (equal weighting of all chips)". That is a
  **per-chip average**, unlike our pooled IoU.

## Comparability

1. **Split.** Ours are on `valid`, theirs on `test`, except the TerraMind Table 8 row, whose split
   is not stated. Ours are also selection scores (max over epochs on the split that chose the
   checkpoint), so they are biased upwards. The like-for-like comparison is our frozen `test` pass.
   Plan 3.8 can only be closed after it.
2. **Metric.** Water IoU and two-class mIoU are the same definitions in our runs and in
   Prithvi-EO-2.0 / TerraTorch / TerraMind / Prithvi-CAFE (IoU of the water class; mean over the
   two classes). **No conversion is needed: `val/mIoU` is already logged** in each run's
   `metrics.csv`, and the table shows both. Pooling over pixels is certain for ours. For the
   papers that used TerraTorch it is implied, not stated. Bonafilia averages per chip, so it is not
   comparable. A per-chip valid IoU could be computed without running any model from the saved
   valid logits in `outputs/logits_valid/s1f11_src/` (valid only). It was not computed here.
3. **F1 is not comparable.** Our `F1_Score` is macro F1 (U-Net 94.84, TerraMind 94.94, Prithvi
   94.83). Prithvi-EO-2.0's mF1 is micro (= pixel accuracy). Our pixel accuracy (≈ 98.0) sits
   next to their 97.6 (300M). Prithvi-CAFE's m-F1 column mixes the two kinds: 97.60 for Prithvi
   300M (copied from Table IV) and about 94–95 for the others. Leave F1 out of the comparison.
4. **Training-recipe differences for the matched pairs.**
   - Prithvi 300M vs Table IV: our decoder is UNetDecoder (theirs UPerNet), our loss dice (theirs
     cross-entropy). We use no 448 resize. Our stopping rule is `val/IoU_Flood`, patience 20, up
     to 200 epochs (theirs 50 epochs). We ran 3 seeds after a 3-point LR sweep (theirs 10 seeds
     after 10 HPO trials). Same 6 bands.
   - TerraMind v1-B vs Table 8: our config follows IBM/terramind's Sen1Floods11 config (UNet
     decoder, dice) but uses S2 L1C only. Table 8 says "S-2" without the processing level, the
     decoder or the split.
   - U-Net: the two published U-Net rows are secondary quotes in Prithvi-CAFE. Their inputs and
     training ([19], [16]) were not read. Our U-Net is ResNet-34/ImageNet on 13 bands.
5. **Run-to-run noise.** A same-settings rerun of U-Net seed 0 moved valid IoU by 1.7 points
   (METHODS_NOTES 2026-10-15). That is close to the 2-point threshold of task 3.8.

## Problems found in the published tables (and in our docs)

- **Prithvi-CAFE Table 1 does not match its own source for Prithvi 600M.** It lists 82.50 IoU /
  90.30 mIoU. Prithvi-EO-2.0 Table IV (p. 12) gives 600M: 82.5 IoU / **89.9** mIoU, and
  600M-TL: 83.1 / 90.3. So the 90.30 is the TL mIoU paired with the non-TL IoU. Cite Table IV
  directly.
- **Prithvi-CAFE's TerraMind v1-B row (82.90 / 90.60 / m-F1 95.01) has no citation.** None of
  82.9, 90.6 or 95.01 appears in the TerraMind paper. The nearest TerraMind number is 90.62 mIoU
  (PANGAEA, S1+S2, frozen encoder, Table 6), which comes with no water IoU. Status:
  **unverified**. Do not quote it as TerraMind's result.
- **The phase-A note in `docs/REFERENCES.md` uses the wrong TerraMind target.** Its table labels
  76.32 correctly as "TerraMind v1-B-single, fine-tuned", but the phase-A note calls it "TerraMind v1-B S2 L1C ≈ 76.3 (TerraMind's
  own number)" as a reproduction target. 76.32 belongs to **v1-B-single**, a different model
  (Table 12 p. 18; Table 2 right p. 6). The setup-matched TerraMind v1-B target is **Table 8:
  82.26 / 89.70**. REFERENCES also lists Prithvi-CAFE as "S2 (6 HLS bands)". Its CNN branch takes
  further channels, so it uses more than 6.
- TerraMind Table 4 (p. 7) lists "Prithvi 2.0 300M, S-2, finetune 80.97" and "DeCUR 82.17" and
  cites no source for them. They are identical to TerraTorch Table II (300M-**TL**, 13 S2L1C
  channels). Its "Baseline [10], S-2, 31.25" matches Bonafilia's hand-labelled FCNN (.3125 AW,
  Table 1), which appears to use S1 input (our inference). Do not use Table 4 as a reference.

## Verdict (task 3.8 rule: investigate any gap > 2 IoU)

- **Matched pairs: no gap above 2 IoU.** Prithvi 300M: +0.95 water IoU / +0.75 mIoU vs Table IV.
  TerraMind v1-B: +1.23 / +0.94 vs Table 8. Ours are higher. That fits `valid`-selection
  optimism and a different split. It is inside the 1.7-point rerun spread. Prithvi's gap is also
  under 2 published std (0.6).
- **U-Net:** within 1 point of the 84.03 / 90.80 row ([19]) and 2.46 above the 80.69 row ([16]).
  The two published U-Nets differ from each other by 3.3 points. Our U-Net lies between them. The
  2.46 gap is against a secondary quote with an unknown setup. Read [16] (U-Prithvi) and [19]
  (Li et al. 2023) before claiming a match with either. It is not a sign of a problem in our runs.
- **Every gap above 2 comes from a mismatched setup**: TerraMind v1-B-single (76.32), the TerraTorch
  300M-TL with 13 bands (80.97, +2.18), Bonafilia (S1, per-chip). None needs investigating as a
  defect.
- **Status:** our results are consistent with the literature on `valid`. The test-vs-test
  comparison is at the top of this file; it closes plan 3.8 / phase-D 1.5.

## Still unverified / to do

- TerraMind Table 8: split and S-2 level (L1C vs L2A) are not stated in the caption or the text
  (p. 8). Check the TerraMind supplementary material or the IBM/terramind repo before using it as
  a test-split target.
- Prithvi-CAFE TerraMind row (82.90 / 90.60): no source.
- U-Net rows [19] Li et al. 2023 and [16] Kostejn et al.: inputs and training not read. PDFs are
  not in `sources/papers/`.
- PANGAEA paper (Marsocci et al.) not in `sources/papers/`. Its Sen1Floods11 inputs and splits
  were not checked.
- METHODS_NOTES (2026-10-16 "Published-number comparison") and the REFERENCES.md corrections
  are done.
