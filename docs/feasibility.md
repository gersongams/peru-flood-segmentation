# Compute feasibility: TerraTorch fine-tuning on Apple MPS

Date: 2026-09-29. Script: `scripts/feasibility_mps.py`. Raw logs are in `outputs/feasibility/*.log`, which git ignores.

## Environment

| | |
|---|---|
| Machine | Apple M4 Max, 36 GB unified memory (`hw.memsize` = 38654705664) |
| OS | macOS 26.6.2 (25G83) |
| Python | 3.12 (uv 0.7.6) |
| torch | 2.14.0, MPS available |
| terratorch | 1.2.13 (PyPI) |
| lightning / torchgeo | 2.6.6 / 0.9.0 |

**Caveat on memory headroom:** I checked the system after the runs, not before them.
- A `com.apple.Virtualization` process (Docker or a VM) was holding about 16 GB.
- Swap was at 21.7 of 22.5 GB used. I read this right after the Prithvi bs=8 run that thrashed, so that run probably caused much of it.
- `torch.mps.recommended_max_memory()` = 28.1 GiB.
- I did not test whether closing the VM would let a larger batch size fit.

## Data

- Source: `https://storage.googleapis.com/sen1floods11/v1.1/...`. I used the public HTTPS endpoint because gsutil is not installed.
- Official split CSVs (`v1.1/splits/flood_handlabeled/*.csv`): **train = 252 chips** (counted as non-empty lines in `flood_train_data.csv`), valid = 89, test = 90, bolivia = 15.
- Subset downloaded: 48 random train chips and 8 random valid chips (seed 0), S2Hand and LabelHand, 156 MB in total, stored in `data/sen1floods11/`. TerraTorch's `Sen1Floods11NonGeo` reads `flood_{train,valid}_data.txt`. Those `.txt` files are **my 48/8 subset lists**, not the official splits. I checked that the dataset loads exactly 48 and 8 chips.
- Chips are 512×512 at full size (no crop). TerraMind gets 13 bands (S2 L1C). Prithvi gets 6 HLS bands.

## Setup measured

- The model is built with `SemanticSegmentationTask(model_factory="EncoderDecoderFactory")` and trained with dice loss, `ignore_index=-1`, AdamW at lr 2e-5, and the full backbone unfrozen.
- **TerraMind v1 Base** (`terramind_v1_base`, modality `S2L1C`) follows the necks and decoder in IBM/terramind `configs/terramind_v1_base_sen1floods11.yaml`:
  - necks: SelectIndices [2,5,8,11], ReshapeTokensToImage, LearnedInterpolateToPyramidal
  - decoder: `UNetDecoder` [512,256,128,64]
  - 103.0M parameters
- **Prithvi-EO-2.0 300M** (`prithvi_eo_v2_300`) follows terrastackai/terratorch `examples/segmentation/segmentation_sen1floods11.yaml`:
  - necks: SelectIndices [5,11,17,23], ReshapeTokensToImage, LearnedInterpolateToPyramidal
  - decoder: `UNetDecoder` [512,256,128,64], `head_channel_list` [256]
  - 324.4M parameters
  - That config's `aux_loss: {aux_head: 1.0}` is omitted because it defines no aux decoder.
- **Prithvi, UperNet variant** (`prithvi_eo_v2_300_upernet`) is my own lighter setup, **not from a published config**:
  - necks: SelectIndices [5,11,17,23], ReshapeTokensToImage (no pyramid neck)
  - decoder: `UperNetDecoder` with 256 channels, running entirely at 1/16 resolution
  - 314.6M parameters
- Timing:
  - Chips are preloaded into RAM, so disk I/O is excluded (loading 48 chips took 1.1 s).
  - `torch.mps.synchronize()` runs around each step, and the first steps are discarded as warmup.
  - Each step is forward + backward + optimizer step. Validation is not included.
- Memory is the maximum of `torch.mps.driver_allocated_memory()` across steps.

## Results (device = mps, 512×512)

| Model | Precision | Batch | Timed steps | s/iter (median) | img/s | Peak MPS driver mem | Steps/epoch (252, drop_last) | Extrapolated train epoch |
|---|---|---|---|---|---|---|---|---|
| TerraMind v1 Base | fp32 | 8 | 50 | **1.237** (1.226–1.261) | 6.47 | **16.65 GiB** | 31 | **0.64 min (~38 s)** |
| TerraMind v1 Base | bf16 autocast | 8 | 30 | 1.198 (1.195–1.205) | 6.68 | 16.59 GiB | 31 | 0.62 min |
| Prithvi-EO-2.0 300M (official config) | fp32 | 8 | – | aborted after step 0 | – | 30.8 GiB, over the 28.1 GiB limit | – | does not fit |
| Prithvi-EO-2.0 300M (official config) | fp32 | 4 | 30 | **1.528** (1.489–1.578) | 2.62 | **20.24 GiB** | 63 | **1.60 min (~96 s)** |
| Prithvi 300M UperNet variant | fp32 | 8 | – | 29 s, 125 s, 239 s (steps 0–2, then killed) | – | 32.81 GiB, swapping | – | not practical |
| Prithvi 300M UperNet variant | fp32 | 4 | 20 | 1.413 (1.396–1.489) | 2.83 | 20.13 GiB | 63 | 1.48 min |

- TerraMind loss fell from 0.648 to 0.400 over 55 fp32 steps. Prithvi (official config) loss fell from 0.586 to 0.317 over 33 steps at batch size 4. These are single noisy values from random batches. They only confirm that gradients flow. They are not evidence of accuracy.
- The extrapolated time counts training steps only. It excludes validation and data loading.
- bf16 autocast barely helps on MPS (about 3% faster, same memory).
- Prithvi at batch size 8 does not raise an OOM error on MPS. It overflows into swap and slows down by more than 100×. The script therefore checks `torch.mps.driver_allocated_memory()` after each step. If it exceeds `torch.mps.recommended_max_memory()`, the script treats that as OOM and drops to the next batch size. That is how the official-config bs=8 run was aborted.

## Errors and workarounds

1. **Prithvi + UperNetDecoder crashes on MPS.** This affects only the UperNet variant. The official config uses UNetDecoder and is not affected. The crash happens with its default `pool_scales=(1,2,3,6)`:
   `RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes` (pytorch/pytorch#96056).
   - Cause: the feature map is 32×32 (512 px / patch 16), and 32 is not divisible by 3 or 6.
   - `PYTORCH_ENABLE_MPS_FALLBACK=1` does **not** help. The op is implemented but raises this RuntimeError, so the CPU fallback never kicks in.
   - Workaround in the script (UperNet variant): `decoder_pool_scales=(1,2,4,8)`. Another option is an input size where the patch grid is divisible by 6, for example 480 px (30×30).
2. TerraMind ran with **no fallback ops**. Its only notice was a TerraTorch warning in `LearnedInterpolateToPyramidal`: "MPS backend: enforcing .contiguous()".
3. The official TerraMind and Prithvi configs use `16-mixed`. The fp32 numbers are the main result here. The bf16 run was for TerraMind only.

## Model weights (Hugging Face model cards, `?blobs=true` API)

| Model | HF repo | File | Size | License |
|---|---|---|---|---|
| TerraMind 1.0 Base | ibm-esa-geospatial/TerraMind-1.0-base | TerraMind_v1_base.pt | 1518.7 MB | apache-2.0 |
| Prithvi-EO-2.0 300M | ibm-nasa-geospatial/Prithvi-EO-2.0-300M | Prithvi_EO_V2_300M.pt | 1326.7 MB | apache-2.0 |

## Notes for real training

- `Sen1Floods11NonGeoDataModule` scales inputs by 1e-4 and normalizes with its own Sen1Floods11 per-band statistics. The official TerraMind config uses TerraMind's pretraining means and stds on raw DN values. This does not affect speed, but it should be fixed before any accuracy experiments.
- Rough estimate from the fp32 numbers above:
  - TerraMind Base: 100 epochs ≈ 100 × 38 s ≈ 64 min of training steps.
  - Prithvi-300M: 100 epochs ≈ 100 × 96 s ≈ 2.7 h. The official config uses batch size 16; only batch size 4 fits here.
  - Both figures exclude validation.
