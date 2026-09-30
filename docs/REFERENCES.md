# References: what to cite, and where it lives

The previous project, `~/Projects/masters-degree-thesis`, is **reference only**. Its code (`app/`) is not reused, but its literature work is:

| Resource | Path (in the old repo) | Use |
|---|---|---|
| Literature index | `docs/papers/INDEX.md` | **Read first.** Comparison table + cross-cutting findings for 23 papers |
| Per-paper notes | `docs/papers/<bibtex_key>.md` | About 3k words each: method, data, results with page cites, limitations |
| Note-writing rules | `docs/papers/_INSTRUCTIONS.md` | Never invent a number; cite the table, not the abstract; flag ⚠️ contradictions |
| Paper → PDF map | `docs/papers/_manifest.json`, `scripts/zotero_index.sh tesis` | Resolve a key to its Zotero PDF (read-only, `~/Zotero/`) |
| Old BibTeX | `overleaf-thesis/3_3_BIBLIOGRAFIA/library.bib` | 54 entries (includes `-1` duplicates). **Copy entries one by one** into `thesis/3_3_BIBLIOGRAFIA/library.bib` as they get cited. Don't copy the whole file. |

Rules carried over:
- Open Zotero only read-only (`immutable=1`). Never write to `~/Zotero/`.
- Recall order: INDEX.md → note → PDF only if the note is insufficient.
- Metrics go into the thesis only with a page, table or URL source.

## Papers from the old library worth citing, by thesis role

### Core: cite these

| Key | Why it matters here | Where |
|---|---|---|
| `konapala_exploring_2021` | S1 vs S2 on **Sen1Floods11** with U-Net: S2 F1 0.90 vs S1 0.62. The same benchmark we use, so a direct comparison point | Ch II SOTA, Ch VI discussion |
| `garshasbi_uncertainty-aware_2025` | Bayesian U-Net with uncertainty on Sen1Floods11 (F1 0.8013, IoU 0.6685). Prior work for RQ3 | Ch II, Ch VI |
| `shastry_mapping_2023` | Optical CNN underpredicted real flood extent by 62%; vegetation caused 74% of the misses. The key limitation of S2 | Ch I problem, limitations, motivates RQ2 |
| `fawakherji_deepflood_2025` | S2-only 43–51 mIoU vs +10–15 with SAR. Evidence for the S1 ablation | Ch II, RQ2 |
| `du_high-precision_2026` | Bolivia was its worst-performing region, which sets expectations for OOD / Peru | Ch VI |
| `li_terrainfloodsense_2025` | Flood mapping through cloud with DEM/HAND/JRC priors | Ch II, future work |
| `tran_surface_2022` | Otsu/NDWI threshold baseline; NDWI overestimates water in paddy fields | Ch II (classical methods) |

### Supporting: architectures and related methods

`saleh_pdca-former_2023`, `saleh_high-precision_2024` (Transformer change detection, S1) · `shu_satvit-seg_2026` (ViT segmentation) · `huang_waterdetectionnet_2024`, `xu_sar_2022` (⚠️ reports no metrics, cite only for architecture) · `chen_flood_2025` (index as a separate encoder branch) · `wu_near-real-time_2023` · `ulloa_sentinel-1_2022` (ConvLSTM time series) · `pech-may_flood_2025` (uses Gao's NDWI, not McFeeters) · `kapoor_qdeepgr4j_2026` (uncertainty recipe).

### Peripheral: framing only

`riche_novel_2024` (susceptibility, ⚠️ inconsistent AUC) · `herath_subgrid_2025` · `zhu_improving_2025` · `liu_deep_2022` · `yin_estimating_2023` · `gau_using_2024` (⚠️ abstract mislabels precision as accuracy).

### In the old .bib but with no note yet (check before citing)

`bentivoglio_deep_2022` (DL flood review, cited in the old Ch I), `ccopi-trucios_river_2023` (possibly Peru-relevant; check), `pech-may_flooded_2023`, `shaheen_swinsegformer_2025`, `danish_graph_2025`, `ding_flood_2025`, `jamali_residual_2024`, `lokula_umv2_2025`, `noori_deep-learning_2025`, `li_cmpf-unet_2024`, `farhadi_flood_2021`, `hajji_enhancing_2025`, `obada_using_2025`, and others. Read the PDF via Zotero and write a note before citing any numbers.

## New references to add (not in Zotero yet)

These were found in the September 2026 SOTA search. **Add each one to Zotero, write a note in the same format, and create a real BibTeX entry.** Don't hand-type BibTeX from memory.

| Reference | Why | Verification status |
|---|---|---|
| Bonafilia et al., *Sen1Floods11*, CVPR Workshops 2020, github.com/cloudtostreet/Sen1Floods11 | Our benchmark | Split counts verified from the CSVs (train 252 / valid 89 / test 90 / bolivia 15) |
| Kaushik, Maurya & Tellman, *Prithvi-CAFE*, arXiv 2601.02315 (2026) | Latest SOTA table; U-Net ≈ FMs in-distribution; +10.8 IoU on Bolivia | **Table 1 checked against the PDF** (Prithvi 2.0 600M 82.50 IoU / 90.30 mIoU; TerraMind v1-B 82.90 / 90.60) |
| Prithvi-EO-2.0, arXiv 2412.02732 (2024) | Foundation model #2 | From web search; verify numbers in the PDF |
| TerraMind, arXiv 2504.11171 (ICCV 2025) | Foundation model #1 (main) | From web search; verify numbers in the PDF |
| TerraTorch, arXiv 2503.20563 (2025) | Fine-tuning toolkit (cite as software) | Used; v1.2.13 |
| Portalés-Julià et al., *WorldFloods v2 / ml4floods*, Sci. Rep. 13:20316 (2023) | Optional extra data | From web search |
| Portalés-Julià et al., Remote Sensing of Environment (2025), cross-dataset generalization of S1/S2 flood models | Motivates the OOD focus | From web search; verify |
| Badillo-Rivera et al., Frontiers in Environmental Science (2025), doi 10.3389/fenvs.2025.1672107 | Piura / Niño Costero 2017 susceptibility with S1 inventory, RF/SVM, 182 samples, random split | Study facts checked on the page. ⚠️ Two reads disagreed on the best validation AUC (0.932 vs >0.907), so **re-read the paper before quoting it** |
| Tellman et al., *Global Flood Database*, Nature (2021) | Context: global flood labels | From web search |
| Nearing et al., Google Flood Hub, Nature 627 (2024) | Context: flood forecasting (out of scope) | From web search |
| Kuro Siwo, NeurIPS Datasets & Benchmarks (2024) | Related S1 benchmark | From web search |
