# Literature: domain adaptation and generalization for flood and water segmentation in Earth observation

**Date and purpose.** Searched 2026-10-14. We want prior work for adapting U-Net, TerraMind v1-B and Prithvi-EO-2.0 models, trained on Sen1Floods11, to the Peru desert coast using only unlabelled Peru Sentinel-2 (S2) imagery.

**Verification levels used below:**
- **FULL TEXT**: I read the numbers in the paper's own text or table (arXiv HTML/PDF, CVF/Nature pages, or ScienceDirect/Wiley full text opened in a browser).
- **ABSTRACT**: I read only the abstract or metadata (arXiv API, OpenAlex, Crossref, publisher page). Any numbers quoted come from the abstract.
- **UNVERIFIED**: known only from a search snippet or from another paper's citation.

Every entry gives the URL I actually opened. "Code: not checked" means I did not look.

---

## 1. Water and flood mapping: generalization and adaptation across datasets, regions and events

### 1.0 Chen, Wang, Zhao & Li (2026), CIM-TTDA, JGR: Machine Learning and Computation (FULL TEXT)

The closest prior work to our plan.

- **Citation:** W. Chen, C. Wang, Z. Zhao, X. Li. "Improving Cross-Event Generalization for SAR-Based Coastal Inundation Mapping via Test-Time Domain Adaptation." *JGR: Machine Learning and Computation* (2026). doi:10.1029/2025JH001145. Open access.
- **URL opened:** https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH001145 (full text via browser). Metadata from OpenAlex.
- **What is adapted to what:**
  - Source: S1S2-Water, permanent water only. Inputs are S1 VV+VH plus slope.
  - Target: three cyclone-driven coastal flood events, eight scenes. Matmo (Vietnam 2019), Ava (Madagascar 2018), Florence (USA 2018).
  - Labels are their own "CI-data set", built from CEMS Rapid Mapping vectors with manual correction.
  - **Sentinel-1 SAR only. No optical input.**
- **Method:** per-event test-time adaptation using only unlabelled post-event SAR.
  - Teacher–student self-training with an EMA teacher (decay 0.9).
  - Pseudo-labels refreshed every epoch, over 4 epochs.
  - Confidence-weighted hard-sample mining and a class-balanced loss.
  - Stochastic restoration to source weights, as in CoTTA.
  - Batch-norm statistics frozen.
- **Hyperparameters:** fixed across events, chosen to follow "established TTDA practices" (§5.1.2). Source models were trained with 4 seeds, and the "best-performing" one was used for the comparison table (§5.3, Text S5). From the main text I could not tell which data that choice was made on.
- **Key results (Table 4, IoU on inundation; IoUp = pixel-weighted, IoUe = event mean):**
  - Source-only U-Net-EfficientNet-B0: IoUp 0.636 / IoUe 0.690. After CIM-TTDA: **0.803 / 0.806**.
  - SegFormer-B1: 0.637 / 0.666 → 0.738 / 0.731.
  - Baselines on a DeepLabV2 source (0.548 / 0.608): CBST 0.713 / 0.698, AdaptSegNet 0.563 / 0.618, TENT 0.543 / 0.604, CSFDA 0.648 / 0.660, UR 0.516 / 0.585. Edge-Otsu threshold 0.547 / 0.578.
  - **Prithvi-EO-2.0-300M-TL** (frozen backbone + FPN, fine-tuned on S1S2-Water) scored the worst of all, **0.478 / 0.474**. Its input is presumably S1, like all the other baselines, but the main text does not say so. It was used only as a baseline and never adapted.
  - Table 5: adapting per event beats adapting across events.
- **Two cautions when reading Table 4:**
  - **CBST and AdaptSegNet ran in a different setting from CIM-TTDA (§5.3).** They were trained offline on a separate, pre-collected unlabelled "Inundation data set" (Text S3) and then evaluated on the CI events. That is offline unsupervised adaptation on a target-side pool, which is **exactly our setting** (unlabelled Peru S2). Per-event test-time adaptation is a different setting.
  - **The "beats SOTA" claim is confounded by the backbone.** CIM-TTDA adapts an EfficientNet source (0.636 → 0.803, +0.167). CBST adapts a DeepLabV2 source (0.548 → 0.713, +0.165). Measured from each method's own source, the gains are about equal.
- **Limitations stated by the authors (§7.2):**
  - SAR only.
  - The source covers permanent water only.
  - Only three events.
  - Larger transformer models are left for future work.
- **Code:** OpenAlex lists a Zenodo record "Core Implementation of CIM-TTDA" (doi:10.5281/zenodo.19403804; metadata only, not opened).
- **What it means for us:** test-time adaptation by self-training for flood mapping now exists, but for SAR, with small CNN/SegFormer models and coastal humid events. It gives us a ready baseline set: CBST, AdaptSegNet, TENT, CSFDA, plus source-only and a threshold method.

### 1.1 Portalés-Julià, Mateo-García & Gómez-Chova (2025), Remote Sensing of Environment (FULL TEXT)

- **Citation:** "Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets." *Remote Sensing of Environment* 328, 114882 (Oct 2025). doi:10.1016/j.rse.2025.114882. Open access, CC-BY-NC. SSRN preprint doi:10.2139/ssrn.5118486.
- **URL opened:** https://www.sciencedirect.com/science/article/pii/S003442572500286X (full text via browser).
- **What is trained and tested on what:**
  - Trained on WorldFloods (S2), Kuro Siwo (S1) and S1S2Water (S1+S2, permanent water only), singly and combined.
  - Held-out test sets: UNOSAT (Nemni et al. 2020; S2; 19 events in East-Central Africa and the Middle East) and hand-labelled Sen1Floods11. They never train on Sen1Floods11, because of spatial autocorrelation between its splits (§2.1).
- **Method:** a cross-dataset benchmark plus training on several datasets at once. **No domain adaptation.**
  - Models: U-Net / U-Net++ (MobileNetV2 encoder), and a dual-stream multimodal U-Net with modality dropout.
  - Baselines: MNDWI threshold at 0 (S2) and Edge-Otsu (S1).
- **Key results (Table 3, all-water IoU):**
  - UNOSAT: multimodal U-Net trained on all datasets 83.55; WorldFloods-trained S2 U-Net 83.42; **MNDWI 82.47**.
  - Sen1Floods11: multimodal U-Net 77.50; S1S2Water-trained S2 U-Net 77.32; **MNDWI 74.39**; WorldFloods-trained S2 U-Net 71.03.
  - The authors (§4.2) say MNDWI comes "very close" to deep learning on unseen sets. Their two explanations: MNDWI was used to build those labels, or the models still generalize poorly.
  - Single-dataset models transfer badly. A WorldFloods model scores 80.36 on S1S2Water, against 94.86 for the in-domain model (Table 3).
- **Relevance to Peru:**
  - Fig. 4 (Somalia coastal scene in UNOSAT): cloud shadows over bright backgrounds cause false positives.
  - §5: flood models overpredict, with low precision, on scenes with little flood water.
  - Their "UNOSAT" dataset is the 2020 Africa / Middle East collection. **It is not our UNOSAT 2023 Peru maps.**
- **Protocol worth copying (§3.3):**
  - Average metrics per flood event.
  - Tiled inference over full scenes (256 px, 32 px overlap).
  - Exclude cloud and invalid pixels.
  - Split permanent from flood water with JRC GSW.
  - Precision-recall curves across thresholds (Fig. 3).
- **Code:** https://github.com/kipoju/udl4fl (non-commercial licence).

### 1.2 WorldFloods v1: Mateo-Garcia et al. (2021), Scientific Reports (FULL TEXT)

- **Citation:** G. Mateo-Garcia, J. Veitch-Michaelis, L. Smith, S. V. Oprea, G. Schumann, Y. Gal, A. G. Baydin, D. Backes. "Towards global flood mapping onboard low cost satellites with machine learning." *Sci. Rep.* 11, 7249 (2021). doi:10.1038/s41598-021-86650-z.
- **URLs opened:** https://www.nature.com/articles/s41598-021-86650-z and https://www.nature.com/articles/s41598-021-86650-z/tables/2
- **Protocol:**
  - The test set uses manually selected flood maps derived from S2 with no temporal misalignment, with errors fixed by hand.
  - No spatial overlap with train/val, and other maps of the same test events are removed from train/val.
  - Noisy labels are kept for training.
- **Table 2 (IoU, total water, 10 m):**
  - NDWI with threshold −0.22: 65.12.
  - NDWI with threshold 0: 39.99. The authors suspect water with suspended matter.
  - Linear model 64.87, SCNN 71.12, U-Net 72.42.
- **Code and data:** WorldFloods via https://tinyurl.com/worldfloods. Code on GitLab (frontierdevelopmentlab).

### 1.3 WorldFloods v2 / ml4floods: Portalés-Julià, Mateo-García, Purcell & Gómez-Chova (2023), Scientific Reports (FULL TEXT, partial)

- **Citation:** "Global flood extent segmentation in optical satellite images." *Sci. Rep.* 13, 20316 (2023). doi:10.1038/s41598-023-47595-7.
- **URL opened:** https://www.nature.com/articles/s41598-023-47595-7
- **Dataset and protocol:**
  - About 75k 256-px S2 L1C tiles from more than 180 events. More than 200 maps manually curated.
  - New validation/test sets with high spatial variability.
  - Global test set: 17 flood maps from 11 events.
  - Multi-output U-Net that predicts cloud and water separately.
- **Main finding:** "MNDWI has been shown to be a strong baseline", but in their experiments the deep-learning models outperform MNDWI on all metrics (Fig. 6; numbers are in the figure and Suppl. Table 1, which I did not transcribe).
  - Lower performance in built-up and forest areas.
  - Fig. 6b gives precision-recall curves for the best models against MNDWI.
- **Code and data:** https://github.com/spaceml-org/ml4floods (code); https://zenodo.org/record/8153514 (maps). Also mirrored on Hugging Face as `isp-uv-es/WorldFloodsv2` (seen in search results only).
- **Tension with §1.1:** on WorldFloods' own test set, deep learning beats MNDWI. On *unseen* datasets (§1.1, §1.6), MNDWI comes close.

### 1.4 Bonafilia et al. (2020), Sen1Floods11, CVPR Workshops (FULL TEXT)

- **Citation:** D. Bonafilia, B. Tellman, T. Anderson, E. Issenberg. "Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1." CVPR Workshops 2020.
- **URL opened:** https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf
- **Setup:** Bolivia is held out as a separate test event. Models use **S1 input**.
- **Table 2 (Bolivia, all-water mean IoU):** S1-weak 0.3871, Otsu-VH 0.3862, hand-labelled 0.3524, S2-weak 0.3160.
- These numbers are not comparable to our S2 IoUs. The paper is cited as the origin of the Bolivia hold-out protocol.
- **Code and data:** github.com/cloudtostreet/Sen1Floods11.

### 1.5 Patel, Sharma, Pasquarella & Gulshan (2021) (FULL TEXT)

- **Citation:** "Evaluating Self and Semi-Supervised Methods for Remote Sensing Segmentation Tasks." arXiv:2111.10079. No journal reference listed.
- **URL opened:** https://arxiv.org/pdf/2111.10079
- **Setup:** Sen1Floods11, **S1 input only**, with a domain-shifted split by event:
  - Train: Ghana, India, Nigeria, Paraguay, USA (275 chips).
  - Validation: Somalia, Sri Lanka, Bolivia (83).
  - Test: Mekong, Pakistan, Spain (88).
  - Unlabelled pool: the 4,385 Sen1Floods11 chips without hand labels. This pool is not restricted to the target regions.
- **Method:** FixMatch (confidence-thresholded pseudo-labels with consistency regularization), optionally on a SimCLR-pretrained backbone.
- **Results (Table IV, water IoU, domain-shifted test, 100 % labels):**
  - Supervised from random init 67.77 ± 0.8; FixMatch from random init 71.56 ± 0.6.
  - Supervised from SimCLR init 70.95 ± 0.1; FixMatch from SimCLR init 70.63 ± 0.5.
  - The column mapping is confirmed by the text in §IV-E: "FixMatch(Random) outperforms Supervised(Random) by 6.0% and 3.8%" at 10 % and 100 % labels.
- **Protocol:** all hyperparameter and checkpoint selection on validation regions that are disjoint from the test regions, then a single test run (§IV-A). Mean ± std over runs.
- **Code:** not seen.

### 1.6 Schmalstieg, Müller & Samek (2026) (ABSTRACT + PDF skim)

- **Citation:** "Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation." arXiv:2609.20441.
- **URL opened:** https://arxiv.org/pdf/2609.20441
- **Setup:**
  - Teacher: Prithvi-EO-2.0 300M fine-tuned on the 252 Sen1Floods11 training chips.
  - The teacher pseudo-labels 23,720 unlabelled global S2 tiles taken from GDACS events.
  - Student: EfficientViT-B0.
  - Out-of-distribution tests on STURM-Flood and WorldFloods-v2. The 15 Bolivia chips are kept as a transfer set.
- **Method:** pseudo-label distillation with confidence filtering. The pool is global, not a specific target region. **The teacher is never adapted.**
- **Results (abstract):**
  - Water IoU on the Sen1Floods11 test split: student 0.787, teacher 0.822.
  - "A fixed MNDWI threshold is competitive with both models on the two clean external benchmarks."
- **Code:** not checked.
- **Why it matters:** it is the nearest work that uses a GFM to pseudo-label unlabelled S2 imagery.

### 1.7 Akiva et al. (2021), H2O-Net, WACV (FULL TEXT, method only)

- **Citation:** P. Akiva, M. Purri, K. Dana, B. Tellman, T. Anderson. WACV 2021. arXiv:2010.05309.
- **URL opened:** https://openaccess.thecvf.com/content/WACV2021/papers/Akiva_H2O-Net_Self-Supervised_Flood_Segmentation_via_Adversarial_Domain_Adaptation_and_Label_WACV_2021_paper.pdf
- **Shift:** across sensors. S2 (which has SWIR) → PlanetScope RGB+NIR → DroneDeploy aerial imagery.
- **Method:** an adversarial network synthesizes the SWIR2 band. The coarse pseudo-labels are water-index-based and are refined from coarse to fine.
- **Result:** the abstract claims +10 % pixel accuracy and +12 % mIoU (Table 2; per-method values not transcribed).
- **Code:** not checked.

### 1.8 Zhao et al. (2025), UUCP, Remote Sensing of Environment (ABSTRACT, publisher page)

- **Citation:** J. J. Zhao, P. Xiao, Y. Dong, C. Geiß, Y. Zhong, H. Taubenböck. "Large-scale mapping of water bodies across sensors using unsupervised deep learning." *RSE* (2025). doi:10.1016/j.rse.2025.114877.
- **URL opened:** https://www.sciencedirect.com/science/article/abs/pii/S0034425725002810
- **Task:** permanent water, across sensors (S2 and Landsat-8) and regions (China, France). No training labels.
- **Method:** unsupervised multi-segment thresholding produces noisy labels, then a network is trained to be robust to label noise.
- **Results (abstract):** Kappa 0.8859 on S2 and 0.8084 on Landsat-8. A Landsat-8-trained model applied to S2 beats traditional water extraction algorithms.
- **Code:** not checked.

### 1.9 Other flood items: domain generalization, semi-supervised learning, or evaluation only

- **Paul & Ganju (2021)**, "Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning," NeurIPS 2021 Climate Change AI workshop. arXiv:2107.08369. ABSTRACT (export.arxiv.org API).
  - Cyclical pseudo-labelling of the unlabelled competition test set using U-Net ensemble confidence, followed by a CRF.
  - IoU 0.7654 vs a 0.60 baseline (ETCI 2021, S1).
  - Code: https://git.io/JW3P8 (as stated in the abstract; not opened).
- **Qiao & Peng (2023)**, "Topology-aware Robust Optimization for Out-of-distribution Generalization" (TRO), ICLR 2023. arXiv:2307.13943. FULL TEXT (https://arxiv.org/html/2307.13943).
  - Sen1Floods11 with the Bolivia event held out (validation is a random 80/20 split of the other events).
  - Table 4, IoU on the Bolivia test set: ERM .430, IRM .338, REx .357, SD .400, DRO .433, TRO .450. The plain ERM baseline is hard to beat.
  - Code: not checked.
- **Türkmenli, Kayabol & Aptoula (2025)**, "A Comparative Domain Generalization Study for SAR Image-Based Flood Segmentation," JURSE 2025. doi:10.1109/jurse60372.2025.11075977. ABSTRACT (OpenAlex).
  - Each Sen1Floods11 event is treated as a source domain. Domain generalization methods give about +3 % IoU and F1.
  - Code: not checked.
  - A follow-up by the same authors, **"KalmanMix: a novel domain generalization method for flood mapping using satellite imagery,"** *Remote Sensing Letters* (2026), doi:10.1080/2150704x.2026.2727104: title and metadata only. UNVERIFIED content.
- **Kaushik, Maurya, Tellman & Zhang (2026)**, "Assessing Geo-Foundational Models for Flood Inundation Mapping: Benchmarking Models for Sentinel-1, Sentinel-2, and PlanetScope," IEEE JSTARS 19:5649–5665. Confirmed on Crossref. doi:10.1109/jstars.2026.3656855, arXiv:2511.01990. FULL TEXT (PDF skim, https://arxiv.org/pdf/2511.01990).
  - FloodPlanet dataset: 19 sites, including Bolivia, Colombia and Paraguay.
  - Leave-one-region-out over 5 sites (Table IV). S2 mIoU (abstract): Clay 0.66, Prithvi 0.64, DOFA 0.64.
  - No domain adaptation.
  - Data: FloodPlanet (the paper links a curated repository). Code: not checked.
- **Kolluru et al. (2026)**, "Land cover and flood type govern the detection limits of satellite-based flood mapping across diverse global flood events," arXiv:2606.07780. FULL TEXT (PDF, https://arxiv.org/pdf/2606.07780).
  - Zero-shot Prithvi-EO-2.0-600M Sen1Floods11 checkpoint applied to 19 out-of-distribution events.
  - Mean IoU 0.46 against Copernicus GFM and 0.49 against OPERA DSWx, versus 0.83 in-sample.
  - High recall (0.90) but low precision (0.49) against GFM, i.e. systematic overdetection.
  - Libya (hot desert) is the best event: F1 0.88, IoU 0.79 against GFM (Table 3). But Libya is a single chip with 23.32 km² of flood, the floods are compact (mean patch 0.75 km², fragmentation index 1.33), and the event is grouped with Italy among the best. Myanmar is the worst: F1 0.24, IoU 0.13.
  - The only South American event is Brazil (Porto Alegre, 2024, a humid riverine flood). Table 3 gives IoU 0.36 against GFM (precision 0.36, recall 1.00) and 0.62 against DSWx. The 4.9 % / 4.6 % precision/recall figures in the text are the *initial* pipeline-broken validation before the failure modes were fixed, not the model's final score.
  - For us: one desert event doing well does not contradict our desert failure mode, because the scenes differ. Cite it as a contrast.
  - Code: uses the public NASA-IMPACT Prithvi-EO-2.0 checkpoint.
- **Li, Lee, Wang, Hsu & Arundel (2023)**, "Assessment of a new GeoAI foundation model for flood inundation mapping," ACM SIGSPATIAL GeoAI workshop. doi:10.1145/3615886.3627747, arXiv:2309.14500. ABSTRACT (OpenAlex).
  - Prithvi-1 evaluated on Sen1Floods11 including Bolivia. It is the source of the "U-Net Base" Bolivia row in the Prithvi-CAFE table.
  - Code: not checked.
- **Tulbure et al. (2025)**, "Leveraging AI multimodal geospatial foundation models for improved near-real-time flood mapping at a global scale," arXiv:2512.02055. ABSTRACT (export.arxiv.org API).
  - TerraMind fine-tuned on FloodsNet (85 events). U-Net recall is higher than every GFM configuration. No domain adaptation.
  - Code: not checked.
- **Panner Selvam, Ramos-Pollan & Kalaitzis (2024)**, "Rapid Adaptation of Earth Observation Foundation Models for Segmentation," arXiv:2409.09907. ABSTRACT + HTML grep.
  - LoRA on the Clay model, S1 flood data. Here "adaptation" means parameter-efficient fine-tuning (PEFT), not domain adaptation.
  - On one out-of-distribution event, LoRA beats a frozen encoder.
  - Code: uses Clay (github.com/Clay-foundation/model).
- **Gonzalez-Calabuig et al. (2025/26)**, "SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation," CVPR EarthVision 2026. arXiv:2511.10370. FULL TEXT (skim).
  - Abstains from predicting by combining input-space OOD detection, embedding-space OOD detection and predictive uncertainty.
  - Includes a WorldFloods flood task. Names deserts as underrepresented in pretraining.
  - Code: https://github.com/vishalned/SHRUG-FM.
  - Relevant to our uncertainty question, not to adaptation.

### 1.10 Water-body adaptation and label-free work, mostly not floods (ABSTRACT or metadata via OpenAlex)

- **Abid et al. (2021)**, "UCL: Unsupervised Curriculum Learning for water body classification from remote sensing imagery," *JAG* 102568. doi:10.1016/j.jag.2021.102568.
  - Clustering of deep features plus curriculum learning, at the scene-classification level.
  - Uses a Sentinel-2 dataset of Pakistan, PakSAT. "outperform supervised methods in domain adaptation".
  - Code: not checked.
- **Zhu et al. (2026)**, "Fully automated water body extraction method using weakly supervised learning and cross-scale knowledge transfer," *J. Appl. Remote Sens.* 20:026512. doi:10.1117/1.jrs.20.026512.
  - **NDWI-derived pseudo-labels** from 10 m imagery are transferred to 0.5 m RGB, then trained across regions. IoU 77.54 % without annotations, over five Chinese cities.
  - Code: not checked.
- **Zhang et al. (2026)**, "Label-Free Water Body Extraction From High-Resolution RGB Imagery Using Cross-Scale Sample Transfer and Self-Training," IEEE TGRS. doi:10.1109/tgrs.2026.3723834. Title only. UNVERIFIED content.
- **Hao et al. (2024)**, "DeepWaterFraction: A globally applicable, self-training deep learning approach for percent surface water area estimation from Landsat mission imagery," *J. Hydrol.* 131512. doi:10.1016/j.jhydrol.2024.131512. Title only. UNVERIFIED content.
- **Liu & Wang (2022)**, SFnet-DA, *Remote Sensing* 14(15):3538. doi:10.3390/rs14153538.
  - Water body extraction from high-resolution imagery with a domain adaptation module. The abstract gives no source → target setup.
- **Yang et al. (2025)**, DASAM, *J. Imaging* 11(12):437. doi:10.3390/jimaging11120437.
  - Segment Anything adapted for water-body **domain generalization**: trained on GF-2, evaluated cross-domain on GLH-water and an S2 water dataset.
- **Hertel, Geiß, Wieland & Taubenböck (2025)**, "Rapid domain adaptation for disaster impact assessment: Remote sensing of building damage after the 2021 Germany floods," *Science of Remote Sensing* 100287. doi:10.1016/j.srs.2025.100287.
  - Benchmarks domain adaptation strategies for building **damage**, not water. "Supervised DA with minimal annotations per class emerged as the most effective."
- **Mateo-Garcia, Laparra, Lopez-Puigdollers & Gómez-Chova (2021)**, "Cross-Sensor Adversarial Domain Adaptation of Landsat-8 and Proba-V Images for Cloud Detection," IEEE JSTARS 14:747–761. doi:10.1109/JSTARS.2020.3031741. Crossref metadata only.
  - CycleGAN-style (CyCADA) adaptation between sensors, for clouds.
- **Hafner, Ban & Nascetti (2022)**, "Unsupervised domain adaptation for global urban extraction using Sentinel-1 SAR and Sentinel-2 MSI data," *RSE* 280:113192. doi:10.1016/j.rse.2022.113192. ABSTRACT (OpenAlex).
  - Built-up area mapping, not water. Unsupervised domain adaptation across regions using S1–S2 prediction consistency.
  - F1 on 60 global sites: 0.694 with adaptation, vs 0.580 (S2 only), 0.574 (S1 only) and 0.651 (input fusion).

---

## 2. General unsupervised domain adaptation for remote-sensing semantic segmentation

### 2.1 Wang et al. (2021), LoveDA, NeurIPS Datasets & Benchmarks (FULL TEXT)

- **Citation:** J. Wang, Z. Zheng, A. Ma, X. Lu, Y. Zhong. arXiv:2110.08733.
- **URL opened:** https://arxiv.org/pdf/2110.08733
- **Shift:** rural ↔ urban, 0.3 m RGB imagery.
- **Model and methods:** DeepLabV2-R50 with:
  - Adversarial training: AdaptSeg, CLAN, FADA, TransNorm.
  - Self-training: CBST, PyCDA, IAST.
  - DDC.
- **Table 6, test-set Water IoU (source-only → best adaptation → oracle):**
  - Rural→Urban: 76.22 → IAST 86.01 (oracle 85.72). mIoU 31.27 → CBST 41.32 (oracle 46.79).
  - Urban→Rural: 49.42 → IAST 64.49 (oracle 65.89). mIoU 31.74 → IAST 38.44 (oracle 45.67).
- **Authors' findings (§4.2):**
  - Self-training beats adversarial training under shifts in class balance.
  - TransNorm, which adapts normalization layers, is worst when source and target come from the same sensor.
- **Code and data:** https://github.com/Junjue-Wang/LoveDA

### 2.2 ISPRS Potsdam ↔ Vaihingen (ABSTRACTS, export.arxiv.org API)

- ST-DASegNet: Zhao et al., arXiv:2301.05526.
- SAM-JOANet: Lyu et al., arXiv:2411.05878.
- Very-high-resolution aerial imagery. No numbers transcribed. Code: not checked.

### 2.3 Lyu et al. (2025), domain adaptation survey (FULL TEXT, searched)

- **Citation:** arXiv:2510.15615.
- **URL opened:** https://arxiv.org/html/2510.15615v1
- **Negative finding:** the full text (about 229k characters) contains **no** matches for "flood", "inundat", "water bod" or "surface water".

### 2.4 Foundation-model domain generalization and test-time adaptation in remote sensing

- **CrossEarth**: Gong et al., IEEE TPAMI 2025. doi:10.1109/tpami.2025.3649001, arXiv:2410.22629. ABSTRACT + HTML grep.
  - Domain generalization, with a benchmark of 32 settings, mostly VHR land cover.
- **CrossEarth-Gate**: arXiv:2511.20302. ABSTRACT. Parameter-efficient fine-tuning across domains.
- **Niloy, Bhaumik & Woo (2024)**: arXiv:2401.02113. ABSTRACT + HTML grep.
  - Backprop-free test-time adaptation that re-estimates batch-norm statistics.
  - Tested on DeepGlobe with synthetic corruptions. Project page: https://sat-tta.github.io
  - My note: Prithvi and TerraMind use LayerNorm, so batch-norm re-estimation would only apply to the U-Net and the decoders.

### 2.5 General computer-vision methods these papers build on (metadata via export.arxiv.org API)

- DACS: arXiv:2007.08702, WACV 2021.
- DAFormer: arXiv:2111.14887, CVPR 2022.
- MIC: arXiv:2212.01322, CVPR 2023.
- TENT: arXiv:2006.10726, ICLR 2021.
- CoTTA: arXiv:2203.13591, CVPR 2022.

CIM-TTDA (§1.0) borrows CoTTA's stochastic restoration and uses TENT as a baseline.

---

## 3. Geospatial foundation models with domain adaptation or test-time adaptation, and Prithvi-CAFE

### 3.1 Prithvi-CAFE: Kaushik, Maurya & Tellman (2026), CV4EO workshop at WACV 2026 (FULL TEXT)

- **Citation:** arXiv:2601.02315.
- **URL opened:** https://arxiv.org/html/2601.02315
- **Code:** https://github.com/Sk-2103/Prithvi-CAFE
- **What it says about Bolivia:**
  - It calls Bolivia a "geographically held-out test site (i.e., Bolivia)" (§5.1) and talks of "generalization to completely unseen locations" (§5.2).
  - It never uses the term "out-of-distribution", and does **no domain adaptation**.
- **Table 1, Bolivia block (water IoU / mIoU):**

  | Model | Water IoU | mIoU | Source |
  |---|---|---|---|
  | Prithvi 1.0 | 72.42 | 82.89 | [16] |
  | U-Prithvi | 79.68 | 87.70 | [16] |
  | U-Net Base | 70.57 | 82.54 | [19] = Li et al. 2023 |
  | DeCUR | – | 85.84 | [23] |
  | Prithvi 2.0 300M LoRA | – | 87.57 | [23] |
  | Prithvi 2.0 300M ViT-Adapter | – | 84.94 | [23] |
  | Prithvi 2.0 300M full fine-tune | – | 82.07 | [23] |
  | **Prithvi-CAFE** | **81.37** | **88.87** | own run |

- **Cautions:**
  - **The rows come from different papers.** The "+10.8 IoU over U-Net" claim therefore compares results across studies.
  - The conclusion says "6.63 mIoU", but Table 1 gives 88.87 − 82.54 = 6.33.

### 3.2 Official Prithvi-EO-2.0 and TerraMind papers (FULL TEXT grep)

- Sources: arXiv:2412.02732 and arXiv:2504.11171 (HTML), searched with a plain case-insensitive count of "bolivia", "held-out", "hold-out", "out-of-distribution", "domain adapt" and "test-time".
- Prithvi-EO-2.0: 0 matches.
- TerraMind: 1 match, only in the reference list (the Hafner 2022 title).
- **Neither paper reports Bolivia or out-of-distribution flood results, nor any domain adaptation.**

### 3.3 Other GFM evidence

- **On-orbit GFM**: Du et al., "First On-Orbit Demonstration of a Geospatial Foundation Model," arXiv:2512.01181. FULL TEXT §4.4.
  - Their "domain adaptation" is **supervised**: task heads of a compressed Prithvi are re-trained on labelled imagery from the Kanyini mission.
- **PANGAEA**: arXiv:2412.04204. ABSTRACT.
  - That U-Net beats GFMs on Sen1Floods11 comes from a search snippet. UNVERIFIED.
- **Hsu, Li & Wang (2024)**: arXiv:2409.00489. "Domain adaptability" there means band adaptation and fine-tuning.
- **Mahlasi et al. (2026)**: arXiv:2602.10720. Performance is sensitive to divergence between downstream inputs and pretraining modalities.

**Bottom line.** I found **no paper that adapts the weights of a geospatial foundation model (Prithvi, TerraMind, Clay, etc.) on unlabelled target-region data**, by unsupervised domain adaptation, self-training or test-time adaptation, for floods or any other task I checked. The nearest works are:
- Schmalstieg 2026: Prithvi is used as a frozen teacher that pseudo-labels unlabelled S2.
- CIM-TTDA 2026: Prithvi-EO-2.0 is only a weak, non-adapted baseline on SAR.
- On-orbit 2025: Prithvi heads are adapted with target labels.

---

## 4. Peru and arid South America

- **Gonzalez, Bhowmick, Beltran, Sankaran & Bengio (2019)**, "Applying Knowledge Transfer for Water Body Segmentation in Peru," NeurIPS 2019 ML4D workshop. arXiv:1912.00957. ABSTRACT (export.arxiv.org API).
  - PeruSAT-1 very-high-resolution rivers and lakes, plus S2. Adding S2 made results worse, which the authors attribute to a distribution shift.
  - Code: not checked.
- **Badillo-Rivera et al. (2025)**, Frontiers in Environmental Science. doi:10.3389/fenvs.2025.1672107. ABSTRACT (OpenAlex).
  - Piura 2017: S1 flood extent mapped with the NDFI index, then machine-learning susceptibility mapping. Not deep-learning segmentation.
- **Roman-Gonzalez et al. (2019)**, IJACSA 10(2). doi:10.14569/ijacsa.2019.0100246. ABSTRACT.
  - S1 change detection for Piura and Chosica in 2017.
- **Alvan Romero & Suarez Alayza (2024)**, Preprints.org. doi:10.20944/preprints202407.1784.v1. ABSTRACT.
  - Cyclone Yaku 2023, La Leche river (Íllimo). Change detection, not deep learning.
- **South American events found in other datasets:**
  - FloodPlanet: Bolivia, Colombia, Paraguay (Kaushik 2026).
  - Sen1Floods11: Bolivia, Paraguay.
  - Kolluru 2026: Brazil (Porto Alegre).
  - All are humid lowland or subtropical. **None is on the arid Pacific coast.**
- **No deep-learning flood segmentation found for the Peruvian desert coast or for Cyclone Yaku.**

---

## Search log

**arXiv API, abstract field:**

| Query | Hits | Relevant results |
|---|---|---|
| flood + "domain adaptation" | 9 | H2O-Net; FloodDAN (forecasting) |
| flood + "test-time" | 5 | none |
| flood + self-training / pseudo-label | 9 | Paul & Ganju |
| flood + "domain generalization" | 2 | AIFloodSense (aerial) |
| water + "domain adaptation" + segmentation | 5 | H2O-Net |
| flood + distribution shift / OOD | 17 | Kolluru, Panner Selvam, SHRUG-FM |

**OpenAlex `title_and_abstract.search`, sorted by citations, top 50 scanned per query:**

| Query | Hits | Relevant results |
|---|---|---|
| flood + "domain adaptation" | 48 | H2O-Net, **CIM-TTDA**, Hertel 2025, SSRN temporal-shift SAR paper (title only) |
| flood + "self-training" | 7 | none |
| flood + "pseudo-label" | 38 | SAR semi-supervised papers; synthetic UAV flood pseudo-labels (Sensors 2025) |
| flood + "test-time adaptation" | 1 | none |
| flood + "domain generalization" | 22 | KalmanMix, JURSE 2025, GeoFusionFlood (low-tier journal; NDWI labels) |
| inundation + "domain adaptation" | 2 | CIM-TTDA code record |
| "water body" + "domain adaptation" | 11 | UCL, SFnet-DA |
| "water body" + "self-training" | 4 | DeepWaterFraction, Zhang 2026 TGRS |
| "water body" + "pseudo-label" | 6 | none for optical satellite floods |
| "water body" + "domain generalization" | 10 | DASAM |
| "surface water" + DA / ST / PL / TTA / DG | 0–4 | none relevant |

**Web searches** (UDA water, flood self-training, TTA flood, GFM + TTA, PANGAEA Bolivia, Peru/Yaku/Niño Costero in Spanish and English) gave the leads cited above.

**Not done:** IEEE Xplore and Scopus directly. Full texts of KalmanMix, ST-DASegNet, SAM-JOANet, DASAM and the Zhang 2026 TGRS paper.

---

## (a) Is unsupervised domain adaptation for optical flood/water mapping an open gap?

**What exists:**
- **Test-time self-training for floods, on SAR only:** CIM-TTDA (2026). It benchmarks CBST, AdaptSegNet, TENT, CSFDA and UR, and gains about +0.15 IoU over source-only.
- **Evaluation across datasets, regions and events without adaptation:** Portalés-Julià 2025, Bonafilia 2020, Kaushik 2026, Kolluru 2026, Prithvi-CAFE.
- **Domain generalization for floods:** TRO, the JURSE 2025 study and KalmanMix.
- **Semi-supervised learning with unlabelled flood imagery, on S1:** Patel 2021 (FixMatch, cross-region, +3.8 IoU) and Paul & Ganju 2021 (transductive).
- **Label-free or index-based pseudo-labels for permanent water, or across sensors:** UUCP, H2O-Net, Zhu 2026, UCL.
- **GFM as a pseudo-label teacher on unlabelled S2:** Schmalstieg 2026.
- **Mature adaptation for land cover:** LoveDA shows self-training recovers most of the Water-class gap.

**What I did not find:**
1. Unsupervised adaptation or test-time adaptation of an **optical (S2)** flood model to a specific unlabelled target region, evaluated against expert labels.
2. Adaptation of a **GFM's own weights** on unlabelled target data.
3. Any flood or water adaptation study aimed at **bright arid backgrounds**.
4. Deep-learning flood mapping of the Peruvian coast or Yaku.

**Verdict: a real but narrow gap.** Claim: "first study of unsupervised adaptation / self-training for **optical** flood segmentation with GFMs, under a documented desert shift."

Do **not** claim:
- the first domain adaptation or test-time adaptation for floods (CIM-TTDA exists), or
- the first adaptation for water (H2O-Net, UUCP and others exist).

**Two caveats:**
- An MNDWI baseline is mandatory, because MNDWI was competitive on external datasets in Portalés-Julià 2025 and Schmalstieg 2026.
- Index-based pseudo-labels are an obvious baseline. Our own label QA shows MNDWI is weak on hazy desert top-of-atmosphere imagery (`docs/peru_label_sources.md`), which motivates learned adaptation.

## (b) Protocols and baselines to copy

1. **Report three levels**, as in LoveDA Table 6 and CIM-TTDA Table 4:
   - source-only (Sen1Floods11 → Peru),
   - each adaptation method,
   - an oracle (Peru-only on `peru-train`).

   Also report the fraction of the source-to-oracle gap that each method closes.
2. **Model selection.**
   - Fix the adaptation hyperparameters and epoch budget in advance, as CIM-TTDA does (4 epochs, fixed learning rate, EMA 0.9).
   - Or select on `peru-dev` only, as Patel does with validation regions disjoint from test.
   - Report both. Never tune on `peru-test`.
3. **Baselines:**
   - **Offline CBST on the unlabelled Peru pool comes first.** It is class-balanced self-training, and the closest published analogue of our setting (CIM-TTDA §5.3, LoveDA). AdaptSegNet (adversarial) goes with it.
   - MNDWI > 0 (and Otsu-tuned MNDWI).
   - Source-only.
   - TENT. For ViTs, apply it to the LayerNorm affine parameters.
   - FixMatch-style consistency, as in Patel.
   - Mean-teacher / EMA self-training with stochastic restoration, as in CIM-TTDA / CoTTA.
   - Index-based pseudo-labels, as in UUCP / H2O-Net / Zhu.
4. **Metrics:**
   - Water IoU per event (unweighted) and pixel-pooled, as CIM-TTDA's IoUe and IoUp.
   - Precision and recall separately. The literature's failure mode under shift is overprediction (Portalés-Julià; Kolluru: recall 0.90, precision 0.49).
   - Precision-recall curves over thresholds (Portalés-Julià Fig. 3; WorldFloods v2 Fig. 6b). This also links to our calibration research question.
5. **Scale.** At least 3 seeds with mean ± std (Patel, Schmalstieg with 4, CIM-TTDA with 4 source seeds). Report adaptation time and memory, which CIM-TTDA does on a single GPU.
6. **Do not mix results across studies or across backbones.** The Prithvi-CAFE Bolivia rows come from different papers. CIM-TTDA compares its EfficientNet against DeepLabV2-based baselines. Re-run every baseline on the same backbone under our protocol, and report the gain over each method's own source model.
7. **Decide the adaptation unit in advance.** Options: adapt once on the 2023 Peru unlabelled pool, then evaluate on `peru-test` and `peru-2017-test`, or adapt separately per event or year. CIM-TTDA's Table 5 finds per-event adaptation beats cross-event adaptation, so this choice changes the results and must be fixed before the final test.
