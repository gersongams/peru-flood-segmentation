# UDA, source-free and test-time adaptation for semantic segmentation: literature check

Context: source = Sen1Floods11 (labelled), target = Peru S2 (≈750 unlabelled chips, 512×512), binary water/not-water.
Models: U-Net ResNet-34, TerraMind v1-B (ViT), Prithvi-EO-2.0 300M (ViT), TerraTorch/Lightning.
Budget: one RTX 4090 for a few hours per run, or an M4 Max.

How this was checked (2026-10-14): each paper's arXiv PDF was downloaded and converted with `pdftotext`, then the abstract, method and result tables were read. Venues come from the arXiv "Comments" field, or from the CVF open-access page where noted.
Every number below was read in the paper at the stated table or section. Anything not verified is marked **UNVERIFIED**.

**Caveat on numbers.** Almost every number is mIoU on GTA5→Cityscapes or Synthia→Cityscapes (19 or 16 classes, synthetic→real). They are given only to show *relative* gains inside each paper. Numbers across papers also mix backbones, so they are not method-only comparisons. For example, AdaptSegNet's 42.4 uses DeepLabV2-R101 and DAFormer's 68.3 uses MiT-B5. None of these numbers transfers to binary water segmentation.

---

## 1. Self-training / pseudo-labelling (teacher–student)

### 1.1 CBST: class-balanced self-training
- **Citation:** Y. Zou, Z. Yu, B.V.K. Vijaya Kumar, J. Wang. "Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training." ECCV 2018 (arXiv comment: "Accepted to ECCV 2018").
- **URL opened:** https://arxiv.org/abs/1810.07911 (PDF read)
- **Idea:** Pseudo-labelling runs in alternating rounds. For each class, only the top-*p* most confident target pixels are kept, using a class-wise normalised confidence. This stops easy or majority classes from dominating. *p* starts at 20% and grows by 5% per round, up to 50% (Sec. 4, "self-paced learning policy"). An optional spatial prior (CBST-SP) can be added.
- **Code:** https://github.com/yzou2/cbst. The link is embedded in the PDF and the GitHub repo returns HTTP 200.
- **Cost:** low. There is no extra network, only offline pseudo-label generation per round plus re-training (several rounds).
- **Result (Table 3, GTA5→Cityscapes, ResNet-38):** source only 35.4, ST 41.5, CBST 45.2, CBST-SP 46.2, CBST-SP+MST 47.0 mIoU.
- **Relevance:** per-class thresholds map directly onto a binary task. They give a separate confidence cut-off for "water" and for "not water", so the rare water class still gets pseudo-labels.

### 1.2 Mean Teacher (EMA teacher)
- **Citation:** A. Tarvainen, H. Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." NeurIPS (NIPS) 2017.
- **URL opened:** https://arxiv.org/abs/1703.01780 (PDF read)
- **Idea:** The teacher's weights are an exponential moving average (EMA) of the student's. A consistency loss pushes the student toward the teacher's predictions. It began as semi-supervised classification and is now the standard teacher in DAFormer, HRDA, MIC and CoTTA.
- **Code:** https://github.com/CuriousAI/mean-teacher (stated in the paper, App. B).
- **Cost:** one extra set of weights in memory and one extra forward pass with no gradient.
- **Result:** not quoted, because it reports classification only.

### 1.3 DACS: cross-domain mixed sampling
- **Citation:** W. Tranheden, V. Olsson, J. Pinto, L. Svensson. "DACS: Domain Adaptation via Cross-domain Mixed Sampling." WACV 2021.
- **URL opened:** https://arxiv.org/abs/2007.08702 (PDF read)
- **Idea:** Source pixels of some classes (ClassMix) are pasted onto target images. The mixed image is trained against a label that combines the source ground truth with the target pseudo-labels. This fixes the class-collapse of naive pseudo-labelling.
- **Code:** https://github.com/vikolss/DACS (stated in paper).
- **Cost:** GTX 1080 Ti, 250k iterations (Sec. 4), so moderate.
- **Result (Table 1):** GTA5→Cityscapes 52.14 mIoU (DeepLabV2-R101, mean of 3 runs).

### 1.4 ProDA: prototypical pseudo-label denoising
- **Citation:** P. Zhang, B. Zhang, T. Zhang, D. Chen, Y. Wang, F. Wen. "Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation." CVPR 2021 (title verified on the CVF open-access page).
- **URL opened:** https://arxiv.org/abs/2101.10979 (PDF read); https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Prototypical_Pseudo_Label_Denoising_and_Target_Structure_Learning_for_Domain_CVPR_2021_paper.html
- **Idea:** Pseudo-labels are corrected online with class prototypes, meaning feature centroids. A consistency term learns the target structure. Training then distils into a self-supervised (SimCLRv2) initialised model.
- **Code:** https://github.com/microsoft/ProDA. The repo exists (HTTP 200), but no link to it was found in the paper text, so the link is **UNVERIFIED as official**.
- **Cost:** high. It uses 4 Tesla V100 GPUs (Sec. 5.1) with multiple stages: adversarial warm-up, then self-training, then distillation.
- **Result (Table 1):** GTA5→Cityscapes 57.5 mIoU, +20.9 over source only (DeepLabV2-R101).
- **Verdict:** too heavy for this budget, and prototypes add little with only two classes.

### 1.5 DAFormer
- **Citation:** L. Hoyer, D. Dai, L. Van Gool. "DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation." CVPR 2022.
- **URL opened:** https://arxiv.org/abs/2111.14887 (PDF read)
- **Idea:** The base is DACS-style self-training with an EMA teacher (α=0.99). Target pseudo-labels are weighted by the fraction of pixels whose max softmax exceeds τ=0.968, and ClassMix, colour jitter and blur are applied. DAFormer adds three changes on top:
  - Rare Class Sampling
  - a Thing-Class ImageNet Feature Distance (FD), which keeps the encoder close to its pre-trained features
  - LR warm-up

  It also uses a SegFormer (MiT-B5) encoder with a context-aware decoder.
- **Code:** https://github.com/lhoyer/DAFormer
- **Cost:** 40k iterations with a batch of 2×512² crops, on one RTX 2080 Ti in about 16 h (Sec. 1 and 4.1). Training memory is 9.6 GB (Table S4).
- **Results:**
  - Table 6: GTA→CS 68.3, Synthia→CS 60.9 mIoU (mean of 3 seeds).
  - Table 1: transformer vs CNN. SegFormer gets 45.6 source-only and 58.2 UDA, against DeepLabV2's 34.3 and 54.2. The authors attribute part of the gain to SegFormer's better source-only generalisation.
  - Table 4: removing LR warm-up drops SegFormer UDA from 58.2 to 51.8. **Warm-up matters for pre-trained transformers.**

### 1.6 HRDA
- **Citation:** L. Hoyer, D. Dai, L. Van Gool. "HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation." ECCV 2022.
- **URL opened:** https://arxiv.org/abs/2204.13132 (PDF read)
- **Idea:** Multi-resolution training. A large low-resolution context crop and a small high-resolution detail crop are fused with a learned scale attention, on top of DAFormer.
- **Code:** https://github.com/lhoyer/HRDA
- **Cost:** heavy. About 22.5 GB training memory (Table 5) on a Titan RTX 24 GB.
- **Result (Table 1):** GTA→CS 73.8, Synthia→CS 65.8 mIoU.
- **Verdict:** a 512² chip is already the native size, so the multi-resolution benefit is doubtful. VFM-UDA (Section 5.1) found HRDA does not help ViT encoders.
- The journal extension (TPAMI) is **UNVERIFIED** and was not checked.

### 1.7 MIC: Masked Image Consistency
- **Citation:** L. Hoyer, D. Dai, H. Wang, L. Van Gool. "MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation." CVPR 2023.
- **URL opened:** https://arxiv.org/abs/2212.01322 (PDF read)
- **Idea:** Random patches of the target image are masked. The student must match the EMA teacher's pseudo-label, which was computed on the full image. This forces the student to use context. It is a plug-in for DAFormer, HRDA or adversarial UDA.
- **Code:** https://github.com/lhoyer/MIC
- **Cost:** about +3% memory over the base method. DAFormer goes from 9.64 to 9.74 GB, and throughput from 0.71 to 0.57 it/s (Table 10).
- **Results:**
  - Table 2: GTA→CS 75.9, Synthia→CS 67.3 mIoU, both with HRDA. CS→DarkZurich test 60.2 (HRDA 55.9). CS→ACDC test 70.4 (HRDA 68.0).
  - Table 7: with DAFormer, 68.3 → 70.6.
  - Table 8: the best patch size and mask ratio are b=64 and r=0.7.

---

## 2. Adversarial alignment

### 2.1 AdaptSegNet: output-space adversarial learning
- **Citation:** Y.-H. Tsai, W.-C. Hung, S. Schulter, K. Sohn, M.-H. Yang, M. Chandraker. "Learning to Adapt Structured Output Space for Semantic Segmentation." CVPR 2018.
- **URL opened:** https://arxiv.org/abs/1802.10349 (PDF read)
- **Idea:** A discriminator tells source softmax maps from target softmax maps, and the segmenter is trained to fool it. Optionally this runs at several levels.
- **Code:** https://github.com/wasidennis/AdaptSegNet
- **Cost:** moderate. One extra discriminator, trained on a single Titan X 12 GB. Adversarial training is known to be unstable and sensitive to λ_adv (the paper's Table 3 is a sensitivity analysis of it).
- **Result (Table 1, ResNet-101):** baseline 36.6 → feature-level 39.3 → single-level output 41.4 → multi-level 42.4 mIoU.

### 2.2 ADVENT: entropy minimisation and adversarial entropy
- **Citation:** T.-H. Vu, H. Jain, M. Bucher, M. Cord, P. Pérez. "ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation." CVPR 2019.
- **URL opened:** https://arxiv.org/abs/1811.12833 (PDF read)
- **Idea:** MinEnt directly minimises the entropy of target predictions. AdvEnt aligns the weighted self-information maps adversarially.
- **Code:** https://github.com/valeoai/ADVENT
- **Cost:** low (MinEnt) to moderate (AdvEnt). Trained on one 1080 Ti.
- **Result (Table 1a, ResNet-101, GTA→CS):** MinEnt 42.3, AdvEnt 43.8, AdvEnt+MinEnt 45.5 mIoU. The retrained AdaptSegNet gets 42.2.
- **Relevance:** MinEnt is the training-time ancestor of TENT. It is a one-line loss and makes a cheap baseline. **Risk for binary:** entropy minimisation can collapse to "all not-water" when water is rare, so it needs a prior or diversity term. This risk is my extrapolation; the paper does not state it.

---

## 3. Image-level appearance alignment

### 3.1 FDA: Fourier Domain Adaptation
- **Citation:** Y. Yang, S. Soatto. "FDA: Fourier Domain Adaptation for Semantic Segmentation." CVPR 2020 (title verified on the CVF open-access page).
- **URL opened:** https://arxiv.org/abs/2004.05498 (PDF read); https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- **Idea:** The low-frequency part of the FFT amplitude spectrum of a *source* image is replaced by that of a random *target* image. The window size is set by β of about 0.01–0.09, and the phase is kept. Training then proceeds on the restyled, labelled source images. There is no extra network.
  - FDA-ENT adds target entropy minimisation.
  - FDA-MBT averages models trained with several β values to produce pseudo-labels, followed by self-supervised training.
- **Code:** https://github.com/YanchaoYang/FDA
- **Cost:** very low, one FFT per image.
- **Results:**
  - Table 1, T=0: FDA alone, without self-training, gives 44.6–45.0 mIoU for β ∈ {0.01, 0.05, 0.09}. That compares with AdaptSegNet's baseline of 36.6 for the same DeepLabV2-R101, which is Table 1 of a *different* paper.
  - Table 2: GTA→CS (ResNet-101) FDA 44.6, FDA-ENT 45.0, FDA-MBT 50.45.
- **Caveat:** the paper uses RGB only. Applying it per band to 6+ Sentinel-2 bands is my extrapolation. Physically it fits a desert-coast vs. global shift well, because low-frequency amplitude carries overall brightness and colour.

### 3.2 Histogram matching / per-band statistics matching
- A classical technique (e.g. `skimage.exposure.match_histograms`). It maps each source band's histogram onto the target's, or the reverse at test time.
- No paper was opened here. Any remote-sensing citation for it is **UNVERIFIED**. It should be treated as a no-training baseline, together with re-computing the normalisation mean/std on Peru imagery.
- The latter interacts with the "official normalisation" rule in CLAUDE.md. It must be framed as an adaptation step, not as a change to the model's normalisation.

---

## 4. Normalisation-statistics adaptation, test-time adaptation and source-free DA

### 4.1 AdaBN
- **Citation:** Y. Li, N. Wang, J. Shi, X. Hou, J. Liu. "Adaptive Batch Normalization for practical domain adaptation." *Pattern Recognition* 80:109–117, 2018. The venue comes from a search result and is not opened; the arXiv preprint title is "Revisiting Batch Normalization For Practical Domain Adaptation," arXiv 1603.04779.
- **URL opened:** https://arxiv.org/abs/1603.04779 (v4 PDF read)
- **Idea:** All BatchNorm running mean and variance values are recomputed on target data. No parameters are learned.
- **Code:** project page https://www.icst.pku.edu.cn/struct/Projects/AdaBN.html. The code is **UNVERIFIED**, but it is trivial to implement.
- **Cost:** one forward pass over the target set.
- **Result (arXiv v4, Table 5):** cloud detection *segmentation* in remote sensing, trained on GF2. On GF1 it goes from 38.95% to 64.50% mIoU, and on Tianhui from 14.54% to 29.66%. This is a satellite-to-satellite shift, which is close to our setting.
- **Applicability to our models (checked in the installed TerraTorch code):**
  - **U-Net ResNet-34:** BatchNorm throughout the encoder and decoder, so AdaBN applies fully.
  - **TerraMind and Prithvi (ViT):** the encoders use LayerNorm, so AdaBN does **nothing** to the encoder. The only BatchNorm layers are:
    - `LearnedInterpolateToPyramidal` neck: one `BatchNorm2d` (`terratorch/models/necks.py`, `fpn1`)
    - `UNetDecoder`: BN blocks (smp `UnetDecoder` with `use_norm="batchnorm"`)

    So AdaBN on the ViT models adapts only the neck and decoder.

### 4.2 TENT: test-time entropy minimisation
- **Citation:** D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell. "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021 (Spotlight). All five authors were checked against the PDF.
- **URL opened:** https://arxiv.org/abs/2006.10726 (PDF read)
- **Idea:** At test time, normalisation statistics are re-estimated and only the channel-wise affine parameters (γ, β) of the normalisation layers are optimised, to minimise prediction entropy. It needs no source data and is online or one epoch.
- **Code:** https://github.com/DequanWang/tent
- **Cost:** very low.
- **Result (Sec. 4.2 text, not a table):** GTA→Cityscapes with HRNet-W18, a BN network. Source 28.8%, BN adaptation 31.4%, TENT 35.8% mIoU (offline, Adam). On a single image, 10 episodic iterations give 36.4%.
- **On ViTs:** TENT is designed for BN. Applying it to LayerNorm affine parameters is my extrapolation. See CoTTA below for evidence that it degrades on a transformer.

### 4.3 CoTTA: continual test-time adaptation
- **Citation:** Q. Wang, O. Fink, L. Van Gool, D. Dai. "Continual Test-Time Domain Adaptation." CVPR 2022.
- **URL opened:** https://arxiv.org/abs/2203.13591 (PDF read)
- **Idea:** Pseudo-labels from an EMA teacher, averaged over weights and over augmentations (the augmentation average is used only when confidence is low). A small fraction of weights is stochastically restored to the source weights to avoid drift and forgetting. All layers are updated.
- **Code:** https://qin.ee/cotta (stated in paper)
- **Cost:** low to moderate, with N augmented forward passes per sample when uncertain.
- **Result (Table 5, Cityscapes→ACDC, SegFormer-B5, mean over 10 rounds):**
  - source 56.7
  - BN-stats adapt 52.0
  - TENT-continual 52.3, degrading over rounds
  - CoTTA 58.6

  The authors explain the failure: Segformer has only one BN layer and transformers mostly use LayerNorm, so BN-adapt and TENT "do not perform well". **This is the key evidence that AdaBN and TENT are weak for our ViT models.**

### 4.4 SHOT: source-free hypothesis transfer
- **Citation:** J. Liang, D. Hu, J. Feng. "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." ICML 2020.
- **URL opened:** https://arxiv.org/abs/2002.08546 (PDF read)
- **Idea:** The source classifier head is frozen. The feature extractor is adapted on target data with information maximisation (confident and diverse predictions) plus clustering-based self-supervised pseudo-labels.
- **Code:** https://github.com/tim-learn/SHOT
- **Result:** **no segmentation experiments.** Segmentation appears only in related work. It is a design reference only.
- **Relevance:** the "diversity" term of information maximisation is a principled fix for the entropy-collapse risk in binary segmentation.
- Source-free DA is not needed in our setting, because the Sen1Floods11 source data is available.

---

## 5. Adapting large pretrained ViT / foundation models

### 5.1 VFM-UDA (DINOv2 + MIC-style self-training)
- **Citation:** B.B. Englert, F.J. Piva, T. Kerssies, D. de Geus, G. Dubbelman. "Exploring the Benefits of Vision Foundation Models for Unsupervised Domain Adaptation." CVPR 2024 Workshops (Second Workshop on Foundation Models). DOI 10.1109/CVPRW63382.2024.00124, from the arXiv comments and DOI field.
- **URL opened:** https://arxiv.org/abs/2406.09896 (v3 PDF read)
- **Idea:** The MiT-B5 encoder of MIC is replaced with a plain DINOv2 ViT. Self-training uses an EMA teacher, and **token-level** masking (in place of image masking) is used for MIC.
- **Code:** https://github.com/tue-mps/vfm-uda
- **Results:**
  - **Table 3, ablation for ViT-B/14 DINOv2, GTA→CS:**

    | Configuration | mIoU |
    |---|---|
    | No adaptation | 62.9 |
    | Pseudo-labelling | 72.8 |
    | + image masking | 76.8 |
    | + token masking | 77.1 |
    | + token masking + FD | 75.1 |
    | + token masking + HRDA | 75.1 |
    | + token masking + FD + HRDA | 76.9 |

    The authors conclude that **FD and HRDA "may be redundant when using ViT-based VFMs"**.
  - **Table 1:** VFM-UDA ViT-B gets 77.1 against MIC's 75.9. ViT-L gets 79.0.
  - **Table 6:** an ImageNet-pre-trained ViT does not improve with size (66.9 / 68.1 / 67.3 for S, B, L), while DINOv2 does (69.7 / 77.1 / 79.0). The pre-training matters.
- **Relevance:** this is the **direct evidence that self-training works with large pretrained ViT backbones**, and that the simplest recipe (EMA teacher + pseudo-labels + token masking) is the right one for ViT foundation models.

### 5.2 Rein: PEFT for VFMs (domain generalisation, not UDA)
- **Citation:** Z. Wei, L. Chen, Y. Jin, X. Ma, T. Liu, P. Ling, B. Wang, H. Chen, J. Zheng. "Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation." CVPR 2024 (title verified on the CVF open-access page).
- **URL opened:** https://arxiv.org/abs/2312.04265 (abstract only)
- **Idea:** A frozen VFM plus about 1% trainable "Rein" parameters. It beats full fine-tuning for **domain generalisation** (no target data).
- **Code:** https://github.com/w1oves/Rein
- **Result (abstract only):** 78.4% mIoU on Cityscapes without real urban training data. No table was read.
- **Relevance:** it suggests that freezing most of the backbone (or using low-LR or PEFT updates) helps keep pre-trained robustness. This ties to the DAFormer FD and warm-up results. It is not a UDA method.

### 5.3 Geospatial-specific (opened but not UDA)
- C.-Y. Hsu, W. Li, S. Wang, "Geospatial foundation models for image analysis: evaluating and enhancing NASA-IBM Prithvi's domain adaptability," arXiv 2409.00489 (abstract read). This is about **supervised fine-tuning, band adaptation and multi-scale features**, not unsupervised adaptation. Not a UDA reference.
- A CVPR 2025 paper on domain generalisation of geospatial foundation models (soft-alignment pseudo-labelling + generative pre-training) appeared in search results (https://cvpr.thecvf.com/virtual/2025/35648) but was **not opened, UNVERIFIED**.
- The LoveDA benchmark (J. Wang, Z. Zheng, A. Ma, X. Lu, Y. Zhong, NeurIPS 2021 Datasets & Benchmarks; https://arxiv.org/abs/2110.08733, abstract read; code https://github.com/Junjue-Wang/LoveDA) is the standard *remote-sensing* UDA benchmark (urban↔rural, 8 UDA methods). Its result tables were **not read**.
- No paper was found applying UDA self-training to Prithvi or TerraMind. Remote-sensing-specific UDA (water and floods) was out of scope for this pass.

---

## 6. Recommendation for this project

The setting is binary, ≈750 target chips, ViT foundation models plus a CNN baseline, and a few GPU-hours.

**Cheap baselines (no or almost no training, run first):**
1. **Statistics alignment at the input.** Use per-band histogram matching, or Peru-recomputed band mean/std, and **AdaBN** for BN layers.
   - For U-Net ResNet-34, AdaBN adapts the whole network. AdaBN Table 5 shows large gains on a satellite-to-satellite segmentation shift.
   - For TerraMind and Prithvi, it touches only the neck and decoder BN, because the encoders use LayerNorm. CoTTA Table 5 shows BN-adapt and TENT *hurt* a LayerNorm transformer.
   - TENT may be run as a secondary baseline for the ResNet only.
2. **FDA.** Restyle Sentinel-2 source chips with Peru low-frequency amplitude (β ≈ 0.01–0.05, per band) and fine-tune on labelled source. FDA's Table 1 (T=0) shows FDA alone, without self-training, already gives most of the gain. It costs one FFT per image and works for all three models.

**Main methods:**
3. **Mean-teacher self-training (DAFormer/DACS-style) with CBST-style per-class thresholds.**
   - EMA teacher (α ≈ 0.99) gives target pseudo-labels.
   - Source + target batches, ClassMix (paste source water onto Peru chips), colour jitter and blur.
   - LR warm-up, which DAFormer Table 4 shows is critical for transformers.
   - Replace DAFormer's single τ=0.968 max-softmax threshold and image-level quality weight with **separate thresholds for water and not-water**, chosen as class-wise percentiles as in CBST (start at about 20% and grow) and tuned on `peru-dev`. With rare water, an image-level "fraction confident" weight is almost always high and does not protect the water class.
   - Drop Rare Class Sampling (meaningless for two classes), FD and HRDA. VFM-UDA Table 3 shows FD and HRDA do not help ViT encoders, and HRDA needs about 22.5 GB (HRDA Table 5).
   - Training budget: DAFormer uses 40k iterations (about 16 h on a 2080 Ti). For 750 target chips this must be scaled down by about 5–10×, with the number of iterations chosen on `peru-dev`.
4. **+ Masked consistency (MIC), as token masking for the ViTs.** The student sees target chips with masked patches or tokens and must match the teacher's full-image pseudo-label.
   - Cost is small: about +3% memory (MIC Table 10).
   - Gains: DAFormer 68.3 → 70.6 (MIC Table 7). For a DINOv2 ViT-B, pseudo-labelling alone gives 72.8 and adding token masking gives 77.1 (VFM-UDA Table 3).
   - TerraMind and Prithvi are MAE-style masked-token models, so token masking is natural for them.
5. **Optional: CoTTA-style stochastic restore and augmentation-averaged teacher.** Use it if self-training drifts or collapses. Note its 58.6 vs source 56.7 on SegFormer (CoTTA Table 5).

**Not recommended:**
- Adversarial alignment (AdaptSegNet / AdvEnt): an extra discriminator, unstable, and superseded by self-training in every later table read.
- ProDA: 4×V100, multi-stage.
- HRDA: memory, and no benefit for 512² chips or ViTs.
- SHOT and other source-free methods: source data is available, and SHOT has no segmentation results.

**Binary-specific risk to control:** pseudo-labelling and entropy objectives can collapse to "all not-water". Mitigations:
- class-wise thresholds
- keeping the supervised source loss in every step
- an optional diversity or prior term (SHOT-style information maximisation)
- monitoring the predicted water fraction on `peru-dev`

**Memory:** for Prithvi-300M, the student, EMA teacher and the masked view's forward pass should fit on a 24 GB 4090 at batch 2–4 of 512² chips. This is **not measured**; check against `docs/feasibility.md`.
