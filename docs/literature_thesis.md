# Literature for the thesis: Peru context, antecedentes, bases teóricas, gap

Compiled 2026-10-16 for §1.1 (context), §2.1 (antecedentes) and §2.2 (bases teóricas).
Binding rules: `docs/WRITING_RULES.md` §3 (antecedentes), §4 (integrity), §8 (citations).
BibTeX for every verified reference: `docs/literature_thesis.bib` (127 entries). Each entry has a
`% verified:` line. Copy entries **one at a time** into `thesis/3_3_BIBLIOGRAFIA/library.bib`, and
only when the entry is cited.

**How this was checked.**
- Four research passes were run, one each for Peru context, Peru/LatAm studies, groups (i)–(iv)
  and bases teóricas.
- Every DOI was resolved on Crossref.
- BibTeX was fetched from doi.org, arXiv or proceedings pages, never typed by hand.
- Spot-checks re-read in the downloaded text on 2026-10-16:
  - INDECI Compendio 2017 national row and Piura peak flow.
  - INDECI 2024 Yaku entry.
  - BCRP S/ 25 655 M.
  - Li et al. 2023, Tables 2–3.
  - Shen et al. 2022 (0.69 / 0.44 / 0.50).
  - Badillo-Rivera 2025, Table 5.
  - PANGAEA 91.42.
  - Breña-Aliaga 2026, Tabla 2.

**Verification tags (per number).**
- **FT**: read in the full text at the stated table/page.
- **FT-note**: carried over from an existing repo note that gives the table/page, not re-read here.
- **ABS**: abstract or metadata only. Do not use in the thesis without reading the table.
- **NO VERIFICADO**: not confirmed; the reason is given.

Pages are journal/printed pages unless marked "PDF p." (arXiv PDF page). Indexing (Scopus/WoS)
was **not** checked for any item. Check it before claiming "revista indexada".

Working copies of the raw research notes, with more detail per item:
- Scratchpad (session-only, not in the repo):
  `/private/tmp/claude-501/-Users-gersongarrido-Projects-peru-flood-segmentation/81330184-5f3b-491a-8eb6-a5779113edd6/scratchpad/lit/`
  (`A_peru_context.md`, `B5_peru_latam.md`, `B14_antecedentes.md`, `C_bases.md`)
- Earlier repo notes: `docs/literature/uda_earth_observation.md`,
  `docs/literature/uda_segmentation_methods.md`, `docs/literature_phaseD.md`.

**Setting described here (matches the repo, not the shorter task brief).**
- Models are trained on Peru only (`peru-train`, UNOSAT 2023 labels).
- Sen1Floods11-trained models serve as the **source-only** comparison.
- Phase D adds unsupervised adaptation (histogram matching, FDA, AdaBN, self-training) from
  Sen1Floods11 to unlabelled Peru imagery. The Peru-trained model is the upper bound.

---

## A. Peru flood context (§1.1)

### A.1 Niño Costero 2017

**Rule:** INDECI revised its totals between editions. Cite **one edition per sentence**. Never mix
*damnificados* and *afectados*, which are separate INDECI categories.

National totals by edition:

| Variable | INDECI, *Compendio estadístico del INDECI 2017: Gestión reactiva* (Dec 2017; data at 04-jul-2017, "95.5 % procesado"), pp. 13–16 | INDECI, *Compendio estadístico del INDECI 2023* (1st ed., Dec 2023), p. 52 = *Compendio … al primer semestre 2024* (Oct 2024), p. 90 |
|---|---|---|
| Damnificados | 285 453 | 283 137 |
| Afectados | 1 454 051 | 1 644 879 |
| Fallecidos | 138 | 169 |
| Heridos / desaparecidos | 459 / 18 | 505 / 19 |
| Viviendas | 63 802 destruidas; 350 181 afectadas (p. 14) | 37 409 colapsadas; 28 533 inhabitables; 381 076 afectadas |
| Carreteras (km) | 4 778 destruidas; 13 311 afectadas (p. 15) | — |
| Puentes destruidos | 449 (p. 15) | — |
| Cultivos (ha) | 91 906 afectadas; 39 705 perdidas (p. 15) | — |
| Tag | FT (spot-checked) | FT |

Notes on these totals:
- The housing categories differ between the editions ("destruidas" vs "colapsadas + inhabitables").
  Do not compare them.
- The figure "1 782 316 personas entre damnificadas y afectadas" comes from Compendio 2017,
  p. 13 (FT).
- The "234 mil km" of roads quoted in the text of p. 14 refers to **rural roads**. Cite the table
  instead.

Period and scope (Compendio 2017):
- The rains ran from the fourth week of December 2016 to 31 May 2017 (p. 13).
- 13 of the 24 departments plus Callao were declared in emergency (p. 13).
- 879 districts and 109 provinces were declared in emergency (p. 16). FT.

**Piura and Lambayeque** (Compendio 2017 only, pp. 13–15; FT):

| | Piura | Lambayeque |
|---|---|---|
| Damnificados / afectados | 89 709 / 375 265 | 44 619 / 138 336 |
| Fallecidos | 18 | 9 |
| Viviendas destruidas / afectadas | 21 412 / 83 957 | 10 051 / 30 285 |
| Cultivos afectados / perdidos (ha) | 15 342 / 8 733 | 4 009 / 2 464 |

- Piura river peak, Compendio 2017, p. 28: *"el rio Piura el 27 de marzo 2017, tuvo su máxima
  avenida con un caudal de 3,800 m3/seg por encima de los datos históricos, afectando la
  totalidad del bajo Piura"*. FT, spot-checked. This is INDECI's figure; no SENAMHI/ANA gauge
  record was opened.
- Use only the 2017-edition national row for department shares.

**Economic losses.** These are estimates. Keep them separate from the reconstruction budget.
- **US$ 3.1 billion (2017 US$).** Source: French, Mechler, Arestegui, MacClune & Cisneros
  (2020), *IJDRR* 47, 101539, doi:10.1016/j.ijdrr.2020.101539, **Table 1**. The table's
  sources are INDECI 2016/2017. FT (accepted manuscript; cite the table).
- **Macroconsult damage by category**, as reproduced by INDECI in Compendio 2023, p. 52:
  - carreteras US$ 1 259 M
  - viviendas US$ 1 123 M
  - puentes US$ 253 M
  - cultivos US$ 243 M
  - II.EE. US$ 171 M
  - canales US$ 38 M
  - caminos rurales US$ 31 M
  - EE.SS. US$ 7 M

  FT, but secondary (the Macroconsult original was not opened). INDECI prints no total. Our own
  sum is US$ 3 125 M; do not present it as INDECI's figure.
- **BCRP**, *Reporte de Inflación junio 2017*, p. 7: Q1-2017 growth of 2.1 % reflected the
  Niño Costero **and** the fall in public and private investment linked to corruption
  investigations. The 2017 projection went from 3.5 % to 2.8 % (pp. 8, 37). FT via Wayback.
  **Not** an estimate of the GDP impact of the Niño Costero alone.
- **NO VERIFICADO:**
  - A BCRP point estimate of the GDP impact ("0.8 pp", snippet only; the BCRP site blocks
    scripted access).
  - World Bank, CEPAL and UNDP loss estimates (no document found).

**Reconstruction budget (not losses).** BCRP, *Reporte de Inflación diciembre 2017*, Recuadro 1,
p. 66 (FT, spot-checked):

> "El gasto total, ascendente a S/ 25 655 millones, comprende S/ 5 446 millones para prevención
> y desarrollo urbano, S/ 19 759 millones para reconstrucción y S/ 450 millones para
> fortalecimiento institucional … representa el 11 por ciento del PBI de las regiones afectadas
> y 4 por ciento del PBI nacional."

- The Plan was approved by DS N.° 091-2017-PCM. The DS annex itself was not opened.
- The August 2017 draft gives S/ 23 338 M. Do not cite the draft as the approved figure.

**Physical framing.** Ramírez & Briones (2017), *Int J Disaster Risk Sci* 8(4):489–492,
doi:10.1007/s13753-017-0151-8, FT:
- p. 489: the event *"developed rapidly with no warning"* and *"Such a localized El Niño was not
  documented since 1925"*.
- p. 490: impacts were similar to those of 1982–83 and 1997–98.

### A.2 Ciclón Yaku and the March–April 2023 rains

- **SENAMHI**, *Efecto del ciclón Yaku y su recorrido* (15-mar-2023), PDF p. 1:
  *"'Yaku', es un ciclón de características tropicales no organizado, de baja presión
  atmosférica…"*. PDF p. 4: *"una circulación inusual en el Pacífico"*. FT.
  URL: https://sinia.minam.gob.pe/sites/default/files/sinia/archivos/public/docs/reporte_-_ciclon_yaku_f_f_.pdf
- **ENFEN**, *Informe Técnico ENFEN Año 9 N.° 7* (09-may-2023), Tabla 2, PDF p. 10:
  - ICEN mar-2023 = 1.51 ("Cálida Moderada"); ICENtmp abr-2023 = 2.21 ("Cálida Fuerte").
  - ONI mar-2023 = −0.16 (neutral).
  - So 2023 was a coastal event, like 2017. FT.
- **Damage attributed to Yaku by INDECI**, *Compendio estadístico de la gestión reactiva al
  primer semestre 2024* (Oct 2024), Tabla 6, p. 86 (FT, spot-checked):
  - Event dates: 4–20 March 2023, with weaker rains continuing for weeks.
  - 8 departments: Áncash, Cajamarca, Ica, La Libertad, Lambayeque, Lima, Piura, Tumbes.
  - People: 568 702 afectados, 107 780 damnificados, 590 lesionados, 6 desaparecidos,
    44 fallecidos.
  - Material: 43 037 viviendas destruidas, 223 945 afectadas; 397 puentes destruidos.
  - **Caveat:** no data cut-off is given, and there is no Piura/Lambayeque breakdown. Write
    "INDECI atribuye a Yaku…".
- **Season-wide counts** (all of the 2022–23 rainy season, all departments; different scope):
  - Source: Defensoría del Pueblo, *Emergencia climática: Gestión de riesgo, Reporte n.° 9*,
    updated 30-may-2023, source COEN-INDECI, FT.
  - National: 99 220 damnificados, 610 775 afectados.
  - Piura: 24 849 / 252 355. Lambayeque: 33 579 / 87 080.
  - URL: https://cdn.www.gob.pe/uploads/document/file/4654507/A-NIVEL-NACIONAL-Y-OTROS-DEPARTAMENTOS-31.05.pdf
- **Emergency declaration.** DS 043-2023-PCM (26-mar-2023) declared a national emergency
  *"por desastre de gran magnitud"* in **Lambayeque, Piura y Tumbes**. The departments are named
  in the first considerando of DS 065-2023-PCM (El Peruano, 25-may-2023). FT.
  URL: https://busquedas.elperuano.pe/dispositivo/NL/2181041-2
- **UNOSAT FL20230321PER** (source of our labels). Sentinel-2, CC BY-SA, both products flagged
  *"preliminary … not yet been validated in the field"*:
  - 22-mar-2023: flood extent **985.9 km²**, analysed area 15 790 km², exposed population about
    13 460.
  - 1-apr-2023: flood extent **992.8 km²**, analysed area 13 115 km².
  - Lambayeque province had about 400 km² of flood and Sechura about 566–604 km².
  - Source: UNOSAT xlsx table on HDX, https://data.humdata.org/dataset/69d4820c-c1bd-4938-9409-877e06c284dc
    and https://data.humdata.org/dataset/3e1cf0c7-5e87-43e6-b04f-2d1c3f982061. FT.
  - ⚠️ UNOSAT's own metadata disagree:
    - product code: EQ20230321PER / EQ20220321PER / FL20230321PER;
    - 22-Mar analysed area: 15 827 vs 15 790 km²;
    - acquisition time: "07:55 UTC" vs "10:36 local".

    Mention this in the data card / label-QA section.
- **International Charter.**
  - Activation **811**, "Cyclone Yaku in Peru": activated 2023-03-14; requested by USGS on
    behalf of INDECI and CONIDA; project manager from CONIDA.
    https://disasterscharter.org/activations/flood-large-in-peru-activation-811- (FT).
  - 2017: Activation **523**, "Flood in Peru": activated 2017-03-31 on behalf of INDECI;
    project management by CONIDA. The URL slug says "call-602".
    https://disasterscharter.org/activations/flood-in-peru-call-602- (FT).
- **Copernicus EMS EMSR199**, "Floods in Peru" (source of our 2017 labels):
  - Activated 22/03/2017 10:08; 21 AOIs, 45 products.
  - Piura AOIs: 08, 10–12, 15–16. **No Lambayeque AOI.**
  - URL: https://mapping.emergency.copernicus.eu/activations/EMSR199/ (FT).
- **NO VERIFICADO:**
  - IFRC DREF for Yaku (none found; do not cite one).
  - OCHA/UNICEF flash updates (ReliefWeb returned 403).

### A.3 Recurrence of El Niño floods on the north coast

- **INDECI, Compendio 2017, p. 11**, table "Intensidad del Fenómeno El Niño en el Perú:
  1578–2017" (FT):
  - 44 events: 17 débiles, 14 moderados, 5 fuertes, 8 extraordinarios.
  - The extraordinary events are 1578, 1720, 1878, 1891, 1925, 1982-83, 1997-98 and 2017.
  - ⚠️ The adjoining text says "7 extraordinarios"; the table has 8, counting 2017. Cite the
    table and note the mismatch.
- **French et al. (2020), Table 1** (FT), INDECI data:

  | | Deaths | Affected | Losses |
  |---|---|---|---|
  | 1982–83 | 512 | 1.27 M | US$ 3.28 bn (1998 US$) |
  | 1997–98 | 366 | 531 104 | US$ 3.5 bn (1998 US$) |
  | 2017 | 138 | 1.45 M | US$ 3.1 bn |

- **ICEN categories** (ENFEN, as reproduced in Compendio 2017, p. 12; FT):
  - débil > 0.4–1.0; moderada > 1.0–1.7; fuerte > 1.7–3.0; extraordinaria > 3.0.
  - Region Niño 1+2. Introduced by ENFEN in 2012.
- **Takahashi & Martínez (2019)**, *Climate Dynamics* 52:7389–7415,
  doi:10.1007/s00382-017-3702-1. The 1925 "coastal" El Niño as a distinct type of very strong
  event. **ABS** (qualitative use only).
- **NO VERIFICADO:**
  - The final ENFEN magnitude of 2017 (sources conflict: "moderada" in the 16-mar forecast vs
    "weak" ICEN per Ramírez & Briones).
  - CAF (2000) 1997–98 loss figures (volume not located).

### A.4 Peruvian institutional use of satellite flood mapping

- **CONIDA** manages the Charter activations for Peru (523 in 2017, 811 in 2023). It acquires
  PerúSAT-1 imagery and delivers it to INDECI/COEN. Sources:
  - Andina, *"Emergencia por lluvias: satélite PerúSAT-1 registra la magnitud del ciclón Yaku"*,
    14-mar-2023. https://andina.pe/agencia/noticia-emergencia-lluvias-satelite-perusat1-registra-magnitud-del-ciclon-yaku-932744.aspx
  - Andina, 13-feb-2017. https://andina.pe/agencia/noticia-observa-las-imagenes-dejan-las-inundaciones-desde-satelite-perusat1-653728.aspx

  Both FT. Neither gives flood-area figures.
- **INGEMMET**, technical report "A6788" (Lambayeque–Cajamarca, Niño Costero 2017; hosted on
  CENEPRED SIGRID):
  - Geomorphological analysis by *"interpretación de fotos áreas e imágenes satelitales de alta
    resolución"* (p. 1). FT.
  - Visual interpretation, not automated mapping.
- **CENEPRED**, *Escenarios de riesgos por lluvias intensas 2018–2019*: no satellite flood
  maps found in the document (grep). **Do not claim** that CENEPRED scenarios use satellite flood
  extents.
- **Defensible statement for §1.1:** *"en los documentos revisados"*, the quantitative
  flood-extent maps of 2017 and 2023 come from international mechanisms (Copernicus EMS, the
  Charter with CONIDA as project manager, UNOSAT). No systematic national product was found.
- **NO VERIFICADO:**
  - ANA (fajas marginales, flood maps).
  - INDECI GeoPortal/SINPAD layers.
  - CENEPRED 2023 scenario (snippet only).
  - CISMID-UNI 2023 work (search result only).

---

## B. Antecedentes (§2.1)

Rules applied:
- Last 5 years (2021–2026) unless marked **[clásico]** or **[pre-2021, justified]**.
  - ⚠️ Counted from a 2026–27 defense date, the early-2021 items fall at or just outside the
    5-year window: Mateo-Garcia (Mar 2021), Konapala (Aug 2021), LoveDA (2021), Patel (Nov 2021).
    Treat them as borderline and justify each in one sentence. Do not relabel the rule.
- Ordered chronologically, oldest to newest. Within a year, the order is approximate
  (publication date where known).
- Groups:
  - (i) DL flood segmentation with S1/S2
  - (ii) geospatial foundation models (GFMs)
  - (iii) domain shift / generalization / UDA
  - (iv) calibration / uncertainty
  - (v) Peru / Latin America
- Peruvian items are marked 🇵🇪. Theses are "no indexada".

### B.1 Candidate list (31, chronological)

1. **Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020).** Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. In *2020 IEEE/CVF CVPR Workshops* (pp. 835–845). https://doi.org/10.1109/CVPRW50498.2020.00113 · (i) · workshop · **[clásico: our benchmark and official splits]**
   - Method: 446 hand-labelled 512×512 chips from 11 events. FCNN on S1 with weak vs hand labels, against Otsu-VH.
   - Result: Bolivia, mean per-chip IoU: hand-labelled .3524, S1-weak .3871 (Table 2, p. 6). FT.
   - ⚠️ Per-chip mean IoU from S1; not comparable with our pooled S2 IoU.
   - Relevance: source dataset; origin of the held-out-region protocol.

2. 🇵🇪 **Gonzalez, J., Sankaran, K., Ayma, V., & Beltran, C. (2020).** Application of semantic segmentation with few labels in the detection of water bodies from PERUSAT-1 satellite's images. *ISPRS Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLII-3/W12-2020*, 183–187 (IEEE LAGIRS 2020). https://doi.org/10.5194/isprs-archives-XLII-3-W12-2020-183-2020 · (v) · conference · **[pre-2021, justified: the clearest Peruvian DL water-segmentation result; PUCP]**
   - Method: U-Net vs HR→VHR knowledge transfer on PerúSAT-1 (680/170/95 split) plus Sentinel-2 patches.
   - Result: U-Net F1 0.9317 / IoU 0.9124 on the 95-image test set (Table 1, p. 186). FT.
   - Caveat: permanent water bodies, not floods. The companion arXiv paper (Gonzalez et al., 2019, arXiv:1912.00957) found that adding Sentinel-2 data hurt (Table 1, p. 4).
   - Relevance: Peruvian precedent; distribution shift between sensors.

3. **Konapala, G., Kumar, S. V., & Khalique Ahmad, S. (2021).** Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning. *ISPRS J. Photogramm. Remote Sens., 180*, 163–173. https://doi.org/10.1016/j.isprsjprs.2021.08.016 · (i) · journal
   - Method: the same U-Net with 32 S1/S2/DEM input combinations on Sen1Floods11, with country-held-out folds.
   - Result: median F1 0.62 for S1 vs 0.88–0.90 for S2 configurations (Table 4, p. 167). FT-note.
   - Relevance: supports S2 as the input; U-Net on Sen1Floods11.

4. **Mateo-Garcia, G., Veitch-Michaelis, J., Smith, L., Oprea, S. V., Schumann, G., Gal, Y., Baydin, A. G., & Backes, D. (2021).** Towards global flood mapping onboard low cost satellites with machine learning. *Scientific Reports, 11*, 7249. https://doi.org/10.1038/s41598-021-86650-z · (i) · journal
   - Method: WorldFloods v1; S2 U-Net vs NDWI thresholds.
   - Result: U-Net IoU 72.42 vs NDWI(−0.22) 65.12 (Table 2). FT-note.
   - Relevance: optical S2 flood segmentation at scale; index baseline.

5. **Wang, J., Zheng, Z., Ma, A., Lu, X., & Zhong, Y. (2021).** LoveDA: A remote sensing land-cover dataset for domain adaptive semantic segmentation. *NeurIPS Datasets and Benchmarks Track, 1*. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4e732ced3463d06de0ca9a15b6153677-Abstract-round2.html · (iii) · conference
   - Method: rural↔urban UDA benchmark with adversarial and self-training methods (DeepLabV2).
   - Result: Water IoU Urban→Rural: source-only 49.42 → IAST 64.49 (oracle 65.89) (Table 6, PDF p. 8). FT.
   - Relevance: the source-only / UDA / oracle protocol we copy. Self-training beats adversarial alignment.

6. **Patel, C., Sharma, S., Pasquarella, V. J., & Gulshan, V. (2021).** *Evaluating self and semi-supervised methods for remote sensing segmentation tasks* (arXiv:2111.10079) [Preprint]. https://arxiv.org/abs/2111.10079 · (iii) · preprint
   - Method: FixMatch / SimCLR on Sen1Floods11 S1 with a split shifted by event.
   - Result: domain-shifted test, 100 % labels: supervised 67.77 ± 0.8 vs FixMatch 71.56 ± 0.6 ("Water mIoU", Table IV, PDF p. 11). FT.
   - Relevance: unlabelled flood imagery; model selection on disjoint regions.

7. 🇵🇪 **Quintana-Ortiz, J. M., Moreno Santillan, R. D., & Pasapera-Gonzalez, J. J. (2021).** Estimation of flooded areas during the El Niño Costero 2017 event using multisensor satellite data. Case study: Lower Piura watershed (Peru). In *2021 IEEE INCAS* (pp. 1–4). https://doi.org/10.1109/INCAS53599.2021.9666917 · (v) · conference
   - Method: per-sensor flood masks from PerúSAT-1, SPOT, KOMPSAT-3A, S1 and Landsat-8. ABS.
   - Result: none in the abstract. **NO VERIFICADO** (paywalled; obtain through UNI).
   - Relevance: the same event and area as `peru-2017-test`; the co-author J. J. Pasapera-Gonzalez may be the CONIDA project manager named on the Charter 811 page (our inference, not verified).

8. 🇵🇪 **Shen, M., Zhang, F., & Wu, C. Y. (2022).** Flood inundation extraction based on decision-level data fusion: A case in Peru. *ISPRS Annals, X-3/W1-2022*, 133–140. https://doi.org/10.5194/isprs-annals-X-3-W1-2022-133-2022 · (v) · conference · study of Peru by foreign authors
   - Method: UNet++ **trained on Sen1Floods11** and applied to the Piura river flood of 27-feb-2017; S1/S2 decision-level fusion.
   - Result:
     - Water IoU: fusion 0.69, S1 alone 0.56 (Table 3, p. 137). FT, spot-checked.
     - Flood IoU against their manual mask: fusion 0.50; **the Copernicus EMS product 0.44** (text, p. 138). FT.
   - ⚠️ The IoU formula is printed with TN (p. 136), probably a typo. The EMS product is not named.
   - Relevance: the closest Peruvian precedent, a Sen1Floods11 model transferred to Piura. Also a warning about how well EMS maps agree with other references.

9. **Li, W., Lee, H., Wang, S., Hsu, C.-Y., & Arundel, S. T. (2023).** Assessment of a new GeoAI foundation model for flood inundation mapping. In *Proc. 6th ACM SIGSPATIAL GeoAI Workshop* (pp. 102–109). https://doi.org/10.1145/3615886.3627747 · (ii)/(iii) · workshop
   - Method: Prithvi-100M vs SegFormer vs U-Net from scratch on 6 S2 bands; mean of 10 runs.
   - Result, water IoU (Tables 2–3, arXiv v4 p. 6; FT, spot-checked):
     - Sen1Floods11 test: U-Net 84.03 vs Prithvi 81.98.
     - Bolivia: Prithvi 76.62 vs U-Net 70.57.
   - New finding: this **is** the "U-Net Base [19]" row of Prithvi-CAFE. It closes an open item in `docs/published_comparison.md` (that file was not edited).
   - Relevance: the CNN wins in distribution and the GFM wins on an unseen region. This is our RQ1.

10. **Jakubik, J., Roy, S., Phillips, C. E., … Ramachandran, R. (2023).** *Foundation models for generalist geospatial artificial intelligence* (arXiv:2310.18660) [Preprint]. https://arxiv.org/abs/2310.18660 · (ii) · preprint
    - Method: Prithvi-EO-1, a temporal ViT-MAE on HLS.
    - Result: Sen1Floods11 test water IoU: pretrained 82.99 vs not pretrained 82.97 at 500 epochs (Table 2, PDF p. 17). FT.
    - ⚠️ Its "baseline 24.21" is Bonafilia's per-chip flood-water S1 cell, so that comparison is not homogeneous.
    - Relevance: origin of Prithvi. Pretraining gives little after long fine-tuning.

11. **Shastry, A., Carter, E., Coltin, B., Sleeter, R., McMichael, S., & Eggleston, J. (2023).** Mapping floods from remote sensing data and quantifying the effects of surface obstruction by clouds and vegetation. *Remote Sensing of Environment, 291*, 113556. https://doi.org/10.1016/j.rse.2023.113556 · (i) · journal
    - Method: CNN on WorldView-2/3, compared with USGS FIM hydraulic extents.
    - Result: validation precision 98.2 % / recall 93.7 % (p. 7); underprediction against FIM, about 74 % of it from vegetation and 9 % from clouds (p. 8). FT-note.
    - Relevance: a structural limit of optical mapping; goes in the limitations.

12. **Portalés-Julià, E., Mateo-García, G., Purcell, C., & Gómez-Chova, L. (2023).** Global flood extent segmentation in optical satellite images. *Scientific Reports, 13*, 20316. https://doi.org/10.1038/s41598-023-47595-7 · (i) · journal
    - Method: WorldFloods v2 (about 75 k S2 L1C tiles, more than 180 events); multi-output U-Net.
    - Result: DL beats MNDWI (text, p. 8). The numbers are only in Fig. 6 and Suppl. Table 1. **NO VERIFICADO** (not transcribed).
    - Relevance: S2 flood segmentation at scale; MNDWI baseline.

13. 🇵🇪 **Rodríguez De La Cruz, J. A., & Moreno Arqque, M. (2023).** *Detección y mapeo de inundaciones mediante imágenes SAR, usando el método K-Means Clustering … Caso: Cuenca Bajo Piura* [Tesis de pregrado, UNMSM]. https://hdl.handle.net/20.500.12672/19587 · (v) · tesis, no indexada
    - Method: S1 (3-feb-2017 and 4-apr-2017) K-Means.
    - Result: "precisión global de 0,987 … Kappa de 0,850". ABS (sample design unknown).
    - Relevance: unsupervised Peruvian baseline on our 2017 area.

14. **Bountos, N. I., Sdraka, M., Zavras, A., Karasante, I., Karavias, A., Herekakis, T., Thanasou, A., Michail, D., & Papoutsis, I. (2024).** Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping. In *NeurIPS 37, Datasets and Benchmarks* (pp. 38105–38121). https://doi.org/10.52202/079017-1204 · (i) · conference
    - Method: manually annotated S1 dataset of 43 events; CNNs vs ViTs vs MAE "FloodViT".
    - Result, GRD: UNet-ResNet50 F1-flood 80.12 vs FloodViT 78.70 (Table 2, PDF p. 8). FT.
    - Relevance: a CNN matches or beats an MAE ViT on floods (SAR).

15. **Szwarcman, D., Roy, S., Fraccaro, P., … Bernabe Moreno, J. (2024).** *Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications* (arXiv:2412.02732) [Preprint]. https://arxiv.org/abs/2412.02732 · (ii) · preprint
    - Method: ViT-MAE 300M/600M with time/location embeddings, pretrained on HLS.
    - Result: Sen1Floods11 test, 300M: water IoU 82.2 (0.6) / mIoU 89.7 (0.3), 10 seeds (Table IV, PDF p. 12). FT.
    - Relevance: foundation model #2 and its reproduction target (ours: 82.03, `docs/published_comparison.md`).

16. **Englert, B. B., Piva, F. J., Kerssies, T., de Geus, D., & Dubbelman, G. (2024).** Exploring the benefits of vision foundation models for unsupervised domain adaptation. In *2024 IEEE/CVF CVPR Workshops* (pp. 1172–1180). https://doi.org/10.1109/CVPRW63382.2024.00124 · (iii) · workshop
    - Method: VFM-UDA, DINOv2 ViT with EMA self-training and token masking.
    - Result: GTA→Cityscapes ViT-B: no adaptation 62.9 → 77.1 mIoU (Table 3, PDF p. 6). FT.
    - Relevance: self-training works on large pretrained ViTs; the basis of our recipe for TerraMind and Prithvi.

17. **Pech-May, F., Aquino-Santos, R., Álvarez-Cárdenas, O., Lozoya Arandia, J., & Ríos-Toledo, G. (2024).** Segmentation and visualization of flooded areas through Sentinel-1 images and U-Net. *IEEE JSTARS, 17*, 8996–9008. https://doi.org/10.1109/JSTARS.2024.3387452 · (v, Mexico) · journal
    - Method: S1 U-Net on Sen1Floods11 with **the authors' own random 70/15/15 split**.
    - Result: IoU 0.7302, F1 0.8441 (Table IV, manuscript p. 9). FT.
    - Caveat: Tabasco is shown only visually; **no local accuracy is reported**.
    - Relevance: a LatAm group transfers a Sen1Floods11 model to its region without measuring local skill. This is the gap our Peru labels fill.

18. 🇵🇪 **Perez Torres, W. I. (2024).** *Segmentación automatizada de lagunas en la cordillera del Vilcanota, región Cusco, un enfoque basado en imágenes satelitales y aprendizaje profundo* [Tesis de pregrado, UNSAAC]. https://hdl.handle.net/20.500.12918/10182 · (v) · tesis, no indexada
    - Method: Landsat; UNetFFT vs U-Net vs NDWI+Otsu; 10 repetitions.
    - Result: MIoU 85.94 ± 1.38 (UNetFFT32) vs 82.44 ± 3.36 (UNet32) vs 46.33 (NDWI+Otsu) (Tabla 4.3, p. 113). FT.
    - Relevance: recent Peruvian DL water segmentation with mean ± sd and an index baseline. Lagoons, not floods.

19. 🇵🇪 **Alvan Romero, N., & Suarez Alayza, W. A. (2024).** *Emergency mapping for flood events using satellite imagery (optical and/or synthetic aperture radar) and benchmarking between open source and commercial geospatial analysis tools* [Preprint]. Preprints.org. https://doi.org/10.20944/preprints202407.1784.v1 · (v) · **preprint, not peer-reviewed** (no later journal version found on 2026-10-16)
    - Method: Yaku, 6–10 March 2023, La Leche river (Íllimo, Lambayeque); S1, COSMO-SkyMed, S2, PerúSAT-1; change detection.
    - Result: optical imagery unusable because of clouds; Jaccard agreement S1 "33%" / COSMO-SkyMed "38%". ABS.
    - Relevance: the only study found on the Yaku flood in Lambayeque; documents the cloud limits of S2.

20. **Garshasbi, M., Alizadeh, H., Mojaradi, B., Saadatpour, M., & Zarei, E. (2025).** Uncertainty-aware flood inundation mapping with a Bayesian deep learning framework using SAR imagery. *IEEE JSTARS, 18*, 26716–26726. https://doi.org/10.1109/JSTARS.2025.3610403 · (iv) · journal
    - Method: Bayesian (variational) U-Net on S1 with Sen1Floods11.
    - Result: test F1 / IoU: BHED 0.8013 / 0.6685 vs U-Net 0.7672 / 0.6224 (Table II, p. 26721). FT-note.
    - ⚠️ No ECE or reliability diagram is reported.
    - Relevance: closest flood UQ antecedent; it leaves calibration unmeasured.

21. **Portalés-Julià, E., Mateo-García, G., & Gómez-Chova, L. (2025).** Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets. *Remote Sensing of Environment, 328*, 114882. https://doi.org/10.1016/j.rse.2025.114882 · (i)/(iii) · journal
    - Method: cross-dataset evaluation; U-Nets vs MNDWI.
    - Result, all-water IoU (Table 3):
      - Sen1Floods11: multimodal U-Net 77.50 vs MNDWI 74.39.
      - UNOSAT (Africa/Middle East 2020): 83.55 vs 82.47.
    - Status: FT-note only. Not re-read on 2026-10-16 (CAPTCHA/403). **Re-check in the PDF before citing.**
    - Relevance: models trained on one dataset transfer poorly, and MNDWI comes close on unseen data.

22. **Jakubik, J., Yang, F., Blumenstiel, B., … Longépé, N. (2025).** TerraMind: Large-scale generative multimodality for Earth observation. In *2025 IEEE/CVF ICCV* (pp. 7383–7394). https://doi.org/10.1109/ICCV51701.2025.00693 · (ii) · conference
    - Method: any-to-any multimodal generative GFM.
    - Result: v1-B, S-2, full fine-tuning: Sen1Floods11 water IoU 82.26 / mIoU 89.70 (Table 8, arXiv v5 PDF p. 8). FT. Split and S2 level not stated.
    - Relevance: foundation model #1 (ours: 81.85).

23. 🇵🇪 **Badillo-Rivera, E., Santiago, R., Poma, I., Chavez, T., Arroyo-Paz, A., Aucahuasi-Almidon, A., Hinostroza, E., Segura, E., Eyzaguirre, L., León, H., & Virú-Vásquez, P. (2025).** Flood susceptibility mapping in El Niño phenomenom integrating multitemporal radar analysis, GIS and machine learning techniques, Piura river basin, Peru. *Frontiers in Environmental Science, 13*, 1672107. https://doi.org/10.3389/fenvs.2025.1672107 · (v) · journal
    - Method: S1 NDFI flood inventory (2017); RF/SVM susceptibility with 182 point samples.
    - Result (Table 5, p. 13; FT, spot-checked):
      - Best test AUC: RF-q 0.932.
      - RF-WoE: test 0.907 / CV 0.936.
    - ⚠️ The abstract gives RF-WoE a training AUC of 0.988; the table gives 0.998. This resolves the earlier disagreement noted in `docs/REFERENCES.md`.
    - Relevance: the most recent peer-reviewed ML study of Piura 2017. Susceptibility, not extent.

24. **Gonzalez-Calabuig, M., Cohrs, K.-H., Nedungadi, V., Osika, Z., Cartuyvels, R., Knoblauch, S., Massant, J., Nath, S., Ebel, P., & Sitokonstantinou, V. (2025).** *SHRUG-FM: Reliability-aware foundation models for Earth observation* (arXiv:2511.10370) [Preprint; accepted at CVPR EarthVision 2026 per the author comment]. https://arxiv.org/abs/2511.10370 · (iv)/(ii) · preprint
    - Method: abstention for GFMs combining OOD detection and uncertainty.
    - Result: WorldFloods flood task AURC 0.115 ± 0.001 vs mutual information 0.159 (Table 2, PDF p. 8). FT.
    - Relevance: GFM reliability on floods; complements ECE.

25. **Marsocci, V., Jia, Y., Le Bellier, G., … Nascetti, A. (2026).** PANGAEA: Assessing geospatial foundation models capabilities through a global and inclusive benchmark. *IEEE Geoscience and Remote Sensing Magazine, 14*(1), 245–285. https://doi.org/10.1109/MGRS.2025.3628194 · (ii) · journal (arXiv:2412.04204)
    - Method: GFMs with **frozen encoders** + UPerNet vs a U-Net from scratch.
    - Result: Sen1Floods11 mIoU: **UNet 91.42** (best), CROMA 90.89, Prithvi(-EO-1) 90.37 (Table 5, arXiv v2 PDF p. 13). FT, spot-checked.
    - Relevance: independent evidence that the U-Net matches or beats GFMs in distribution. Caveat: frozen encoders.

26. **Kaushik, S., Maurya, L., Tellman, B., & Zhang, Z. (2026).** Assessing geo-foundational models for flood inundation mapping: Benchmarking models for Sentinel-1, Sentinel-2, and PlanetScope. *IEEE JSTARS, 19*, 5649–5665. https://doi.org/10.1109/JSTARS.2026.3656855 · (ii)/(iii) · journal
    - Method: Prithvi 2.0, Clay, DOFA and U-Net on FloodPlanet (19 sites, including Bolivia, Colombia and Paraguay).
    - Result: S2: Clay 0.72, U-Net 0.67, Prithvi 0.64 (Table III, arXiv PDF p. 7; header ambiguous IoU/mIoU). FT. The leave-one-region-out averages are ABS.
    - Relevance: on S2, U-Net ≈ Prithvi; includes South American sites.

27. **Kaushik, S., Maurya, L., & Tellman, B. (2026).** *Prithvi-Complimentary Adaptive Fusion Encoder (CAFE): Unlocking full-potential for flood inundation mapping* (arXiv:2601.02315) [Preprint; CV4EO @ WACV 2026 per the author comment]. https://arxiv.org/abs/2601.02315 · (ii) · preprint
    - Method: Prithvi + CNN branch with adaptive fusion.
    - Result: Sen1Floods11 test IoU 83.41 vs U-Net Base 84.03; Bolivia 81.37 vs 70.57 (Table 1, PDF p. 6). FT.
    - ⚠️ The TerraMind 82.90 row is uncited. The Bolivia rows mix studies.
    - Relevance: latest SOTA table; the CNN still leads in distribution.

28. **Chen, W., Wang, C., Zhao, Z., & Li, X. (2026).** Improving cross-event generalization for SAR-based coastal inundation mapping via test-time domain adaptation. *JGR: Machine Learning and Computation, 3*(4), e2025JH001145. https://doi.org/10.1029/2025JH001145 · (iii) · journal
    - Method: CIM-TTDA, per-event test-time self-training (EMA teacher, CoTTA restore) on S1.
    - Result, IoUp (Table 4):
      - U-Net-EffNet-B0: 0.636 → 0.803.
      - CBST: 0.548 → 0.713.
      - Prithvi-EO-2.0-300M-TL, not adapted: 0.478.
    - Status: FT-note, not re-read.
    - Relevance: the closest UDA/TTA flood antecedent. SAR only, and the GFM is never adapted.

29. **Kolluru, V., Shinde, R., Marouane, A., Helbling, C., Shah, D., Drew, O., Gurung, I., Maskey, M., & Ramachandran, R. (2026).** *Land cover and flood type govern the detection limits of satellite-based flood mapping across diverse global flood events* (arXiv:2606.07780) [Preprint]. https://arxiv.org/abs/2606.07780 · (iii)/(ii) · preprint
    - Method: the public Prithvi-EO-2.0 Sen1Floods11 checkpoint applied zero-shot to 19 OOD events.
    - Result: mean IoU 0.46 against Copernicus GFM; recall 0.90 vs precision 0.49 (text, PDF p. 7; per-event Table 3, PDF p. 29). FT.
    - Relevance: a source-only GFM loses much skill and over-predicts. This parallels our source-only Peru result.

30. 🇵🇪 **Breña-Aliaga, J. C., & Cruz-Machacuay, J. (2026).** Estimación de umbrales y dinámica de inundaciones integrando Random Forest, el índice SDWI de Sentinel-1, el modelo HAND y datos SAR en la cuenca del río Tumbes. *Ingeniería del Agua, 30*(3), 211–228. https://doi.org/10.4995/ia.25917 · (v) · journal
    - Method: RF + SDWI + HAND on S1, against a PlanetScope reference; nine validation events, 2023–2025.
    - Result: fusion IoU 69.02 ± 3.46 %, Kappa 0.7676 (Tabla 2, p. 221). FT, spot-checked. The abstract gives a different OA and F1 (91.0 % and 0.80); cite the table.
    - Relevance: Peruvian north-coast ML flood mapping including 2023, with an IoU comparable in kind.

31. **Banerjee, A., & Daou, D. (2026).** *Explainable flood segmentation on Sentinel-1 SAR imagery using CNN and transformer architectures* (arXiv:2606.16302) [Preprint]. https://arxiv.org/abs/2606.16302 · (iv) · preprint
    - Method: U-Net vs SegFormer on Sen1Floods11 (S1), with ECE and XAI metrics.
    - Result: test ECE U-Net 0.0675 vs SegFormer-b2 0.0694 (Table 5, PDF p. 12). FT.
    - Relevance: the only ECE-on-Sen1Floods11 comparison of CNN vs transformer found (S1).

**Counts by group** (primary group, 31 total):
- (i) 7: items 1, 3, 4, 11, 12, 14, 21
- (ii) 7: items 9, 10, 15, 22, 25, 26, 27
- (iii) 5: items 5, 6, 16, 28, 29
- (iv) 3: items 20, 24, 31
- (v) 9: 8 Peruvian (items 2, 7, 8, 13, 18, 19, 23, 30) and 1 Mexican (item 17)

More Peru/LatAm items are in B.2, for 24 Peru/LatAm studies in total.

### B.2 Supplementary items (cite if space allows; all verified in the `.bib`)

**Groups (i)–(iv)**
- **Mehrtash et al. (2020)**, *IEEE TMI* 39(12):3868–3878, doi:10.1109/TMI.2020.3006437. ECE for segmentation; ensembles vs MC dropout. Prostate L_CE ECE 8.08 % → ensemble 4.12 % (Table II, arXiv PDF p. 7). FT. ⚠️ Duplicated rows in Table II; quote only the L_CE/Prostate rows.
- **Akiva et al. (2021)**, H2O-Net, WACV, doi:10.1109/WACV48630.2021.00016. Index pseudo-labels plus adversarial SWIR synthesis. mIoU 74.52 vs NDWI 63.26 (Table 2). FT.
- **Tellman et al. (2021)**, *Nature* 596:80–86, doi:10.1038/s41586-021-03695-w. 913 events, 2000–2018; +58–86 M people exposed. ABS (framing only).
- **Hafner, Ban & Nascetti (2022)**, *RSE* 280:113192. S1/S2 UDA, urban. ABS.
- **Cong et al. (2022)**, SatMAE, NeurIPS 35, doi:10.52202/068431-0015. FT (Table 3).
- **Lacoste et al. (2023)**, GEO-Bench, NeurIPS 36 D&B. No flood task.
- **Qiao & Peng (2023)**, TRO, ICLR. Sen1Floods11 Bolivia IoU: ERM .430 vs TRO .450 (Table 4, p. 9). FT.
- **Hertel et al. (2023)**, *RSE* 285:113388. Bayesian CNN for S1 water. ABS.
- **Gawlikowski et al. (2023)**, *AI Review* 56:1513–1589. UQ survey.
- **Konidakis et al. (2025)**, IGARSS, doi:10.1109/IGARSS55030.2025.11242848. Conformal prediction for **S2** flood segmentation. ABS, no numbers.

**Group (v), Peru and Latin America** (details in the scratchpad `B5_peru_latam.md`)
- Roman-Gonzalez et al. (2019), IJACSA. S1 Piura 2017 threshold; no metric. [pre-2021]
- Gonzalez Villarreal (2020), PUCP master's thesis, closed access.
- Alvan Romero, Cigna & Tapete (2020), *Applied Sciences*. SAR, Lima ENSO floods. ABS.
- Anchelía & Mori (2020), UNMSM thesis. RF vs AHP susceptibility, Chancay–Lambayeque. ABS.
- Soria-Ruiz et al. (2022), *Atmosphere*. Mexico, GB/RF. ABS.
- Pech-May et al. (2023), *Computación y Sistemas* 27(2). Tabasco, S1 U-Net; F1 93 % (Table 2, p. 455). FT; weak protocol.
- Ccopi-Trucios et al. (2023), *Sustainability*. Andes GIS risk; peripheral.
- Arias-Choquehuanca et al. (2023), *DYNA* 90(229). Iquitos, S1 GEE. ⚠️ The abstract counts disagree with Table 2.
- Asay Ramirez (2023), ESAN thesis. S1 Piura/Lambayeque 2017; no numbers.
- Velasquez Hurtado et al. (2024), SPIE. Colombia Mojana. ABS.
- Pech-May et al. (2025), *CyS* 29(1). Tabasco S1+S2 U-Net; flood F1 89.38 % (Table 3, p. 403). FT.
- Rathore et al. (2025), *Natural Hazards Research*. Brazil 2024. ABS.
- Breña Aliaga et al. (2026), *Remote Sensing* 18(10):1493. Tumbes, S1 Otsu; IoU 45.6 %. ABS.
- Rizo Zamora et al. (2026), *RBGF*. Rio Grande do Sul; spatial-block CV; flood mask F1 0.681 (p. 2469). FT.

**Search coverage for (v).**
- Searched: OpenAlex (ES/EN/PT), ALICIA (CONCYTEC), web.
- **Not** queried directly: RENATI, nor the UNI, UNALM, UDEP, UNP and UNPRG repositories. Do this before submission.
- No Sentinel + DL/ML flood studies from 2021–2026 were found for Argentina, Chile or Bolivia, and none that qualify for Ecuador.
- Leads to watch:
  - Breña Aliaga's 2026 Zenodo records (Piura S1 + deep learning; no paper yet).
  - The NeurIPS 2026 CCAI workshop paper *"Does confidence track visibility?"* (S2 flood confidence under clouds). ABS only, author list truncated, **not in the .bib**.

---

## C. Bases teóricas (§2.2): canonical references

All are verified in the `.bib`. Details and status: scratchpad `C_bases.md`.

| Topic | Reference (key) | DOI / URL | Notes |
|---|---|---|---|
| Sentinel-2 | Drusch et al. 2012 (`drusch_sentinel2_2012`) | 10.1016/j.rse.2011.11.026 | Mission paper; tables not read (paywall) |
| Sentinel-2 specs | ESA 2015, *Sentinel-2 User Handbook* Issue 1 Rev 2, 24/07/2015 (`esa_sentinel2_2015`) | https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_User_Handbook | Facts below |
| Current specs, L1C, PB 04.00 | ESA SentiWiki "S2 Mission", "S2 Products", "S2 Processing" (accessed 2026-10-16) | https://sentiwiki.copernicus.eu/web/s2-mission (…/s2-products, …/s2-processing) | |
| NDWI | McFeeters 1996 (`mcfeeters_use_1996`) [clásico] | 10.1080/01431169608948714 | (Green − NIR)/(Green + NIR); page NO VERIFICADO |
| MNDWI | Xu 2006 (`xu_modification_2006`) [clásico] | 10.1080/01431160600589179 | (Green − MIR)/(Green + MIR); the S2 mapping B3/B11 is ours |
| Vegetation "NDWI" (to disambiguate) | Gao 1996 (`gao_ndwi_1996`) | 10.1016/S0034-4257(96)00067-3 | |
| FCN | Long, Shelhamer & Darrell 2015 (`long_fully_2015`) | 10.1109/CVPR.2015.7298965 | |
| U-Net | Ronneberger et al. 2015 (`ronneberger_unet_2015`) | 10.1007/978-3-319-24574-4_28 | LNCS vol. 9351 not verified, so omitted |
| ResNet | He et al. 2016 (`he_deep_2016`) | 10.1109/CVPR.2016.90 | Encoder of our U-Net |
| Transformer | Vaswani et al. 2017 (`vaswani_attention_2017`) | NeurIPS proceedings URL | |
| ViT | Dosovitskiy et al. 2021 (`dosovitskiy_image_2021`) | https://openreview.net/forum?id=YicbFdNTTy | |
| MAE | He et al. 2022 (`he_masked_2022`) | 10.1109/CVPR52688.2022.01553 | |
| Foundation models | Bommasani et al. 2021 (`bommasani_opportunities_2021`) | arXiv:2108.07258 | |
| UPerNet / DeepLab (optional) | Xiao et al. 2018; Chen et al. 2018 | 10.1007/978-3-030-01228-1_26; 10.1109/TPAMI.2017.2699184 | |
| DA theory | Ben-David et al. 2010 (`bendavid_theory_2010`) | 10.1007/s10994-009-5152-4 | Online 2009; the volume year 2010 is used |
| DA surveys | Wang & Deng 2018; Csurka 2017; Tuia, Persello & Bruzzone 2016 (remote sensing) | 10.1016/j.neucom.2018.05.083; 10.1007/978-3-319-58347-1_1; 10.1109/MGRS.2016.2548504 | |
| DANN | Ganin et al. 2016 (`ganin_domainadversarial_2016`) | https://jmlr.org/papers/v17/15-239.html | |
| FDA | Yang & Soatto 2020 (`yang_fda_2020`) | 10.1109/CVPR42600.2020.00414 | IEEE pp. 4084–4094 (CVF pp. 4085–4095) |
| AdaBN | Li et al. 2018 (`li_adaptive_2018`) | 10.1016/j.patcog.2018.03.005 | |
| CBST | Zou et al. 2018 (`zou_unsupervised_2018`) | 10.1007/978-3-030-01219-9_18 | |
| Mean Teacher | Tarvainen & Valpola 2017 | NeurIPS URL | |
| TENT | Wang et al. 2021 (`wang_tent_2021`) | https://openreview.net/forum?id=uXl3bZLkr3c | |
| Histogram matching | Gonzalez & Woods 2018, 4th ed., Pearson (`gonzalez_digital_2018`) | ISBN 978-0-13-335672-4 | Section/page NO VERIFICADO |
| Calibration, ECE, temperature scaling | Guo et al. 2017 (`guo_calibration_2017`) | https://proceedings.mlr.press/v70/guo17a.html | |
| ECE origin | Pakdaman Naeini et al. 2015 (`pakdamannaeini_obtaining_2015`) | 10.1609/aaai.v29i1.9602 | Cite as "Pakdaman Naeini et al." |
| Deep ensembles | Lakshminarayanan et al. 2017 | NeurIPS URL | |
| MC dropout | Gal & Ghahramani 2016 | PMLR v48 | |
| Calibration under shift | Ovadia et al. 2019 | NeurIPS URL | Links DA and calibration |
| Error-detection AUROC | Hendrycks & Gimpel 2017 | https://openreview.net/forum?id=Hkg4TI9xl | |
| Reliability diagrams (optional) | Niculescu-Mizil & Caruana 2005; DeGroot & Fienberg 1983 | 10.1145/1102351.1102430; 10.2307/2987588 | DeGroot end page NO VERIFICADO |
| IoU | Jaccard 1912; Everingham et al. 2010 | 10.1111/j.1469-8137.1912.tb05611.x; 10.1007/s11263-009-0275-4 | |
| Segmentation metrics (optional) | Csurka et al. 2013; Taha & Hanbury 2015 | 10.5244/C.27.32; 10.1186/s12880-015-0068-x | |
| Bootstrap | Efron 1979; Efron & Tibshirani 1993 | 10.1214/aos/1176344552; 10.1007/978-1-4899-4541-9 | Two DOI records exist for the book; the Springer US 1993 one is used |
| Dice loss | Milletari et al. 2016 | 10.1109/3DV.2016.79 | |
| Otsu | Otsu 1979 | 10.1109/TSMC.1979.4310076 | |
| Software | PyTorch (Paszke et al. 2019); TerraTorch (Gomes et al. 2025, IGARSS, 10.1109/IGARSS55030.2025.11243570); TorchGeo (Stewart et al. 2022, 10.1145/3557915.3560953) | | TerraTorch now has an IGARSS DOI; prefer it over the arXiv version |

**Sentinel-2 facts.** Source: *Sentinel-2 User Handbook*, ESA, Issue 1 Rev 2, 24/07/2015.
- 13 bands: four at 10 m, six at 20 m, three at 60 m (§1.1, p. 9). FT.
- Revisit: 10 days per satellite, 5 days with the constellation (§1.12, p. 51). FT.
- Nominal central wavelengths, Tables 3–4, p. 53 (FT):
  - B2 490 nm, B3 560 nm, B4 665 nm, B8 842 nm (10 m)
  - B11 1 610 nm, B12 2 190 nm (20 m)
  - ⚠️ SentiWiki gives the S2A equivalent wavelength of B8 as 832.8 nm. Use one source consistently.
- L1C = top-of-atmosphere reflectance (§1.10.4, p. 45; SentiWiki "S2 Products"). FT.
- Processing baseline 04.00 (from 25-jan-2022) adds RADIO_ADD_OFFSET, with
  L1C_TOA = (DN + offset)/QUANTIFICATION_VALUE (SentiWiki "S2 Products"). FT.
  - The value **−1000** comes only from an ESA STEP forum post by the OPT-MPC S2 Technical
    Manager (25-mar-2022). It is not a formal document.
  - QUANTIFICATION_VALUE = 10000: **NO VERIFICADO** (read it from the product metadata).

---

## D. Comparison matrix (draft) and research gap

### D.1 Matrix

Only rows whose number was read at table level (FT) or carried from a table-level note (FT-note)
are included.

| # | Study | Method | Data / sensor | Region | Metric | Result (source) | Gap relative to this thesis |
|---|---|---|---|---|---|---|---|
| 1 | Li et al. 2023 | Prithvi-100M vs U-Net, fine-tuned | Sen1Floods11, 6 S2 bands | Global; Bolivia held out | Water IoU | U-Net 84.03 vs Prithvi 81.98 in distribution; Bolivia Prithvi 76.62 vs U-Net 70.57 (Tables 2–3) | One held-out region, humid; no Peru; no adaptation; no calibration |
| 2 | Szwarcman et al. 2024 (Prithvi-EO-2.0) | ViT-MAE GFM, fine-tuned | Sen1Floods11, S2 HLS bands | Global | Water IoU | 300M: 82.2 (0.6) (Table IV) | In-distribution only |
| 3 | Jakubik et al. 2025 (TerraMind) | Multimodal generative GFM | Sen1Floods11, S-2 | Global | Water IoU | v1-B 82.26 (Table 8) | In-distribution only; split not stated |
| 4 | Marsocci et al. 2026 (PANGAEA) | Frozen GFMs vs U-Net | Sen1Floods11 | Global | mIoU | U-Net 91.42 best vs CROMA 90.89 vs Prithvi-EO-1 90.37 (Table 5) | Frozen encoders; in-distribution |
| 5 | Kaushik et al. 2026 (JSTARS) | GFMs vs CNNs | FloodPlanet S1/S2/PlanetScope | 19 sites incl. Bolivia, Colombia, Paraguay | IoU (S2) | Clay 0.72, U-Net 0.67, Prithvi 0.64 (Table III) | No arid coast; no adaptation; no calibration |
| 6 | Kolluru et al. 2026 | Prithvi-EO-2.0 zero-shot | S2, 19 OOD events | Global (one LatAm event) | IoU vs Copernicus GFM | Mean 0.46; recall 0.90 vs precision 0.49 (text p. 7; Table 3) | Source-only; no adaptation; references are automatic products |
| 7 | Portalés-Julià et al. 2025 | U-Nets cross-dataset vs MNDWI | WorldFloods, Kuro Siwo, S1S2Water → Sen1Floods11, UNOSAT | Global | All-water IoU | Sen1Floods11: U-Net 77.50 vs MNDWI 74.39 (Table 3; FT-note, re-check) | No GFMs; no adaptation |
| 8 | Chen et al. 2026 (CIM-TTDA) | Test-time self-training | S1 SAR | Vietnam, Madagascar, USA | IoUp | U-Net-EffNet 0.636 → 0.803; unadapted Prithvi-EO-2.0 0.478 (Table 4; FT-note) | SAR only; the GFM is not adapted; humid coasts |
| 9 | Shen et al. 2022 | UNet++ trained on Sen1Floods11, S1/S2 fusion | S1 + S2 | **Piura, 2017** | Water IoU | Fusion 0.69 (Table 3); EMS product vs their manual mask IoU 0.44 (p. 138) | No GFMs; one date; small manual reference |
| 10 | Breña-Aliaga & Cruz-Machacuay 2026 | Random Forest + SDWI + HAND | S1; PlanetScope reference | **Tumbes, 2023–25** | IoU | 69.02 ± 3.46 % (Tabla 2) | SAR + classical ML; no DL, GFMs or calibration |
| 11 | Garshasbi et al. 2025 | Bayesian U-Net | S1, Sen1Floods11 | Global; Iran | IoU | 0.6685 vs U-Net 0.6224 (Table II; FT-note) | SAR; no ECE / temperature scaling; no GFMs |
| 12 | Banerjee & Daou 2026 | U-Net vs SegFormer + ECE | S1, Sen1Floods11 | Global | ECE | 0.0675 vs 0.0694 (Table 5) | SAR; in-distribution; no recalibration under shift |

### D.2 Research gap (draft, 4 sentences; the authors should rewrite it in their own words, per WRITING_RULES §9)

1. On the standard Sentinel-2 benchmark, geospatial foundation models do not beat a well-trained
   U-Net:
   - Li et al. (2023): U-Net 84.03 vs Prithvi 81.98 water IoU, both fine-tuned.
   - Marsocci et al. (2026, PANGAEA): U-Net 91.42 mIoU, the best, although there the foundation
     models had **frozen encoders**.

   The evidence that foundation models help comes from held-out regions. The South American ones
   are humid lowlands: Bolivia (Li et al., 2023) and the Bolivia, Colombia and Paraguay sites of
   FloodPlanet (Kaushik et al., 2026).
2. Outside the benchmark, source-only models lose much of their skill and over-predict
   (Kolluru et al., 2026: mean IoU 0.46, precision 0.49).
   - The one desert event there, Libya, scored well (IoU 0.79). But it is a single chip with
     compact floods, so it does not show that arid backgrounds are easy.
   - Flood adaptation exists for SAR (Chen et al., 2026) and across sensors (Akiva et al., 2021,
     H2O-Net). We found none that adapts a foundation model on unlabelled Sentinel-2 imagery of
     the target region.
3. For Peru, the studies found map the 2017 and 2023 floods with SAR thresholds, K-Means or
   Random Forest, or with a Sen1Floods11-trained CNN validated against a small manual mask (Shen
   et al., 2022). None trains or evaluates deep-learning or foundation models against the expert
   UNOSAT/Copernicus EMS maps of the arid north coast.
4. In a preliminary search ("búsqueda preliminar", not IEEE Xplore or Scopus), we found no study
   that measures the calibration (ECE, temperature scaling, ensembles) of **optical** flood
   segmentation, under shift or otherwise. The closest works either omit ECE (Garshasbi et al.,
   2025) or measure ECE on SAR in distribution only (Banerjee & Daou, 2026). Konidakis et al.
   (2025) apply conformal prediction to Sentinel-2 floods (abstract only).

**What this thesis contributes.** It compares TerraMind, Prithvi-EO-2.0 and U-Net:
- trained on Peruvian expert labels;
- against Sen1Floods11 source-only models;
- with unsupervised adaptation to unlabelled Peruvian Sentinel-2 imagery;
- and with calibration and uncertainty, on a desert-background shift;
- evaluated on two events (Yaku 2023 and Niño Costero 2017).

**Do not claim:**
- "first domain adaptation for floods" (Chen et al., 2026 exists);
- "first adaptation for water" (H2O-Net, UUCP exist);
- "first DL flood mapping in Peru" (Shen et al., 2022 exists).

Phrase gaps as "no se encontró" ("not found"). IEEE Xplore, Scopus and RENATI were not searched
directly.

---

## E. Open items before citing

1. Re-read Portalés-Julià 2025, Table 3 in the PDF (ScienceDirect blocked scripted access).
2. Obtain Quintana-Ortiz et al. 2021 (IEEE, UNI library). It covers the same event and area as
   `peru-2017-test`.
3. Search RENATI and the UNI, UNALM, UDEP, UNP and UNPRG repositories directly for theses.
4. Check Scopus/WoS indexing for every journal antecedent before writing "revista indexada".
5. Update `docs/published_comparison.md` and `docs/REFERENCES.md`: the U-Net Base [19] setup is
   now known (Li et al. 2023, Tables 2–3), and the Badillo-Rivera AUC question is resolved
   (Table 5). Neither file was edited in this task.
6. A `docs/METHODS_NOTES.md` entry for this literature pass was **not** written, because the task
   allowed only these two files.
