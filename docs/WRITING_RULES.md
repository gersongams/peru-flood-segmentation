# Writing rules for the thesis (and the papers)

Adopted 2026-10-16 by the authors. This file is binding for every chapter, whoever drafts it.
**Precedence**, from highest to lowest:
1. EPG-UNI **Guía N°2** (thesis), approved by the Consejo Directivo N°05 on 28-03-2023 and signed by
   the director of the FIIS graduate unit. It says the rules apply "en forma estricta" (p. 4).
2. EPG-UNI **Guía N°1**, Anexo III (format standards; the only format annex the guides have).
3. **APA 7** ("versión vigente"; the guides name no edition).
4. Course syllabus MIA-403 v2 (S2), and v1.1 (S1) where v2 is silent.
5. Our own conventions (sections 5–9).

Page numbers refer to the printed page of each guide (PDF page = printed + 1). Full extraction
with quotes: `docs/guide_rules_extracted.md`. Items marked **(default)** were set by us where the
guides are silent. They can be changed, but only here, never ad hoc in a chapter.

## 1. Structure (Guía N°2 §4.1, pp. 7, 9–10). Hard rule

The template was an undergraduate FIIS template (6 chapters). It is replaced by the research
master's structure:

```
Carátula (cover)
Dedicatoria · Agradecimientos · [Copia de documentos: added only after the defense]
Índice de contenidos · Lista de tablas · Lista de figuras
Resumen (Spanish) · Abstract (English)
Introducción
CAPÍTULO I. PROTOCOLO DE LA INVESTIGACIÓN
  1.1 Identificación y descripción del problema de estudio
      1.1.1 Formulación del problema general
      1.1.2 Formulación de los problemas específicos
  1.2 Objetivos (1.2.1 general · 1.2.2 específicos)
  1.3 Hipótesis y variables (1.3.1 general · 1.3.2 específicas · 1.3.3 variables)
  1.4 Metodología
CAPÍTULO II. MARCO TEÓRICO
  2.1 Antecedentes de la investigación · 2.2 Bases teóricas · 2.3 Definición de términos
CAPÍTULO III. DESARROLLO DEL TRABAJO DE INVESTIGACIÓN
  3.1 Análisis de los datos y resultados
  3.2 Discusión e interpretación de los resultados
  3.3 Contrastación de las hipótesis
CAPÍTULO IV. CONCLUSIONES Y RECOMENDACIONES (4.1 · 4.2)
REFERENCIAS
ANEXOS: 1 Matriz de consistencia · 2 Matriz de operacionalización (required by the course, S2) ·
        3 Reproducibilidad (repo, commits, seeds, versions; required by the course, S1/S2)
```

- The headings above are fixed. Lower levels (e.g. 1.1.3, 3.1.1) can be added freely.
- Chapter III will be long. Split 3.1 into subsections (data and the Peru dataset; models and
  training; experimental protocol; results on dev; results on test; calibration).
- Planned mapping of our material:
  - §1.1: the Peru flood context and the gap.
  - §1.4: type/level/design plus a summary of the protocol. The technical method goes in §3.1.
  - §2.1: antecedentes.
  - §2.2: Sentinel-2, segmentation, CNNs and Transformers, foundation models, domain
    adaptation, calibration.
  - §3.1 also holds the system: its development and its technical evaluation (RF1–RF7, time per
    scene; Chapter I v2).
  - §3: everything in `docs/phaseD_dev.md`, `docs/phaseD_test.md`, the failure, calibration and
    domain-shift write-ups, and `docs/published_comparison.md`.

## 2. Format (Guía N°1 Anexo III pp. 43–44; Guía N°2 p. 20). Hard rule

| Item | Rule |
|---|---|
| Paper, margins | A4. Left and top 3.0 cm, right and bottom 2.5 cm |
| Font | **Arial 11** in the body. A smaller size is allowed only in tables, figures and annexes. The cover follows G2 p. 20 (Poppins 22 bold for the university name, Arial 16/14/12 for the rest) |
| Spacing | 1.5 lines. Single spacing for notes, block quotations, list paragraphs (not between items), lists of tables/figures and annexes. **References: double spacing with a hanging indent** (G2 p. 19 overrides G1's single spacing) |
| Italics | **Only for words that are not Spanish.** Never for emphasis, never for numbers |
| Page numbers | Top right, number only: no header text, no word "página". The front matter is in lowercase Roman numerals, from the cover (counted, not printed) to the Resumen/Abstract. The text is in Arabic numerals, from the Introducción on. The Introducción and each chapter-opening page are counted but not printed. Pages with tables and figures are numbered |
| Length | **At most 130 pages in total**, references and annexes included |
| Title | Capitals, centred, **at most 20 words**, non-Spanish words in italics |
| Headings | "REFERENCIAS" (not "Bibliografía") |

## 3. Content rules from the guides and the course. Hard rule

- **Problems:** stated as **questions** (S2 wants "pregunta(s) precisa(s)"; G1 allows either).
  Write one general problem and **three** specific ones.
- **Objectives:** begin with a verb in the **infinitive** (Evaluar, Determinar, Comparar…). Write
  one general objective and **three** specific ones, each matching one specific problem. They must
  be measurable (S2 rubric).
- **Hypotheses:** one general and three specific, each **empirically testable** and matching one
  objective (G2 p. 34).
  - They come from the research-question hypotheses written in `docs/PLAN.md` on 2026-10-12,
    before any results.
  - They are not reworded to fit the results. A refuted or partly refuted hypothesis is reported
    as such in §3.3.
  - Tested with the pre-specified evidence: paired chip-bootstrap CIs and the frozen test pass.
- **Variables:** independent = the "solution" (model / adaptation method); dependent = the
  "problem" (segmentation quality measured by IoU, and calibration measured by ECE/AUROC). Give
  dimensions and indicators, since the operationalization matrix is required by the course.
- **Justification:** three labelled parts: **teórica, práctica, metodológica** (G2 p. 33).
- **Scope:** three labelled parts: **temático, espacial, temporal** (G2 p. 33). Limitations and
  threats to validity are stated in §1.4 and again in §3.2.
- **Antecedentes:**
  - Each one gives **method and main results**, with a table/page source for any number.
  - Order them **chronologically, oldest to newest**.
  - Use studies from the **last 5 years**, prefer indexed journals.
  - Older work (U-Net 2015, Sen1Floods11 2020) only as "clásicos", with one sentence saying why.
  - Include Peruvian/regional studies where they exist.
  - Close with a comparison matrix and the gap (S2 week 3).
- **Discussion §3.2**, in this order (G2 p. 18):
  1. A short summary of the main results.
  2. Explanations and comparison with other studies, both agreeing and contradicting.
  3. The important findings, tied to the literature.
  4. The authors' interpretation.
  5. **Limitations and weaknesses.**
  6. Implications.
- **Conclusions:** **exactly 3**, one per specific objective, each answering its objective and
  summarising the evidence behind it. **Recommendations: exactly 3**, one following from each
  conclusion (G2 p. 19).
- **Resumen:** in the **past tense**. It covers introduction, objectives, methodological design,
  results and conclusions. It is followed by an English Abstract with the same content.
  **(default)** Each is at most 250 words and ends with 4–6 *palabras clave* / *keywords*.
- **Introducción:** discursive prose, without numbered subsections.
- **Traceability** (S2 §XI): problem → question → objective → method → evidence → result →
  conclusion. The consistency matrix (Anexo 1) is the check.
- "Using AI/ML/DL is not by itself a contribution" (S2 §XI). The contribution must be stated
  against the state of the art.

## 4. Scientific integrity. Hard rule, from `CLAUDE.md`

- **Never invent a number.**
  - Every number in the text, a table or a figure comes from a results file in this repo, or
    from a cited **table/page/URL** of a source (cite tables, not abstracts).
  - In the `.tex`, put a comment after each number or table with its source, e.g.
    `% fuente: outputs/phaseD/test/peru-test_groups.csv`. Comments do not print and keep the
    text auditable.
- **Dev vs test.**
  - Always say which split a number comes from.
  - Model choices are justified with dev / `valid` numbers only.
  - Test numbers are reported as they came out of the single frozen pass, including results
    that go against us (the 2017 reversal, the partly refuted calibration claim).
- **Uncertainty.** Report means ± std over 3 seeds. Comparisons are stated with their 95 %
  bootstrap CI.
  - Do not call a difference "better" when its CI includes 0. Say "no se distingue" instead.
  - No significance language beyond what the CI supports.
- **Corrections and deviations** recorded in `docs/METHODS_NOTES.md` appear in the text where
  they matter: the 7.3 deviation, the AdaBN reproduction gap, the excluded rows.
- **Post-hoc analyses** are labelled as post-hoc.

## 5. Language and style **(default)**

- **Voice:** impersonal with *se* or the passive: "se entrenaron tres semillas", "se observó".
  Never "yo". Use "nosotros" only in the Agradecimientos. Refer to the work as "esta
  investigación" or "la presente tesis".
- **Tense:**
  - Past for what was done and found (methods, results, Resumen).
  - Present for established facts and for what the thesis contains ("el Capítulo III presenta…").
  - Future only in the plan-like parts of §1 if needed.
- **Sentences:**
  - One idea per sentence; aim for less than about 30 words.
  - Paragraphs open with their main point.
  - No rhetorical questions outside §1.1.
  - No filler ("cabe mencionar que", "es importante destacar que", "en este sentido").
- **Precise wording:** say "IoU de agua" rather than "precisión" when IoU is meant. *Precisión*
  is reserved for the precision metric.
- **Gender-neutral wording** where natural ("las personas afectadas").

## 6. Terminology **(default)**

- Use the established Spanish term when one exists. When the English term is needed (no settled
  Spanish equivalent, or the field uses it), write it in *italics* with the Spanish gloss at
  first use, then italics every time (G1 p. 43). LaTeX macro: `\ingles{...}`.
- **Not italic:**
  - proper names and products: Sentinel-2, Sen1Floods11, TerraMind, Prithvi, U-Net, TerraTorch,
    PyTorch;
  - acronyms: IoU, ECE, AUROC, FDA, AdaBN, MNDWI, NDWI, CNN, ViT;
  - equations and numbers.
- Expand each acronym at first use. Acronyms stay in their usual English form.

| English | Spanish to use |
|---|---|
| foundation model | modelo fundacional |
| dataset | conjunto de datos |
| fine-tuning | ajuste fino |
| (semantic) segmentation | segmentación (semántica) |
| deep learning | aprendizaje profundo |
| encoder / decoder | codificador / decodificador |
| domain shift / adaptation | cambio de dominio / adaptación de dominio |
| unsupervised domain adaptation | adaptación de dominio no supervisada (sin etiquetas del dominio objetivo) |
| source / target domain | dominio de origen / dominio objetivo |
| training / validation / test split | partición de entrenamiento / validación / prueba |
| seed | semilla |
| learning rate | tasa de aprendizaje |
| epoch / batch | época / lote |
| pseudo-labels / self-training | pseudoetiquetas / autoentrenamiento |
| histogram matching | igualación de histogramas |
| calibration / reliability diagram | calibración / diagrama de fiabilidad |
| temperature scaling | escalado de temperatura |
| ensemble | ensamble |
| uncertainty | incertidumbre |
| ground truth / label | etiqueta de referencia / etiqueta |
| chip / tile | recorte (512 × 512 px) |
| water index | índice espectral de agua |
| upper bound | cota superior |
| benchmark | conjunto de referencia (*benchmark* only if needed) |
| batch normalization | *batch normalization* (normalización por lotes) |
| backbone | *backbone* (codificador preentrenado) |
| patch | *patch* (parche) |
| prompt, token | *token* |

## 7. Numbers and units **(default** unless noted)

- **Decimal point** (the template sets `\decimalpoint`), e.g. 0.799. A thin space groups
  thousands: 1 291, written `\num{1291}`.
- **IoU, precision, recall:** reported as **proportions with 3 decimals** (0.799 ± 0.005), the
  same as our results files. That avoids rounding errors between files and text.
  - Differences between methods are written as proportions too (+0.022), with their CI.
  - In comparisons with the literature, which reports percentages, the published value is
    quoted as published and the unit is stated.
- **ECE:** in percent with 2 decimals (0.60 %), as in our tables.
- **Units:** SI with a space before them (10 m, 512 × 512 px, 5.12 km), written with `siunitx`.
  Dates in the text: "22 de marzo de 2023". In tables: 2023-03-22.
- **Numbers in words:** zero to nine as words in prose ("tres semillas"), except with units or
  in comparisons.

## 8. Citations, figures, tables

- **APA 7** via `apacite` with `natbibapa`: "Bonafilia et al. (2020)" (`\citet`) or
  "(Bonafilia et al., 2020)" (`\citep`). Add a page or table as `\citep[Tabla~3, p.~137]{key}`.
  - Each bib entry is added **one at a time** to `3_3_BIBLIOGRAFIA/library.bib`, checked against
    the source (DOI/URL), and only once it is cited.
  - Datasets and software are cited too (Sen1Floods11, TerraTorch, UNOSAT and EMS products).
- **Tables and figures** follow APA 7.
  - **Above:** "**Tabla 3.2**" in bold, then the title in italics on the next line.
  - **Below:** "*Nota.* …" with the source, e.g. "Elaboración propia a partir de
    `outputs/…`" or "Adaptado de Autor (año, p. x)".
  - Numbering is by chapter.
  - Every table and figure is cited in the text before it appears.
- **Figures:**
  - Made by scripts in this repo (reproducible), saved as PDF (vector) or high-resolution PNG.
  - Legible in grey scale.
  - Labels in Spanish with the decimal point.
- **No screenshots of other authors' figures** unless cited as "Tomado de".

## 9. Workflow and use of AI **(authors' decision, 2026-10-16)**

- **Who writes:** Claude drafts sections, with every number and citation traced. The authors
  review and edit before a section counts as final. The authors are responsible for authorship,
  accuracy, citation and integrity (S2 p. 3).
- **AI detection risk:** S1 p. 1 requires that work "supere detectores … de generación automática
  (LLM)", and S2 does not withdraw this. The authors should therefore **rewrite in their own words
  the argumentative parts**: Introducción, §1.1, §3.2 Discusión, Cap. IV. Declare AI assistance as
  S2 allows.
- **Repo:** `thesis/` is the Overleaf repo.
  - `git pull` before editing.
  - One section per commit.
  - It must compile with XeLaTeX (`latexmk`; `latexmkrc` enforces it).
  - **Push to Overleaf only with the authors' approval.**
- **Before a section is marked done:**
  - it compiles;
  - every number has a `% fuente:` comment;
  - every citation is in the `.bib`;
  - the terminology follows §6;
  - it has no italics except for non-Spanish words;
  - the page count is on track for ≤ 130 pages.

## 10. Open items (to settle with the course or the Unidad de Posgrado)

1. **Title.** The plan was approved with "Sistema de análisis satelital de inundaciones… para la
   detección de áreas de riesgo", which the thesis no longer does. Agreed: a new title of at most
   20 words. **Check whether changing an approved title needs a formal request** (G1 p. 48 allows
   at most 2 topic changes).
2. **Course draft numbering.** S2 uses its own chapter numbering for the course draft. Confirm
   that the course accepts Guía N°2's structure.
3. **Arial on Overleaf.** The template falls back to TeX Gyre Heros (a Helvetica clone) if Arial
   is missing. The final PDF must be built where Arial exists. Check this before submission.
4. **Adviser:** none assigned yet. The cover shows "ASESOR: (por asignar)".
