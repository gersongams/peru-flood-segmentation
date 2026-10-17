# Chapter I backbone v2: the system as the general objective (proposal for approval)

2026-10-16. The authors asked to reframe Chapter I around building an AI application that uses
the models. **v1** (below, approved and written earlier the same day) is superseded once v2 is
approved. Evaluation of the system: **technical only** (authors' choice, 2026-10-16).

## Integrity rule applied
- HE1, and the merged HE2 (old HE2 + HE3), keep the **pre-registered wording unchanged**.
  Their outcomes are already known, so they are not reworded.
- **HE3 is new and is fixed now (2026-10-16), before the system exists or is measured.** That
  is legitimate only because the system has not been evaluated yet.
- The threshold and the requirements below must not change after the measurement.

## Title (proposal; ≤ 20 words; closest to the approved plan title)
> SISTEMA DE ANÁLISIS SATELITAL DE INUNDACIONES BASADO EN MODELOS DE SEGMENTACIÓN PROFUNDA PARA EL
> MAPEO DE ÁREAS INUNDADAS EN EL PERÚ (19 words)

Only the end of the approved title changes: "para la detección de áreas de riesgo" becomes "para el
mapeo de áreas inundadas en el Perú". It is still worth checking whether this needs a formal
request.

## Problema general
> ¿Cómo desarrollar un sistema de análisis satelital de inundaciones para el Perú, basado en
> modelos de segmentación profunda, que genere mapas de la extensión del agua y de su
> incertidumbre a partir de imágenes Sentinel-2, incluso cuando no se dispone de etiquetas
> peruanas?

## Problemas específicos
1. **PE1.** ¿Qué modelo de segmentación (los modelos fundacionales geoespaciales TerraMind y
   Prithvi, o una U-Net) y qué régimen de entrenamiento ofrecen el mejor desempeño en
   Sen1Floods11 y en el Perú?
2. **PE2.** ¿En qué medida la adaptación de dominio sin etiquetas peruanas reduce la pérdida de
   desempeño de los modelos en el Perú, y está calibrada la incertidumbre que el sistema
   mostrará?
3. **PE3.** ¿Cumple el sistema implementado los requisitos funcionales y el tiempo de
   procesamiento necesarios para usarlo después de un evento de inundación?

## Objetivo general
> Desarrollar un sistema de análisis satelital de inundaciones para el Perú, basado en modelos de
> segmentación profunda, que genere mapas de la extensión del agua y de su incertidumbre a partir
> de imágenes Sentinel-2.

## Objetivos específicos
1. **OE1.** Comparar el IoU de agua de TerraMind v1-B, Prithvi-EO-2.0 300M y U-Net en
   Sen1Floods11 y en los eventos peruanos de 2023 y 2017, con tres regímenes de entrenamiento,
   para seleccionar el modelo del sistema.
2. **OE2.** Determinar en qué medida la adaptación de dominio no supervisada y la calibración de
   la incertidumbre mejoran la fiabilidad de los mapas en el Perú, e identificar los estratos en
   los que se concentran los errores.
3. **OE3.** Implementar el sistema como una aplicación web interactiva y evaluar su cumplimiento
   de los requisitos funcionales y su tiempo de procesamiento por escena Sentinel-2.

## Hipótesis general
> Los modelos fundacionales geoespaciales pierden menos desempeño que la U-Net al pasar de
> Sen1Floods11 al Perú. La adaptación de dominio sin etiquetas y la calibración de la
> incertidumbre mejoran parcialmente la fiabilidad de los mapas. Un sistema que integra estos
> modelos cumple sus requisitos funcionales y procesa una escena Sentinel-2 en un tiempo
> compatible con la respuesta posterior a un evento.

## Hipótesis específicas
| | Hypothesis (Spanish) | Status |
|---|---|---|
| **HE1** | En Sen1Floods11 no existe una diferencia apreciable en IoU de agua entre los modelos fundacionales y la U-Net (el IC 95 % de la diferencia incluye 0). En el Perú, los modelos fundacionales pierden menos IoU de agua que la U-Net. | pre-registered (PLAN, 2026-10-11), unchanged |
| **HE2** | Los errores de los modelos en el Perú se concentran en pocos estratos (nubes, vegetación inundada, terreno desértico o costero). La adaptación de dominio no supervisada aumenta el IoU de agua respecto del modelo sin adaptar (IC 95 % > 0), sin alcanzar al modelo entrenado en el Perú. Sin calibrar, las probabilidades están sobreconfiadas; el escalado de temperatura y los ensambles reducen el ECE, y la incertidumbre concentra los errores (AUROC > 0.5). | old HE2 + HE3, pre-registered, wording unchanged |
| **HE3** | El sistema cumple el 100 % de los requisitos funcionales RF1–RF7 y genera el mapa de agua y de incertidumbre de una escena Sentinel-2 completa (aproximadamente 110 × 110 km) en no más de **15 minutos**, incluida la descarga, en un equipo de escritorio (Apple M4 Max). | **new, fixed 2026-10-16 before any measurement** |

**Functional requirements**, fixed now and verified by a test procedure in Chapter III:
- **RF1** Select an event or date and an area of interest.
- **RF2** Obtain the Sentinel-2 L1C image from Copernicus Data Space, or load one already
  processed.
- **RF3** Generate the water-probability map with the selected model (ensemble of 3 seeds).
- **RF4** Show the uncertainty and the unobserved areas (clouds, no data).
- **RF5** Overlay the reference map when one exists (UNOSAT / EMS) and show agreement metrics.
- **RF6** Compute the flooded area per date and show it on a timeline.
- **RF7** Export the maps (GeoTIFF / GeoJSON).

The **15 min** threshold is the authors' operational requirement, not a literature figure. The
authors can confirm or change it **now**, never after measuring.

## Variables
- **Independent:** model and training / adaptation strategy (unchanged), plus the **system
  configuration** (selected model, inference pipeline).
- **Dependent:** quality of the flood mapping (unchanged dimensions), plus **system
  performance**:
  - functional requirements met (%);
  - processing time per scene (min).

## Methodology changes
- **Type:** applied (technological development).
- **Design:**
  - experimental, as controlled computational experiments, for OE1–OE2;
  - development and technical evaluation of the system for OE3, with an iterative-incremental
    method.
- **New stage 6:** system development and technical evaluation.
- **Chapter III gains a section:** "Desarrollo del sistema", covering architecture,
  implementation and the RF/time tests.

## Consequences
- **The viewer becomes a thesis component.** The 2026-10-16 rule that "the viewer produces no
  thesis results" (`.viewer-wip/uncommitted_viewer_changes.diff`) will be rewritten. Its
  technical evaluation (RF checklist, processing time) becomes a thesis result, logged and
  pre-specified as above. Model quality numbers still come only from the frozen test pass.
- **RF2 and the time hypothesis need on-demand processing:** a small backend that fetches a
  scene and runs the model. This is new work: Stage 3 of the viewer plan moves into the thesis.
- **Expected verdicts:** HE1 is supported with a caveat; HE2 is partly supported (unchanged from
  v1). HE3 is unknown until the system is built and measured.

---

# Chapter I backbone v1 (superseded if v2 is approved)

2026-10-16. **Approved by the authors on 2026-10-16 and written into `thesis/2_CAPITULO1/Capitulo1.tex`
(thesis commit after `32568ac`) and Anexos 1–2.** Drafted by Claude (`docs/WRITING_RULES.md` §3, §9). Thesis wording is in Spanish. Notes on
provenance and the expected verdict are in English.

## Provenance of the hypotheses (rule: hypotheses are not fitted to the results)

| Source | Date | Status |
|---|---|---|
| `docs/PLAN.md` RQ1–RQ3 and their hypotheses | first committed 2026-10-11 (`3762328`), before any model result | pre-registered |
| RQ4: "Defined after RQ2 (phase D)". The phase-D proposal (`openspec/changes/phase-d-peru-adaptation/proposal.md`) frames adaptation as closing the source-only → Peru-trained gap | 2026-10-14 (`02da940`) | after the dev failure analysis, **before any test evaluation**. The thesis must say this |

The four RQs are merged into three specific objectives, as the authors decided on 2026-10-16
(G2 p. 19 requires exactly three conclusions). RQ2 ("where and why they fail") and RQ4
("adaptation") become one objective, because the adaptation targets the failure RQ2 found.
**One sub-claim of the RQ1 hypothesis is dropped as untested:** "need fewer Peru chips". No
learning-curve experiment was run. §1.2.2 states this as a scope limitation.

---

## Problema general

> ¿En qué medida los modelos fundacionales geoespaciales TerraMind y Prithvi y una red U-Net,
> entrenados con el conjunto Sen1Floods11, segmentan correctamente las áreas inundadas en
> imágenes Sentinel-2 del Perú, con una confianza bien calibrada, y en qué medida la adaptación
> de dominio sin etiquetas peruanas reduce su pérdida de desempeño frente a modelos entrenados
> con datos del Perú?

## Problemas específicos

1. **PE1.** ¿Superan los modelos fundacionales geoespaciales a la U-Net en la segmentación de
   agua, tanto en el dominio de entrenamiento (Sen1Floods11) como en el Perú, y cuánto cambia el
   desempeño al entrenar con datos peruanos?
2. **PE2.** ¿En qué condiciones fallan estos modelos en el Perú y en qué medida la adaptación
   de dominio no supervisada, que usa solo imágenes peruanas sin etiquetar, reduce esa pérdida?
3. **PE3.** ¿Están calibradas las probabilidades que producen los modelos en el Perú, y señala
   su incertidumbre los píxeles mal clasificados?

## Objetivo general

> Evaluar el desempeño y la calibración de modelos fundacionales geoespaciales y de una U-Net
> en la segmentación de inundaciones en imágenes Sentinel-2 del Perú, y determinar en qué medida
> la adaptación de dominio sin etiquetas peruanas reduce su pérdida de desempeño respecto de
> modelos entrenados con datos del Perú.

## Objetivos específicos

1. **OE1.** Comparar el IoU de agua de TerraMind v1-B, Prithvi-EO-2.0 300M y U-Net en la
   partición de prueba oficial de Sen1Floods11 y en los eventos peruanos de 2023 y 2017. Se
   compara el entrenamiento con Sen1Floods11, con datos peruanos y con Sen1Floods11 seguido de
   ajuste fino en el Perú.
2. **OE2.** Identificar los estratos en los que se concentran los errores en el Perú y
   determinar qué proporción de la brecha entre el modelo sin adaptar y el modelo entrenado en
   el Perú cierra la adaptación de dominio no supervisada.
3. **OE3.** Evaluar la calibración (ECE) y la capacidad de la incertidumbre para detectar errores
   (AUROC) de los modelos en el Perú, con y sin escalado de temperatura y ensambles de semillas.

## Hipótesis general

> Los modelos fundacionales geoespaciales pierden menos desempeño que la U-Net al pasar de
> Sen1Floods11 al Perú. Sus errores en el Perú se concentran en pocos estratos, y la adaptación
> de dominio sin etiquetas reduce parcialmente esos errores. Además, el escalado de temperatura
> y los ensambles mejoran su calibración.

## Hipótesis específicas

| | Hypothesis (Spanish, for the thesis) | Source (verbatim, PLAN 2026-10-11 unless noted) |
|---|---|---|
| **HE1** | En Sen1Floods11 no existe una diferencia apreciable en IoU de agua entre los modelos fundacionales y la U-Net (el IC 95 % de la diferencia incluye 0). En el Perú, los modelos fundacionales pierden menos IoU de agua que la U-Net. | "No meaningful gap in-distribution; the foundation models lose less on Peru" (the "fewer Peru chips" part is dropped as untested) |
| **HE2** | Los errores de los modelos en el Perú se concentran en pocos estratos (nubes, vegetación inundada, terreno desértico o costero). La adaptación de dominio no supervisada aumenta el IoU de agua respecto del modelo sin adaptar (IC 95 % > 0), sin alcanzar al modelo entrenado en el Perú. | RQ2: "Most errors concentrate in a few strata: clouds, flooded vegetation, desert/coastal terrain." Adaptation part: phase-D proposal, 2026-10-14 (see provenance) |
| **HE3** | Sin calibrar, las probabilidades de los modelos están sobreconfiadas. El escalado de temperatura y los ensambles reducen el error de calibración esperado (ECE), y la incertidumbre concentra los errores (AUROC de detección de errores > 0.5). | "Raw softmax is over-confident; ensembles + temperature scaling lower ECE; high uncertainty concentrates errors." |

## Variables

| Variable | Dimensiones | Indicadores |
|---|---|---|
| **Independiente:** modelo de segmentación y estrategia de entrenamiento / adaptación | Arquitectura | U-Net (ResNet-34), TerraMind v1-B, Prithvi-EO-2.0 300M |
| | Régimen de entrenamiento | Sen1Floods11; Perú (`peru-train`); Sen1Floods11 → ajuste fino en el Perú |
| | Método de adaptación sin etiquetas | Ninguno; igualación de histogramas (en entrenamiento / en prueba); FDA en entrenamiento; AdaBN |
| | Tratamiento de la incertidumbre | Logits sin calibrar; escalado de temperatura; ensamble de 3 semillas |
| **Dependiente:** calidad del mapeo de inundaciones | Exactitud de la segmentación | IoU de agua (principal), IoU tolerante a bordes, precisión, *recall*, mIoU |
| | Generalización al Perú | Caída de IoU Sen1Floods11 → Perú; proporción de la brecha cerrada |
| | Calibración | ECE (%) |
| | Detección de errores | AUROC de la incertidumbre (−\|z\|) |
| **Interviniente** (controlled / reported) | Evento y etiquetas | 2023 (UNOSAT) vs 2017 (EMS); recortes dudosos; estratos (NDWI, distancia al borde, nubes) |

Scale: ratio (all indicators are continuous proportions or percentages).

## Metodología (§1.4), proposed classification

- **Enfoque:** cuantitativo.
- **Tipo:** aplicada.
- **Nivel:** explicativo: the effect of the model and the method on performance, under control.
- **Diseño:** experimental, made of **controlled computational experiments**. The independent
  variable is manipulated; everything else is fixed: official and spatial splits, 3 seeds,
  configurations frozen before the single test evaluation.
  - G1 p. 24 links "experimental" to full control of variables. The text must justify the term
    with that control.
  - The alternative label "cuasiexperimental" is weaker and does not fit, since the model is
    manipulated and nothing else varies. Authors' call.
- **Método:** hipotético-deductivo.
- **Población y muestra:**
  - **Population:** Sentinel-2 L1C images of floods.
  - **Sample:** Sen1Floods11 (hand-labelled chips: official train 252 / valid 89 / test 90) and
    the Peru set (1,291 recortes from 2023 in a spatial split, plus 204 from 2017).
  - The sample is not random. The thesis states this as a threat to external validity.
- **Técnicas e instrumentos:**
  - **Recolección:** download from CDSE; automatic labels from expert maps (UNOSAT, Copernicus
    EMS).
  - **Análisis:** pooled metrics; a paired chip bootstrap (2,000 replicates) for 95 % CIs;
    stratified analysis.

---

## Expected verdict per hypothesis, from results already in hand (for the authors only)

This is not thesis text. It shows how §3.3 will read, so the hypotheses are approved knowing
the outcome. Sources: `docs/phaseD_test.md`, `docs/published_comparison.md`,
`docs/failure_analysis_dev.md`, METHODS_NOTES 2026-10-16.

| | Verdict | Evidence |
|---|---|---|
| HE1 | **Supported, with a caveat** | In-domain: U-Net − Prithvi +0.013 [−0.022, 0.062]; U-Net − TerraMind CI also includes 0. Drop to `peru-test`: U-Net −0.257, TerraMind −0.192, Prithvi −0.080, so both FMs lose less. TerraMind's margin is within its seed spread, and no CI for the difference of drops has been computed (it needs an independent bootstrap). Training on Peru data raises every model to 0.815–0.828 |
| HE2 | **Partly supported** | Strata: errors concentrate at label edges, in low-contrast water and in small water bodies. The desert drives the false positives of source-only models. Clouds matter **less** than expected and flooded vegetation is not a main stratum, so the stratum list is partly wrong. Adaptation on `peru-test`: U-Net trHM +0.222 [0.132, 0.309]; TerraMind trFDA +0.082 [0.016, 0.156]; neither reaches the Peru-trained model. Prithvi: no adaptation helped on dev, so it has no adapted row. On 2017 the adapted rows are below source-only (labels measure only clear-water recall) |
| HE3 | **Partly supported** | Raw → temperature/ensemble lowers ECE for every row on test. AUROC > 0.5 for most rows, **but** the Peru-trained Prithvi has 0.44 on `peru-test`. Near-edge calibration degrades on test for all rows. The uncertainty does not flag low-contrast water (dev) |

The general hypothesis is therefore **partly supported**. That is a legitimate, reportable
outcome. Rewording the hypotheses now to match would break the "not fitted to results" rule.
