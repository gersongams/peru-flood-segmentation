# Chapter I backbone: problems, objectives, hypotheses, variables (proposal for approval)

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
