---
title: "Informe de sprint, semana 6: EDA, ingeniería de características y experimento A/B de los datos"
author:
  - Gerson Andre Garrido Mansilla
  - Renzo Elvis Cerron Tome
date: 2026-10-23
lang: es
---

# Informe de sprint, semana 6

**Proyecto de tesis.** Segmentación de áreas inundadas en imágenes Sentinel-2 de Perú con modelos
fundacionales geoespaciales. Maestría en Ciencias con mención en Inteligencia Artificial, UNI.

**Rama del entregable.** `avance-sem6-sprint`. Los avances anteriores están en `avance-sem4-eda` y
`avance-sem5-feature-engineering`, ambos incluidos en `main`.

**Material del sprint.**

| Parte | Cuaderno (ejecutado) | Documento de apoyo |
|---|---|---|
| Análisis exploratorio (semana 4) | `notebooks/eda.ipynb` | `docs/peru_data_card.md` |
| Ingeniería de características (semana 5) | `notebooks/feature_engineering.ipynb` | `docs/feature_engineering.md` |
| Experimento A/B de los datos (semana 6) | `notebooks/baseline_variantes.ipynb` | `notebooks/results/sprint6_results.csv` |

Todas las cifras de este informe provienen de esos cuadernos o de los documentos del proyecto que se
citan en cada sección. El registro de corridas está en `docs/EXPERIMENT_LOG.md` y el de decisiones
metodológicas en `docs/METHODS_NOTES.md`.

## 1. Contexto y métrica central

**Objetivo del proyecto.** Segmentar el agua de inundación en imágenes Sentinel-2 de Perú. El caso
realista es una inundación nueva **sin etiquetas peruanas**: solo hay un conjunto etiquetado de otras
regiones (Sen1Floods11) e imágenes de Perú sin etiquetar.

**Objetivos del sprint.**
1. Conocer los datos y sus riesgos antes de modelar: fuga de información, desbalance de clases, cambio
   de dominio y calidad de las etiquetas (EDA, semana 4).
2. Medir qué aportan las características diseñadas a mano frente a la reflectancia cruda (ingeniería
   de características, semana 5).
3. Probar con un experimento A/B si **cambiar solo los datos de entrenamiento** corrige el problema
   que encontraron las semanas 4 y 5 (semana 6).

**Métrica central.** IoU de agua agrupado sobre todos los píxeles etiquetados de `peru-dev`:

$$\mathrm{IoU} = \frac{TP}{TP + FP + FN}.$$

Se elige el IoU y no la exactitud porque el agua es minoritaria (sección 3): un modelo que no detecta
agua tendría una exactitud alta. **Métricas secundarias:** precisión y recall del agua, e IoU en
`peru-test` (evaluación única).

## 2. Datos

| Conjunto | Fuente | Recortes | Uso |
|---|---|---|---|
| Sen1Floods11 (etiquetas manuales) | [@bonafilia_sen1floods11_2020] | 252 / 89 / 90 (train / valid / test oficiales) | Entrenamiento y selección del experimento A/B; no se vuelve a partir |
| Perú 2023, ciclón Yaku (Lambayeque y Piura) | Mapas de UNOSAT [@unosat_lambayequepiura22mar_2023; @unosat_lambayequepiura1apr_2023] | 753 / 174 / 172 (`peru-train` / `peru-dev` / `peru-test`) | Imágenes de `peru-train` sin etiquetas para las variantes; `peru-dev` para comparar; `peru-test` una sola vez |
| Perú 2017, Niño Costero | Copernicus EMS EMSR199 [@copernicus_emsr199_2017] | 204, de ellos 101 sin contacto con `peru-train` | Prueba en otro evento |

- **Imágenes.** Sentinel-2 L1C, 13 bandas, recortes de 512 × 512 píxeles a 10 m (5.12 × 5.12 km),
  en UTM 17S. Las imágenes de Perú se descargan de la API de Copernicus Data Space.
- **Armonización radiométrica.** Se elimina el desplazamiento de +1000 ND de los productos con
  *processing baseline* ≥ 04.00, para que los valores de Perú sean comparables con Sen1Floods11.
- **Etiquetas de Perú sin anotación manual.** Los polígonos expertos se rasterizan con reglas fijas:
  1 = agua, 0 = no agua, −1 = ignorado. Se ignoran los píxeles fuera del área analizada por UNOSAT,
  bajo nubes o sin datos.
- **Partición espacial.** Bloques de 4 × 4 recortes (unos 20 km), asignados al azar con semilla fija.
  Los recortes de entrenamiento vecinos de dev o prueba se descartan como *buffer* (192 recortes).
  Una misma ubicación queda siempre en la misma partición en las dos fechas.

## 3. Análisis exploratorio (semana 4)

Cuaderno: `notebooks/eda.ipynb`. Solo se leen etiquetas de `peru-train`, `peru-dev` y de las
particiones train y valid de Sen1Floods11. De las particiones de prueba solo se muestran metadatos.

**Hallazgos principales.**

- **Desbalance.** El agua es una clase minoritaria, y su proporción cambia entre particiones:

  | Conjunto | Agua (% de píxeles etiquetados) | Recortes sin agua |
  |---|---|---|
  | `peru-train` | 4.6 % | 64.5 % |
  | `peru-dev` | 12.4 % | 36.2 % |
  | Sen1Floods11 train | 9.5 % | 13.1 % |
  | Sen1Floods11 valid | 11.0 % | 13.5 % |

- **Fuga de información.** Ninguna ubicación aparece en dos particiones (0 casos encontrados).
- **Cambio de dominio.** El suelo seco de Perú (desierto de Sechura) es mucho más brillante que el
  de Sen1Floods11; el agua se parece más entre ambos dominios (`docs/domain_shift_dev.md`).
- **Calidad de etiquetas.** 16 de 753 recortes de entrenamiento y 8 de 174 de dev contradicen
  claramente al índice MNDWI (`docs/peru_doubtful_chips.md`).
- **Correlación entre bandas** (`peru-train`): igual o mayor a 0.97 dentro de cada grupo (visible
  B02–B05, borde rojo/NIR B06–B8A, SWIR B11–B12), entre 0.70 y 0.91 entre grupos.
- **Nubes.** Los detectores estándar (s2cloudless, SCL) marcan el desierto como nube, así que se usa
  una regla propia de nube espesa.

**Decisiones tomadas a partir del EDA.**

| Riesgo | Decisión |
|---|---|
| Fuga espacial | Partición por bloques con *buffer* |
| Desbalance | IoU de agua agrupado como métrica principal (no exactitud); precisión y recall por separado |
| Distinta proporción de agua entre particiones | Comparar siempre dentro de la misma partición |
| Cambio de dominio | Estudiar cómo transferir desde Sen1Floods11 sin etiquetas peruanas (experimento de la semana 6) |
| Ruido en etiquetas | Etiquetas congeladas; subconjunto sin recortes dudosos como análisis de sensibilidad |

## 4. Ingeniería de características (semana 5)

Cuaderno: `notebooks/feature_engineering.ipynb`. Informe: `docs/feature_engineering.md`.

**Experimento.** Clasificador por píxel (*gradient boosting*) con cuatro conjuntos de variables:
- 13 bandas;
- 8 índices: NDWI, MNDWI [@xu_modification_2006], MNDWI2, NDVI, NDMI, AWEI_nsh, AWEI_sh y WRI;
- bandas más índices;
- todo lo anterior más 5 variables de contexto local (ventanas de 3 × 3 y 7 × 7).

Hay dos escenarios: (a) entrenado con etiquetas de Perú y (b) entrenado solo con Sen1Floods11. La
evaluación se hace en `peru-dev`, sobre 1/4 de los píxeles, con 3 semillas.

| Escenario | Mejor clasificador | Umbral de MNDWI |
|---|---|---|
| (a) etiquetas de Perú | 0.930 ± 0.001 (bandas crudas) | 0.924 (umbral 0.28, ajustado en `peru-train`) |
| (b) solo Sen1Floods11 | 0.71–0.74 (según el conjunto) | 0.836 (umbral 0.15, ajustado en Sen1Floods11) |

**Conclusiones que alimentan la semana 6.**
- Con etiquetas del dominio, las variables diseñadas no añaden información.
- Sin etiquetas de Perú, los clasificadores entrenados en Sen1Floods11 confunden el desierto brillante
  con agua (precisión de 0.72 a 0.74).
- **El problema principal es el cambio de dominio del suelo seco, no la falta de variables.** Por eso
  el experimento de esta semana cambia los datos y no el modelo.

## 5. Línea base

**`UNet_s1f11_raw`.** U-Net con codificador ResNet-34 [@ronneberger_unet_2015; @he_deep_2016]
preentrenado en ImageNet, 13 bandas L1C, entrenada con las imágenes crudas de Sen1Floods11 `train`.
Es lo que se obtendría hoy ante una inundación nueva en Perú, sin etiquetas: **0.836 ± 0.022** de IoU
en `peru-dev`, con precisión 0.848 y recall 0.985. Su error dominante son los falsos positivos en el
desierto. En `peru-test` obtiene 0.576.

## 6. Experimentos A/B: un solo cambio por variante

**Lo que comparten las tres filas** (`docs/EXPERIMENT_LOG.md`, 2026-10-14/15 y 2026-10-15/16):

| Elemento | Valor fijo |
|---|---|
| Modelo | U-Net, codificador ResNet-34 (ImageNet), 13 bandas L1C |
| Datos de entrenamiento | Sen1Floods11 `train` (252 recortes) |
| Selección | Mejor época por IoU de agua en Sen1Floods11 `valid` |
| Configuración | `configs/unet.yaml`: pérdida Dice, AdamW, tasa de aprendizaje 3e-4, lote 8, aumento D4 |
| Parada | Máximo 200 épocas, parada temprana con paciencia 20 |
| Semillas | 0, 1 y 2 |
| Hardware | RunPod RTX 4090, precisión mixta de 16 bits |

**Las variantes.** Solo cambia la transformación de las imágenes de entrenamiento (y de `valid`, para
que la selección vea el mismo estilo de entrada). En los tres casos el modelo final se aplica a las
imágenes de Perú **sin modificar**.

| ID | Rol | Qué cambia | Por qué |
|---|---|---|---|
| `UNet_s1f11_raw` | *Baseline* | Nada | Referencia sin adaptación |
| `UNet_s1f11_trHM` | Var1 | Igualación de histogramas por banda: cada valor se lleva al cuantil equivalente de las imágenes de `peru-train` sin etiquetas (1 001 cuantiles) | Si el problema es que el suelo seco peruano tiene otra distribución de brillo, igualar las distribuciones debería quitarlo |
| `UNet_s1f11_trFDA_b0.01` | Var2 | FDA: la amplitud de Fourier de baja frecuencia (β = 0.01) de cada imagen se reemplaza por la de una imagen de `peru-train` al azar; se conserva la fase | Alternativa que transfiere solo el "estilo" global de la imagen y conserva su estructura |

**Filas de referencia.** No son variantes, porque cambian más de una cosa:
- la U-Net entrenada con etiquetas de Perú: cota superior;
- el umbral de MNDWI de la semana 5.

## 7. Resultados comparables

### 7.1 Tabla estándar

| Rol | ID | Qué cambió | IoU `peru-dev` (media ± de) | Semillas 0 / 1 / 2 | Precisión / recall dev | IoU `peru-test` [IC 95 %] | Costo / latencia | Log |
|---|---|---|---|---|---|---|---|---|
| *Baseline* | `UNet_s1f11_raw` | nada | 0.8364 ± 0.0224 | 0.8433 / 0.8545 / 0.8114 | 0.848 / 0.985 | 0.576 [0.467, 0.669] | 5.8 s/época; 336–648 s por corrida; inferencia sin costo extra | `outputs/s1f11_src_unet/` |
| Var1 | `UNet_s1f11_trHM` | histogramas igualados a Perú | **0.9118 ± 0.0026** | 0.9148 / 0.9098 / 0.9108 | 0.962 / 0.946 | **0.798 [0.704, 0.865]** | +0.11 s/recorte en CPU (≈ 25 s/época con 4 procesos de carga); inferencia igual | `outputs/s1f11_trhm_unet/` |
| Var2 | `UNet_s1f11_trFDA_b0.01` | amplitud de Fourier de Perú | 0.6151 ± 0.1122 | 0.7251 / 0.5009 / 0.6193 | no registrado | no evaluado (descartado en dev) | +0.26 s/recorte en CPU; inferencia igual | `outputs/s1f11_trfda_unet/` |
| Referencia | `UNet_peru` | usa etiquetas de Perú (cota superior) | 0.9415 ± 0.0019 | 0.9397 / 0.9434 / 0.9414 | — | 0.828 [0.732, 0.895] | — | `outputs/runs_cloud/unet/` |
| Referencia | `MNDWI_s1f11_t0.15` | umbral sin aprendizaje (semana 5) | 0.836 (1/4 de los píxeles) | — | — | — | casi nulo | `outputs/feature_engineering/` |

Fuentes: `docs/EXPERIMENT_LOG.md` (2026-10-13/14, 2026-10-14/15, 2026-10-15, 2026-10-15/16, 2026-10-16),
`docs/METHODS_NOTES.md` (2026-10-16, "Pod session 2 results" y "Feature engineering"),
`docs/phaseD_test.md`. El costo de GPU es de 0.89 USD/h (RTX 4090). La tabla está también en
`docs/figs/sprint6/tabla_ab.csv`, y las referencias en `docs/figs/sprint6/tabla_ab_referencias.csv`.

### 7.2 Gráfico

![Izquierda: IoU de agua en `peru-dev` por semilla (puntos) y media ± desviación estándar (barra), con la cota superior de la U-Net entrenada con etiquetas de Perú. Derecha: precisión y recall en dev.](figs/sprint6/fig_ab_datos.png)

La clase sugiere una curva PR para clasificación. Requiere los logits de cada modelo, que no están en
esta máquina (`outputs/` no se versiona), así que se muestran la precisión y el recall en el punto de
operación del modelo (probabilidad 0.5).

### 7.3 Confirmación en prueba (evaluación única)

| Comparación (diferencias pareadas por recorte) | Diferencia | IC 95 % |
|---|---|---|
| Var1 − *baseline*, `peru-test` | +0.222 | [0.132, 0.309] |
| Var1 − U-Net con etiquetas de Perú, `peru-test` | −0.030 | [−0.048, −0.013] |
| Var1 − *baseline*, `peru-2017-test` (101 recortes) | −0.064 | [−0.083, −0.046] |

Fuente: `docs/METHODS_NOTES.md`, 2026-10-16 ("One-time test evaluation"). También en
`docs/figs/sprint6/tabla_ab_prueba.csv`.

## 8. Validación

- **Partición agrupada, sin fuga.** `peru-dev` y `peru-test` salen de bloques espaciales con *buffer*:
  es una partición por grupos, en la que una ubicación nunca aparece en dos particiones. Sen1Floods11
  usa sus particiones oficiales.
- **Divide primero, transforma después.** Los cuantiles de la igualación de histogramas y las
  amplitudes de FDA se calculan **solo** con imágenes de `peru-train`, sin leer sus etiquetas. Nunca se
  usan imágenes de `peru-dev` ni de prueba para ajustar la transformación.
- **Semillas y particiones fijas.** Las tres filas usan las semillas 0, 1 y 2 y las mismas particiones.
- **La selección no usa dev.** La mejor época se elige en Sen1Floods11 `valid`; `peru-dev` solo
  compara variantes.
- **Prueba una sola vez.** `peru-test` y `peru-2017-test` se evaluaron el 2026-10-16, desde
  configuraciones congeladas con el hash de cada checkpoint (`configs/frozen.yaml`).
- **Sin validación cruzada K-fold.** Las reglas del proyecto prohíben volver a partir Sen1Floods11, y
  un K-fold sobre bloques de Perú obligaría a reentrenar cada fila K veces en GPU alquilada. En su
  lugar se reportan 3 semillas e intervalos *bootstrap* por recortes.
- **Ruido entre corridas.** Repetir la semilla 0 del *baseline* con la misma configuración dio una
  diferencia de 0.017 de IoU en `valid` (no determinismo de cuDNN y precisión mixta). Las diferencias
  del experimento (+0.075 y −0.221 en dev) son mucho mayores.
- **Diferencia operativa.** Var1 y Var2 se entrenaron con 8 procesos de carga de datos en lugar de 4,
  para que la transformación en CPU no frenara el entrenamiento. Cambia el orden aleatorio del
  aumento de datos, no los datos ni el modelo.

## 9. Conclusión y decisión

La regla de elección entre HM y FDA se fijó **antes** de ver resultados: el IoU en dev de la semilla
0 (`docs/METHODS_NOTES.md`, 2026-10-15). HM obtuvo 0.9148 y FDA 0.7251. Solo la variante elegida pasó
a la evaluación única en prueba.

**Se adopta Var1 (`UNet_s1f11_trHM`).**
- **Mejora consistente.** Supera al *baseline* en todas las semillas (+0.072, +0.055 y +0.100 en dev)
  y reduce la variación entre semillas (desviación estándar 0.003 frente a 0.022).
- **Corrige el error que diagnosticaron las semanas 4 y 5.** La precisión sube de 0.848 a 0.962: se
  eliminan la mayoría de los falsos positivos del desierto. El recall baja de 0.985 a 0.946.
- **Se confirma en prueba.** +0.222 [0.132, 0.309] sobre el *baseline*. Queda a 0.030 de la U-Net
  entrenada con etiquetas peruanas, sin usar ninguna.
- **Costo viable.** La transformación cuesta 0.11 s por recorte en CPU, solo durante el
  entrenamiento; la inferencia no cambia.

**Se descarta Var2 (`UNet_s1f11_trFDA_b0.01`).** Baja el IoU a 0.615 ± 0.112, es inestable entre
semillas y es la variante más costosa (0.26 s por recorte). No se evaluó en prueba, como manda el
protocolo.

## 10. Reproducibilidad

**Comandos de entrenamiento** (RunPod, desde la raíz del repositorio; `docs/EXPERIMENT_LOG.md`):

```
# Baseline: UNet_s1f11_raw (commit 8a8fee4)
DATASET=s1f11 RUN_NAME=s1f11_src_unet bash scripts/cloud/run_queue.sh unet "0 1 2" \
  --optimizer.init_args.lr=3e-4 --trainer.max_epochs=200

# Var1: UNet_s1f11_trHM (commit e8001a4)
DATASET=s1f11 RUN_NAME=s1f11_trhm_unet bash scripts/cloud/run_queue.sh unet "0 1 2" \
  --optimizer.init_args.lr=3e-4 --trainer.max_epochs=200 -c configs/phaseD/train_hm.yaml \
  --data.init_args.num_workers=8

# Var2: UNet_s1f11_trFDA_b0.01 (commit e8001a4)
DATASET=s1f11 RUN_NAME=s1f11_trfda_unet bash scripts/cloud/run_queue.sh unet "0 1 2" \
  --optimizer.init_args.lr=3e-4 --trainer.max_epochs=200 -c configs/phaseD/train_fda.yaml \
  --data.init_args.num_workers=8
```

**Evaluación en `peru-dev` y cuaderno:**

```
bash scripts/phaseD2_dev_logits.sh                 # logits en peru-dev de los modelos
uv run python scripts/phaseD_dev_table.py          # tabla de dev (outputs/phaseD/)
uv run python notebooks/build_baseline_nb.py       # cuaderno, tablas y figuras de este informe
```

**Logs.** Cada corrida guarda `config.yaml`, `metrics.csv` y `run.json`, con su commit y su estado
`dirty`, en `outputs/s1f11_src_unet/`, `outputs/s1f11_trhm_unet/` y `outputs/s1f11_trfda_unet/` (subcarpetas `unet/lightning_logs/seed0` a `seed2`). El registro de todas las corridas está
en `docs/EXPERIMENT_LOG.md`. La tabla de este informe, con la fuente de cada cifra, está en
`notebooks/results/sprint6_results.csv`.

**Instantánea de los datos.**
- Perú v1, congelado el 2026-10-14: 2 990 archivos fijados por SHA-256 en
  `docs/peru_v1_checksums.txt`.
- Ocupan 5.3 GB (`data/peru`); Sen1Floods11, 977 MB.
- El 2026-10-23 se reconstruyeron desde cero en una segunda máquina y coincidieron bit a bit
  (`uv run python scripts/freeze_peru.py --check`).

**Figuras para LaTeX.** `docs/figs/sprint6/fig_ab_datos.pdf` y `docs/figs/sprint6/fig_anexo_modelos.pdf` (vectoriales), además de los PNG enlazados aquí.

**Versiones** (`uv.lock`): Python 3.12, PyTorch 2.14.0, Lightning 2.6.6, TerraTorch 1.2.13 y TorchGeo
0.9.0.

## 11. Amenazas a la validez

| Tipo | Amenaza | Mitigación o estado |
|---|---|---|
| Interna | Etiquetas de UNOSAT preliminares, sin validación en campo | Sin los recortes dudosos, cada fila de prueba sube entre 0.005 y 0.016; el único cambio de orden es entre dos filas de TerraMind, ajenas a este experimento (`docs/phaseD_test.md`) |
| Interna | Var1 y Var2 usaron 8 procesos de carga y el *baseline* 4 | Solo afecta el orden aleatorio del aumento de datos; la diferencia observada supera mucho el ruido entre corridas (0.017) |
| Externa | Un solo evento de entrenamiento en Perú | Prueba en otro evento (2017): allí Var1 queda por debajo del *baseline* |
| De constructo | Las etiquetas de 2017 solo definen suelo seco donde MNDWI ≤ 0, así que miden sobre todo el recall del agua clara | Se informa por separado; la decisión se limita al evento de 2023 |
| De conclusión estadística | 3 semillas y una sola partición espacial; sin K-fold | Diferencias pareadas con IC *bootstrap* en prueba |

## 12. Riesgos y próximos pasos

- **Riesgo: deriva hacia otros eventos.** Var1 sacrifica agua tenue para eliminar falsos positivos. En
  el evento de 2017 eso la deja 0.064 por debajo del *baseline*.
- **Acción inmediata del siguiente sprint.** Medir en `peru-dev` cuánta agua tenue (MNDWI entre 0 y
  0.3) pierde Var1 por estrato, y pasar este informe a LaTeX con la plantilla de la tesis.

## 13. Guion de la demo (exposición de la semana 6)

| Paso | Qué decir | Qué mostrar |
|---|---|---|
| 1. Contexto (1 min) | Inundación nueva en Perú sin etiquetas; solo hay Sen1Floods11 e imágenes peruanas sin etiquetar. Métrica: IoU de agua en `peru-dev`. | Cuaderno, sección inicial |
| 2. Lo que dijeron los datos (1 min) | El EDA y la semana 5 mostraron que el problema es el desierto brillante, no las variables. | Esta guía, secciones 3 y 4 |
| 3. Línea base (30 s) | U-Net entrenada en Sen1Floods11: 0.836, con muchos falsos positivos (precisión 0.848). | Tabla estándar, fila *baseline* |
| 4. Variantes (1 min) | Mismo modelo y configuración; solo cambian las imágenes de entrenamiento: igualar histogramas o transferir la amplitud de Fourier, con imágenes de Perú sin etiquetas. | Cuaderno, sección 1 |
| 5. Resultado (1 min) | HM sube a 0.912 en todas las semillas, gracias a la precisión; FDA cae a 0.615 y es inestable. | `fig_ab_datos.png` |
| 6. Decisión (1 min) | Se adopta HM: en prueba, +0.222 sobre el *baseline* y a 0.030 de entrenar con etiquetas de Perú, sin costo en la inferencia. FDA se descarta y no se evalúa en prueba. | Cuaderno, sección 5 |
| 7. Riesgo y siguiente paso (30 s) | En otro evento (2017), HM pierde agua tenue; el próximo sprint lo mide por estrato. | Esta guía, sección 12 |

## Anexo: comparación de modelos con etiquetas de Perú (contexto)

Este anexo no forma parte del A/B de la semana 6: cambia el modelo, no los datos. Los tres modelos se
entrenaron con `peru-train` (etiquetas de Perú), con la configuración oficial de cada uno, 3 semillas,
y selección en `peru-dev` (`docs/EXPERIMENT_LOG.md`, 2026-10-13/14).

| Modelo | IoU `peru-dev` (media ± de) | IoU `peru-test` [IC 95 %] | Precisión / recall `peru-test` | IoU `peru-2017-test` (101) |
|---|---|---|---|---|
| U-Net (ResNet-34) | 0.9415 ± 0.0019 | 0.828 [0.732, 0.895] | 0.966 / 0.853 | 0.878 |
| TerraMind v1-B [@jakubik_terramind_2025] | 0.9265 ± 0.0025 | 0.815 [0.718, 0.885] | 0.960 / 0.843 | 0.892 |
| Prithvi-EO-2.0 300M [@szwarcman_prithvi-eo-2_2024] | 0.9218 ± 0.0007 | 0.824 [0.731, 0.890] | 0.966 / 0.850 | 0.878 |

![Anexo: IoU de agua en `peru-dev` de los modelos entrenados con etiquetas de Perú, por semilla y media ± desviación estándar.](figs/sprint6/fig_anexo_modelos.png)

- En dev, la U-Net supera a los dos modelos fundacionales. El 73 % y el 86 % de esa ventaja está en la
  franja de 1 px del borde de la etiqueta (`docs/METHODS_NOTES.md`, 2026-10-14).
- En `peru-test` las tres medias quedan a menos de 0.014 entre sí, con IC solapados. No se calcularon
  diferencias pareadas entre ellas, así que no se afirma un orden.
- Con etiquetas del mismo evento el modelo importa poco: un umbral de MNDWI ajustado en `peru-train`
  ya llega a 0.924 en dev (semana 5). Por eso esta semana el experimento se centra en los datos.

Tabla también en `docs/figs/sprint6/tabla_anexo_modelos.csv`.
