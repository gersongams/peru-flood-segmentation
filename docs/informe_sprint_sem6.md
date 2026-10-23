---
title: "Informe de sprint, semana 6: EDA, ingeniería de características y experimento *baseline* + variantes"
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
| *Baseline* + variantes (semana 6) | `notebooks/baseline_variantes.ipynb` | `notebooks/results/sprint6_results.csv` |

Todas las cifras de este informe provienen de esos cuadernos o de los documentos del proyecto que se
citan en cada sección. El registro de corridas está en `docs/EXPERIMENT_LOG.md` y el de decisiones
metodológicas en `docs/METHODS_NOTES.md`.

## 1. Objetivos del sprint

1. Conocer los datos y sus riesgos antes de modelar: fuga de información, desbalance de clases,
   cambio de dominio y calidad de las etiquetas (EDA).
2. Medir qué aportan las características diseñadas a mano (índices espectrales y contexto local)
   frente a la reflectancia cruda (ingeniería de características).
3. Comparar un *baseline* convolucional con dos variantes basadas en modelos fundacionales
   geoespaciales, con el mismo protocolo y las mismas etiquetas peruanas (experimento).

## 2. Datos

| Conjunto | Fuente | Recortes | Uso |
|---|---|---|---|
| Sen1Floods11 (etiquetas manuales) | [@bonafilia_sen1floods11_2020] | 252 / 89 / 90 (train / valid / test oficiales) | Dominio fuente; no se vuelve a partir |
| Perú 2023, ciclón Yaku (Lambayeque y Piura) | Mapas de UNOSAT [@unosat_lambayequepiura22mar_2023; @unosat_lambayequepiura1apr_2023] | 753 / 174 / 172 (`peru-train` / `peru-dev` / `peru-test`) | Entrenamiento, selección y prueba |
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
- **Versión congelada.** El conjunto de Perú v1 está fijado por sumas SHA-256
  (`docs/peru_v1_checksums.txt`). En este sprint se reconstruyó desde cero en una segunda máquina,
  y los 2 990 archivos coincidieron bit a bit con la v1 (`docs/EXPERIMENT_LOG.md`, 2026-10-23).

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
| Cambio de dominio | Entrenar con `peru-train`; adaptación sin etiquetas en una fase posterior |
| Ruido en etiquetas | Etiquetas congeladas; subconjunto sin recortes dudosos como análisis de sensibilidad |

## 4. Ingeniería de características (semana 5)

Cuaderno: `notebooks/feature_engineering.ipynb`. Informe: `docs/feature_engineering.md`.

**Lo que ya existía en el proyecto.** Los modelos profundos reciben reflectancia cruda, pero el
proyecto ya transforma y selecciona entradas en varios puntos:
- selección de bandas por modelo (6 bandas HLS para Prithvi);
- normalización oficial de cada modelo;
- armonización radiométrica;
- índices espectrales como línea base, como control de calidad de etiquetas y para estratificar
  errores;
- igualación de histogramas y FDA como transformaciones de dominio.

**Experimento.** Clasificador por píxel (*gradient boosting*) con cuatro conjuntos de variables:
- 13 bandas;
- 8 índices: NDWI, MNDWI [@xu_modification_2006], MNDWI2, NDVI, NDMI, AWEI_nsh, AWEI_sh y WRI;
- bandas más índices;
- todo lo anterior más 5 variables de contexto local (ventanas de 3 × 3 y 7 × 7).

Hay dos escenarios: (a) entrenado con etiquetas de Perú y (b) entrenado solo con Sen1Floods11. La
evaluación se hace en `peru-dev`, sobre 1/4 de los píxeles, con 3 semillas.

**Resultados.**

| Escenario | Mejor clasificador | Umbral de MNDWI |
|---|---|---|
| (a) etiquetas de Perú | 0.930 ± 0.001 (bandas crudas) | 0.924 (umbral 0.28, ajustado en `peru-train`) |
| (b) solo Sen1Floods11 | 0.71–0.74 (según el conjunto) | 0.836 (umbral 0.15, ajustado en Sen1Floods11) |

**Conclusiones.**
- Con etiquetas del dominio, las variables diseñadas no añaden información: el modelo ya combina las
  bandas por sí solo.
- Sin etiquetas de Perú, un índice simple e invariante transfiere mejor que un modelo flexible. Los
  clasificadores entrenados en Sen1Floods11 confunden el desierto brillante con agua (precisión de
  0.72 a 0.74).
- El problema principal es el cambio de dominio del suelo seco, no la falta de variables.
- Para el experimento de esta semana, el umbral de MNDWI ajustado en `peru-train` (0.924) queda como
  referencia sin aprendizaje profundo.

## 5. Experimento *baseline* + variantes (semana 6)

Cuaderno: `notebooks/baseline_variantes.ipynb`. El cuaderno presenta resultados ya registrados: no
entrena ni vuelve a evaluar. Cada cifra se lee de `notebooks/results/sprint6_results.csv`, donde cada
fila indica su fuente.

### 5.1 Pregunta e hipótesis

**Pregunta.** Con las mismas etiquetas peruanas y el mismo protocolo, ¿segmenta mejor el agua un
modelo fundacional geoespacial Transformer que una CNN convencional?

**Hipótesis de trabajo.** El preentrenamiento a gran escala en observación de la Tierra da a los
modelos fundacionales una ventaja sobre una CNN preentrenada en ImageNet.

### 5.2 Diseño

| Rol | Modelo | Preentrenamiento | Entradas | Tasa de aprendizaje / lote |
|---|---|---|---|---|
| ***Baseline*** | U-Net [@ronneberger_unet_2015] con codificador ResNet-34 [@he_deep_2016] | ImageNet | 13 bandas L1C | 1e-4 / 8 |
| **Variante 1** | TerraMind v1-B [@jakubik_terramind_2025], decodificador UNet | Multimodal, observación de la Tierra | 13 bandas L1C | 2e-5 / 8 |
| **Variante 2** | Prithvi-EO-2.0 300M [@szwarcman_prithvi-eo-2_2024], decodificador UNet | HLS | 6 bandas HLS | 1e-4 / 4 |
| Referencia | Umbral de MNDWI | — | 2 bandas | — |

- **Variables controladas.** Los tres modelos comparten:
  - los datos: entrenamiento en `peru-train` y elección de la mejor época en `peru-dev`;
  - la pérdida (Dice, ignorando los píxeles con etiqueta −1);
  - el optimizador (AdamW con ReduceLROnPlateau);
  - el aumento de datos (rotaciones y reflejos D4);
  - el criterio de parada: máximo 100 épocas, parada temprana con paciencia 20 sobre el IoU de agua
    en dev.

  Cada modelo usa su normalización oficial y los hiperparámetros oficiales de su configuración para
  Sen1Floods11.
- **Variable independiente:** la arquitectura y su preentrenamiento.
- **Variable dependiente:** IoU de agua agrupado,

  $$\mathrm{IoU} = \frac{TP}{TP + FP + FN},$$

  sobre todos los píxeles etiquetados de la partición. También se reportan el mIoU, la precisión,
  el recall y el IoU tolerante a bordes, que excluye los píxeles a ≤ 1 px del borde de la etiqueta.
- **Protocolo.**
  - 3 semillas por modelo; se reporta la media ± la desviación estándar muestral.
  - La selección se hace solo en `peru-dev`.
  - `peru-test` y `peru-2017-test` se evaluaron una sola vez (2026-10-16), desde configuraciones
    congeladas con el hash de cada checkpoint (`configs/frozen.yaml`).
  - Entrenamiento en GPU RTX 4090 (RunPod), entre 22 y 35 min por corrida.

### 5.3 Resultados en `peru-dev`

| Rol | Modelo | Semilla 0 | Semilla 1 | Semilla 2 | IoU media ± de | mIoU media ± de | Δ IoU frente al *baseline* |
|---|---|---|---|---|---|---|---|
| *Baseline* | U-Net | 0.9397 | 0.9434 | 0.9414 | **0.9415 ± 0.0019** | 0.9665 ± 0.0011 | — |
| Variante 1 | TerraMind v1-B | 0.9275 | 0.9237 | 0.9283 | 0.9265 ± 0.0025 | 0.9579 ± 0.0014 | −0.0150 |
| Variante 2 | Prithvi-EO-2.0 300M | 0.9212 | 0.9225 | 0.9217 | 0.9218 ± 0.0007 | 0.9552 ± 0.0004 | −0.0197 |
| Referencia | MNDWI > 0.28 (ajustado en `peru-train`) | — | — | — | 0.9237 | — | −0.0178 |

Fuentes: `docs/EXPERIMENT_LOG.md` (2026-10-13/14) y, para MNDWI, `docs/METHODS_NOTES.md`
(2026-10-16). El umbral se evaluó sobre 1/4 de los píxeles de dev.

![IoU de agua en `peru-dev` por semilla (puntos) y media ± desviación estándar (barra).](figs/sprint6/fig_dev_semillas.png)

- El *baseline* obtiene el mayor IoU. Las diferencias con las variantes (0.015 y 0.020) son varias
  veces mayores que la variación entre semillas.
- Solo la U-Net supera con claridad al umbral de MNDWI (+0.018). TerraMind queda prácticamente al
  mismo nivel (+0.003) y Prithvi por debajo (−0.002).
- **TerraMind no está infraentrenado.** Sus mejores épocas (92, 93, 88) estuvieron cerca del tope de
  100. Con un tope de 200 épocas, la semilla 0 obtuvo 0.9270 en la época 77, frente a 0.9275 con el
  tope de 100.
- **La ventaja de la U-Net está sobre todo en el borde de la etiqueta.** Al excluir los píxeles a
  ≤ 1 px del borde, cambia así (puntos de IoU, `docs/METHODS_NOTES.md`, 2026-10-14):

  | Comparación | IoU normal | Sin borde ≤ 1 px |
  |---|---|---|
  | U-Net − TerraMind | +1.50 | +0.40 |
  | U-Net − Prithvi | +1.97 | +0.28 |

  El 73 % y el 86 % de la ventaja se concentran en esa franja. Las etiquetas de UNOSAT son polígonos
  rasterizados, menos fiables justo en los bordes.

### 5.4 Resultados en las particiones de prueba (evaluación única)

| Rol | Modelo | IoU `peru-test` (media ± de) | IC 95 % | IoU tolerante | Precisión | Recall | Caída dev → prueba | IoU `peru-2017-test` (101 recortes) |
|---|---|---|---|---|---|---|---|---|
| *Baseline* | U-Net | 0.828 ± 0.002 | 0.732–0.895 | 0.861 | 0.966 | 0.853 | 0.114 | 0.878 |
| Variante 1 | TerraMind v1-B | 0.815 ± 0.002 | 0.718–0.885 | 0.852 | 0.960 | 0.843 | 0.112 | 0.892 |
| Variante 2 | Prithvi-EO-2.0 300M | 0.824 ± 0.007 | 0.731–0.890 | 0.866 | 0.966 | 0.850 | 0.097 | 0.878 |

Fuente: `docs/phaseD_test.md`. El IC del 95 % es un *bootstrap* por recortes de la media de las
semillas.

![Izquierda: `peru-test`, media e IC 95 % (círculos) y media en dev (rombos). Derecha: `peru-2017-test`.](figs/sprint6/fig_prueba.png)

- **`peru-test`.** El orden de las medias se mantiene (U-Net 0.828, Prithvi 0.824, TerraMind
  0.815), pero los IC se solapan casi por completo. No se calcularon diferencias pareadas entre estos
  tres modelos, así que no se afirma que uno supere a otro en prueba.
- **Caída de dev a prueba.** Es parecida en los tres (0.097–0.114). En la U-Net, el recall baja de
  0.97 a 0.85. Los bloques de prueba son más difíciles o están etiquetados de otra forma; no es un
  fallo de un modelo concreto.
- **Precisión alta.** Es de 0.96–0.97 en los tres modelos. En prueba domina la omisión de agua.
- **`peru-2017-test`.** TerraMind queda primero (0.892). Estas etiquetas solo definen suelo seco
  donde MNDWI ≤ 0, así que la partición mide sobre todo el recall del agua clara.

## 6. Discusión

- **La hipótesis no se confirma en este escenario.** Con etiquetas del mismo evento, el *baseline*
  U-Net no es superado por los modelos fundacionales. Esto coincide con la semana 5: en Perú 2023 la
  separación entre agua y suelo es casi un umbral espectral, así que hay poco margen para que el
  preentrenamiento aporte.
- **Parte del ranking depende de los bordes.** Fuera de la franja de 1 px, la ventaja de la U-Net es
  de 0.3–0.4 puntos. La U-Net trabaja a la resolución completa de 10 m; los Transformer procesan la
  imagen en parches y recuperan la resolución en el decodificador.
- **El valor de los modelos fundacionales hay que buscarlo en otro escenario.** La semana 5 mostró
  que el problema principal es transferir sin etiquetas peruanas, por el cambio de dominio del suelo
  seco. Ese es el siguiente bloque de la tesis.

## 7. Amenazas a la validez

| Tipo | Amenaza | Mitigación o estado |
|---|---|---|
| Interna | Etiquetas de UNOSAT preliminares, sin validación en campo, poco fiables en los bordes | IoU tolerante a bordes; subconjunto sin recortes dudosos |
| Interna | Hiperparámetros oficiales de cada modelo, no ajustados para Perú | Igual criterio de parada para todos; tope de épocas comprobado en TerraMind |
| Interna | Selección de la mejor época en `peru-dev` | Prueba evaluada una sola vez, con configuraciones congeladas |
| Externa | Un solo evento de entrenamiento y una región (costa norte de Perú) | Prueba en otro evento (2017); su etiquetado mide sobre todo recall |
| De constructo | El IoU agrupado lo dominan los recortes con mucha agua | IC por *bootstrap* de recortes; precisión y recall por separado |
| De conclusión estadística | 3 semillas, una sola partición espacial; sin diferencias pareadas entre modelos en prueba | No se afirma un orden en prueba |
| De conclusión estadística | El umbral de MNDWI se evaluó sobre 1/4 de los píxeles de dev | La comparación con MNDWI se presenta como aproximada |

## 8. Próximos pasos

1. Presentar la comparación sin etiquetas peruanas (modelos entrenados en Sen1Floods11 y adaptados
   a Perú), ya ejecutada en el proyecto (`docs/phaseD_dev.md`, `docs/phaseD_test.md`).
2. Pasar este informe a LaTeX con la plantilla de la tesis (UNI).
3. Trabajo futuro: probar MNDWI como canal adicional de entrada de la U-Net, solo en dev.

## 9. Reproducibilidad

```
uv sync
uv run python scripts/download_sen1floods11.py
uv run python scripts/build_peru_unosat.py         # requiere credenciales CDSE en .env
uv run python scripts/split_peru_unosat.py
uv run python scripts/build_peru_emsr2017.py
uv run python scripts/freeze_peru.py --check       # verifica la v1 (2 990 archivos)
uv run python notebooks/build_eda_nb.py
uv run python scripts/feature_engineering.py
uv run python notebooks/build_feature_engineering_nb.py
uv run python notebooks/build_baseline_nb.py       # tablas y figuras en docs/figs/sprint6/
```

El cuaderno de EDA lee además `outputs/band_stats` y `outputs/label_qa`, generados por
`scripts/band_stats.py` y `scripts/label_qa.py`. El cuaderno de esta semana corre sin `outputs/`.
