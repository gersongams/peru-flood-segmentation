# Ingeniería de características — avance semana 5

Rama: `avance-sem5-feature-engineering` (no se fusiona con `main`).
Autores: Gerson Andre Garrido Mansilla, Renzo Elvis Cerron Tome. Fecha: 2026-10-16.

**Tema de la tesis.** Segmentación de áreas inundadas en imágenes Sentinel-2 en Perú. Se
comparan dos modelos fundacionales geoespaciales, TerraMind v1-B y Prithvi-EO-2.0 300M, con
una U-Net de referencia. Datos:
- Sen1Floods11, con sus particiones oficiales;
- Perú 2023, con etiquetas de UNOSAT;
- Perú 2017, con etiquetas de Copernicus EMS (EMSR199).

**Alcance de este avance.** Todo se evalúa solo en `peru-dev`, la partición de selección, y
el trabajo es exploratorio. No se lee ninguna imagen ni etiqueta de `peru-test` ni de
`peru-2017-test`; el script lo verifica con un `assert`. Nada de lo que aparece aquí cambia la
comparación congelada de la tesis (`configs/frozen.yaml`).

## 1. Qué ingeniería de características ya existe en el proyecto

Los modelos profundos de la tesis reciben **reflectancia cruda** (Sentinel-2 L1C) y aprenden
sus propias características. Es una decisión deliberada y coincide con los resultados
publicados en Sen1Floods11. Aun así, el proyecto ya transforma y selecciona las entradas en
varios puntos:

| Paso | Qué hace | Dónde |
|---|---|---|
| Selección de bandas | U-Net y TerraMind reciben las 13 bandas L1C. Prithvi recibe sus 6 bandas de preentrenamiento HLS: Blue, Green, Red, NIR estrecho (B8A), SWIR1, SWIR2. | `configs/*.yaml` (`output_bands`, `backbone_bands`) |
| Normalización por modelo | Media y desviación estándar oficiales de cada modelo, copiadas de su configuración oficial para Sen1Floods11. Para la U-Net se usan las de la partición `train` de Sen1Floods11. | `src/floodseg/normalization.py`, `scripts/compute_train_stats.py` |
| Armonización radiométrica | En las imágenes L1C de Perú, desde el *processing baseline* 04.00, se elimina el desplazamiento de +1000 ND con `harmonizeValues` de Sentinel Hub. Así los valores son comparables con Sen1Floods11. | `scripts/build_peru_unosat.py`; verificación en `docs/METHODS_NOTES.md` (2026-10-14, distribuciones por banda) |
| Índices espectrales como línea base | MNDWI = (B03 − B11)/(B03 + B11) y NDWI = (B03 − B08)/(B03 + B08), con umbral fijo, umbral de Otsu sin etiquetas y umbral ajustado en dev (oráculo). | `scripts/index_baseline.py`, `scripts/phaseD_dev_table.py` |
| Índices para control de calidad de etiquetas | Se marcan como dudosas las zonas donde la etiqueta y el MNDWI se contradicen: agua con MNDWI < −0.1, o seco con MNDWI > 0.5. | `scripts/label_qa.py` |
| Índices para análisis de errores | Estratos por NDWI/MNDWI, cobertura del suelo y distancia a nubes. Muestran que el agua de bajo contraste (NDWI cercano a 0) es el estrato más difícil. | `scripts/analyze_dev.py`, `scripts/diag_2017_recall.py` |
| Transformaciones de dominio de la entrada | Igualación de histogramas por banda (1,001 cuantiles) y FDA (amplitud de baja frecuencia de Fourier, β = 0.01) entre Sen1Floods11 y las imágenes de Perú sin etiquetas, en entrenamiento y en prueba. | `src/floodseg/input_alignment.py`, `src/floodseg/train_alignment.py` |
| Diagnóstico del desplazamiento de dominio | Diferencia de medias estandarizada y distancia de Wasserstein por banda y por clase. El suelo seco de Perú (desierto de Sechura) difiere mucho más que el agua. | `scripts/band_stats.py` |

## 2. Experimento nuevo: características diseñadas para un clasificador por píxel

Script: `scripts/feature_engineering.py`. Resultados: `outputs/feature_engineering/`.

**Pregunta.** ¿Las características diseñadas (índices espectrales y contexto local) superan a
las 13 bandas crudas en un clasificador clásico por píxel? Se prueba en dos escenarios:
- (a) entrenado con etiquetas de Perú;
- (b) entrenado solo con Sen1Floods11 y aplicado a Perú, que es el escenario sin etiquetas
  peruanas que estudia la tesis.

**Cuaderno con el desarrollo completo, en español:** `notebooks/feature_engineering.ipynb`.
Ya está ejecutado y se puede ver en GitHub sin correrlo. Incluye las definiciones, mapas de
características de un recorte, separabilidad, resultados e importancia por permutación.

**Reproducir.**

```
uv run python scripts/feature_engineering.py             # unos 10 min en un Apple M4 Max
uv run python notebooks/build_feature_engineering_nb.py   # regenera y ejecuta el cuaderno
```

### Resultados y conclusiones

Todas las cifras son de `peru-dev`, evaluadas sobre 1/4 de los píxeles, con 3 semillas
(media ± desviación estándar). Fuente: `outputs/feature_engineering/results.csv`.

**(a) Con etiquetas de Perú, las características diseñadas no mejoran al clasificador.**

| Conjunto de características | IoU de agua en `peru-dev` |
|---|---|
| Bandas crudas | 0.930 ± 0.001 |
| Bandas + índices | 0.926 ± 0.003 |
| Bandas + índices + contexto | 0.929 ± 0.005 |
| Solo índices | 0.913 ± 0.004 |

- El *gradient boosting* ya combina las bandas por sí mismo: en Perú, B11 y B12 solas separan
  el agua del desierto con un AUROC cercano a 0.98.
- Un solo umbral de MNDWI ajustado con las etiquetas de `peru-train` (t = 0.28) llega a 0.924.
  Es casi lo mismo que el umbral "oráculo" ajustado en dev (0.925). En Perú 2023, la frontera
  entre agua y suelo seco es casi un umbral de MNDWI.
- La U-Net entrenada en Perú llega a 0.9415. La diferencia con el mejor clasificador por píxel
  (unos 0.01 de IoU) es lo que aporta el contexto espacial que aprende la red. Las ventanas
  de contexto diseñadas (3×3 y 7×7) no recuperan esa diferencia.

**(b) Sin etiquetas de Perú, una característica simple e invariante transfiere mejor que un
modelo flexible.**

| Modelo, entrenado solo con Sen1Floods11 | IoU de agua en `peru-dev` |
|---|---|
| *Gradient boosting*, según el conjunto de características | 0.71 a 0.74 |
| MNDWI > 0.15, umbral ajustado con las etiquetas de Sen1Floods11 | 0.836 |
| U-Net entrenada solo en Sen1Floods11 (medida antes) | 0.836 ± 0.022 |

- Los clasificadores entrenados en Sen1Floods11 tienen recall de 0.98 a 0.99 pero precisión de
  solo 0.72 a 0.74. Confunden el desierto brillante de Perú con agua, el mismo error que
  muestran los modelos profundos sin adaptar.
- La importancia por permutación explica por qué. Estos modelos dependen de bandas crudas
  (B12, B08, B06), cuyo valor cambia entre dominios. En cambio, el modelo entrenado en Perú
  depende sobre todo de índices de agua (AWEI_nsh, MNDWI2, MNDWI).
- Añadir índices y contexto al modelo flexible **no** mejora la transferencia (0.713 a 0.736).
  La flexibilidad le permite aprender la apariencia del suelo seco de Sen1Floods11.
- La separabilidad univariada de los índices de agua es alta y estable entre dominios: AUROC
  de 0.98 en Sen1Floods11 y de 0.997 en `peru-dev` para MNDWI. La de las bandas visibles, en
  cambio, se invierte. En Sen1Floods11 el agua es algo más brillante que el suelo seco en el
  visible; en Perú es más oscura que el desierto.

**Implicaciones para la tesis.**
1. Mantener entradas crudas en los modelos profundos sigue siendo razonable. Con etiquetas del
   dominio, las características diseñadas no añaden información.
2. El problema principal es el **cambio de dominio del suelo seco**, no la falta de
   características. Es lo que atacan los métodos de adaptación de la tesis: la igualación de
   histogramas y FDA.
3. Hay una línea base nueva sin etiquetas peruanas: MNDWI > umbral ajustado en Sen1Floods11.
   Obtiene 0.836 en dev, igual que la U-Net sin adaptar. Como se definió **después** de la
   evaluación única en prueba, no se evaluará en `peru-test`. Queda como resultado
   exploratorio y como trabajo futuro, junto con añadir MNDWI como canal de entrada de la
   U-Net.

**Limitaciones.**
- Solo se evalúa en `peru-dev`, con 1/4 de los píxeles y una sola partición. Las comparaciones
  con la U-Net son aproximadas, porque la U-Net se evaluó sobre todos los píxeles.
- El clasificador es por píxel. La importancia por permutación con características
  correlacionadas reparte la importancia entre ellas.
