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

(Resultados: en ejecución; se completan en el siguiente commit.)
