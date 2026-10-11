# Propuesta: conjunto de prueba de Perú a partir de Copernicus EMS EMSR199

Estado: borrador, 2026-10-11. Contexto: el asesor pidió **no usar Bolivia** como prueba fuera de distribución y construir una evaluación con datos de **Perú**.

## 1. ¿Existe un dataset etiquetado de inundaciones en Perú?

No encontramos uno público.

- **Sen1Floods11**: 11 eventos, ninguno en Perú (sí Bolivia y Colombia).
- **WorldFloods v2** (509 eventos): filtrado por coordenadas de `dataset_metadata.csv`, **ningún evento en Perú**.
- **Kuro Siwo, C2S-MS Floods, GEOID-Flood**: sin confirmación de eventos en Perú; falta revisar sus listas.
- **Copernicus EMS EMSR199** (Niño Costero 2017): **sí existe**, con mapas de delimitación hechos por expertos. Es la base propuesta.

## 2. Qué trae EMSR199

Activación del 22/03/2017, "Floods in Peru". **21 áreas (AOI), 45 productos**, de los cuales **18 son delimitaciones de inundación** (polígonos de zona inundada). El resto son mapas de referencia (pre-evento) y de grado de daño.

Datos crudos: `data/emsr199/raw/` (git-ignored). Resumen: `data/emsr199/summary.csv`. Mapas PDF: `sources/emsr199_maps/`.

| Zona | Área AOI | Inundado | Sensor usado por EMS | Fecha imagen |
|---|---|---|---|---|
| Salinas (Piura) | 1 936 km² | 805 km² (42 %) | sin dato en el shapefile | sin dato |
| Parachique (Bajo Piura / Sechura) | 1 907 km² | 790 km² (41 %) | Cosmo-SkyMed + Radarsat | 26 y 30 mar 2017 |
| Piura | 1 892 km² | 154 km² (8 %) | Cosmo-SkyMed + Radarsat | 26 y 30 mar 2017 |
| Trujillo | 1 887 km² | 148 km² (8 %) | varios | 23–27 mar 2017 |
| Piura Norte | 918 km² | 33 km² (4 %) | "Other" | 23–26 mar 2017 |
| Iquitos | 200 km² | 16 km² (8 %) | Cosmo-SkyMed | 24 mar 2017 |
| Lima, Ate, Chaclacayo, Huaura, Roma, Ciudad de Dios, Sullana | 14–237 km² c/u | 0.4–3.7 km² c/u | Radarsat, TerraSAR-X, Sentinel-1, Cosmo-SkyMed | 23–24 mar (Huaura: 24 abr) |
| Comas | 183 km² | 0 (sin capa de inundación) | — | — |

Cifras de las versiones MONIT01 (la más reciente) cuando existen. Áreas calculadas con geopandas en la proyección UTM de cada AOI.

**Lo útil está en Piura**: Salinas, Parachique, Piura y Piura Norte suman ~1 780 km² de inundación etiquetada. Lima y las zonas pequeñas son huaicos/desbordes locales, pocos km² y difíciles de ver a 10 m.

## 3. Problemas

1. **Las etiquetas EMS se hicieron sobre radar** (Cosmo-SkyMed, Radarsat, TerraSAR-X, Sentinel-1), no sobre Sentinel-2. Hay que emparejarlas con una escena S2 cercana en fecha.
2. **Nubes.** Escenas Sentinel-2 L2A entre el 10/03 y el 15/04/2017 (Planetary Computer STAC; nubosidad por tile completo, no por AOI):
   - Piura: 11, 18 y 31 mar con 52–99 % de nubes. **Primera fecha despejada: 7 abr 2017** (0–5 %).
   - Trujillo: despejado el 7 y 14 abr (1–28 %).
   - Lima / Huaura: despejado el 25 mar, 4 abr, 14 abr.
   - Iquitos: 83–100 % en todas las escenas. **No sirve.**
3. **Desfase de fecha.** Entre las etiquetas (26–30 mar) y la escena S2 despejada (7 abr) hay ~1 semana; parte del agua pudo retirarse. Los polígonos no se pueden copiar sin revisión.

## 4. Propuesta

1. **Base**: Piura (Salinas, Parachique, Piura, Piura Norte) y Trujillo con Sentinel-2 del **7 abr 2017**.
2. **Etiqueta inicial** = polígonos EMS rasterizados a 10 m en chips de 512×512 (mismo formato que Sen1Floods11: 1 agua, 0 no agua, −1 nube / sin dato).
3. **Corrección manual** de cada chip sobre la imagen S2 del 7 abr. Objetivo: ~30–60 chips con agua.
4. **Acuerdo entre anotadores**: ambos autores etiquetan un subconjunto común; reportar IoU / kappa entre ellos.
5. **Documentar** en la tesis: "etiquetas derivadas de Copernicus EMS EMSR199 y revisadas manualmente sobre Sentinel-2".
6. **Opcional**: Sentinel-1 del 23–30 mar (coincide en fecha con EMS) como apoyo para la revisión o para la ablación S1 (RQ2).

## 5. Consecuencia para el protocolo

Sin Bolivia, Sen1Floods11 queda para entrenamiento, validación y test **en distribución**; **Perú pasa a ser la única prueba de generalización**. Actualizar `docs/PLAN.md` (§3 RQ1, §4, §7) cuando el asesor lo confirme.

## 6. Siguiente paso

Descargar las escenas S2 del 7 abr 2017 de Piura y Trujillo, superponer los polígonos EMS y estimar visualmente cuánta agua sigue presente → cuánta corrección hace falta.

## Fuentes

- Copernicus EMS, EMSR199 "Floods in Peru": https://mapping.emergency.copernicus.eu/activations/EMSR199/
- UN-SPIDER, activación del Charter en Perú: https://un-spider.org/es/node/10947
- WorldFloods v2: https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2
- Kuro Siwo: https://arxiv.org/abs/2311.12056
- C2S-MS Floods: https://source.coop/c2sms/c2smsfloods
- GEOID-Flood: https://arxiv.org/abs/2608.02315
