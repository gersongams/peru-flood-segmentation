# Guía de etiquetado: conjunto Perú (EMSR199)

Versión 1, 2026-10-11. Ambos autores siguen esta guía **antes** de etiquetar el subconjunto común de acuerdo (≥10 chips). Si la guía cambia después, sube la versión y se re-etiqueta ese subconjunto.

## 1. Qué se etiqueta

Cada chip es de 512×512 píxeles a 10 m sobre la escena Sentinel-2 elegida (7 abr 2017 salvo otra más cercana y despejada). La etiqueta inicial son los polígonos EMS rasterizados; la tarea es **corregirla para que describa lo que se ve en la imagen Sentinel-2**, no lo que había en la fecha del radar.

| Valor | Clase | Incluye |
|---|---|---|
| **1** | Agua | Agua abierta visible en la fecha de la imagen: inundación, ríos, lagunas, mar, estanques, agua turbia o con sedimento |
| **0** | No agua | Suelo, vegetación, zonas urbanas, arena húmeda sin lámina de agua visible |
| **−1** | Ignorar | Nube, sombra de nube, bruma densa, sin dato, bordes de escena, píxeles dudosos que no se pueden decidir |

Igual que Sen1Floods11: el agua permanente **también es 1**. La distinción permanente / inundación se hace después con capas externas (JRC), no a mano.

## 2. Composiciones para revisar

1. **Color verdadero** (B4, B3, B2): contexto general.
2. **Falso color SWIR** (B12, B8, B4): el agua aparece oscura (azul/negra); es la vista principal para decidir.
3. **NDWI / MNDWI** como apoyo, nunca como etiqueta automática (en zonas desérticas y salinas da falsos positivos).
4. Opcional: Sentinel-1 de 23–30 mar 2017 para entender dónde estuvo el agua; **no** marcar agua solo porque aparece en radar si no se ve en S2.

## 3. Reglas para casos difíciles

| Caso | Regla |
|---|---|
| Agua que EMS marcó pero ya no se ve en S2 (se retiró) | **0**. Anotar el chip en el manifiesto como `receded`. |
| Agua visible en S2 que EMS no marcó | **1**. |
| Suelo saturado, barro oscuro sin lámina visible | **0**; si no se puede decidir, **−1**. |
| Vegetación inundada (agua bajo dosel, no visible) | **0** en S2 (no se ve); anotar `flooded_vegetation` en el manifiesto si el radar o EMS lo sugiere. |
| Salinas, desierto oscuro, sombras topográficas | **0** salvo lámina de agua clara en falso color SWIR. |
| Nube fina por la que se ve el suelo | Etiquetar debajo si la decisión es clara; si no, **−1**. |
| Bordes agua / tierra mezclados | Asignar el píxel a la clase mayoritaria; no dejar huecos. |
| Objetos menores a ~2 píxeles | No dibujarlos a mano salvo que ya estén en la etiqueta inicial. |

## 4. Procedimiento

1. Abrir el chip en QGIS con las tres vistas del §2 y la etiqueta inicial superpuesta.
2. Marcar nubes y sin dato como −1 primero.
3. Borrar agua que no se ve; añadir agua que sí se ve.
4. Guardar como GeoTIFF int16 con el mismo nombre y georreferencia que el chip.
5. En el manifiesto: `reviewed = true`, iniciales, fecha, minutos dedicados y notas (`receded`, `flooded_vegetation`, `uncertain`).
6. Subconjunto de acuerdo: cada autor etiqueta **sin ver** la etiqueta del otro. Se calculan IoU de agua y kappa de Cohen sobre los píxeles que ninguno marcó −1.

## 5. Qué no hacer

- No usar predicciones de nuestros modelos como etiqueta inicial (sesga la evaluación).
- No corregir un chip después de evaluar modelos en él sin crear una nueva versión del conjunto.
