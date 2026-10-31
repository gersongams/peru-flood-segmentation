## 5. Conclusiones

**Variables independientes (sección 2).**

- El esquema de entrenamiento es el factor que más pesa. Explica el 53 % de la varianza del IoU
  de agua en `peru-dev` (η² = 0.526). La arquitectura explica el 18 % y la interacción, el 17 %.
- Entrenar con datos del Perú mejora a los tres modelos: la U-Net pasa de 0.836 a 0.942,
  TerraMind de 0.703 a 0.927 y Prithvi de 0.870 a 0.922. Con datos del Perú, los tres quedan
  cerca entre sí.
- Entre los métodos de adaptación, la FDA aplicada en la prueba falla por completo y explica por
  sí sola el 92 % de la varianza. Sin ella, lo que más pesa es la interacción (η² = 0.496): el
  efecto del método depende del modelo. La FDA en entrenamiento mejora a TerraMind (0.837 frente
  a 0.703) y empeora a la U-Net (0.615 frente a 0.836). La igualación de histogramas en
  entrenamiento mejora a ambos (U-Net 0.912, TerraMind 0.844).
- Para la matriz de consistencia, esto significa que la dimensión X2 (esquema de entrenamiento)
  es la que más influye en la exactitud (Y1). Por eso la tesis estudia la adaptación sin datos
  etiquetados del Perú (OE2): busca acercarse al modelo entrenado en el Perú sin usar sus
  etiquetas.

**Bandas de entrada de la U-Net (sección 3).**

- La U-Net depende sobre todo de la banda SWIR 1 (B11). Al permutarla, el IoU de agua baja de
  0.940 a casi cero (caída de 0.88).
- Le siguen el NIR estrecho (B8A, caída de 0.37) y el borde rojo 1 (B05, 0.33). El rojo (B04),
  los cirros (B10) y el borde rojo 2 (B06) casi no influyen (caídas menores que 0.01).
- La SWIR 2 (B12) sola apenas influye (0.015), aunque está muy correlacionada con la SWIR 1. El
  modelo aprendió a usar la B11 y no la reemplaza con la B12.
- Por grupos, el orden es SWIR (0.88), borde rojo (0.37), NIR (0.23), visible (0.10) y
  atmosféricas (0.01). Las importancias no se suman: permutar un grupo completo puede bajar el
  IoU menos que permutar una sola de sus bandas.
- El resultado tiene sentido físico. El agua absorbe casi toda la radiación en el SWIR, mientras
  que la arena del desierto costero es muy brillante en esa región. Por eso el SWIR separa bien el
  agua del suelo seco en la costa peruana.

**Comparación con la semana 5 (sección 4).** El clasificador por píxel también depende del SWIR:
sus características más importantes son AWEI_nsh, MNDWI2 y MNDWI (índices que usan el SWIR) y
las bandas B12 y B11. Los dos enfoques coinciden en que el SWIR es la información clave.

**Limitaciones.**

- La permutación de bandas usa una sola semilla y dos permutaciones, solo en `peru-dev`.
- La importancia por permutación mide cuánto depende el modelo de una banda, no si la banda causa
  la presencia de agua.
- Las bandas correlacionadas se reparten la importancia; por eso se reportan también los grupos.
- La descomposición de la varianza usa tres semillas por configuración y no incluye intervalos
  de confianza.

**Pendiente.** Medir la importancia de bandas en TerraMind y Prithvi con el mismo programa
(`scripts/variable_importance_dev.py`) y en las tres semillas.
