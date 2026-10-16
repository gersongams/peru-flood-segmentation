## 5. Conclusiones

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
