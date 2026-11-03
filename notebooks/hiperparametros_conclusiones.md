## 8. Conclusiones

**Qué se optimizó.**

- Se hizo una búsqueda en malla de la tasa de aprendizaje para los tres modelos entrenados en
  Sen1Floods11, con 10 corridas (semilla 0) y selección en `valid`. Se eligió 3e-4 para la U-Net,
  5e-5 para TerraMind y 1e-4 para Prithvi.
- También se subió el tope de 100 a 200 épocas para las corridas finales, porque dos corridas del
  barrido tuvieron su mejor resultado cerca del tope. Con 200 épocas, ninguna corrida final llegó
  al tope.

**Qué se ganó.**

- En `valid`, el IoU de agua de la U-Net subió de 0.819 a 0.832 y el de TerraMind de 0.817 a 0.835.
  Prithvi se mantuvo (0.833 frente a 0.832), porque su mejor tasa es la que ya tenía.
- En TerraMind, la mejora se confirma con tres semillas: 0.835 ± 0.001 con 5e-5 frente a
  0.814 ± 0.015 con 2e-5. Además, sus resultados varían mucho menos entre semillas.
- En la U-Net, la ventaja de 3e-4 sobre la segunda tasa (0.017) es igual a la diferencia que se
  obtuvo al repetir la misma corrida (0.017). Por eso no se puede afirmar que 3e-4 sea claramente
  la mejor tasa. En Prithvi, las tres tasas quedan a menos de 0.009 entre sí.

**La mejora no se trasladó al Perú.** Con la tasa elegida, TerraMind obtiene 0.703 ± 0.091 en
`peru-dev`, frente a 0.692 ± 0.100 con 2e-5. La diferencia es mucho menor que la variación entre
semillas. La U-Net queda en 0.836 ± 0.022, frente a 0.835 con su tasa original (una sola semilla).
Ajustar los hiperparámetros dentro del dominio de origen no resuelve el cambio de dominio. Esto
coincide con la importancia de variables de la semana 7: el esquema de entrenamiento pesa mucho más
en el IoU que los demás factores.

**Sensibilidad.**

- En el autoentrenamiento con AdaBN y semilla 0, ninguna combinación de α y q supera al modelo sin
  adaptar en TerraMind (0.764). En la U-Net, las combinaciones quedan entre 0.890 y 0.914 a las 4000
  iteraciones, por encima del modelo sin adaptar (0.843). Este resultado no es estable: con las
  semillas 1 y 2, la configuración por defecto pierde entre 0.01 y 0.11, y la media de las tres
  semillas (0.818 ± 0.110) queda por debajo del modelo sin adaptar (0.836 ± 0.022).
- El umbral de MNDWI es muy sensible: el IoU de agua en `peru-dev` pasa de 0.486 con t = 0 a 0.924
  con t = 0.25. Ese umbral se ajustó con las etiquetas de `peru-dev`, así que es una referencia
  optimista.

**Limitaciones.**

- **Los modelos entrenados en el Perú no tuvieron barrido propio.** La U-Net (0.9415), TerraMind
  (0.9265) y Prithvi (0.9218) usan las tasas de sus archivos de configuración. Por eso, sus
  resultados no representan el mejor valor posible con una búsqueda en `peru-dev`.
- Solo se buscó la tasa de aprendizaje, con búsqueda en malla y una semilla. No se usó búsqueda
  aleatoria ni bayesiana. El decaimiento de pesos, el tamaño de lote, la función de pérdida, el
  decodificador, el planificador, β de FDA y los parámetros del *gradient boosting* quedaron fijos.
- La tasa 5e-5 de TerraMind está en el borde de su malla; una tasa mayor podría ser mejor. Para
  Prithvi se fijó de antemano una regla de ampliación de la malla, que no hizo falta aplicar.
- Las cifras de `valid` son optimistas, porque la tasa, la parada temprana y el punto de control
  se eligen con esa misma partición. Solo sirven para ordenar configuraciones.
- El entrenamiento en GPU no se puede reproducir bit a bit. Una diferencia menor de unos 2 puntos
  de IoU entre configuraciones, con tres semillas, debe interpretarse con cuidado.
- La revisión del tope de épocas en el Perú se hizo solo para TerraMind con la semilla 0. La U-Net
  con semilla 1 también tuvo su mejor resultado cerca del tope (época 98) y no se revisó.

**Pendiente.** Un barrido de la tasa de aprendizaje con `peru-train` y selección en `peru-dev`
mostraría cuánto se puede ganar en los modelos entrenados en el Perú. Este cuaderno no usa ninguna
partición de prueba.
