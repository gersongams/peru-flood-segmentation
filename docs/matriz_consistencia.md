# Matriz de consistencia (entregable del curso)

PDF: [`entregables/matriz_consistencia/matriz_consistencia.pdf`](../entregables/matriz_consistencia/matriz_consistencia.pdf)
· fuente LaTeX: `entregables/matriz_consistencia/matriz_consistencia.tex`
(`latexmk -xelatex matriz_consistencia.tex`).

Versión simplificada para la entrega de la semana 7 (problema, objetivos, hipótesis, variables
por hipótesis e indicadores, una página):
[`entregables/matriz_consistencia/matriz_consistencia_simplificada.pdf`](../entregables/matriz_consistencia/matriz_consistencia_simplificada.pdf).

## 1. Contenido

- **Tabla A. Matriz de consistencia.** Plantilla de la Guía N°2 EPG-UNI (p. 27): problema de
  estudio, problemas, objetivos, hipótesis, variables y dimensiones, diseño metodológico. El
  contenido es el del Capítulo I de la tesis (§1.2 a §1.4), redactado con las fórmulas habituales
  de las matrices de tesis peruanas («¿De qué manera…?», «¿En qué medida…?», objetivos en
  infinitivo, variables X e Y con dimensiones numeradas).
- **Tabla B. Matriz de trazabilidad.** Una fila por afirmación contrastable (1a a 3): problema,
  objetivo e hipótesis; dimensión; indicador; criterio de contrastación; instrumento (programa del
  repositorio); evidencia (archivo de resultados); estado. Responde al criterio 3 de la rúbrica del
  curso, «trazabilidad completa» (sílabo v2, p. 4).

## 2. Fuentes de cada cifra de la Tabla B

| Afirm. | Cifra | Fuente |
|---|---|---|
| 1a | U-Net − Prithvi +0.013 [−0.022; 0.062]; U-Net − TerraMind +0.015 [−0.018; 0.051] | `outputs/phaseD/test/bootstrap_diffs.csv` (test, full) |
| 1b | D Prithvi 0.177 [0.102; 0.253]; TerraMind 0.064 [0.001; 0.120] | `outputs/phaseD/test/bootstrap_drop_diff.csv` (peru-test/full) |
| 2a | veredicto | Capítulo III, §3.3 (afirmación 2a); `outputs/phaseD/analysis/error_shares.csv` |
| 2b | U-Net +0.222 [0.132; 0.309]; TerraMind +0.082 [0.016; 0.156]; 2017: −0.063, −0.103 | `bootstrap_diffs.csv` (peru-test full; peru-2017-test no_2023_train_overlap) |
| 2c | AUROC 0.598–0.783; Prithvi Perú 0.439 | `outputs/phaseD/test/calibration.md`; Capítulo III, §3.3 |
| 3 | RF1, RF3, RF5 superados (preliminar) | `outputs/system_eval/rf/rf1.json`, `rf3_17MNP_2023-03-22.json`, `rf5_chips.json` |

Las cifras se verificaron contra estos archivos el 2026-10-28. La HE3 sigue pendiente de la
medición formal (`docs/system_evaluation_protocol.md`).

## 3. Pendiente de los autores antes de entregar

El profesor observó que la redacción sonaba poco natural y poco peruana. Las celdas de la
Tabla A son un borrador: **los autores deben dar la redacción final a cada una** (unas 15 oraciones).
La Tabla B es un registro de datos y solo necesita revisión.

- [ ] Problema de estudio
- [ ] Problema general y PE1–PE3
- [ ] Objetivo general y OE1–OE3
- [ ] Hipótesis general y HE1–HE3 (mantener el sentido: la HE1 y la HE2 se fijaron antes de la prueba)
- [ ] Variables y dimensiones
- [ ] Diseño metodológico
- [ ] Revisar la Tabla B

## 4. Cambios de redacción respecto de la tesis

La matriz ya usa estas formas; el Capítulo I y los Anexos 1 y 2 todavía usan las de la izquierda.
Si los autores las aceptan, hay que aplicarlas también en la tesis para que coincidan.

| Antes (calco o poco natural) | Ahora |
|---|---|
| sin etiquetas peruanas de entrenamiento | sin datos etiquetados del Perú |
| régimen de entrenamiento | esquema de entrenamiento |
| estratos (de error) | condiciones (en las que se concentran los errores) |
| fiabilidad de los mapas | confiabilidad de los mapas |
| incluso cuando no se dispone | aun cuando no se dispone |
| mejoran parcialmente | mejoran en parte |
| un tiempo compatible con la respuesta posterior a un evento | un tiempo adecuado para la respuesta después de un evento |
| la incertidumbre concentra los errores | la incertidumbre permite detectar errores |
| ¿Cómo desarrollar…? / ¿Cumple el sistema…? | ¿De qué manera se puede desarrollar…? / ¿En qué medida el sistema cumple…? |
| hipótesis general en una oración con punto y coma | tres oraciones cortas |

Otras señales de texto poco natural en la tesis: cadenas largas de sustantivos («la pérdida de
desempeño y de confiabilidad de los modelos entrenados fuera del país»), oraciones de más de
30 palabras y la repetición literal de la misma frase en párrafos seguidos.
