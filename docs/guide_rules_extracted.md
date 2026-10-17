# UNI thesis writing and formatting rules: extracted from the guides and syllabi

Read on 2026-10-16. Sources, all in `thesis-guidelines/`:

| Code | File | Notes |
|---|---|---|
| **G1** | `GUIA_Ndeg01_Plan.pdf`: *Guía N°1, Guía metodológica para la elaboración y presentación del Plan de Tesis (Maestrías y Doctorados)*, EPG-UNI 2023, approved by Consejo Directivo N°05, 28-03-2023 | 60 scanned pages |
| **G2** | `GUIA_No.2_Desarrollo.pdf`: *Guía N°2, … presentación de la Tesis de Grado (Maestrías y Doctorados)*, EPG-UNI 2023, same approval | 48 scanned pages |
| **S1** | `MIA-403_Proyecto_de_Investigacion_II_(Silabo)_2026-2_v.1.1.pdf` | 9 pages, older sprint/MLOps version |
| **S2** | `MIA-403_Proyecto_de_Investigacion_II_Silabo_v_2.pdf`: "SÍLABO ACTUALIZADO 2026-2", professors Glen Rodríguez R. and Wester Zela | 5 pages; treated as the current one |
| **SES2** | `Sesion 2.pdf` (Dr. Wester Zela, matriz de consistencia) | 7 pages; read as a bonus, not requested |

**Page references.** For G1 and G2, "p. N" is the page number printed on the page. In both guides PDF page 2 is printed "0", so **PDF page = printed page + 1** from PDF page 3 on. For example, G1 p. 43 is PDF page 44. For S1, S2 and SES2, "p. N" is the PDF page.

**Which syllabus is newer.** I treat S2 as current because of its filename (`v_2`) and its header "SÍLABO ACTUALIZADO 2026-2". This is an inference. The PDF metadata does not support it: both files were created on 2026-09-18, and S1 is one minute later. So S1 requirements not repeated in S2 are not treated as withdrawn.

**Readability.** Every page of G1 (1–60) and G2 (1–48) was readable. One exception: the side label on the G2 p. 7 diagram is cut off ("NUMERACIÓN EN ROMANOS Y …").

**Labels used below:**
- **[HARD]**: obligatory wording in the source ("deberá", "debe", "tiene que", "se debe", "imprescindiblemente", a checklist item the reviewer marks SÍ/NO, or a rubric criterion).
- **[REC]**: a recommendation or a definition only.
- **[EX]**: what an example or template shows, with no rule stated.
- **[NOT SPECIFIED]**: none of the documents covers it.

**Scope.** Our program is a *Maestría en Ciencias* (research master's), which is the "Maestría de Investigación" track in both guides. The *Especialización/Profesionalizante* sections (G1 §5, §6.2; G2 §5, §6.2) do not apply and are left out, except where they show a contrast.

---

## 0. Overriding general rules

| Rule | Source | Status |
|---|---|---|
| "Las instrucciones de forma y estilos que se detallan a continuación, deben ser aplicadas **en forma estricta** en la elaboración del Plan de las Tesis de Investigación (Académicas)…" | G1 p. 4 | HARD |
| The same sentence for the thesis: "…deben ser aplicadas en forma estricta en la elaboración de las Tesis de Investigación (Académicas)…" | G2 p. 4 | HARD |
| "En la redacción de los documentos y en las citaciones correspondientes, deben considerarse el **estilo APA (Versión vigente)**…" | G1 p. 4; G2 p. 4 | HARD |
| "El plan de tesis y la tesis tienen que estar enmarcadas en las líneas de investigación de los Programa de Posgrado de la UNI." | G1 p. 4 (also G2 p. 5, G1 p. 49) | HARD |
| The thesis "debe ser **original e inédito**". | G2 p. 5 | HARD |
| A master's thesis can have **up to two authors**: "de hasta dos egresados para obtener el grado de Maestría". Also: "Solo para el caso de Maestría, pueden sustentar dos tesistas … la nota que obtengan será individual". | G2 p. 5; G1 p. 5 | HARD (two authors allowed) |
| A thesis written in another language: "El tesista deberá presentar una **copia traducida al idioma español**." This implies Spanish is the default language. | G2 pp. 5–6 | HARD |
| "Grado de Excelencia en Maestría": requires "un artículo publicado en una Revista Indexada". | G1 p. 6; G2 p. 5 | HARD (only if that distinction is wanted) |

**Which APA edition.** None of the documents names one; they all say "versión vigente". The current edition is **APA 7th ed. (2020)**. That is my inference, not something the documents state.

---

## 1. Required structure

### 1.1 Plan de Tesis, Maestría de Investigación (G1 §4.1, pp. 9–10) [HARD]

The plan "deberá considerar **imprescindiblemente** las siguientes 03 partes: cuerpo preliminar, texto, referencias y anexos" (G1 p. 7).

```
Carátula
Datos Generales
  1) Título del Plan de Tesis (Anteproyecto de Investigación)
  2) Nombre(s) de autor(es)
  3) Nombre del asesor o asesores
  4) Área involucrada: [ ] Unidad de Posgrado: ____  [ ] Escuela de Posgrado
  5) Lugar o institución donde se desarrolla el Proyecto
  6) Duración estimada del Proyecto en meses (fecha de inicio/fecha de culminación)
CAPÍTULO I. PLANTEAMIENTO DEL PROBLEMA
  1.1 Planteamiento del Problema
    1.1.1 Identificación y Descripción del Problema de Estudio
    1.1.2 Antecedentes de la Investigación
    1.1.3 Formulación del Problema
      1.1.3.1 Formulación del [Problema] General
      1.1.3.2 Formulación de los Problemas Específicos
  1.2 Justificación y Alcances
    1.2.1 Justificación
    1.2.2 Alcances
CAPÍTULO II. OBJETIVOS
  2.1 Objetivo General
  2.2 Objetivos Específicos
CAPÍTULO III. MARCO TEÓRICO
  3.1 Bases Teóricas
  3.2 Definición de Términos
  3.3 Hipótesis y Variables (cuando corresponda / si aplica)
    3.3.1 Hipótesis
    3.3.2 Variables
CAPÍTULO IV. DISEÑO METODOLÓGICO
  4.1 Tipo de Investigación
  4.2 Nivel de Investigación
  4.3 Métodos de Trabajo
  4.4 Población y Muestra (cuando corresponda / si aplica)
  4.5 Tipo de Diseño
  4.6 Técnicas e Instrumentos de Recolección de Datos
  4.7 Técnicas e Instrumentos de Análisis y Procesamiento de Datos
  4.8 Etapas de Intervención del Estudio
CAPÍTULO V. ADMINISTRACIÓN DEL PLAN DE TESIS
  5.1 Cronograma
  5.2 Presupuesto
  5.3 Financiamiento
REFERENCIAS
ANEXOS: Anexo 1: Matriz de Consistencia
```

The doctoral plan (G1 pp. 11–12) adds "1.2.3 Limitaciones" and "Anexo 2: Matriz de Operacionalización de Variables". Both are doctoral-only in the guide. See conflict C4 for limitaciones.

### 1.2 Tesis, Maestría de Investigación (G2 §4.1, pp. 9–10) [HARD]

"El contenido del trabajo de investigación de la tesis de posgrado deberá considerar **imprescindiblemente** las siguientes 03 partes: cuerpo preliminar, texto, referencias y anexos" (G2 p. 7). G2 p. 9 adds: "NOTA: La estructura del Plan de Tesis, ver en la Guía N°1".

```
Carátula
Datos Generales
  > Dedicatoria
  > Agradecimientos
  > Copia de Documentos
  > Índice de Contenidos
  > Lista de Tablas, Ilustraciones y Cuadros
  > Resumen - Abstract
  > Introducción
CAPÍTULO I. PROTOCOLO DE LA INVESTIGACIÓN
  1.1 Identificación y Descripción del Problema de estudio
    1.1.1 Formulación del Problema General
    1.1.2 Formulación de los Problemas Específicos
  1.2 Objetivos
    1.2.1 Formulación del Objetivo General
    1.2.2 Formulación de los Objetivos Específicos
  1.3 Hipótesis y variables
    1.3.1 Formulación de la Hipótesis General
    1.3.2 Formulación de las Hipótesis Específicas
    1.3.3 Variables
  1.4 Metodología
CAPÍTULO II. MARCO TEÓRICO
  2.1 Antecedentes de la investigación
  2.2 Bases Teóricas
  2.3 Definición de Términos
CAPÍTULO III. DESARROLLO DEL TRABAJO DE INVESTIGACIÓN
  3.1 Análisis de los Datos y Resultados
  3.2 Discusión e interpretación de los Resultados
  3.3 Contrastación de la Hipótesis (Cuando corresponda / Si Aplica)   [p. 9 misprints "Contratación"]
CAPÍTULO IV. CONCLUSIONES Y RECOMENDACIONES
  4.1 Conclusiones
  4.2 Recomendaciones
REFERENCIAS
ANEXOS: Anexo 1: Matriz de Consistencia
```

- **Order of the front matter** (G2 p. 7 diagram, bottom to top): Carátula → Dedicatoria → Agradecimientos → Copia de Documentos → Índice de Contenidos → Lista de Tablas, Ilustraciones y Cuadros → Resumen y Abstract → Introducción. [HARD]
- **"Copia de Documentos"** is added **after the defense**: "(Se incorporará después de la sustentación): Acta de Sustentación, Certificado de idioma extranjero, Grado de Bachiller/Maestría (según el caso). Certificado de Notas y/o de Finalización de Estudios y Recibos de Pago" (G2 p. 7). [HARD, but only for the final version]
- **Matriz de operacionalización**: "Anexo 2: Matriz de Operacionalización de Variables (**Solo para Doctorado**)" (G2 p. 19; G1 p. 35). See conflict C3: the course syllabus asks for it anyway.

### 1.3 Annexes and forms that must accompany the thesis

| Item | Source | Status |
|---|---|---|
| **Matriz de Consistencia**, laid out landscape. Columns: *Problema de Estudio · Problemas (general / específicos) · Objetivos (general / específicos) · Hipótesis (general / específicas) · Variables y dimensiones (Variable independiente + DIMENSIONES; Variable dependiente + DIMENSIONES) · Diseño Metodológico (Tipo de investigación; Nivel de investigación; Métodos; Diseño; Población y muestra; Técnicas e instrumentos de recolección de datos; Técnicas e instrumentos de análisis y procesamiento de datos)*. A "Título:" row goes on top; "Autor:" and "Elaboración:" lines go below. | G1 p. 41; G2 p. 27 | HARD (template) |
| Matriz de Operacionalización. Columns: *Variable · Definición conceptual · Definición operacional · Operacionalización (Dimensiones · Indicadores · Índices y/o ítems) · Escala de medición*. "Autor:" and "Elaboración:" lines go below. | G1 p. 42; G2 p. 28 | Doctoral only in the guides; asked for by S2 (see C3) |
| **Declaración de Autenticidad y No Plagio** (form): "Declara que el presente trabajo de tesis ha sido elaborado por el mismo y no existe plagio/copia de ninguna naturaleza … Deja constancia que las citas de otros autores han sido debidamente identificadas…" | G2 p. 46 (Anexo XIII) | HARD (signed form) |
| Autorización externa para realizar la investigación. Needed only when an outside organization's data or instruments are used. | G2 p. 47 (Anexo XIV) | Only when applicable |
| "Resumen de la Tesis, **publicado en la Gaceta UNI**" (a document needed for the defense). | G2 p. 30 | HARD (procedural) |
| Reproducibility annex: "Diapositivas y anexo reproducible (instrucciones, seeds, versiones)". | S1 p. 7 (Semana 16) | HARD for the course |
| "anexo/repo reproducible". | S2 p. 3 (Semana 16) | HARD for the course |
| Course deliverables: "Tablas/figuras finales reproducibles + tarjetas de datos/modelo"; "Preparar model card y data card finales (sesgos/limitaciones)". | S1 p. 7 (Semana 14) | Course deliverable (S1 only) |

---

## 2. Content rules per part, including how to word things

### 2.1 Title

- "Debe ser **claro, conciso y específico**, reflejar adecuadamente el problema y el objetivo de la tesis, así como las variables. Asimismo, reflejar el método que permita solucionar el problema y/o el efecto de la variable más influyente, de ser el caso delimitar el problema tanto en el tiempo como en el espacio." (G1 p. 15) **[HARD]**
- "El título se escribe en **letras mayúsculas** …, centrándolo en la parte media superior, de ocupar más de un renglón, dejar **doble espacio**. El título debe ser continuo, considerar **no exceder 20 palabras** y las **palabras distintas al español escribirlas en cursiva**, sin considerar los artículos /e, la, para, etc.)" (G1 p. 15; repeated p. 36) **[HARD]**
  - *Interpretation, not quoted rule:* the parenthetical is grammatically ambiguous. The most likely reading is that articles and prepositions are not counted toward the 20 words. It could also mean they are not italicized. To be safe, keep the title at 20 words or fewer counting every word.
  - English terms in the title, such as *Transformer*, *Sentinel-2* (a proper noun) or *foundation models*, go in italics.
- Plan evaluation form: "¿El enunciado del título de la Investigación es claro y preciso, orientado a dar respuesta al problema?" and "¿Refleja los objetivos el tema abordado?" (G1 p. 51). **[HARD, checklist]**

### 2.2 Problem: description and formulation

- 1.1.1: identify a specific problem and describe "cada una de sus partes: Problema Específico, el Objeto de Estudio, Alcance Espacial y el Alcance Temporal (si corresponde)". The route is Línea de investigación → Tema particular → Tema específico. (G1 p. 18) [REC: method]
- "Una buena formulación del problema implica necesariamente la delimitación del campo de investigación, establece claramente los límites…" (G1 p. 18). [HARD]
- "Se formulan los problemas principales y derivados (o específicos)… **Los problemas pueden formularse como preguntas o de manera declaratoria.**" (G1 p. 19). Either form is allowed here, but S2 expects questions (see C6).
- Problema general: "denominado también problema principal (de estudio ó científico) **a partir del cual se redacta el objetivo general**" (G2 p. 15). Problemas específicos "se disgregan del problema general planteado, cuya formulación orienta a la formulación de los objetivos específicos, **determinando previamente los indicadores de las variables correspondientes**" (G2 p. 15). [HARD]
- Thesis review checklist, Anexo V (G2 p. 33). **[HARD, checklist]**
  - 1.1 "Identifica con claridad el problema de estudio (Problema específico, **Variable dependiente**)".
  - 1.2 "El problema general se desprende de la descripción del problema".
  - "Problemas específicos se desagregan del problema general y **se consideran sus dimensiones**".
  - "El problema general está formulado de manera correcta".
- Plan form: "¿En la formulación del problema, el Investigador ha propuesto **la variable independiente de la solución del problema**?" (G1 p. 51). [HARD, checklist]

### 2.3 Objectives

- Objetivo general: "Expresa lo que se pretende alcanzar con la investigación" (G1 p. 19). It "proporciona la dirección general y precisa de lo que el tesista quiere lograr al final de la tesis" (G2 p. 15).
- Objetivos específicos: "Explican las acciones que se deben llevar a cabo para alcanzar el objetivo general" (G1 p. 19). They are "objetivos más pequeños y lógicamente alineados, con el objetivo general" (G2 p. 15).
- **Wording [HARD, checklist G2 p. 33]**: "Objetivo general, es **coherente con el problema general, redactados con verbos en infinitivo**"; "Objetivo específico, se desprenden del general, **están alineados con los problemas específicos**".
- So each specific objective should map one-to-one onto a specific problem and a specific hypothesis. This is implied by the matrix and the checklist rather than stated.
- S2 rubric criterion 3: "Objetivos **medibles**" (S2 p. 4). Plan form: "¿Los objetivos específicos permiten el logro del objetivo general…?" (G1 p. 51).

### 2.4 Justification and scope

- Justification: "Se exponen detalladamente todas aquellas razones que validan la realización del estudio". It covers "i. La importancia, el valor y la pertinencia…; ii. Sus aportes, ya sea a nivel de conocimiento, práctico o metodológico; iii. Las soluciones que ofrece…" (G1 p. 19). [HARD]
- Thesis checklist: "Justifica la relevancia de la investigación"; "**Justificación: teórica, práctica y metodológica**" (G2 p. 33). [HARD, checklist]. Using these three labelled sub-parts is the safe choice.
- Scope: "debe tener alcance espacial, alcance temporal temático y recursos con los que se realizará la investigación" (G1 p. 19). Thesis checklist 1.4 "Alcance del Estudio: Temático, Espacial, Temporal" (G2 p. 33). [HARD]
- Limitations: see C4. They are expected in the evaluation forms and in the Discussion (G2 p. 18).

### 2.5 Antecedentes and literature

- "Se hace referencia a investigaciones relevantes sobre el problema o cuestiones afines y se presentan de manera resumida, **la metodología utilizada y los resultados principales de cada una** de ellas." (G1 p. 18; G2 pp. 16–17) [HARD]
- "Para las investigaciones de posgrado se debe tener en cuenta la **revisión exhaustiva** de los textos y artículos científicos publicados en **revista indexadas**…" (G1 p. 18; G2 p. 17) [HARD]
- "**Los antecedentes deben ser de los últimos cinco años.** En algunas investigaciones, podrá usarse antecedentes de mayor antigüedad, dependiendo del tema de investigación." (G1 p. 18; G2 p. 17) [HARD, with an exception]
- **Order**: "Ha considerado antecedentes en **orden cronológico del más antiguo al más reciente**" (G2 p. 33, thesis checklist 2.1). The plan form asks: "¿…se especifica la evolución de los antecedentes **de la variable dependiente** ordenados desde el más antiguo hasta el más reciente?" (G1 p. 51). **[HARD, checklist]**
- S2 rubric criterion 2 (3 pts excellent): "Revisión **crítica y actualizada**; compara estudios, identifica brecha, sustenta decisiones…". An "Insuficiente" grade covers "Revisión escasa, desarticulada o basada principalmente en fuentes inadecuadas" (S2 p. 4).
- S2 §XI: "El estado del arte debe demostrar una brecha de investigación; **la cantidad de referencias por sí sola no determina su calidad**." (S2 p. 4)
- S2 week 3 deliverable: "matriz de antecedentes/estado del arte y brecha" (a comparative matrix) (S2 p. 2). [Course HARD]
- **No minimum number of references** appears anywhere. [NOT SPECIFIED]

### 2.6 Marco teórico, bases teóricas, definición de términos

- Bases teóricas: "Las bases teóricas de las variables de estudio, variable dependiente y variable independiente y las dimensiones de estas variables, **respetando el derecho de autor**." (G1 p. 19). It "es la teoría científica de las variables … y … de sus dimensiones" (G2 p. 17). Thesis checklist: "Las bases teóricas se han desarrollado de acuerdo a sus variables de Estudio" (G2 p. 33). [HARD]
- Definición de términos, also called "Marco Conceptual": "son definiciones **cortas y precisas** de las variables del estudio, sus dimensiones y los conceptos claves… Recomendables hacer las citas correspondientes" (G2 p. 17). Its four functions are listed in G1 p. 20 and G2 p. 17. The checklist asks: "Ha instalado la definición de términos de sus variables de estudio y de sus dimensiones" (G2 p. 33). [HARD]

### 2.7 Hypotheses and variables

- A hypothesis is "una **propuesta tentativa**, acerca de las relaciones entre dos o más variables" (G1 p. 20; G2 p. 15).
- "El objetivo que plantea la demostración de una causa-efecto, asociación, correlación, etc. **debe tener hipótesis**; si plantea la exploración o descripción de un fenómeno, no requiere hipótesis, **pero sí será necesario colocar el subtítulo con un texto que justifique porqué el plan no la requiere**." (G1 p. 20) [HARD]
- Hipótesis general: "predicción o explicación provisoria (mientras no sea contrastada) de la relación entre 2 o más variables" (G1 p. 20). Hipótesis específicas "se derivan de la general … si se demuestran estas se está demostrando la hipótesis General" (G2 p. 16).
- Thesis checklist 2.3 (G2 p. 34): "Formula la hipótesis general y específica de manera clara"; "Están alineadas con los problemas, objetivos e hipótesis"; "**Son susceptibles de verificación empírica**". [HARD, checklist]
- Variables (G2 p. 16): "deben medirse, controlarse y observarse".
  - Dependiente: "se utiliza para medir y describir el problema de estudio".
  - Independiente: "factores … que son la causa o que influyen en la solución del problema".
  - G1 p. 21 also defines the *variable interviniente* (extraneous variable).
- Thesis checklist 2.4 (G2 p. 34): "Ha definido correctamente la **variable problema (variable dependiente)**"; "… la **variable solución (variable independiente)**"; "Tiene la tesis Matriz de Consistencia". [HARD, checklist]
- 3.3 Contrastación de la hipótesis (si aplica): "El tesista indicará la validez de la hipótesis." (G2 p. 18)
- S2 rubric 3: "hipótesis/preguntas pertinentes; **variables/dimensiones/indicadores e instrumentos** definidos cuando corresponda; **trazabilidad completa**" (S2 p. 4).

### 2.8 Methodology (Cap. IV in the plan; §1.4 in the thesis)

The plan must name each of the following [HARD, from the structure]: Tipo, Nivel, Métodos de trabajo, Población y muestra (si aplica), Tipo de diseño, Técnicas e instrumentos de recolección, Técnicas e instrumentos de análisis y procesamiento, Etapas de intervención.

- **Menus offered** [REC/definitions, G1 pp. 22–30]:
  - Tipo según objetivo: básica / aplicada.
  - Según profundidad: exploratoria / descriptiva / correlacional / explicativa (causal).
  - Según datos: cualitativa / cuantitativa / mixta.
  - Según manipulación: experimental / cuasiexperimental / no experimental.
  - Según inferencia: deductiva / inductiva / hipotético-deductiva.
  - Según temporalidad: longitudinal / transversal.
  - Nivel (Dankhe, 1986): exploratorio / descriptivo / correlacional / explicativo.
  - Métodos: deductivo, inductivo, histórico, descriptivo, experimental. The experimental method "requiere que se plantee una hipótesis predictiva" (G1 p. 28).
  - Diseño: the Hernández Sampieri concept map, 6th ed. (G1 p. 30).
- Thesis checklist 3.1 (G2 p. 34) expects:
  - "**Por enfoque**: Cuantitativo / Cualitativo".
  - "**Por nivel**: Exploratoria, descriptiva, aplicada, correlacional, explicativa".
  - "**Diseño**: Experimental y No Experimental".
  - "**Métodos**: Hipotético-deductivo, analítico, inductivo, deductivo, sintético".
  - 3.2 "Población: Define el escenario de estudio; Muestra: Establece el criterio de selección de la muestra".
  - 3.3 and 3.4: present the collection and analysis techniques and instruments. **[HARD, checklist]**
- G1 p. 24: an experimental study gives "control total sobre las variables… el laboratorio tiene que estar certificado". That text targets physical labs. [Definition; it may draw a reviewer's comment if "experimental" is claimed.]
- G1 p. 29: "En la investigación experimental … se recomienda muestras pequeñas que suelen ser de por lo menos 30 sujetos." [REC]
- 4.8 Etapas de intervención = "los Pasos Metodológicos o Etapas" (G1 p. 34).
- G2 p. 16 (thesis §1.4 Metodología): state "cómo la metodología … posibilitará dar respuestas congruentes a las preguntas" and identify "las fuentes de información o datos". [REC]
- S2 rubric criterion 4 (4 pts excellent): "Diseño justificado y reproducible; datos/población/muestra, instrumentos, procedimiento, protocolo, métricas, validación y **amenazas a la validez** definidos y coherentes." S2 week 6 adds that Cap. III must include "enfoque, tipo y diseño, población/muestra o fuentes de datos, técnicas e instrumentos, procedimiento, **ética** y amenazas a la validez". S2 week 7 adds the "protocolo … baselines, variables de control, partición de datos …, métricas, criterios de éxito, reproducibilidad y plan de análisis" (S2 pp. 2, 4). [Course HARD]

### 2.9 Administración (plan only)

- 5.1 Cronograma: "Detallar cada una de las actividades … **Se elaborará utilizando el diagrama de Gantt.**" (G1 p. 34) [HARD]
- 5.2 Presupuesto, broken into (a) equipos y bienes duraderos; (b) materiales, instrumentos e insumos; (c) servicios tecnológicos …; (d) pasajes, viáticos …, including "gastos por publicación científica (APC, traducción, corrección…)" (G1 p. 34). [HARD: "Tomar en cuenta los siguientes rubros"]
- 5.3 Financiamiento: "autofinanciado, financiado con fondos concursables…". The plan form asks whether it "indica el tipo" (G1 pp. 35, 52). [HARD, checklist]
- Plan form item 11 "Ubicación del proyecto": "¿Justifica la localización, ubicación donde se desarrollará el proyecto … y describe brevemente las características del contexto y entorno?" (G1 p. 52). [HARD, checklist]

### 2.10 Front matter (thesis)

- Dedicatoria and Agradecimientos (G2 p. 14). [HARD (listed); content is free]
- Índice de Contenidos: "Indica las partes … especificando la página correspondiente" (G2 p. 14). The checklist asks for it "De acuerdo a la Guía" and "Con numeración de páginas" (G2 p. 33). [HARD]
- Lista de Tablas, Ilustraciones y Cuadros: "Se debe elaborar un índice de Tablas y Figuras, indicando el **número consecutivo, el título y la página** respectiva." (G2 p. 14) [HARD]
- **Resumen / Abstract**:
  - "El resumen tiene que presentar el contenido básico de la tesis en relación con: **introducción, objetivos, diseño metodológico, resultados y conclusiones**. El resumen tiene que constar en **dos lenguas: en la lengua original de la tesis, y habitualmente en inglés**." (G2 p. 5) [HARD]
  - "Al inicio del documento se deberá **redactar en pasado** una breve introducción, objetivos de la tesis y de ser el caso qué experimentos se realizaron, los resultados obtenidos y un condensado de las conclusiones." (G2 p. 14) **[HARD: past tense]**
  - **Word limit: [NOT SPECIFIED]. Keywords: [NOT SPECIFIED].** The APA 7 fallback (an abstract of at most 250 words plus *Palabras clave*) is my inference only.
- Introducción: "debe resumir con claridad y precisión, **de manera discursiva**, los principales componentes del problema y de la Investigación" (G2 p. 14). [HARD] The plan form also expects an introduction (G1 p. 51): "¿Explica el interés del proyecto, las razones de la elección del tema, destaca los problemas …, su ubicación en contexto y describe qué es lo que busca lograr…?" and "¿**Indica las fuentes de investigación** utilizadas para obtener el tema de tesis?" (see C5).

### 2.11 Results, discussion, conclusions, recommendations (thesis)

- 3.1 Results: "presentará los resultados … **en forma lógica** … como tabla, figuras, gráficos u otros…; **de manera ordenada y comprensible**…; Los resultados obtenidos servirán para formular las conclusiones." (G2 p. 17) [HARD]
- 3.2 Discussion (G2 p. 18), required sequence [HARD: "desarrollará de la forma siguiente"]:
  - a. "Iniciar con un breve resumen de los resultados más importantes, sin caer en detalles."
  - b. Explanations, "comparación de otras investigaciones relacionadas", covering both supporting and contradicting results.
  - c. "todos los hallazgos importantes, enriquecidos con las referencias previas."
  - d. "juicios e interpretaciones del autor … contrastarlas con otros investigadores."
  - "**Especificar las limitaciones y debilidades que tuvo el estudio.**" Implications for the field. "Al formular cada conclusión se resumen las pruebas que la respaldan." Recommendations for future research and "aspectos que no fueron abordados".
- 4.1 Conclusions: answer "¿Y qué?"; "una reflexión final" (G2 p. 18). S2 §XI: "**Las conclusiones deben responder a los objetivos y estar respaldadas por resultados o evidencia.**" (S2 p. 4) [HARD]
- 4.2 Recommendations: "**De cada conclusión debe desprenderse una recomendación. Tiene que desarrollar tres conclusiones y por consiguiente son tres recomendaciones.**" (G2 p. 19). **[HARD as worded]** Taken literally, this means exactly three conclusions and three recommendations, one recommendation per conclusion.

### 2.12 Traceability (course)

- S2 §XI: "Debe existir **trazabilidad problema → pregunta → objetivo → método → evidencia → resultado/discusión → conclusión**." (S2 p. 4) [HARD]
- "La utilización de IA/ML/DL no constituye por sí misma una contribución científica; la contribución debe diferenciarse del estado del arte y sustentarse con evidencia." (S2 p. 4) [HARD]
- "Cuando corresponda, documentar datos, preprocesamiento, baselines, métricas, validación, reproducibilidad, sesgos, limitaciones y amenazas a la validez." (S2 p. 4) [HARD]

---

## 3. Citation and references

| Rule | Source | Status |
|---|---|---|
| APA "Versión vigente" for writing and citations. | G1 p. 4; G2 p. 4 | HARD |
| "REFERENCIAS De acuerdo al APA vigente." | G1 p. 35 | HARD |
| "La lista de referencias bibliográficas en el estilo APA (Versión vigente) se inserta al final del documento, **a doble espacio y con sangría francesa** para cada entrada. … autor, año de la publicación, título y datos de la publicación." | G2 p. 19 | HARD (see C9 for spacing) |
| Plan form: "¿Cumple las normas APA? Versión vigente." and "¿Las Referencias está actualizada en un **máximo de 5 años (salvo los textos clásicos)**?" | G1 p. 53 (also p. 55) | HARD, checklist |
| Thesis checklist: "Presenta las referencias de acuerdo a lo estipulado en la Norma APA (Edición vigente)". | G2 p. 35 | HARD, checklist |
| Respect copyright and cite all concepts ("respetando el derecho de autor"). | G1 p. 19; G2 p. 17 | HARD |
| Heading name: "REFERENCIAS". The guides use "Referencias", not "Bibliografía", as the section title. G1 p. 43 does mention "Bibliografía" when discussing spacing. | G1 pp. 10, 35; G2 p. 10 | HARD (heading name) |
| S1 tools listed: "Zotero/Mendeley … Turnitin". | S1 p. 7 | REC |

- **Minimum number of references**: [NOT SPECIFIED].
- **Age**: antecedentes ≤ 5 years and references ≤ 5 years, except classics or when the topic justifies older work.

---

## 4. Language and style rules

| Topic | What the documents say | Status |
|---|---|---|
| Thesis language | Spanish is implied. A thesis in another language needs a translated Spanish copy (G2 p. 6). The abstract goes in the original language plus "habitualmente en inglés" (G2 p. 5). | HARD |
| Foreign words / anglicisms | "**No use letra cursiva excepto para las palabras cuyo origen no sea de un idioma al español.**" (G1 p. 43). Title: "las palabras distintas al español escribirlas en cursiva" (G1 p. 15). So English terms (*dataset*, *fine-tuning*, *encoder*, *foundation model*) go in italics, and italics are used **for nothing else**: no italics for emphasis. | HARD |
| Numbers | "**No use cursiva para los números.**" (G1 p. 43) | HARD |
| Tense | The Resumen must be written **in the past tense** (G2 p. 14). No other tense rule. | HARD (abstract only) |
| Grammar | Plan form: "¿La redacción gramatical del Plan de tesis es correcta, coherente y cumple las normas APA? Versión vigente" (G1 p. 53). The introduction is written "de manera discursiva" (G2 p. 14). | HARD, checklist |
| Person/voice (impersonal, 3rd person, 1st person) | **[NOT SPECIFIED]** in either guide or syllabus. The guides refer to "el tesista" / "el investigador" in the third person. | NOT SPECIFIED |
| Abbreviations/acronyms | **[NOT SPECIFIED]** | NOT SPECIFIED |
| Decimal separator | **[NOT SPECIFIED]**. The guide's own text uses a decimal comma ("3,0 cm", "2,5 cm", G1 p. 43), but the diagram on p. 44 writes "3.0 cm". This is only an observation. | NOT SPECIFIED |
| Units / SI | **[NOT SPECIFIED]** | NOT SPECIFIED |
| Gender-neutral language | **[NOT SPECIFIED]** | NOT SPECIFIED |
| Equations | **[NOT SPECIFIED]** | NOT SPECIFIED |
| Use of generative AI | "Las herramientas de IA generativa **podrán utilizarse como apoyo**, manteniendo el estudiante responsabilidad sobre autoría, exactitud, citación, confidencialidad e integridad académica." (S2 p. 3). S2 competency: "uso responsable de herramientas de IA en investigación" (S2 p. 1). | HARD (conditions) |
| AI detection | "Los informes y productos generados **deberán superar la revisión por detectores de plagio y de generación automática mediante modelos de lenguaje (LLM)**." (S1 p. 1) | HARD in S1; S2 does not repeat it |
| Plagiarism | The thesis must be "original e inédito" (G2 p. 5). The signed declaration of no plagiarism is in Anexo XIII (G2 p. 46). S2 week 14: "Revisión de Borrador de Plan de Tesis/Borrador de Tesis **con Turnitin**" (S2 p. 3). **No Turnitin percentage threshold appears anywhere** [NOT SPECIFIED]. | HARD (no % given) |

---

## 5. Physical format: G1 Anexo III "Estándares para la presentación del Plan de Tesis", pp. 43–44

G2 has **no** format-standards annex of its own. G1 Anexo III is the only one, and its text keeps saying "la tesis", so it is in practice the format standard for the thesis too.

| Item | Rule (quote) | Status |
|---|---|---|
| Paper | "La tesis se presenta en papel bond, **tamaño A4**, peso 80 gr." | HARD |
| Margins | "Margen **izquierdo y superior 3,0 cm**. Margen **derecho e inferior 2,5 cm**." The diagram on p. 44 shows the same. | HARD |
| Line spacing | "El texto de la tesis se hará a espacio de **1,5 interlíneas**." Single spacing is allowed for: "Notas y citas textuales; Párrafos de listas, pero no entre los elementos listados; Bibliografía; Índice de ilustraciones y tablas; Anexos." | HARD |
| Font | "Utilice como fuente para el tipo de letra, **Arial 11**." | HARD |
| Font in figures/tables | "Use el mismo tipo de letra para enumerar las ilustraciones y las tablas, el cual puede ser diferente del tipo de letra usado para el texto…" "**Podrá usar tamaño reducido de letras solamente en los anexos, en las ilustraciones y tablas.**" | HARD |
| Italics | Only for words that are not Spanish. No italics for numbers. | HARD |
| Pagination: preliminaries | "**Números romanos en minúscula** para el cuerpo preliminar…, comenzando por la página de la portada…, que **no se enumera, pero se considera**, se termina en el resumen (de la portada hasta el resumen)". | HARD (see C10) |
| Pagination: text | "**Números arábigos** para el texto. **La introducción y el inicio de cada capítulo no se enumeran, pero se considera** (de la introducción hasta los anexos)." Chapter-opening pages carry no printed number. | HARD |
| Number position | "Tanto la paginación de numeración romana como la arábiga, se escriben en la **derecha superior** (APA Edición vigente)." | HARD |
| Pages with tables/figures | "Las páginas en las que aparecen tablas y figuras también deben enumerarse." | HARD |
| The word "página" | "**No use la palabra 'página'** antes de la numeración de las páginas." | HARD |
| **Page limit** | "Las **Tesis de Maestría deberán tener un máximo de 130 páginas, incluyendo gráficas, tablas, referencias y anexos.**" | HARD |
| Header | "Sólo se considera la paginación del APA (vigente) y lo indicado en la presente guía." So the header carries only the page number: no running head, no chapter titles. | HARD |
| Alignment | "Los párrafos deben estar alineados como se muestra" (a margin diagram only). Justified vs. left-aligned text is not stated. | EX |
| Plan form | "¿El documento está presentado con numeración correcta?"; "¿…cumple con los estándares de presentación…?" | HARD, checklist (G1 p. 53) |

### 5.1 Carátula (cover page)

- **Plan**: example on G1 p. 17; description on G1 pp. 15–16. **Thesis**: example on G2 p. 20.
- **Required content [HARD]** (G1 p. 15): institution name, UNI logo, title, propósito de graduación, author(s), adviser(s), place and date. The G2 checklist (p. 32) adds "Unidad de Posgrado de la Facultad……" and "Escuela de Posgrado", "Logotipo UNI", "Título de la Tesis", "Grado Académico", "Nombres y Apellidos del tesista", "Nombres y Apellidos del Asesor", "Lugar y fecha".
- **Placement [HARD]**:
  - Institution name in the centre.
  - Logo "en el centro superior el escudo oficial de la UNI".
  - Title in capitals, upper-middle part.
  - Propósito de graduación "en la parte media baja, centrado y destacado".
  - Authors: "Nombres y apellidos completos **en mayúsculas** … centrado, inmediatamente abajo del propósito de graduación".
  - Place and date: "En el primer renglón, se indica la ciudad y el país, en el segundo renglón, se indica el año" (G1 pp. 5, 15–16).
- **Fonts on the example cover [EX, but annotated as the template]** (G2 p. 20 for the thesis; G1 p. 17 for the plan):

| Line | Font annotation |
|---|---|
| "UNIVERSIDAD NACIONAL DE INGENIERÍA" | **Poppins 22, negritas** (the scan renders it non-bold) |
| "UNIDAD DE POSGRADO FACULTAD ……… / ESCUELA DE POSGRADO" | Arial 16 negritas. For us: Facultad de Ingeniería Industrial y de Sistemas |
| UNI shield | — |
| "TESIS" (plan: "PLAN DE TESIS") + "**"TÍTULO DE LA TESIS"**" (in quotation marks, capitals) | Arial 14 negrita |
| "PARA OBTENER EL GRADO ACADÉMICO DE MAESTRO/DOCTOR EN CIENCIAS CON MENCIÓN EN: (colocar el nombre de la mención)" | Arial 12. For us: "…MAESTRO EN CIENCIAS CON MENCIÓN EN INTELIGENCIA ARTIFICIAL" (wording assumed) |
| "ELABORADA POR:" | Arial 12 negrita |
| "NOMBRE DEL GRADUANDO" | Arial 12 negrita |
| "ASESOR:" | Arial 12 negrita |
| "NOMBRE DEL ASESOR" | Thesis cover only; the arrow points to Arial 14 negrita |
| "LIMA, PERÚ" | Arial 14 negrita |
| "AÑO" | Arial 12 negrita |

- **Spine (lomo)**: "debe de constar el título de la Tesis y se colocará de tal forma que se pueda leer de abajo hacia arriba a partir de la inscripción de la Escuela de Posgrado o de la Unidad de Posgrado y el nombre y apellido del Tesista" (G2 p. 6). [HARD, printed copy]
- **Binding**: "05 juegos empastados"; for the Maestro degree the colour is "**Verde Bosque**"; "Dos (02) CD o USB de la Tesis"; "01 CD o USB con la presentación"; "Constancia de Idioma Inglés nivel Intermedio" (G2 p. 29). [HARD, procedural]

### 5.2 Figures, tables, captions

| Item | What the documents say | Status |
|---|---|---|
| Lists of tables and figures | Required in the front matter, with consecutive number, title and page (G2 p. 14). | HARD |
| Numbering font | The same font for numbering illustrations and tables; reduced size allowed inside them (G1 p. 43). | HARD |
| Caption position (above/below) | **[NOT SPECIFIED]** in the guides. They defer to APA "vigente". APA 7 puts the **number (bold) and title (italic) above** the table or figure, and a "*Nota.*" below. That is an inference from APA, not stated by UNI. | NOT SPECIFIED |
| "Fuente:" line | **[EX only]**. G1 p. 30 reproduces Hernández Sampieri's figure with "Autor: Roberto Hernández Sampieri / Fuente: Metodología de la Investigación 6ta Edición" **below** the figure. The matriz templates carry "Autor:" and "Elaboración:" lines below the table (G1 pp. 41–42; G2 pp. 27–28). The common Peruvian convention "Fuente: Elaboración propia" fits these templates but is not prescribed. | EX |
| Table vs "cuadro" vs "ilustración" terminology | The guides use "Tablas, Ilustraciones y Cuadros" (structure) and "Tablas y Figuras" (G2 p. 14) interchangeably. | EX |

---

## 6. Evaluation rubrics and checklists for the written document

### 6.1 S2 §X Rúbrica, Trabajo Final: Borrador integral de tesis (S2 p. 4), 20 pts

| # | Criterion | Excellent | Good | In progress | Insufficient |
|---|---|---|---|---|---|
| 1 | Problema, pregunta y relevancia | **3**: problem clearly delimited, precise question(s), solid justification, gap identified, scientific/technological relevance argued | 2 | 1 | 0.5 |
| 2 | Estado del arte y marco teórico | **3**: critical and up-to-date review; compares studies, identifies the gap, supports decisions, links the theory to the problem and method | 2 | 1 ("Predomina el resumen de fuentes") | 0.5 |
| 3 | Objetivos, hipótesis/preguntas, variables y consistencia | **3**: measurable objectives; relevant hypotheses/questions; variables/dimensions/indicators and instruments defined when applicable; full traceability | 2 | 1 | 0.5 |
| 4 | Metodología y protocolo | **4**: justified, reproducible design; data/population/sample, instruments, procedure, protocol, metrics, validation and **threats to validity** defined and coherent | 3 | 2 | 1 |
| 5 | Evidencia, resultados y discusión | **2**: sufficient, traceable evidence; correct analysis; comparison with baselines/literature; critical, supported discussion | 1.5 | 1 | 0.5 |
| 6 | Calidad e integración del borrador | **2**: complete, coherent, well written draft with references, tables/figures, annexes, **formato institucional**, traceability and **integridad académica** | 1.5 | 1 | 0.5 |
| 7 | Plan de Tesis aprobado | **3**: formal evidence of approval **by week 14**; otherwise 0 | — | — | 0 |

- "Regla de aplicación: los criterios metodológicamente no aplicables … no deben penalizarse. El docente debe registrar 'No aplica'…" (S2 p. 4).
- Grade weights: EP (Plan de Tesis + protocolo + defensa intermedia, week 9) 25%; EF (borrador v2 + defensa, week 16) 25%; TI (progressive work) 50%. NF = (EP + EF + 2·TI)/4 (S2 pp. 3–4).
- S1 (taken to be the older version; see the note at the top). Its weights are EP week 8, EF week 15. Its intermediate report is "aprox. 10-15 págs" (S1 p. 5) but "informe 5–7 págs" on p. 2, which is internally inconsistent. Its mini-report is 2–3 pages (S1 pp. 2, 4–5). It also mentions a rubric "exactitud + diseño experimental + claridad" for the midterm demo (S1 p. 5). No written-document rubric beyond this.

### 6.2 G1 Anexo VIII 8.1: plan evaluation form for Maestría/Doctorado de Investigación (G1 pp. 51–53), SÍ/NO per item, signed by the adviser

1. **Título**: clear and precise, answering the problem; reflects the objectives.
2. **Índice**: structured according to the Guide.
3. **Introducción**: interest, reasons, problems, context, aims; sources used.
4. **Contenidos**: justification and references; evolution of the antecedentes of the dependent variable, oldest to newest.
5. **Problema**: problem reality stated clearly; the independent variable is proposed as the solution.
6. **Objetivos**: general objective formulated correctly; specific objectives lead to the general one.
7. **Alcances y limitaciones**: clear scope; limitations stated "con claridad y precisión".
8. **Justificación**: clear and precise; the type of justification fits.
9. **Metodología**: clear and concrete; collection instruments and analysis techniques.
10. **Marco teórico**: antecedentes, bases teóricas, "¿Presenta Hipótesis y variables?", relevant concepts.
11. **Ubicación del proyecto**.
12. **Esquema de contenido** according to the Guide.
13. **Cronograma**.
14. **Presupuesto**.
15. **Financiamiento** (type).
16. **Fuentes**: APA vigente; references ≤ 5 years, classics excepted.
17. **Otros**: correct, coherent grammar plus APA; correct page numbering; presentation standards met.

### 6.3 G2 Anexo V: thesis review checklist (G2 pp. 32–35)

- **Purpose**: "verificar el **rigor científico** de la investigación" (G2 p. 32).
- **Items**:
  - Carátula: each element listed in §5.1.
  - Índice: per the Guide, with page numbers.
  - I. Planteamiento: 1.1 problem identification (specific problem, dependent variable); 1.2 problem formulation; 1.3 justification (teórica, práctica, metodológica); 1.4 scope (temático, espacial, temporal); 1.5 objectives (infinitive verbs, aligned).
  - II. Marco teórico: 2.1 chronological antecedentes; 2.2 bases by variable; 2.3 definition of terms; 2.3 hypotheses (clear, aligned, empirically verifiable); 2.4 variables (problem/dependent, solution/independent, consistency matrix, operationalization matrix for doctorate only).
  - III. Diseño metodológico: 3.1 type (enfoque, nivel, diseño, métodos); 3.2 population and sample; 3.3–3.4 techniques and instruments.
  - Referencias: APA.
  - Anexos: matriz de consistencia "formulada correctamente"; other annexes.
  - IV. Desarrollo; V. Conclusiones y recomendaciones: free-text comments.
- "NOTA: El Máximo de Revisiones es de tres Revisiones" (G2 p. 35).
- This checklist's section labels ("I. Planteamiento del problema", "III. Diseño metodológico") follow the **plan** structure, not the G2 thesis chapters (see C1).

### 6.4 Approval mechanics (for context)

| Rule | Source |
|---|---|
| The plan is approved by the adviser from the 2nd cycle on; at most 3 plan revisions; the topic allows at most 2 revisions and 2 changes. | G1 pp. 45, 48 |
| The thesis is approved by the adviser in the 4th cycle; at most 5 years from admission; at most 3 revisions. | G2 pp. 24, 42 |
| The thesis is also evaluated by an optional external adviser and 2 specialists. | G2 p. 37 |
| Reviewers and advisers have at most 30 days per review. | G1 p. 6 |
| Defense: up to 30 min presentation, up to 15 min questions, 5 min deliberation. Jury: Decano (president), Director de Unidad de Posgrado, asesor, two specialists. | G2 p. 30 |
| If failed: 2 months to address observations. | G2 p. 31 |
| Course simulated defense: 12–15 min plus questions. | S2 p. 3; S1 p. 7 |

---

## 7. Conflicts and inconsistencies

| # | Conflict | Sources | Suggested handling |
|---|---|---|---|
| **C1** | **Plan and thesis chapter structures differ.** The plan has 5 chapters: I Planteamiento (antecedentes inside), II Objetivos, III Marco teórico (hypotheses inside), IV Diseño metodológico, V Administración. The thesis has 4: I Protocolo (problem, objectives, hypotheses, §1.4 Metodología), II Marco teórico (antecedentes here), III Desarrollo (results, discussion, contrastación), IV Conclusiones. The G2 Anexo V checklist still uses the plan's section labels. | G1 pp. 9–10 vs G2 pp. 9–10, 33–34 | Use the G1 structure for the Plan de Tesis and the G2 structure for the thesis. **Checked `thesis/` (read-only):** the FIIS template has 6 chapters, separate Conclusiones/Recomendaciones, plus Prólogo and Símbolos sections. The chapters are I PLANTEAMIENTO DEL PROBLEMA (diagnóstico, formulación, objetivos, hipótesis, justificación, alcances y delimitaciones), II FUNDAMENTO TEÓRICO (antecedentes, bases, definición de términos), III MÉTODO DE LA INVESTIGACIÓN, IV DESARROLLO DE LA SOLUCIÓN, V RESULTADOS, VI DISCUSIÓN DE LOS RESULTADOS (contrastación de hipótesis). This matches **neither** G2 §4.1 (4 chapters) nor the G1 plan. It is closest to the S2 sequence and to G2's specialization structure. The authors need to decide which to follow, possibly after asking the teacher or the Unidad de Posgrado FIIS. |
| **C2** | **S2 assigns chapter numbers that fit neither guide**: "Capítulo I" (problem + state of the art), "Capítulo II" (marco teórico), "Capítulo III" (diseño metodológico), plus separate "Capítulo de Resultados" and "Capítulo de Discusión". S1 week 15 lists yet another set: "problema/objetivos, estado del arte resumido, metodología, resultados, discusión, amenazas a la validez, ética, trabajo futuro". | S2 p. 2–3; S1 p. 7 | The guides (institutional and "en forma estricta") outrank the syllabi for the official document. The extra content S2/S1 want (threats to validity, ethics, separate discussion depth) can go inside the G2 chapters (§1.4, §3.2). Confirm with the teacher which numbering the course draft must follow. |
| **C3** | **Matriz de operacionalización**: the guides make it doctoral-only. S2 week 5 requires "matriz de operacionalización y matriz de consistencia v1", and rubric criterion 3 wants "variables/dimensiones/indicadores e instrumentos". G2 also asks for dimensions and indicators of variables (p. 15 problems, p. 33 checklist). | G1 pp. 7, 35, 42; G2 pp. 19, 28 vs S2 pp. 2, 4 | Option: include it as an extra annex. That satisfies the course and does no harm under the guides. Authors' decision. |
| **C4** | **Limitaciones**: the master's plan structure omits them (only the doctoral 1.2.3 has them). The plan evaluation form used for master's asks whether they are "determinadas con claridad y precisión". G2 §3.2 requires limitations in the Discussion. S2 requires "amenazas a la validez". | G1 pp. 9 vs 11, 52; G2 p. 18; S2 p. 2 | Option: include a Limitaciones subsection (1.2.3) in the plan and limitations/threats to validity in the thesis. Authors' decision. |
| **C5** | **Introducción**: the plan structure (G1 p. 9) has no Introduction, but the plan evaluation form item 3 grades one. | G1 p. 9 vs p. 51 | Option: add a short Introduction to the plan. Authors' decision. |
| **C6** | **Problem wording**: G1 allows "preguntas o de manera declaratoria". S2 asks for "pregunta(s) precisa(s)" and "problema → pregunta → objetivo" traceability. | G1 p. 19 vs S2 pp. 2, 4 | Option: phrase the general and specific problems as questions. That satisfies both sources. |
| **C7** | **Hypotheses**: G1 says exploratory/descriptive work needs none, but the subtitle must stay with a justification. G2's thesis structure always includes 1.3 Hipótesis, and its checklist expects them to be "susceptibles de verificación empírica". The matriz has a mandatory Hipótesis column. | G1 p. 20; G2 pp. 9, 34; G1 p. 41 | Options for the authors: (a) state empirically testable comparative hypotheses, so that G2 §3.3 Contrastación and the G2 checklist apply; or (b) declare the study exploratory/descriptive and keep the subtitle with a justification, per G1 p. 20. That risks the G2 checklist item and leaves the matriz Hipótesis column empty. The guides lean towards (a). |
| **C8** | **Age of sources**: antecedentes "de los últimos cinco años" with exceptions (G1 p. 18; G2 p. 17); references "máximo de 5 años (salvo los textos clásicos)" (G1 p. 53). S2 says quantity doesn't define quality. These are consistent but strict. | — | Antecedentes from 2021 or later for a 2026 plan. Older works (e.g. U-Net 2015, Sen1Floods11 2020) need the "textos clásicos" or topic justification. |
| **C9** | **Reference-list spacing**: G1 Anexo III allows **single** spacing for "Bibliografía"; G2 p. 19 says "**a doble espacio** y con sangría francesa". | G1 p. 43 vs G2 p. 19 | G2 is the thesis-specific rule: use double spacing with a hanging indent. Single spacing is defensible for the plan. |
| **C10** | **Front-matter page numerals**: G1 Anexo III says lowercase Roman for the front matter and Arabic for the text. The G2 p. 7 diagram labels the front-matter bracket "NUMERACIÓN EN ARÁBIGOS", while its right-side label (cut off) reads "NUMERACIÓN EN ROMANOS Y…". G1 p. 15 says the preliminaries "se numeran con números romanos". | G1 pp. 15, 43 vs G2 p. 7 | Follow the explicit text (Roman, then Arabic). The diagram label looks like an error. |
| **C11** | **Introduction numbering**: G1 p. 43 says Roman numerals run "de la portada hasta el resumen" and Arabic numbering starts at the Introducción, which "no se enumera, pero se considera". So the Introduction is the first Arabic page, unprinted. | G1 p. 43 | Start Arabic page 1 at the Introduction and suppress the printed number on it and on every chapter-opening page. |
| **C12** | **Page limit applies to which document?** "Las Tesis de Maestría deberán tener un máximo de 130 páginas, incluyendo gráficas, tablas, referencias y anexos" sits in G1, the plan guide. G2 has no format annex at all. | G1 p. 44 | Treat it as binding for the final thesis, annexes and references included. |
| **C13** | **Typography of the cover vs body**: the body must be Arial 11, but the cover uses Poppins 22 for the university name. The annotation says "negritas", while the rendered sample is regular weight. | G1 pp. 17, 43; G2 p. 20 | Use Poppins only on the cover, and Arial 11 for everything else. In XeLaTeX, load Arial via fontspec; check font availability on Overleaf, where Arial may need uploading or a metric clone such as Liberation Sans/Arimo, which a strict reviewer could notice. |
| **C14** | **Number of conclusions**: G2 p. 19 demands exactly three conclusions and three recommendations. S2 says conclusions must answer the objectives. With more than 3 specific objectives these collide. | G2 p. 19 vs S2 p. 4 | Options: (a) 3 specific objectives, giving a clean 3/3/3 mapping; or (b) more objectives with conclusions grouped into 3. Authors' decision. Ask the teacher whether "tres" is read literally. |
| **C15** | **Anti-AI detection**: S1 says products "deberán superar la revisión por detectores … de generación automática (LLM)". S2 allows generative AI "como apoyo" with the author responsible, and does not repeat the detector clause. | S1 p. 1 vs S2 p. 3 | S2 looks like the newer version, but nothing formally withdraws the S1 clause. Treat the LLM-detector requirement as **live**: the draft goes through Turnitin (S2 week 14), whose AI indicator may be used. |
| **C16** | **Midterm report length** in S1: "informe 5–7 págs" (p. 2) vs "Reporte aprox. 10-15 págs" (p. 5). | S1 | S2 (taken to be current) gives no page count. Ask the teacher. |
| **C17** | **"Tipo de investigación" vocabulary**: G1 offers básica/aplicada × exploratoria…explicativa, etc. The G2 checklist puts "aplicada" under *nivel* and lists *métodos* "Hipotético-deductivo, analítico, inductivo, deductivo, sintético", which G1 does not list. | G1 pp. 22–28 vs G2 p. 34 | State the type on every axis the G2 checklist uses (enfoque, nivel, diseño, métodos) and give the G1 type (básica/aplicada). The choice is the authors'. Caution: G1 p. 24 defines "experimental" as "control total sobre las variables … el laboratorio tiene que estar certificado". Alternatives to weigh: *cuasiexperimental*, or an explicitly "experimental computacional" design that justifies control through fixed splits and seeds. Enfoque cuantitativo and método hipotético-deductivo are uncontroversial if hypotheses are stated. |
| **C18** | Minor copy errors: G2 p. 4 says Guía N°2 is "para la elaboración del Plan de Tesis"; G2 p. 9 has "Contratación" for "Contrastación"; G2 p. 18 addresses the reader in 2nd person ("tu ensayo"). | — | Ignore. |

---

## 8. Supplementary: SES2 (Dr. Wester Zela, course session 2)

An ML example of the matriz de consistencia (SES2 p. 5). Its columns are:

> Problema de Estudio (a question: "¿Cómo mejorar la precisión…?") | Objetivo General (infinitive: "Desarrollar un modelo…") | Objetivos Específicos (infinitive verbs: Analizar, Seleccionar, Entrenar, Evaluar, Optimizar) | Hipótesis ("El uso de algoritmos de ML mejora significativamente la precisión … frente a métodos tradicionales") | Variables (independiente: the algorithms; dependiente: the classification; input variables) | Metodología (enfoque cuantitativo, diseño experimental, phases) | Técnicas e Instrumentos (dataset, libraries, cross-validation, confusion matrix, ROC-AUC) | **Resultados Esperados**

- The "Resultados esperados" column is **not** in the guide template.
- SES2 p. 4 lists the kinds of validity to consider: instrumentos, técnica, diseño, propósito, generalización. [REC]
- The example confirms the teacher's reading for ML theses:
  - independent variable = the model/algorithm;
  - dependent variable = the task performance;
  - problem phrased as a question;
  - objectives in the infinitive;
  - a comparative hypothesis.

---

## 9. Quick checklist for our LaTeX template (derived from the above)

- [ ] A4; margins: left 3.0, top 3.0, right 2.5, bottom 2.5 cm.
- [ ] Arial 11; line spacing 1.5. Single spacing only in notes and quotes, inside list paragraphs, references (but G2 says double, see C9), lists of figures/tables, and annexes.
- [ ] Italics only for non-Spanish words; never for numbers or emphasis.
- [ ] Page numbers top right. Lowercase Roman from the cover (cover unprinted) to the Resumen/Abstract. Arabic from the Introducción, with no printed number on the Introduction or chapter-opening pages. No running head. No word "página".
- [ ] ≤ 130 pages in total, including references and annexes.
- [ ] Cover per G2 p. 20: title in capitals and quotes, ≤ 20 words, foreign words in italics; authors' full names in capitals.
- [ ] Front matter in this order: Dedicatoria, Agradecimientos, [Copia de documentos after the defense], Índice, Lista de tablas y figuras (number, title, page), Resumen (Spanish, past tense), Abstract (English), Introducción.
- [ ] Chapters per G2 §4.1: I Protocolo, II Marco teórico, III Desarrollo, IV Conclusiones y recomendaciones (exactly 3 + 3). Then Referencias (APA 7, hanging indent), then Anexos: Matriz de consistencia (+ Operacionalización, + reproducibility annex for the course).
- [ ] Problems as questions; objectives with infinitive verbs, one per specific problem; hypotheses empirically testable; justification split into teórica, práctica, metodológica; scope split into temático, espacial, temporal.
- [ ] Antecedentes ≤ 5 years (classics justified), in chronological order oldest → newest, each with its method and main results.
- [ ] Signed Declaración de autenticidad y no plagio (G2 Anexo XIII); Turnitin pass.
