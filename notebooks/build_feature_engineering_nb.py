"""Builds and runs notebooks/feature_engineering.ipynb (Spanish, course deliverable "Avance sem 5").
The notebook only reads the outputs of scripts/feature_engineering.py and one `peru-dev` chip for
the feature maps; it reads no test split.

    uv run python scripts/feature_engineering.py
    uv run python notebooks/build_feature_engineering_nb.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
CONCLUSIONS = (HERE / "fe_conclusiones.md").read_text()

cells = [
    nbf.v4.new_markdown_cell("""# Ingeniería de características para segmentación de inundaciones en Perú

**Avance semana 5.** Autores: Gerson Andre Garrido Mansilla y Renzo Elvis Cerron Tome
(Maestría en Ciencias con mención en IA, UNI).

**Contexto de la tesis.** El objetivo es segmentar áreas inundadas en imágenes Sentinel-2 de
Perú. Se comparan dos modelos fundacionales geoespaciales (TerraMind v1-B y
Prithvi-EO-2.0 300M) con una U-Net de referencia. Datos:
- Sen1Floods11, con sus particiones oficiales;
- Perú 2023, con etiquetas UNOSAT;
- Perú 2017, con etiquetas Copernicus EMS.

Los modelos profundos reciben reflectancia cruda y aprenden sus propias características. En
este cuaderno se estudia de forma explícita **qué aportan las características diseñadas**: los
índices espectrales y el contexto espacial local. Para ello se usa un clasificador clásico por
píxel (*gradient boosting*), en dos escenarios:
- **(a)** entrenado con etiquetas de Perú (`peru-train`);
- **(b)** entrenado solo con Sen1Floods11 y aplicado a Perú. Es el escenario sin etiquetas
  peruanas que estudia la tesis.

**Protocolo.** Toda la evaluación se hace en `peru-dev`, la partición de selección. Las
particiones de prueba (`peru-test`, `peru-2017-test`) no se leen. Este trabajo es exploratorio
y no modifica la comparación final de la tesis.

El inventario completo de la ingeniería de características que ya usa el proyecto está en
`docs/feature_engineering.md`. Incluye la selección de bandas por modelo, la normalización
oficial, la armonización radiométrica, la igualación de histogramas, FDA y el control de
calidad de etiquetas con MNDWI.

Código del experimento: `scripts/feature_engineering.py`."""),
    nbf.v4.new_code_cell("""import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, rasterio

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("fe", REPO / "scripts" / "feature_engineering.py")
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
%matplotlib inline
OUT = REPO / "outputs" / "feature_engineering"
pd.set_option("display.precision", 4)
print(len(fe.ALL), "características:", fe.ALL)"""),
    nbf.v4.new_markdown_cell("""## 1. Características

Reflectancia r = ND / 10 000, con las 13 bandas L1C de Sentinel-2. Notación: G = B03 (verde),
R = B04 (rojo), NIR = B08, SWIR1 = B11, SWIR2 = B12.

| Grupo | Característica | Definición | Motivación |
|---|---|---|---|
| Bandas | B01–B12 (13) | Reflectancia cruda | Línea base: lo que reciben los modelos profundos |
| Índices | NDWI | (G − NIR)/(G + NIR) | El agua absorbe el NIR (McFeeters, 1996) |
| | MNDWI | (G − SWIR1)/(G + SWIR1) | El SWIR separa el agua del suelo construido o desnudo (Xu, 2006) |
| | MNDWI2 | (G − SWIR2)/(G + SWIR2) | Variante con SWIR2 |
| | NDVI | (NIR − R)/(NIR + R) | Vegetación; separa la vegetación del agua |
| | NDMI | (NIR − SWIR1)/(NIR + SWIR1) | Humedad del suelo y de la vegetación |
| | AWEI_nsh | 4(G − SWIR1) − (0.25 NIR + 2.75 SWIR2) | Índice de agua automatizado, sin sombras (Feyisa et al., 2014) |
| | AWEI_sh | B + 2.5G − 1.5(NIR + SWIR1) − 0.25 SWIR2 | Variante que suprime sombras |
| | WRI | (G + R)/(NIR + SWIR1) | Razón de agua |
| Contexto | MNDWI_mean3, MNDWI_mean7 | Media local de MNDWI en ventanas de 3×3 y 7×7 | Un píxel aislado no tiene contexto espacial |
| | MNDWI_std7 | Desviación estándar local de MNDWI en 7×7 | Textura; resalta bordes y ruido |
| | NIR_mean7, NDVI_mean7 | Media local en 7×7 | Entorno: vegetación o desierto |

Conjuntos comparados: `bands` (13), `indices` (8), `bands+indices` (21) y
`bands+indices+context` (26)."""),
    nbf.v4.new_markdown_cell("""### Ejemplo visual en un recorte de `peru-dev`

Se elige el recorte de dev cuya fracción de agua está más cerca de 0.4, para que se vean agua y
suelo seco a la vez."""),
    nbf.v4.new_code_cell("""m = pd.read_csv(REPO / "data/peru/unosat2023/manifest.csv")
m["stem"] = m.date + "_" + m.chip
dev = set((REPO / "data/peru/unosat2023/splits/dev.txt").read_text().split())
d = m[m.stem.isin(dev)]
stem = d.loc[(d.water_frac - 0.4).abs().idxmin(), "stem"]
f, y, ok = fe.load(REPO / f"data/peru/unosat2023/s2/{stem}.tif", REPO / f"data/peru/unosat2023/label/{stem}.tif")
rgb = np.clip(f[[3, 2, 1]].transpose(1, 2, 0) / 0.3, 0, 1)
show = [("RGB", rgb, None), ("Etiqueta (agua=1, ignorado=gris)", np.where(y < 0, 0.5, y), "gray"),
        ("MNDWI", f[fe.ALL.index("MNDWI")], "RdBu"), ("NDWI", f[fe.ALL.index("NDWI")], "RdBu"),
        ("AWEI_sh", f[fe.ALL.index("AWEI_sh")], "RdBu"), ("NDVI", f[fe.ALL.index("NDVI")], "BrBG"),
        ("MNDWI_mean7", f[fe.ALL.index("MNDWI_mean7")], "RdBu"), ("MNDWI_std7", f[fe.ALL.index("MNDWI_std7")], "magma")]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax, (t, a, cm) in zip(axs.flat, show):
    im = ax.imshow(a, cmap=cm, vmin=None if cm in (None, "gray", "magma") else -1, vmax=None if cm in (None, "gray", "magma") else 1)
    ax.set_title(t); ax.axis("off")
    if cm not in (None, "gray"): fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"peru-dev: {stem}"); plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## 2. Separabilidad univariada (AUROC)

Para cada característica se calcula el AUROC entre agua y suelo seco, es decir, qué tan bien la
característica sola ordena los píxeles de agua por encima de los secos. Se muestra
max(AUROC, 1 − AUROC), porque una característica que separa en sentido inverso es igual de útil.
Se compara en tres conjuntos: Sen1Floods11 `train`, `peru-train` y `peru-dev`.

Una característica es **robusta al cambio de dominio** si su AUROC se mantiene alto en
Sen1Floods11 y en Perú."""),
    nbf.v4.new_code_cell("""au = pd.read_csv(OUT / "auroc.csv").set_index("feature")
sep = ((au - 0.5).abs() + 0.5).rename(columns={"auroc_s1f11-train": "Sen1Floods11 train", "auroc_peru-train": "peru-train", "auroc_peru-dev": "peru-dev"})
sep = sep[["Sen1Floods11 train", "peru-train", "peru-dev"]]
display(sep.sort_values("peru-dev", ascending=False).round(3))
ax = sep.plot.bar(figsize=(14, 4), width=0.8); ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC agua vs. seco")
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## 3. Clasificador por píxel: ¿qué conjunto de características funciona mejor?

**Clasificador.** `HistGradientBoostingClassifier` de scikit-learn, con 200 iteraciones y tasa
de aprendizaje 0.1.

**Entrenamiento.**
- Píxeles al azar con la proporción natural de clases: 600 por recorte de `peru-train` (753
  recortes) o 2 000 por recorte de Sen1Floods11 `train` (252 recortes).
- 3 semillas por configuración; la semilla fija la muestra y el modelo.

**Evaluación.**
- En `peru-dev`, sobre una de cada dos filas y columnas (1/4 de los píxeles).
- Métrica: IoU de agua agrupado sobre todos los píxeles etiquetados. Se reporta la media ± la
  desviación estándar entre semillas.

**Línea base.** Un umbral de MNDWI ajustado con las etiquetas de **entrenamiento**, a diferencia
del umbral "oráculo" ajustado en dev. Como referencia, ya medidos antes en dev:
- MNDWI > 0: IoU 0.486;
- umbral oráculo ajustado en dev (+0.27): 0.925;
- U-Net entrenada en Perú: 0.9415 ± 0.0019;
- U-Net entrenada solo en Sen1Floods11: 0.8364 ± 0.0224.

Fuente: `docs/METHODS_NOTES.md`, entrada del 2026-10-15, "Phase-D starting point"."""),
    nbf.v4.new_code_cell("""res = pd.read_csv(OUT / "results.csv")
display(res.round(4))
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
for ax, n, t in zip(axs, ["peru-train", "s1f11-train"], ["(a) entrenado en Perú", "(b) entrenado en Sen1Floods11"]):
    r = res[res.train == n].reset_index(drop=True)
    colors = ["tab:orange" if f.startswith("MNDWI") else "tab:blue" for f in r.features]
    ax.barh(range(len(r)), r.iou_mean, xerr=r.iou_std.fillna(0), color=colors)
    ax.set_yticks(range(len(r)), r.features)
    ax.set_xlim(0, 1); ax.set_title(t + " → peru-dev"); ax.set_xlabel("IoU de agua")
    for i, v in enumerate(r.iou_mean): ax.text(v + 0.01, i, f"{v:.3f}", va="center")
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## 4. Importancia por permutación

Se usa el modelo con las 26 características, semilla 0. Se mide cuánto baja el índice de
Jaccard (IoU) en `peru-dev` cuando se desordenan los valores de una característica, con 3
repeticiones sobre 1 200 píxeles al azar por recorte de dev.

Una importancia alta significa que el modelo **depende** de esa característica en Perú."""),
    nbf.v4.new_code_cell("""imp = pd.read_csv(OUT / "importance.csv")
piv = imp.pivot(index="feature", columns="train", values="importance").reindex(fe.ALL)
display(piv.sort_values("peru-train", ascending=False).round(4))
ax = piv.sort_values("peru-train").plot.barh(figsize=(10, 8)); ax.set_xlabel("caída de Jaccard al permutar")
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell(CONCLUSIONS),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                "language_info": {"name": "python"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
nbf.write(nb, HERE / "feature_engineering.ipynb")
print("wrote", HERE / "feature_engineering.ipynb")
