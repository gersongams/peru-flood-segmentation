# Optimización de hiperparámetros: avance del sprint

Rama: `avance-sem8-hiperparametros`. Cuaderno:
[`notebooks/optimizacion_hiperparametros.ipynb`](../notebooks/optimizacion_hiperparametros.ipynb).
Se genera con `uv run python notebooks/build_hiperparametros_nb.py`. No hay corridas nuevas ni se lee
ninguna partición de prueba.

| Modelo | Tasa elegida (Sen1Floods11) | IoU `valid` (3 semillas) | IoU `peru-dev` | Tasa en el Perú (sin barrido) | IoU `peru-dev` |
|---|---|---|---|---|---|
| U-Net | 3e-4 | 0.8315 ± 0.0015 | 0.8364 ± 0.0224 | 1e-4 | 0.9415 ± 0.0019 |
| TerraMind v1-B | 5e-5 | 0.8349 ± 0.0014 | 0.7027 ± 0.0913 | 2e-5 | 0.9265 ± 0.0025 |
| Prithvi-EO-2.0 300M | 1e-4 | 0.8315 ± 0.0047 | 0.8701 ± 0.0011 | 1e-4 | 0.9218 ± 0.0007 |

Conclusiones y limitaciones: `notebooks/hiperparametros_conclusiones.md`. Método y correcciones:
`docs/METHODS_NOTES.md` (2026-11-03). Corridas: `docs/EXPERIMENT_LOG.md`.
