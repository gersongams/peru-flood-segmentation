# Inundaciones Perú — web app

Frontend of the flood-analysis system (objective 3). React + TypeScript + Vite, MapLibre.
The backend is `floodseg.system.server` (FastAPI), proxied at `/api`.

```sh
uv run uvicorn floodseg.system.server:app --port 8000   # from the repo root
cd app && pnpm install && pnpm dev                       # http://localhost:5173
```

Screens: Visor, Nuevo análisis, Referencia, Línea de tiempo, Evidencia (and Exportar).
Requirements and timing protocol: `docs/system_evaluation_protocol.md`.

3D incident view (`terrain.html`, Three.js): http://localhost:5173/terrain.html. Its data is
derived and git-ignored (`app/public/terrain/`); build it from the processed 17MNP scenes with

```sh
uv run python scripts/system/terrain_export.py   # relief, AI water per date, zone statistics, weather
uv run python scripts/system/terrain_frames.py   # Sentinel-2 photo per acquisition (needs CDSE credentials)
```
