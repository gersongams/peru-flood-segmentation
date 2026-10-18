# Inundaciones Perú — web app

Frontend of the flood-analysis system (objective 3). React + TypeScript + Vite, MapLibre.
The backend is `floodseg.system.server` (FastAPI), proxied at `/api`.

```sh
uv run uvicorn floodseg.system.server:app --port 8000   # from the repo root
cd app && pnpm install && pnpm dev                       # http://localhost:5173
```

Screens: Visor, Nuevo análisis, Referencia, Línea de tiempo, Evidencia (and Exportar).
Requirements and timing protocol: `docs/system_evaluation_protocol.md`.
