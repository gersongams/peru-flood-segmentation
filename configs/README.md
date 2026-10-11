# Configs

TerraTorch (LightningCLI) YAML. A run combines a model config with a trainer config:

```bash
uv run terratorch fit -c configs/<model>.yaml -c configs/trainer/mps.yaml -c configs/trainer/demo.yaml  # local demo
uv run terratorch fit -c configs/<model>.yaml -c configs/trainer/cuda.yaml --seed_everything 1          # full run on a rented GPU
```

- `trainer/mps.yaml`, `trainer/cuda.yaml`: device and precision only. Model configs never set them.
- `trainer/demo.yaml`: few epochs on a fraction of the data. Demo runs may start from a dirty tree; full runs may not.
- Every model config sets `trainer.default_root_dir: outputs/runs/<model>` and includes the
  `floodseg.provenance.RunProvenance` and `floodseg.callbacks.MPSMemoryGuard` callbacks.
  Each run writes `outputs/runs/<model>/lightning_logs/version_<n>/run.json`.
