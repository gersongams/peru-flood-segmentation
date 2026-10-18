"""Inference of a frozen 3-seed ensemble over a whole tile, on the evaluation's chip grid.

The tile is cut into non-overlapping 512 px windows aligned to the same 5 120 m UTM grid as the
Peru chips (lower-left corners at multiples of 5 120 m, scripts/build_peru_unosat.py). Each window
is predicted alone, as in the frozen evaluation, so on every chip the map equals the evaluated
configuration. Windows that cross the tile edge are padded with 0 (no data). Per pixel: seed mean
of sigmoid(z1 − z0) (probability) and seed std (uncertainty), as scripts/viewer/infer_all.py.

Decision (authors, 2026-10-18, METHODS_NOTES): overlapping windows with blending were tried
first; they agree with the frozen pass less at chip borders (IoU 0.934 on 17MNP) while scoring
the same on peru-dev labels, so the evaluated window layout was kept.
"""

from collections.abc import Callable

import numpy as np
import torch
import yaml
from rasterio.transform import Affine
from terratorch.tasks import SemanticSegmentationTask

from floodseg.target_pool import REPO, model_input_spec, sha256

WIN = 512
CHIP_M = 5120  # chip size in metres (512 px × 10 m)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def chip_grid(transform: Affine, h: int, w: int) -> tuple[list[int], list[int]]:
    """Row and column offsets (px, may be negative) of the chip-grid windows covering the tile."""
    res = transform.a
    left, top = transform.c, transform.f
    col0 = int(round((np.floor(left / CHIP_M) * CHIP_M - left) / res))
    row0 = int(round((top - np.ceil(top / CHIP_M) * CHIP_M) / res))
    return list(range(row0, h, WIN)), list(range(col0, w, WIN))


def window(stack: np.ndarray, r: int, c: int) -> np.ndarray:
    """(13, 512, 512) window at row r, col c, zero-padded where it leaves the tile."""
    _, h, w = stack.shape
    out = np.zeros((stack.shape[0], WIN, WIN), stack.dtype)
    r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + WIN, h), min(c + WIN, w)
    out[:, r0 - r:r1 - r, c0 - c:c1 - c] = stack[:, r0:r1, c0:c1]
    return out


class Ensemble:
    """The frozen checkpoints of one `configs/frozen.yaml` group, loaded once (warm server)."""

    def __init__(self, group_id: str = "peru/unet", device: str = DEVICE):
        frozen = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())
        g = next(x for x in frozen["groups"] if x["id"] == group_id)
        spec = model_input_spec(REPO / "configs" / f"{g['model']}.yaml")
        self.group_id, self.device = group_id, device
        self.idx = [spec["dataset_bands"].index(b) for b in spec["output_bands"]]
        self.mean = torch.tensor(spec["means"], dtype=torch.float32, device=device)[:, None, None]
        self.std = torch.tensor(spec["stds"], dtype=torch.float32, device=device)[:, None, None]
        self.checkpoints, self.nets = {}, []
        for seed, r in g["runs"].items():
            path = REPO / r["checkpoint"]
            assert sha256(path) == r["sha256"], f"{group_id} {seed}: checkpoint changed since the freeze"
            self.checkpoints[seed] = {"path": r["checkpoint"], "sha256": r["sha256"]}
            net = SemanticSegmentationTask.load_from_checkpoint(path, map_location="cpu").model
            self.nets.append(net.to(device).eval())

    @torch.no_grad()
    def predict_windows(self, raw: np.ndarray) -> np.ndarray:
        """(B, 13, 512, 512) uint16 DN → (seeds, B, 512, 512) float32 water probability."""
        x = torch.from_numpy(raw[:, self.idx].astype(np.float32)).to(self.device)
        x = (x - self.mean) / self.std
        out = []
        for n in self.nets:
            o = n(x).output.float()
            out.append(torch.sigmoid(o[:, 1] - o[:, 0]))
        return torch.stack(out).cpu().numpy()

    def predict_tile(self, stack: np.ndarray, nodata: np.ndarray, transform: Affine, batch: int = 16,
                     progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Whole-tile (13, H, W) uint16 → probability (seed mean) and uncertainty (seed std),
        float32 (H, W). Windows entirely in no-data are skipped (left at 0)."""
        _, h, w = stack.shape
        rows, cols = chip_grid(transform, h, w)
        boxes = [(r, c) for r in rows for c in cols
                 if not nodata[max(r, 0):min(r + WIN, h), max(c, 0):min(c + WIN, w)].all()]
        prob = np.zeros((h, w), np.float32)
        unc = np.zeros((h, w), np.float32)
        for i in range(0, len(boxes), batch):
            part = boxes[i:i + batch]
            p = self.predict_windows(np.stack([window(stack, r, c) for r, c in part]))
            mean, std = p.mean(0), p.std(0)
            for j, (r, c) in enumerate(part):
                r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + WIN, h), min(c + WIN, w)
                prob[r0:r1, c0:c1] = mean[j, r0 - r:r1 - r, c0 - c:c1 - c]
                unc[r0:r1, c0:c1] = std[j, r0 - r:r1 - r, c0 - c:c1 - c]
            if progress:
                progress(i + len(part), len(boxes))
        if self.device == "mps":
            torch.mps.empty_cache()
        return prob, unc
