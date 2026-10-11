"""Official input normalization per model, on raw Sentinel-2 L1C DN (Sen1Floods11 S2Hand).

Values are copied from each model's official Sen1Floods11 config; the URL is the
source. Model configs in `configs/` must use exactly these values
(tests/test_sen1floods11.py checks it).
"""

import json
from pathlib import Path

S2_L1C_BANDS = [
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
    "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2",
]  # fmt: skip

_STATS = Path(__file__).resolve().parents[2] / "configs" / "stats" / "sen1floods11_train_s2.json"


def _unet() -> dict:
    stats = json.loads(_STATS.read_text())
    return {
        "bands": S2_L1C_BANDS,
        "means": stats["means"],
        "stds": stats["stds"],
        "source": "scripts/compute_train_stats.py (official train split only)",
    }


NORMALIZATION = {
    "terramind_v1_base": {
        "bands": S2_L1C_BANDS,
        "means": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849,
                  3040.560, 3306.481, 1473.847, 506.070, 2472.825, 1838.929],
        "stds": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131,
                 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419],
        "source": "https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_sen1floods11.yaml (S2L1C)",
    },
    "prithvi_eo_v2_300": {
        "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        "means": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
        "stds": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
        "source": "https://github.com/terrastackai/terratorch/blob/main/examples/segmentation/segmentation_sen1floods11.yaml",
    },
}  # fmt: skip


def get(model: str) -> dict:
    """Bands, means, stds and source URL for `model` (`unet`, `terramind_v1_base`, `prithvi_eo_v2_300`)."""
    if model == "unet":
        return _unet()
    return NORMALIZATION[model]
