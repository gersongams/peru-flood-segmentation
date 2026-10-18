"""Phase-D result rows: Spanish names (as in thesis Chapter III, tables 3.6-3.8) and CSV loaders.

Inputs: outputs/phaseD/test/*.csv (written by scripts/score_frozen.py and the bootstrap script).
No numbers live here; every value is read from the CSVs at plotting time.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "outputs" / "phaseD" / "test"

MODELS = ["unet", "terramind_v1_base", "prithvi_eo_v2_300"]
SHORT = {"unet": "U-Net", "terramind_v1_base": "TerraMind", "prithvi_eo_v2_300": "Prithvi"}

# Regime -> method name in the thesis tables.
METHOD = {
    "s1f11_src": "origen",
    "trhm": "igualación de histogramas (entren.)",
    "trfda": "FDA (entren.)",
    "adabn": "AdaBN",
    "hm": "igualación de histogramas (infer.)",
    "peru": "entrenado en el Perú",
}
# The 11 rows of tables 3.6/3.7 (no s1f11_ft rows), in table order.
TABLE_ROWS = [
    "trhm/unet", "trfda/terramind_v1_base", "s1f11_src/prithvi_eo_v2_300",
    "trhm/terramind_v1_base", "adabn/unet", "hm/terramind_v1_base",
    "s1f11_src/unet", "s1f11_src/terramind_v1_base",
    "peru/unet", "peru/terramind_v1_base", "peru/prithvi_eo_v2_300",
]


def split_group(group: str) -> tuple[str, str]:
    regime, model = group.split("/")
    return regime, model


def method_name(group: str) -> str:
    regime, model = split_group(group)
    name = METHOD[regime]
    if regime == "peru" and model == "unet":
        name = "entrenada en el Perú"  # la U-Net (femenino), as in the thesis tables
    return name


def row_name(group: str) -> str:
    """'U-Net, igualación de histogramas (entren.)' etc., matching the thesis tables."""
    _, model = split_group(group)
    return f"{SHORT[model]}, {method_name(group)}"


def load(name: str, **filters) -> pd.DataFrame:
    df = pd.read_csv(TEST / name)
    for col, val in filters.items():
        df = df[df[col] == val]
    return df.reset_index(drop=True)


def pick(df: pd.DataFrame, groups, key: str = "group") -> pd.DataFrame:
    """Rows of df for the given groups, in that order; fails if any is missing or duplicated."""
    out = df.set_index(key)
    missing = [g for g in groups if g not in out.index]
    assert not missing, f"missing rows: {missing}"
    assert not out.index.duplicated().any(), "duplicated groups after filtering"
    return out.loc[list(groups)]


def diff_row(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """One row of bootstrap_diffs.csv (already filtered by split/subset) for a - b."""
    m = df[(df["a"] == a) & (df["b"] == b)]
    assert len(m) == 1, f"expected one diff row for {a} - {b}, got {len(m)}"
    return m.iloc[0]
