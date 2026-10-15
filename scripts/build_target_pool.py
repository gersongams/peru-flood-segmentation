"""Write the default phase-D target-pool manifest (task 2.3): `configs/pools/peru-train.csv`.

One row per `peru-train` image (stem, sha256). Checks before writing: the pool guard
(no held-out chip, no chip outside Peru v1) and that each sha256 equals the one in the Peru v1
data manifest (`data/peru/unosat2023/manifest.csv`, column `s2_sha256`). Labels are not read.

    uv run python scripts/build_target_pool.py
"""

import pandas as pd

from floodseg.target_pool import PERU_IMAGES, POOL_DIR, REPO, check_pool, default_pool_stems, sha256


def main() -> None:
    stems = default_pool_stems()
    check_pool(stems)
    v1 = pd.read_csv(REPO / "data" / "peru" / "unosat2023" / "manifest.csv")
    v1 = dict(zip(v1.date + "_" + v1.chip, v1.s2_sha256))
    rows = [(s, sha256(PERU_IMAGES / f"{s}.tif")) for s in stems]
    bad = [s for s, h in rows if v1.get(s) != h]
    if bad:
        raise SystemExit(f"{len(bad)} images differ from the Peru v1 manifest: {bad[:5]}")
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    out = POOL_DIR / "peru-train.csv"
    pd.DataFrame(rows, columns=["stem", "sha256"]).to_csv(out, index=False)
    dates = pd.Series([s[:10] for s in stems]).value_counts().sort_index().to_dict()
    print(f"{out.relative_to(REPO)}: {len(rows)} images, per date {dates}; sha256 match Peru v1 manifest")


if __name__ == "__main__":
    main()
