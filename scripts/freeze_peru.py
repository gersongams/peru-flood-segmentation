"""Freeze the Peru dataset: check every chip against its manifest hash and write
docs/peru_v1_checksums.txt (tracked in git), which pins the manifests and split
lists. Re-run with --check to verify an existing copy (e.g. on a rented GPU).

    uv run python scripts/freeze_peru.py           # write the record
    uv run python scripts/freeze_peru.py --check   # verify against the record
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PERU = REPO / "data" / "peru"
RECORD = REPO / "docs" / "peru_v1_checksums.txt"
SETS = {"unosat2023": lambda r: f"{r.date}_{r.chip}", "emsr2017": lambda r: r.chip}
PINNED = ["unosat2023/manifest.csv", "unosat2023/splits/train.txt", "unosat2023/splits/dev.txt",
          "unosat2023/splits/test.txt", "unosat2023/splits/split.csv", "emsr2017/manifest.csv"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_chips() -> int:
    n, bad = 0, []
    for name, stem in SETS.items():
        m = pd.read_csv(PERU / name / "manifest.csv")
        for r in m.itertuples():
            for kind, col in (("s2", "s2_sha256"), ("label", "label_sha256")):
                f = PERU / name / kind / f"{stem(r)}.tif"
                if not f.exists() or sha(f) != getattr(r, col):
                    bad.append(str(f.relative_to(PERU)))
                n += 1
        extra = {p.stem for k in ("s2", "label") for p in (PERU / name / k).glob("*.tif")} - {stem(r) for r in m.itertuples()}
        bad += [f"{name}: not in manifest: {e}" for e in sorted(extra)]
    if bad:
        sys.exit("checksum mismatch:\n" + "\n".join(bad[:20]))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    n = check_chips()
    lines = [f"{sha(PERU / p)}  data/peru/{p}" for p in PINNED]
    if args.check:
        want = [l for l in RECORD.read_text().splitlines() if l and not l.startswith("#")]
        if want != lines:
            sys.exit("pinned files differ from docs/peru_v1_checksums.txt")
        print(f"ok: {n} chip files match their manifests; manifests and splits match the v1 record")
        return
    RECORD.write_text("# Peru dataset v1 (scripts/freeze_peru.py). Chip files are pinned by the\n"
                      f"# s2_sha256/label_sha256 columns of the manifests below; {n} files verified.\n"
                      + "\n".join(lines) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}: {n} chip files verified")


if __name__ == "__main__":
    main()
