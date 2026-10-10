"""Download the Sen1Floods11 hand-labeled chips listed in the official split CSVs.

Uses the public HTTPS endpoint of the `sen1floods11` GCS bucket (no gsutil needed).
Existing files with a non-zero size are skipped, so the script is safe to re-run.

    uv run python scripts/download_sen1floods11.py                  # S2Hand, LabelHand, JRCWaterHand
    uv run python scripts/download_sen1floods11.py --layers S1Hand  # add Sentinel-1
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS = ["train", "valid", "test", "bolivia"]
ROOT = Path(__file__).resolve().parents[1] / "data" / "sen1floods11" / "v1.1"
SPLIT_DIR = ROOT / "splits" / "flood_handlabeled"
CHIP_DIR = ROOT / "data" / "flood_events" / "HandLabeled"


def chip_ids() -> list[str]:
    """Chip ids (e.g. `Ghana_103272`) from the official split CSVs, in split order."""
    ids = []
    for split in SPLITS:
        path = SPLIT_DIR / f"flood_{split}_data.csv"
        if not path.exists():
            url = f"{BASE_URL}/splits/flood_handlabeled/{path.name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_get(url))
        with path.open() as f:
            ids += [row[0].removesuffix("_S1Hand.tif") for row in csv.reader(f) if row]
    return ids


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def download(layer: str, chip: str) -> int:
    dest = CHIP_DIR / layer / f"{chip}_{layer}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    data = _get(f"{BASE_URL}/data/flood_events/HandLabeled/{layer}/{dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", nargs="+", default=["S2Hand", "LabelHand", "JRCWaterHand"]
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    ids = chip_ids()
    jobs = [(layer, chip) for layer in args.layers for chip in ids]
    print(f"{len(ids)} chips x {len(args.layers)} layers = {len(jobs)} files")

    total, failed = 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                total += fut.result()
            except Exception as e:  # report and continue; re-run to retry
                failed.append((futures[fut], e))
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {total / 1e6:.0f} MB new")

    for job, e in failed:
        print(f"FAILED {job}: {e}")
    if failed:
        raise SystemExit(f"{len(failed)} downloads failed; re-run to retry")


if __name__ == "__main__":
    main()
