"""Verify the downloaded Sen1Floods11 official splits and write TerraTorch's split lists.

    uv run python scripts/verify_sen1floods11.py

Exits non-zero, naming the chips, if any file is missing or invalid.
"""

from floodseg.sen1floods11 import OFFICIAL_SIZES, check_chip, official_ids, write_split_lists


def main() -> None:
    failed = {}
    for split in OFFICIAL_SIZES:
        ids = official_ids(split)
        bad = {chip: p for chip in ids if (p := check_chip(chip))}
        failed.update(bad)
        print(f"{split}: {len(ids)} chips, {len(ids) - len(bad)} ok")
    for chip, problems in failed.items():
        print(f"BAD {chip}: {'; '.join(problems)}")
    if failed:
        raise SystemExit(f"{len(failed)} chips failed; re-run scripts/download_sen1floods11.py")
    write_split_lists()
    print("Wrote flood_{train,valid,test}_data.txt")


if __name__ == "__main__":
    main()
