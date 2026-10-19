"""HE3 timing runs (docs/system_evaluation_protocol.md §2): 6 tiles × 3 repetitions.

Requires the server running warm (`uv run uvicorn floodseg.system.server:app --port 8000`) on a
clean working tree at a tagged commit. For each run: clear the tile/date cache, record downlink
capacity with `networkQuality -s -c`, send the job request as the app does, wait for it, and read
the server's timestamps (timer start = request received, stop = GeoTIFFs written). Runs are
interleaved across tiles and at least 10 min apart for the same tile. A run that fails because of
an external service (HTTP 5xx, timeout, CDSE outage) is logged and repeated, at most 3 times; any
other failure is a system failure and is recorded as such.
→ outputs/system_eval/timing_runs.csv (one row per run) and timing_failures.csv

    uv run python scripts/system_timing.py            # the protocol measurement
    uv run python scripts/system_timing.py --dry-run  # one run of one tile, no tag/clean check
"""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone

import requests

from floodseg.system import cdse, pipeline
from floodseg.target_pool import REPO

API = "http://localhost:8000"
TILES = ["17MNP", "17MNQ", "17MPN", "17MNN", "17MPP", "17MPQ"]
DATE = "2023-03-22"
REPS = 3
MIN_GAP_S = 10 * 60
MAX_RETRIES = 3
OUT = REPO / "outputs" / "system_eval"
COLUMNS = ["tile", "date", "rep", "start", "stop", "minutes", "network_mbps", "commit", "failures",
           "product", "download_s", "read_s", "inference_s", "write_s", "macos", "archive_reused"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO).stdout.strip()


def network_mbps() -> float | None:
    r = subprocess.run(["networkQuality", "-s", "-c"], capture_output=True, text=True, timeout=180)
    try:
        return round(json.loads(r.stdout)["dl_throughput"] / 1e6, 1)
    except (ValueError, KeyError):
        return None


def tile_date(tile: str) -> str:
    """Protocol: 2023-03-22, or the nearest available date (logged) if the tile has none."""
    if cdse.find_l1c(tile, DATE):
        return DATE
    near = cdse.acquisitions(tile, "2023-03-12", "2023-04-01")
    best = min(near, key=lambda p: abs((datetime.fromisoformat(p["date"]) - datetime.fromisoformat(DATE)).days))
    print(f"{tile}: no product on {DATE}, using {best['date']}", flush=True)
    return best["date"]


def one_run(tile: str, date: str) -> dict:
    pipeline.clear(tile, date)
    job = requests.post(f"{API}/api/jobs", json={"event": "yaku2023", "tile": tile, "date": date}, timeout=30).json()
    if job.get("cached"):
        raise RuntimeError("cache not empty at start")
    while job["status"] in ("queued", "running"):
        time.sleep(2)
        job = requests.get(f"{API}/api/jobs/{job['id']}", timeout=30).json()
    return job


def is_external(err: str) -> bool:
    e = err.lower()
    return any(k in e for k in ("httperror: 5", "timeout", "timed out", "connectionerror", "remotedisconnected", "incomplete download"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    commit = git("rev-parse", "HEAD")
    if not a.dry_run:
        if git("status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("working tree not clean: commit first (protocol: the system at a tagged commit)")
        if not git("tag", "--points-at", "HEAD"):
            raise SystemExit("HEAD has no tag: tag the system version first")
    status = requests.get(f"{API}/api/status", timeout=10).json()
    if not status["models_loaded"]:
        raise SystemExit("server not warm")
    macos = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()
    tiles = TILES[:1] if a.dry_run else TILES
    reps = 1 if a.dry_run else REPS
    dates = {t: tile_date(t) for t in tiles}

    OUT.mkdir(parents=True, exist_ok=True)
    runs_path = OUT / ("timing_dry_run.csv" if a.dry_run else "timing_runs.csv")
    fail_path = OUT / "timing_failures.csv"
    last_end: dict[str, float] = {}
    with open(runs_path, "a", newline="") as fr, open(fail_path, "a", newline="") as ff:
        w = csv.DictWriter(fr, COLUMNS)
        wf = csv.writer(ff)
        if fr.tell() == 0:
            w.writeheader()
        if ff.tell() == 0:
            wf.writerow(["when", "tile", "date", "rep", "attempt", "error", "external"])
        for rep in range(1, reps + 1):
            for tile in tiles:
                wait = MIN_GAP_S - (time.time() - last_end.get(tile, 0))
                if wait > 0 and tile in last_end:
                    print(f"waiting {wait / 60:.1f} min before {tile} rep {rep}", flush=True)
                    time.sleep(wait)
                failures = 0
                while True:
                    mbps = network_mbps()
                    job = one_run(tile, dates[tile])
                    if job["status"] == "done":
                        break
                    ext = is_external(job["error"] or "")
                    failures += 1
                    wf.writerow([datetime.now(timezone.utc).isoformat(), tile, dates[tile], rep, failures, job["error"], ext])
                    ff.flush()
                    if not ext or failures >= MAX_RETRIES:
                        raise SystemExit(f"{tile} rep {rep}: {'system' if not ext else 'repeated external'} failure: {job['error']}")
                meta = json.loads((pipeline.scene_dir(tile, dates[tile]) / "meta.json").read_text())
                assert not meta["archive_reused"], "archive reused: the run is not valid for HE3"
                st = {k: v.get("seconds") for k, v in job["stages"].items()}
                w.writerow({"tile": tile, "date": dates[tile], "rep": rep, "start": job["received"], "stop": job["stop"],
                            "minutes": round(job["seconds"] / 60, 3), "network_mbps": mbps, "commit": commit,
                            "failures": failures, "product": meta["product"], "download_s": st.get("download"),
                            "read_s": st.get("read"), "inference_s": st.get("inference"), "write_s": st.get("write"),
                            "macos": macos, "archive_reused": meta["archive_reused"]})
                fr.flush()
                last_end[tile] = time.time()
                print(f"{tile} rep {rep}: {job['seconds'] / 60:.2f} min ({mbps} Mbps)", flush=True)


if __name__ == "__main__":
    main()
