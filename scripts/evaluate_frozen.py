"""Gated one-time evaluation of the frozen configurations (phase-D task 6.2; plan-flood-research
7.1). Writes float32 water logits z = logit_water − logit_other per chip; scoring is separate
(scripts/score_frozen.py). This script reads images only, never labels (it runs under the Peru
label guard).

Gate for the held-out splits `peru-test` and `peru-2017-test` (spec `adaptation-evaluation`,
"Premature test evaluation"): the script refuses to run unless
- `configs/frozen.yaml` is tracked and unchanged against HEAD, and no tracked file is modified;
- the group is listed there for this pass (`--pass`, default phaseD);
- every checkpoint and fitted artifact still has its frozen sha256;
- the run's output folder does not exist yet (one look). `--correction "<reason>"` reruns a run
  after an evaluation-code fix; the reason is stored in run.json and must be logged.
`peru-dev` runs through the same code with none of these checks and writes to
outputs/frozen_check/ by default, to reproduce the dev table before the test pass.
Sen1Floods11 `test` (90 official chips; added 2026-10-16 for the section-7 pass, METHODS_NOTES
"Section-7 pass") is held out too; per the freeze it is evaluated for the Sen1Floods11-trained
source rows (`s1f11_src`, method none) only, so for this split the group filter is that regime
instead of `--pass`.

Per group `method`: `none` (checkpoint as is), `adabn` (BN statistics re-estimated on the
`peru-train` pool, as scripts/adabn.py), `hm_test` (pool → Sen1Floods11 histogram map of
outputs/phaseD/alignment/params.npz applied to each image, as scripts/input_alignment_dev.py).

    uv run python scripts/evaluate_frozen.py peru-dev  [--group trhm/unet ...]
    uv run python scripts/evaluate_frozen.py peru-test [--group ...]
    uv run python scripts/evaluate_frozen.py peru-2017-test [--group ...]
    uv run python scripts/evaluate_frozen.py peru-test --pass section7
    uv run python scripts/evaluate_frozen.py test
→ <out_root>/<regime>/<model>/seed<s>/<stem>.npy + run.json
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from terratorch.tasks import SemanticSegmentationTask

from floodseg.adapt_eval import adapt_bn
from floodseg.input_alignment import hist_match
from floodseg.sen1floods11 import CHIP_DIR, official_ids
from floodseg.target_pool import REPO, forbid_peru_labels, model_input_spec, normalize, sha256

FROZEN = REPO / "configs" / "frozen.yaml"
UNOSAT = REPO / "data" / "peru" / "unosat2023"
EMSR = REPO / "data" / "peru" / "emsr2017"
HELD_OUT = {"peru-test", "peru-2017-test", "test"}
TEST_REGIMES = {"s1f11_src"}  # rows frozen for Sen1Floods11 `test` (METHODS_NOTES "Phase-D freeze")
OUT_ROOT = {"peru-dev": REPO / "outputs" / "frozen_check" / "logits_dev",
            "peru-test": REPO / "outputs" / "logits_test",
            "peru-2017-test": REPO / "outputs" / "logits_2017",
            "test": REPO / "outputs" / "logits_s1f11_test"}
BATCHES = {"unet": 16, "terramind_v1_base": 8, "prithvi_eo_v2_300": 4}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()


def split_images(split: str) -> list[Path]:
    if split == "peru-dev":
        return [UNOSAT / "s2" / f"{s}.tif" for s in (UNOSAT / "splits" / "dev.txt").read_text().split()]
    if split == "peru-test":
        return [UNOSAT / "s2" / f"{s}.tif" for s in (UNOSAT / "splits" / "test.txt").read_text().split()]
    if split == "peru-2017-test":
        return [EMSR / "s2" / f"{c}.tif" for c in pd.read_csv(EMSR / "manifest.csv").chip]
    if split == "test":
        return [CHIP_DIR / "S2Hand" / f"{i}_S2Hand.tif" for i in official_ids("test")]
    raise ValueError(split)


def eligible(g: dict, split: str, pas: str) -> bool:
    return g["regime"] in TEST_REGIMES and g["method"] == "none" if split == "test" else g["pass"] == pas


def check_gate(frozen: dict, groups: list[dict], pas: str, split: str) -> None:
    if git("ls-files", "configs/frozen.yaml") != "configs/frozen.yaml":
        raise SystemExit("gate: configs/frozen.yaml is not tracked; commit the freeze first")
    modified = [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line]
    if modified:
        raise SystemExit(f"gate: tracked files modified, commit first: {modified[:5]}")
    for g in groups:
        if not eligible(g, split, pas):
            raise SystemExit(f"gate: {g['id']} is not frozen for {split} in pass {pas!r}")
        for s, r in g["runs"].items():
            if sha256(REPO / r["checkpoint"]) != r["sha256"]:
                raise SystemExit(f"gate: checkpoint of {g['id']} {s} changed since the freeze")
    for k, a in frozen["artifacts"].items():
        if sha256(REPO / a["path"]) != a["sha256"]:
            raise SystemExit(f"gate: artifact {k} ({a['path']}) changed since the freeze")


def predict(model: torch.nn.Module, spec: dict, images: list[Path], batch: int, tf, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch):
            part = images[i:i + batch]
            xs = []
            for p in part:
                with rasterio.open(p) as src:
                    raw = src.read()
                xs.append(normalize(tf(raw) if tf else raw, spec))
            o = model(torch.stack(xs).to(DEVICE)).output.float()
            for p, z in zip(part, (o[:, 1] - o[:, 0]).cpu().numpy().astype(np.float32)):
                np.save(out / f"{p.stem}.npy", z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=sorted(OUT_ROOT))
    ap.add_argument("--group", nargs="*", help="regime/model ids (default: every group of the pass)")
    ap.add_argument("--pass", dest="pas", default="phaseD", choices=["phaseD", "section7"])
    ap.add_argument("--out_root", type=Path)
    ap.add_argument("--correction", help="reason for rerunning an existing held-out run (log it)")
    a = ap.parse_args()

    frozen = yaml.safe_load(FROZEN.read_text())
    groups = [g for g in frozen["groups"] if (g["id"] in a.group if a.group else eligible(g, a.split, a.pas))]
    if a.group and len(groups) != len(a.group):
        raise SystemExit(f"not frozen: {sorted(set(a.group) - {g['id'] for g in groups})}")
    held = a.split in HELD_OUT
    if held:
        if a.out_root:
            raise SystemExit("gate: held-out splits write to their fixed output folder")
        check_gate(frozen, groups, a.pas, a.split)
    out_root = a.out_root or OUT_ROOT[a.split]
    images = split_images(a.split)
    hm = np.load(REPO / frozen["artifacts"]["hm_test_params"]["path"])
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))

    with forbid_peru_labels():
        for g in groups:
            spec = model_input_spec(REPO / "configs" / f"{g['model']}.yaml")
            batch = BATCHES[g["model"]]
            tf = (lambda x: hist_match(x, hm["q_pool"], hm["q_s1f"])) if g["method"] == "hm_test" else None
            for seed, r in g["runs"].items():
                out = out_root / g["regime"] / g["model"] / seed
                if held and out.exists() and not a.correction:
                    raise SystemExit(f"gate: {out.relative_to(REPO)} exists; held-out splits are evaluated once")
                t0 = time.time()
                task = SemanticSegmentationTask.load_from_checkpoint(REPO / r["checkpoint"], map_location="cpu").to(DEVICE)
                n_bn = n_pool = None
                if g["method"] == "adabn":
                    n_bn, n_pool = adapt_bn(task.model, spec, DEVICE, batch)
                predict(task.model, spec, images, batch, tf, out)
                rec = dict(split=a.split, group=g["id"], role=g["role"], method=g["method"], seed=seed,
                           checkpoint=r["checkpoint"], checkpoint_sha256=r["sha256"], frozen_sha256=sha256(FROZEN),
                           files=len(images), bn_layers=n_bn, pool_images=n_pool, device=DEVICE,
                           seconds=round(time.time() - t0, 1), git_commit=commit, git_dirty=dirty,
                           utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), correction=a.correction)
                (out / "run.json").write_text(json.dumps(rec, indent=2))
                print(json.dumps(rec), flush=True)
                del task
                if DEVICE == "mps":
                    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
