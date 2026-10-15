"""Offline `peru-dev` evaluation of self-training checkpoints (phase D; spec: dev is scored
outside the adaptation run).

For a run folder from scripts/self_train.py: load the source architecture, load the teacher
(default; the pre-registered evaluated model) or student weights of checkpoint iter<N>
(default: the final one), run one AdaBN pass on the pool (pre-registered rule), and save dev
logits to outputs/logits_dev/<regime>/<model>/seed<s>/ with run.json. Score with
scripts/phaseD_dev_table.py.

    uv run python scripts/eval_adapted.py outputs/phaseD/self_train/st/unet/seed0 [--iter N] [--weights student] [--regime NAME]

`--no_adabn` (post-hoc diagnostic, not pre-registered): skip the AdaBN pass and keep the BN
buffers the teacher had at the end of training; regime gets the suffix `_noadabn`.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from terratorch.tasks import SemanticSegmentationTask

from floodseg.adapt_eval import adapt_bn, predict_dev, predict_dev_arrays
from floodseg.target_pool import REPO, model_input_spec

BATCHES = {"unet": 16, "terramind_v1_base": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--weights", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--regime", default=None)
    ap.add_argument("--all", action="store_true",
                    help="evaluate every saved iteration: intermediate ones as packed binary predictions "
                         "(<run>/dev_pred/iter<N>.npz), the final one as float32 logits")
    ap.add_argument("--no_adabn", action="store_true", help="post-hoc diagnostic: no AdaBN pass before prediction")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    rec = json.loads((run / "run.json").read_text())
    model, seed = rec["model"], rec["seed"]
    ckpts = {int(re.search(r"iter(\d+)", p.name).group(1)): p for p in (run / "checkpoints").glob("iter*.pt")}
    final = max(ckpts)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    task = SemanticSegmentationTask.load_from_checkpoint(REPO / rec["source_ckpt"], map_location="cpu")
    spec = model_input_spec(REPO / "configs" / f"{model}.yaml")
    iters = sorted(ckpts) if a.all else [a.iter if a.iter is not None else final]
    for it in iters:
        task.model.load_state_dict(torch.load(ckpts[it], map_location="cpu")[a.weights])
        net = task.model.to(device)
        n_bn, n_pool = (0, 0) if a.no_adabn else adapt_bn(net, spec, device, BATCHES[model])
        if it != final and a.all:
            z = predict_dev_arrays(net, spec, device, BATCHES[model])
            (run / "dev_pred").mkdir(exist_ok=True)
            np.savez_compressed(run / "dev_pred" / f"iter{it}.npz", **{s: np.packbits(v > 0) for s, v in z.items()})
            print(json.dumps(dict(run=str(run.relative_to(REPO)), iter=it, packed=True)), flush=True)
            continue
        regime = a.regime or f"{run.parts[-3]}" + ("" if a.weights == "teacher" else "_student") + ("" if it == final else f"_iter{it}") \
            + ("_noadabn" if a.no_adabn else "")
        out = REPO / "outputs" / "logits_dev" / regime / model / f"seed{seed}"
        n = predict_dev(net, spec, device, BATCHES[model], out)
        info = dict(run=str(run.relative_to(REPO)), iter=it, weights=a.weights, adabn_bn_layers=n_bn, adabn_pool_images=n_pool,
                    dev_files=n, device=device)
        (out / "run.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
