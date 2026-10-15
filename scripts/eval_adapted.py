"""Offline `peru-dev` evaluation of self-training checkpoints (phase D; spec: dev is scored
outside the adaptation run).

For a run folder from scripts/self_train.py: load the source architecture, load the teacher
(default; the pre-registered evaluated model) or student weights of checkpoint iter<N>
(default: the final one), run one AdaBN pass on the pool (pre-registered rule), and save dev
logits to outputs/logits_dev/<regime>/<model>/seed<s>/ with run.json. Score with
scripts/phaseD_dev_table.py.

    uv run python scripts/eval_adapted.py outputs/phaseD/self_train/st/unet/seed0 [--iter N] [--weights student] [--regime NAME]
"""

import argparse
import json
import re
from pathlib import Path

import torch
from terratorch.tasks import SemanticSegmentationTask

from floodseg.adapt_eval import adapt_bn, predict_dev
from floodseg.target_pool import REPO, model_input_spec

BATCHES = {"unet": 16, "terramind_v1_base": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--weights", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--regime", default=None)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    rec = json.loads((run / "run.json").read_text())
    model, seed = rec["model"], rec["seed"]
    ckpts = {int(re.search(r"iter(\d+)", p.name).group(1)): p for p in (run / "checkpoints").glob("iter*.pt")}
    it = a.iter if a.iter is not None else max(ckpts)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    task = SemanticSegmentationTask.load_from_checkpoint(REPO / rec["source_ckpt"], map_location="cpu")
    task.model.load_state_dict(torch.load(ckpts[it], map_location="cpu")[a.weights])
    net = task.model.to(device)
    spec = model_input_spec(REPO / "configs" / f"{model}.yaml")
    n_bn, n_pool = adapt_bn(net, spec, device, BATCHES[model])
    regime = a.regime or f"{run.parts[-3]}" + ("" if a.weights == "teacher" else "_student") + ("" if it == max(ckpts) else f"_iter{it}")
    out = REPO / "outputs" / "logits_dev" / regime / model / f"seed{seed}"
    n = predict_dev(net, spec, device, BATCHES[model], out)
    info = dict(run=str(run.relative_to(REPO)), iter=it, weights=a.weights, adabn_bn_layers=n_bn, adabn_pool_images=n_pool,
                dev_files=n, device=device)
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
