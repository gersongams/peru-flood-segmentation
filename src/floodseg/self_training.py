"""Teacher–student self-training for phase D (spec `domain-adaptation`, design decisions 3–6,
details fixed in docs/METHODS_NOTES.md 2026-10-15 before implementation).

Each step: a labelled Sen1Floods11 batch (source Dice loss) and an unlabelled Peru pool batch.
The EMA teacher pseudo-labels the pool batch with per-class top-q % confidence; source water
is pasted onto the target images (binary DACS); the student, with brightness/contrast/blur
applied only to its input, is trained with pixel cross-entropy on the mixed image. Optional
masked-consistency term (MIC): student on the target image with 64 px patches masked, against
the same pseudo-labels.
"""

import copy
import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

IGNORE = -1


def pseudo_labels(logits: torch.Tensor, q: float) -> tuple[torch.Tensor, list[float]]:
    """Per predicted class, keep exactly the top-q fraction most confident pixels of the batch.

    logits: (B, 2, H, W). Confidence = logit margin |z_water − z_dry|, which ranks pixels as the
    softmax probability does but does not saturate (float32 softmax ties at 1.0 for margins
    above ~17, which would let a threshold keep more than q). Ties are broken by top-k.
    Returns labels (B, H, W) with IGNORE elsewhere, and the cut-off per class as the smallest
    kept logit margin (nan if the class is not predicted; the probability is its sigmoid, which
    rounds to 1.0 above ~17)."""
    z = logits.float()
    margin = (z[:, 1] - z[:, 0]).abs()
    pred = z.argmax(1)
    out = torch.full_like(pred, IGNORE).flatten()
    m, pr = margin.flatten(), pred.flatten()
    cuts = []
    for c in range(z.shape[1]):
        idx = (pr == c).nonzero().squeeze(1)
        if idx.numel() == 0:
            cuts.append(float("nan"))
            continue
        k = max(1, int(math.ceil(q * idx.numel())))
        top = torch.topk(m[idx], k)
        out[idx[top.indices]] = c
        cuts.append(float(top.values.min()))
    return out.view_as(pred), cuts


def dacs_mix(src_x, src_y, tgt_x, tgt_pl, perm=None):
    """Paste the labelled water pixels (label 1) of a source image onto each target image."""
    if perm is None:
        perm = torch.randperm(src_x.shape[0], device=src_x.device)[: tgt_x.shape[0]] if src_x.shape[0] >= tgt_x.shape[0] \
            else torch.randint(0, src_x.shape[0], (tgt_x.shape[0],), device=src_x.device)
    m = (src_y[perm] == 1)
    x = torch.where(m[:, None], src_x[perm], tgt_x)
    y = torch.where(m, torch.ones_like(tgt_pl), tgt_pl)
    return x, y


def student_aug(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Brightness/contrast (p = 0.8) and Gaussian blur (p = 0.5), per image, all bands alike."""
    b = x.shape[0]
    dev = x.device

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(b, 1, 1, 1, generator=gen).to(dev)

    on = (torch.rand(b, 1, 1, 1, generator=gen) < 0.8).to(dev)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    jit = (x - mean) * u(0.8, 1.2) + mean + u(-0.2, 0.2)
    x = torch.where(on, jit, x)
    out = []
    for i in range(b):
        if torch.rand(1, generator=gen).item() < 0.5:
            s = 0.1 + 0.9 * torch.rand(1, generator=gen).item()
            r = torch.arange(5, device=dev, dtype=x.dtype) - 2
            k = torch.exp(-r**2 / (2 * s * s))
            k = k / k.sum()
            c = x.shape[1]
            xi = F.conv2d(F.pad(x[i:i + 1], (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
            xi = F.conv2d(F.pad(xi, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)
            out.append(xi)
        else:
            out.append(x[i:i + 1])
    return torch.cat(out)


def patch_mask(x: torch.Tensor, patch: int = 64, ratio: float = 0.7, gen: torch.Generator | None = None) -> torch.Tensor:
    """Set random patch × patch blocks (fraction `ratio`) to 0, the normalized mean."""
    b, _, h, w = x.shape
    keep = (torch.rand(b, 1, math.ceil(h / patch), math.ceil(w / patch), generator=gen) >= ratio).to(x.device, x.dtype)
    keep = F.interpolate(keep, scale_factor=patch, mode="nearest")[..., :h, :w]
    return x * keep


class SelfTraining(pl.LightningModule):
    """Wraps a TerraTorch SemanticSegmentationTask loaded from the source checkpoint."""

    def __init__(self, task, normalize_source, lr: float, iters: int = 4000, alpha: float = 0.99,
                 q_start: float = 0.2, q_end: float = 0.5, warmup_frac: float = 0.1, weight_decay: float = 0.05,
                 masking: dict | None = None, collapse_ratio: float = 0.5, collapse_init_iters: int = 50):
        super().__init__()
        self.student = task.model
        self.criterion = task.criterion
        self.teacher = copy.deepcopy(self.student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.normalize_source = normalize_source  # the source datamodule's Normalize (batch dict → batch dict)
        self.cfg = dict(lr=lr, iters=iters, alpha=alpha, q_start=q_start, q_end=q_end, warmup_frac=warmup_frac,
                        weight_decay=weight_decay, masking=masking, collapse_ratio=collapse_ratio,
                        collapse_init_iters=collapse_init_iters)
        self.water_frac_ema = None
        self.water_frac_init = []
        self.collapse_events = []
        self.below_floor = False

    def train(self, mode: bool = True):
        """Lightning calls train() on the whole module; the teacher always stays in eval mode
        (no dropout, BN running statistics)."""
        super().train(mode)
        self.teacher.eval()
        return self

    def q(self) -> float:
        c = self.cfg
        return c["q_start"] + (c["q_end"] - c["q_start"]) * min(1.0, self.global_step / max(1, c["iters"] - 1))

    def logits(self, net, x):
        return net(x).output

    @torch.no_grad()
    def update_teacher(self):
        a = min(1 - 1 / (self.global_step + 1), self.cfg["alpha"])
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(a).add_(ps.detach(), alpha=1 - a)
        for bt, bs in zip(self.teacher.buffers(), self.student.buffers()):
            bt.copy_(bs)

    def training_step(self, batch, _):
        src = self.normalize_source(batch["src"])
        sx, sy = src["image"], src["mask"].long()
        tx = batch["tgt"]["image"]
        loss_src = self.criterion(self.logits(self.student, sx), sy)

        with torch.no_grad():
            t_logits = self.logits(self.teacher, tx).float()
        pl_, cuts = pseudo_labels(t_logits, self.q())
        mx, my = dacs_mix(sx, sy, tx, pl_)
        loss_mix = F.cross_entropy(self.logits(self.student, student_aug(mx)).float(), my, ignore_index=IGNORE)
        loss = loss_src + loss_mix
        logs = {"loss/src": loss_src, "loss/mix": loss_mix, "pl/margin_cut_dry": cuts[0], "pl/margin_cut_water": cuts[1],
                "pl/q": self.q(), "pl/used_frac": (pl_ != IGNORE).float().mean()}
        if self.cfg["masking"]:
            loss_mask = F.cross_entropy(self.logits(self.student, patch_mask(tx, **self.cfg["masking"])).float(), pl_,
                                        ignore_index=IGNORE)
            loss = loss + loss_mask
            logs["loss/mask"] = loss_mask
        self.watch_collapse(float((t_logits.argmax(1) == 1).float().mean()))
        logs["target/water_frac_ema"] = self.water_frac_ema
        self.log_dict({k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in logs.items()}, on_step=True, prog_bar=False)
        return loss

    def watch_collapse(self, frac: float):
        self.water_frac_ema = frac if self.water_frac_ema is None else 0.98 * self.water_frac_ema + 0.02 * frac
        n = self.cfg["collapse_init_iters"]
        if len(self.water_frac_init) < n:
            self.water_frac_init.append(frac)
            return
        floor = self.cfg["collapse_ratio"] * sum(self.water_frac_init) / n
        below = self.water_frac_ema < floor
        if below and not self.below_floor:  # record each episode once, at its start
            print(f"COLLAPSE WARNING iter {self.global_step}: target water fraction {self.water_frac_ema:.4f} < floor {floor:.4f}",
                  flush=True)
            self.collapse_events.append(dict(iter=int(self.global_step), water_frac=self.water_frac_ema, floor=floor))
        self.below_floor = below

    def on_train_batch_end(self, *_):
        self.update_teacher()

    def configure_optimizers(self):
        c = self.cfg
        opt = torch.optim.AdamW(self.student.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        warm = max(1, int(c["warmup_frac"] * c["iters"]))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
