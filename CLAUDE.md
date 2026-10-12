# CLAUDE.md

Master's thesis (UNI, Maestría en Ciencias con mención en IA; authors Gerson Andre Garrido Mansilla and Renzo Elvis Cerron Tome). The thesis also targets a paper.

**Research:** flood extent segmentation in Sentinel-2 imagery with Transformer geospatial foundation models (TerraMind v1-B, Prithvi-EO-2.0 300M) against a U-Net baseline. Focus areas: generalization to Peru (a new Peru dataset used both to fine-tune and to test; Bolivia dropped per the advisor) and calibrated uncertainty.

## Read first

- `docs/PLAN.md`: research questions, data, models, protocol, phases, open decisions.
- `docs/REFERENCES.md`: what to cite and where the literature notes live.
- `docs/feasibility.md`: measured MPS training speed and memory; known MPS pitfalls.

## Environment

- Python 3.12, **uv** (`uv sync`, `uv run ...`). Core library: **TerraTorch** (with Lightning and TorchGeo).
- Hardware: Apple M4 Max, 36 GB, **MPS** (no CUDA). TerraMind v1-B fits batch 8; Prithvi-300M fits batch 4. Swapping happens silently instead of an OOM error, so watch memory.
- Prithvi + UperNet crashes on MPS (adaptive pool); use pool scales (1,2,4,8) or the UNet decoder.
- `data/`, `outputs/`, checkpoints are git-ignored.

## Experiment rules

- Use the **official Sen1Floods11 splits** (train 252 / valid 89 / test 90 / bolivia 15). Never re-split.
- Select models on `valid` / `peru-dev` only. Train on Peru only with `peru-train`. `test` and `peru-test` are evaluated once, at the end. `bolivia` is not used.
- Run ≥3 seeds per model and report mean ± std. Log config, seed and git commit for every run.
- Match each model's official input normalization.

## Thesis (Overleaf)

- `thesis/` is a **separate git repo** whose remote is the Overleaf project; it is git-ignored here. Never commit it into this repo.
- Always `cd thesis && git pull` before editing. Push to Overleaf only when the user approves.
- UNI FIIS template: `main.tex` + `TesisUNI.cls`, chapters in `2_CAPITULO1..6/`. Must compile with **XeLaTeX** (`latexmk -xelatex main.tex`). Thesis text is in **Spanish**; talk to the user in English.
- The bibliography include is commented out in `main.tex` until the first `\cite`. Add BibTeX entries to `thesis/3_3_BIBLIOGRAFIA/library.bib` one at a time.
- Never hardcode the Overleaf token.

## Literature (old project, reference only)

`~/Projects/masters-degree-thesis` holds the previous attempt. **Don't reuse its code.** Do reuse its literature base:
- `docs/papers/INDEX.md` → `docs/papers/<key>.md` → PDF via `scripts/zotero_index.sh tesis` (Zotero at `~/Zotero/`, **read-only**, `immutable=1`).
- **Never invent a number.** Every metric in the thesis or paper needs a page, table or URL source. Cite tables, not abstracts.

## Shell

zsh aliases replace classic tools (`grep`→`rg`, `find`→`fd`, `cat`→`bat`). Use `command grep` etc. when you need GNU flags.
