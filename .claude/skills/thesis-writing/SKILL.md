---
name: thesis-writing
description: Draft or revise a chapter or section of the UNI master's thesis (Spanish, LaTeX in thesis/). Use whenever writing thesis prose, objectives, hypotheses, results or discussion text for the Overleaf repo. Enforces docs/WRITING_RULES.md, including the claim-precision lessons of §11.
---

# Thesis writing (UNI, EPG Guía N°2)

Binding rules: `docs/WRITING_RULES.md`. Read it fully before drafting, especially §3 (content),
§4 (integrity), §6 (terminology), §7 (numbers), §8 (citations) and **§11 (claim precision)**.
The thesis is in Spanish; talk to the user in English.

## Before drafting

1. In `thesis/`, run `git pull`. On a credential error, stop and tell the user. Never hardcode
   a token.
2. Collect the sources for the section:
   - result files: `outputs/…`, `docs/phaseD_test.md`, `docs/phaseD_dev.md`,
     `docs/published_comparison.md`;
   - decisions: `docs/METHODS_NOTES.md`;
   - literature: `docs/literature_thesis.md` and `.bib`.
3. List the claims the section will make, and the source of each, **before** writing prose.
4. Content that changes the research (objectives, hypotheses, scope) goes to the authors first as
   a proposal in `docs/`. Write it into `thesis/` only after they approve.

## While drafting

- **Structure.** Keep the Guía N°2 headings exactly. Sub-levels may be added. Course-only
  sections (for example ethics) need the authors' consent; they removed §1.4.7.
- **Every number** gets a `% fuente: <file or citation with Tabla/página>` comment.
- **Wording** follows §11:
  - scope and cut-off date for statistics;
  - printed vs PDF pages;
  - no absolute negatives;
  - "CI includes 0" ≠ equivalence;
  - "without Peru labels" scoped to fitting, not selection;
  - requirements are not results;
  - "agua" vs "inundación".
- **Caveats** go once, in the methodology or the limitations. Do not scatter them through the
  argument (§11.13).
- **Voice.** Impersonal and academic. Do not imitate informality to look "human" (§11.14).
  Vary the paragraph shape (§11.15).
- **Terminology** follows the §6 table. Write foreign words with `\ingles{}`. Citations use
  `\citet` / `\citep[Tabla~x, p.~y]{key}`.
- **Bibliography.** Copy entries into `thesis/3_3_BIBLIOGRAFIA/library.bib` one at a time,
  only when cited, each with a `% verified:` line. Protect proper nouns with braces. Drop
  English working notes from `note` fields.

## After drafting

1. Compile with `latexmk -interaction=nonstopmode main.tex` (`latexmkrc` forces XeLaTeX). Check:
   - no `^!` errors;
   - no undefined citations or references;
   - `main.blg` has 0 warnings.
2. Render the changed pages to PNG and look at them: tables fit, landscape annexes fit, no
   heading numbers like "1.1.3.0.1".
3. Run the **thesis-claim-audit** skill on the changed section, or at least its checklist.
4. Commit in `thesis/` with a Spanish message. **Push to Overleaf only after the authors
   approve.**
5. Tell the authors which argumentative paragraphs they should rewrite in their own voice
   (WRITING_RULES §9).
