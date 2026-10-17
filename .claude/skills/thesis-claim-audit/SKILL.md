---
name: thesis-claim-audit
description: Audit a thesis chapter's citations, figures and claims against their sources (papers, official reports, project result files), the way the Chapter I audit was done (docs/chapter1_citation_audit.md). Use after drafting or revising any thesis section, or when the user asks to verify citations or claims.
---

# Thesis claim audit

Model: `docs/chapter1_citation_audit.md`. Rules checked: `docs/WRITING_RULES.md` §4 and §11.
Output: `docs/<chapter>_citation_audit.md`, in Spanish (the authors read it), plus a list of
corrections. Text changes to the thesis are proposed, not pushed.

## Procedure

1. **Extract every claim** in the section: numbers, dates, attributions, "no se identificó…",
   comparisons, and descriptions of methods, data and decisions.
2. **For each cited reference:**
   - Open the primary source: local PDF in `sources/` (git-ignored), DOI, arXiv, or the official
     URL. Notes in `docs/literature_thesis.md` and the `% verified:` lines are **leads, not
     proof**.
   - Check the figure in its table or text. Record the locator, saying whether the page is the
     printed page or the PDF page.
   - Check what the number counts: scope, units, cut-off date, edition, and the definition of the
     reference product.
   - Check the exact act or result the sentence claims (declared vs extended; compiled table vs
     controlled comparison; superlatives).
   - Keep local copies of the sources consulted in a git-ignored folder.
3. **For each project claim** (dataset sizes, splits, protocol, hypotheses' dates, method
   details), verify it against the repo:
   - manifests;
   - `configs/frozen.yaml`;
   - the scripts (for example, bootstrap replicates and what is resampled);
   - commit hashes and dates (`git log`);
   - `docs/METHODS_NOTES.md`.
4. **Check the §11 traps:**
   - absolute negatives;
   - CI-includes-0 read as equivalence;
   - "without Peru labels" covering selection too;
   - optimistic labels (2017 MNDWI agreement);
   - requirements stated as achieved;
   - "inundación" used where only water is measured;
   - caveats scattered through the argument instead of in the limitations.
5. **Write the audit table** with columns *Fuente | Qué se comprobó | Resultado y tratamiento*.
   Add a section for internal claims and a section *Límites y asuntos pendientes*.
6. **Propose the text corrections** to the authors. After approval, apply them with the
   thesis-writing skill (compile, render, commit, push only with approval).

## Never

- Treat an AI-text detector score as evidence of anything, or as a criterion for truth.
- Change a pre-registered hypothesis's prediction to match results. Clarify its
  operationalization in a note instead.
- Mark a claim verified without having opened the source.
