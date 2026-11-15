# Neural prototypes

This report covers the completed setup/punchline MiniLM prototype. DeBERTa has
not completed training and is excluded from these results.

Validation-selected pipeline: **Previous + setup / punchline**.

Train: 2019–2023; validation: 2024; historical test: Jan 2025–Feb 2026.

| Model | Validation AP | Jul–Dec gate AP | Test AP | Test top-10% success |
|---|---:|---:|---:|---:|
| Previous ensemble + HGB | 0.2524 | 0.2660 | 0.2792 | 32.1% |
| Reweighted ranking blend | 0.2523 | 0.2648 | 0.2757 | 32.1% |
| Setup / punchline MiniLM | 0.2506 | 0.2652 | 0.2804 | 32.8% |
| Previous + setup / punchline | 0.2566 | 0.2708 | 0.2855 | 33.3% |

AP is average precision (called PR-AUC elsewhere in the repo). Top-10% success is precision among the highest-ranked tenth, not accuracy.

New prototypes use single-seed runs; the reference averages three seeds. Neural checkpoints use full-2024 validation AP. Blend weights use Jan–Jun only; Jul–Dec gates replacement at +0.002 AP. That gate is not untouched because neural checkpoints and the old reference already used 2024.

Test has been inspected in earlier project work. These are exploratory historical comparisons, not a fresh prospective benchmark. Paired week-bootstrap intervals in uncertainty.json do not include model-selection uncertainty.

A candidate is not promoted because it happens to have the best historical test score. Read selection.json for the frozen validation decision.
