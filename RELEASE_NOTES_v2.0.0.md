# v2.0.0 — corrected cohort and reproducibility artifact

This release makes the corrected `v2` result set canonical and packages the patient-disjoint `v2pd` sensitivity analysis.

## Results

- `v2`: 3,397 cycles across 2023–2025; 348 model configurations.
- Verification-index AUROC: 0.891.
- Macro-F1 AUROC: 0.720.
- Difference: 0.171; 95% cluster-bootstrap CI [0.023, 0.329].
- `v2pd`: AUROC 0.899 versus 0.715; difference 0.185.

## Artifact verification

```bash
python scripts/verify_release.py
```

Expected final line:

```text
OK ✓ release artifact matches the manuscript
```

The verifier recomputes both primary AUROCs from 348 configuration rows and checks SHA-256 values for the release package.

## Data boundary

No registry extract or patient-level row is included. Full model retraining requires authorized access to three clinical workbooks. Public verification covers configuration-level statistics, ablations, cohort aggregates, and figures.

## Corrections from the historical run

- The 2025 workbook is selected by sheet name instead of position.
- The primary cohort is restricted to stimulated autologous cycles.
- The threshold is calibrated on 2024 and frozen before 2025 testing.
- The patient-disjoint analysis removes later cycles from patients present in earlier years.
