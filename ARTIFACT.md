# Artifact evidence map

This file links each manuscript result to a public file and a verification command.

## Primary result

| Claim | Value | Public evidence |
|---|---:|---|
| configurations | 348 | `results/v2/stress2_all.csv` |
| correct / defective | 12 / 336 | `defective` column in the same file |
| verification AUROC | 0.891 | recomputed by `scripts/verify_release.py` |
| Macro-F1 AUROC | 0.720 | recomputed by `scripts/verify_release.py` |
| AUROC difference | 0.171 | `results/v2/stress2_summary.json` |
| cluster-bootstrap CI | [0.023, 0.329] | `results/v2/stress2_summary.json` |
| frozen threshold | 0.700 | `results/v2/stress2_summary.json` |
| false verification rate | 0.086 | `results/v2/stress2_summary.json` |
| erroneous rejection rate | 0.500 | `results/v2/stress2_summary.json` |
| conformal coverage | 0.912 | `results/v2/stress2_summary.json` |

## Negative results

| Claim | Value | Public evidence |
|---|---:|---|
| full geometric index AUROC | 0.891 | `results/v2/stress2_ablation.csv` |
| index without `R` | 0.921 | `results/v2/stress2_ablation.csv` |
| arithmetic aggregation | 0.909 | `results/v2/stress2_ablation.csv` |
| defect-specific AUROC for `R` | 0.693 | `results/v2/stress2_ablation.csv` |

## Patient-disjoint sensitivity analysis

| Claim | Value | Public evidence |
|---|---:|---|
| train / calibration / test cycles | 1259 / 896 / 901 | `results/v2pd/final_meta.json` |
| verification AUROC | 0.899 | `results/v2pd/stress2_summary.json` |
| Macro-F1 AUROC | 0.715 | `results/v2pd/stress2_summary.json` |
| AUROC difference | 0.185 | `results/v2pd/stress2_summary.json` |
| cluster-bootstrap CI | [0.030, 0.349] | `results/v2pd/stress2_summary.json` |

## Verification command

```bash
python scripts/verify_release.py
```

The command uses no clinical registry file. It recomputes the primary AUROCs from configuration-level rows and verifies every release-file hash.

## Evidence boundary

The artifact reproduces aggregate statistical checks and figures. It cannot recreate cohort formation or model training without the restricted registry workbooks.

No patient-level row, patient key, name, date of birth, telephone number, or national identifier is included.
