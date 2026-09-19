# Explainable verification of AI model configurations

Public repository: <https://github.com/Syrym-Zhakypbekov/ivf-explainable-verification>

This repository contains the code and aggregate artifacts for a criteria-governed method that verifies AI model configurations. Ovarian-response prediction provides the temporally structured test bed; the method is not a clinical treatment system.

The canonical release result is `v2`. It uses named registry sheets and stimulated autologous cycles. The earlier files directly under `results/` are retained only as a documented sensitivity history.

## Verify the released results

Requirements: Python 3.10 or newer. No third-party package is needed for this verification.

```bash
python scripts/verify_release.py
```

Expected output:

```text
v2: n=348 AUROC(V)=0.891 AUROC(Macro-F1)=0.720
v2pd: n=348 AUROC(V)=0.899 AUROC(Macro-F1)=0.715
OK ✓ release artifact matches the manuscript
```

The command reads `results/v2/stress2_all.csv`, independently recomputes both AUROCs, checks cohort counts, and verifies SHA-256 values from `ARTIFACT_MANIFEST.json`.

## Main results

| Result | Corrected v2 | Patient-disjoint v2pd |
|---|---:|---:|
| Configurations | 348 | 348 |
| Correct / defective | 12 / 336 | 12 / 336 |
| Verification-index AUROC | 0.891 | 0.899 |
| Macro-F1 AUROC | 0.720 | 0.715 |
| AUROC difference | 0.171 | 0.185 |
| 95% cluster-bootstrap CI | [0.023, 0.329] | [0.030, 0.349] |
| False verification rate | 0.086 | 0.101 |
| Erroneous rejection rate | 0.500 | 0.500 |

Primary files: `results/v2/stress2_summary.json` and `results/v2/stress2_all.csv`. Sensitivity files: `results/v2pd/`.

## Verification operator

The configuration-level index is

```text
V = D · T · (F · S · C · R)^(1/4)
```

| Component | Role |
|---|---|
| `D` | veto for data admissibility |
| `T` | veto for temporal admissibility |
| `F` | explanation-fidelity diagnostic |
| `S` | retraining-stability diagnostic |
| `C` | domain-consistency diagnostic |
| `R` | conformal-set informativeness under separately checked coverage |

`D=0` or `T=0` rejects the configuration independently of its predictive score. The other components diagnose distinct methodological properties.

## Evidence boundary

The study validates defect detection for model configurations. It does not validate individual clinical predictions or treatment recommendations.

The following negative results are part of the artifact:

- removing `R` increased overall AUROC from 0.891 to 0.921;
- arithmetic aggregation reached 0.909, compared with 0.891 for the geometric form;
- the frozen threshold rejected 6 of 12 correct configurations;
- `F` and `S` are configuration-level proxy measures;
- the input-noise intervention did not validate `R` as a universal uncertainty detector.

## Artifact contents

```text
results/v2/      corrected aggregate results cited in the manuscript
results/v2pd/    patient-disjoint sensitivity results
figures/v2/      corrected-run PNG and vector PDF figures
src/             private-data computation pipeline
scripts/         public aggregate verifier
```

See `ARTIFACT.md` for the claim-to-file map. See `WHERE_EVERYTHING_IS.md` for commands, expected outputs, and the full data boundary.

## Data availability

No registry extract or patient-level row is included. The public stress-test file contains one row per experimental configuration and no person-level features.

Authorized users can perform the full recomputation after placing `2023.xlsx`, `2024.xlsx`, and `2025.xlsx` in the untracked `data/` directory. The complete run takes approximately 60–90 minutes on 16 CPU cores.

## Environment

The reported run used Python 3.13.15. Exact package versions are listed in `requirements-lock.txt`. The fixed base seed is `20260802`.

## Authors

Artem Bykov and Syrym Zhakypbekov, International Information Technology University, Almaty, Kazakhstan.
