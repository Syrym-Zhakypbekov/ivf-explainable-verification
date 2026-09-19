# Reproducibility and artifact map

This repository publishes code, configuration-level outputs, aggregate cohort summaries, and figures for the model-verification study.

The clinical registry workbooks are not public. They contained identifiable health information before de-identification. Therefore, this repository supports two distinct verification levels:

1. **Public artifact verification:** checks the released `v2` and `v2pd` aggregates and recomputes the reported AUROCs from 348 configuration records.
2. **Full private-data recomputation:** rebuilds cohorts and models when authorized registry workbooks are placed in `data/`.

## Canonical result sets

| Directory | Meaning | Main result |
|---|---|---|
| `results/v2/` | corrected named-sheet cohort; stimulated autologous cycles | AUROC 0.891 versus 0.720 |
| `results/v2pd/` | patient-disjoint temporal sensitivity analysis | AUROC 0.899 versus 0.715 |
| `results/` root files | historical pre-correction outputs retained for audit | not cited as the main KBS result |

The KBS manuscript cites `results/v2/` as its primary result and `results/v2pd/` as sensitivity evidence.

## Public verification

Requirements: Python 3.10 or newer. The verifier uses only the standard library.

```bash
python scripts/verify_release.py
```

Expected final lines:

```text
v2: n=348 AUROC(V)=0.891 AUROC(Macro-F1)=0.720
v2pd: n=348 AUROC(V)=0.899 AUROC(Macro-F1)=0.715
OK ✓ release artifact matches the manuscript
```

The verifier performs these checks:

- all 348 configuration rows are present;
- 12 configurations are correct and 336 are defective;
- AUROC is recomputed independently from `stress2_all.csv`;
- the summary and ablation files contain the cited values;
- the patient-disjoint summary contains the reported sensitivity result;
- all release files match `ARTIFACT_MANIFEST.json`.

Runtime is below five seconds on a laptop because model training is not repeated.

## Repository layout

```text
src/                    analysis pipeline
scripts/verify_release.py
results/v2/             canonical corrected aggregate artifact
results/v2pd/           patient-disjoint sensitivity artifact
figures/v2/             figures generated from the corrected run
ARTIFACT.md             claim-to-file map and evidence boundary
ARTIFACT_MANIFEST.json  SHA-256 manifest for the release files
```

## Full private-data recomputation

Authorized users place three workbooks in an untracked directory:

```text
data/2023.xlsx
data/2024.xlsx
data/2025.xlsx
```

The `.gitignore` file excludes `data/` and all spreadsheet files. Never commit registry extracts or per-cycle clinical outputs.

The canonical server path uses Nix:

```bash
bash src/run_v2.sh
bash src/run_v2pd.sh
```

`run_v2.sh` prints `DONE_V2` after the complete chain. `run_v2pd.sh` prints `DONE_V2PD`. The stress test is the longest stage and takes approximately 60–90 minutes on 16 CPU cores with three numerical threads.

For an existing Python environment:

```bash
python -m pip install -r requirements.txt
PY=python COHORT=v2 THREADS=3 bash src/run_local.sh
```

The scripts resolve the repository root automatically. `IVF_BASE` can override that root, and `OUT_DIR` can redirect outputs.

## Pipeline

| Script | Purpose | Main outputs |
|---|---|---|
| `src/01_audit.py` | feature and timing audit | `data_dictionary.csv`, `audit_summary.txt` |
| `src/05_models.py` | correct versus leakage comparison | `table_models.csv` |
| `src/07_calibrated.py` | threshold calibration and isolated defects | `final_table.csv`, `final_meta.json` |
| `src/08_semisynth.py` | known-mechanism explanation recovery | `semisynth_recovery.csv` |
| `src/09b_stress_fixed.py` | 348-configuration stress test | `stress2_all.csv`, summary, ablation |
| `src/20_figures_en.py` | English scientific figures | `figures/v2/` equivalents |
| `src/22_cohort_characteristics.py` | aggregate cohort description | cohort summary CSV files |

The fixed seed is `20260802`. The primary split trains on 2023, calibrates on 2024, and tests on 2025.

## Data boundary

The public configuration-level file contains one row per experimental configuration, not one row per patient. It includes model metrics and controlled-defect labels only.

The public artifact does not contain names, national identifiers, telephone numbers, registry rows, or per-cycle clinical features. Full cohort reconstruction cannot be independently reproduced without authorized access to the registry extracts.

## Historical outputs

Files directly under `results/` record the earlier cohort definition. Their main AUROC was 0.855. They remain available only to document why the cohort correction changed the magnitude.

Do not cite those root files as the main result. Use `results/v2/` and the tagged release asset.
