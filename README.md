# Explainable verification of AI decisions — ovarian response prediction

Reproducibility materials for the study on **explainable verification of AI decision
correctness**, demonstrated on ovarian response prediction in ART (IVF) cycles.

**Core claim:** a statistically accurate model can be methodologically incorrect, and
standard quality metrics fail to detect this.

## Key results

| Indicator | V | Macro-F1 |
|---|---|---|
| AUROC of defect detection (348 configurations) | **0.855** [0.800; 0.905] | 0.616 [0.499; 0.718] |
| AUROC on the data-leakage defect | **1.000** | **0.111** |
| Difference in AUROC, 95% CI (cluster bootstrap) | **[0.101; 0.391]** — excludes zero | — |
| False verification rate at θ = 0.694 | 0.089 (target ≤ 0.10) | — |
| Conformal prediction coverage at α = 0.10 | 0.904 (target 0.900) | — |

An AUROC of **0.111** for macro-F1 on the leakage defect is below 0.5: the accuracy
metric does not merely fail to notice the methodological defect — it systematically
ranks the defective model **above** the correct one.

The integral index also outperforms every individual component taken alone (best single
component: C = 0.745), which shows the discriminative ability arises from the joint
non-compensatory aggregation rather than from one strong component.

## The verification index

```
V(x) = D(x) · T(x) · [ F(x) · S(x) · C(x) · R(x) ]^(1/4)
```

| Component | Meaning |
|---|---|
| **D** | data admissibility — impossible values, contradictions, missing mandatory features |
| **T** | temporal admissibility — was the feature known *before* the decision moment |
| **F** | explanation fidelity — does the explanation match the model's actual behaviour |
| **S** | stability — does the explanation survive retraining on bootstrap samples |
| **C** | domain consistency — agreement with clinical knowledge |
| **R** | reliability — split conformal prediction, size of the admissible class set |

D and T enter as multiplicative **vetoes**: a zero in either nullifies the index
regardless of the remaining components. Under a weighted sum the leaking model would
score ≈ 0.65 and pass verification.

## Repository layout

```
src/        computation pipeline (Python)
results/    result tables (CSV/JSON), 348 stress-test configurations
figures/en  figures for journal submission — PDF + EPS (vector) + PNG 600 dpi
figures/ru  the same figures in Russian, for the dissertation
docs/       article and experiment report (Russian text)
```

### Pipeline

| Script | Purpose |
|---|---|
| `01_audit.py` | feature audit and temporal marking of 374 columns |
| `07_calibrated.py` | threshold calibration, baseline experiment on 6 isolated defects |
| `08_semisynth.py` | semi-synthetic experiment with a known ground-truth explanation |
| `09b_stress_fixed.py` | large-scale stress test, 348 configurations |
| `10_doctors.py` | de-identified case package for expert review |
| `13`–`16`, `20` | figure generation (Russian and English versions) |

Data are read strictly read-only; MD5 checksums of the source files are unchanged.
Random seed 20260802 throughout.

## Figures

| File | Content |
|---|---|
| `fig1_leakage` | accuracy vs verifiability under leakage, 4 algorithms |
| `fig2_components_heatmap` | component heat map — each defect lowers exactly its own component |
| `fig3_significance` | ROC curves + bootstrap CI of the AUROC difference |
| `fig4_separation` | separation of 348 configurations |
| `fig5_intensity` | index vs defect intensity, 6 defect types, threshold line |
| `fig6_monotonicity` | behaviour by defect type in native intensity units |
| `fig7_ablation` | ablation: contribution of each component |
| `fig8_semisynthetic` | predictive accuracy vs explanation recovery |
| `fig9_all_indicators` | all indicators compared, AUROC with confidence intervals |

Prepared to journal requirements: 170 mm width (MDPI two-column), minimum font size 7 pt
after typesetting (verified programmatically), fonts embedded as TrueType. Vector PDF/EPS
is the submission format — the "1000 dpi for line art" requirement applies to raster only.

## Limitations (stated explicitly)

1. **Component R** provides the coverage guarantee but is insensitive to its own defect:
   the index varies within 0.627–0.631 as noise grows eightfold, and the "without R"
   ablation yields the same AUROC of 0.855. R currently contributes statistical
   correctness, not detection ability.
2. **False rejection rate is 0.25** at θ = 0.694 — the price of a strict threshold chosen
   to keep false verification ≤ 0.10.
3. Baseline model accuracy is low (macro-F1 0.392) because the feature set is restricted
   to those known before the decision. This follows from the methodology, not from a
   defect in the verification method.
4. Defective configurations are constructed deliberately. This is a property of the
   design: controlled defect injection is the only way to have known ground truth when
   validating a detection instrument.
5. Components F, S and C are evaluated at model level, not per observation.

## Study type

This is a **theoretical-methodological study on AI verification**, not a clinical
validation of an ovarian response prediction system. The method was tested on data with
known ground truth, where the correct answer is known by construction and expert
assessment is not required to validate the method itself. Clinical validation is a
separate study with a different design; the case package for it is prepared.

## Data

Source clinical datasets are **not** included in this repository. No patient-level data,
identifiers or outcome files are committed. Only aggregated result tables are published.

## Authors

Bykov A. A., Zhakypbekov S. — International Information Technology University, Almaty.
