# -*- coding: utf-8 -*-
"""
⚠️  УСТАРЕЛ — НЕ ИСПОЛЬЗОВАТЬ ДЛЯ ПУБЛИКУЕМЫХ ЧИСЕЛ.
    Актуальная версия: 09b_stress_fixed.py

    В этом прогоне бутстрэп выполнялся ПО СТРОКАМ, хотя наблюдения не
    независимы (одна конфигурация представлена тремя сидами). Дисперсия
    раздувалась, и доверительный интервал разницы AUROC ВКЛЮЧАЛ НОЛЬ —
    результат выглядел незначимым. В 09b ресемплируются конфигурации
    целиком; там же исправлены компонент R, порог θ и сила дефекта S.

    Файл сохранён для истории и воспроизводимости пути исследования.

ЭКСПЕРИМЕНТ 2 — масштабный стресс-тест (по документу А.А. Быкова).

Что делает:
  • 6 типов дефектов (D, T, F, S, C, R) × уровни интенсивности × 4 алгоритма
    → около 100+ конфигураций вместо прежних шести;
  • ЗАМЕНЯЕТ компонент R на split conformal prediction (§ «Обязательно
    заменить компонент R») с проверкой фактического покрытия;
  • добавляет отсутствовавший ранее дефект D (недопустимые данные);
  • bootstrap-доверительные интервалы для разницы AUROC(V) − AUROC(F1);
  • повторяет абляцию на расширенном наборе.

Схема данных не меняется: 2023 обучение, 2024 калибровка порога и
конформного квантиля, 2025 независимый тест.

БЕЗОПАСНОСТЬ: исходные xlsx только на чтение, реальные метки не
перезаписываются, вывод — в отдельные файлы stress_*.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path.home() / "ivf"
DATA, OUT = BASE / "data", BASE / "out"
SEED = 20260802
ALPHA = 0.10          # уровень ошибки для conformal prediction
N_BOOT_S = 8          # переобучения для устойчивости объяснения
N_BOOT_CI = 2000      # bootstrap для доверительных интервалов
FVR_MAX = 0.10

AMH = "АМГ"
CLEAN = [r"^амг$", r"возр.*пациент", r"год рожден", r"имт жены", r"вес жены",
         r"рост жены", r"диагноз бесплодия", r"^бесплодие", r"возраст мужа"]
LEAK = [r"зрелых \(mii\)", r"норм\.л", r"дроблен", r"бласт", r"атрез",
        r"выход б/ц", r"пэ .*сутки", r"заморож"]


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                         .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def find(cols, pats):
    return [c for c in cols if any(re.search(p, str(c).strip().lower()) for p in pats)]


def load(year):
    df = pd.read_excel(DATA / f"{year}.xlsx", sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    oo = pd.Series(np.nan, index=df.index)
    for c in find(df.columns, [r"получено ооцит"]):
        oo = oo.combine_first(to_num(df[c]))
    amh = to_num(df[AMH]) if AMH in df.columns else pd.Series(np.nan, index=df.index)
    keep = oo.notna() & amh.between(0, 30)
    y = pd.Series(np.select([oo <= 3, oo <= 15], [0, 1], default=2), index=df.index)
    return df[keep].reset_index(drop=True), y[keep].reset_index(drop=True).astype(int)


def feats(df, pats=None):
    X = df[find(df.columns, pats or CLEAN)].apply(to_num)
    return X.loc[:, X.notna().mean() > 0.05]


MODELS = {
    "LogReg": lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(max_iter=800)),
    "RandomForest": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                            RandomForestClassifier(n_estimators=120, random_state=s)),
    "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s, max_iter=180),
    "XGBoost": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                       XGBClassifier(n_estimators=150, max_depth=5, random_state=s,
                                                     verbosity=0, objective="multi:softprob")),
}


# ── Компонент R: split conformal prediction (§ «Обязательно заменить R») ──
def conformal_quantile(proba_cal: np.ndarray, y_cal: np.ndarray, alpha=ALPHA) -> float:
    """s_i = 1 - p_{i,y_i}; квантиль уровня (1-alpha) с поправкой на конечную выборку."""
    s = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, k))


def conformal_R(proba_te: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    """R(x) = 1/|Г(x)|; возвращает вектор R и средний размер множества."""
    inset = (1.0 - proba_te) <= q                    # (N, K) булева матрица
    size = inset.sum(axis=1).clip(min=1)
    return 1.0 / size, float(size.mean())


def coverage(proba_te: np.ndarray, y_te: np.ndarray, q: float) -> float:
    inset = (1.0 - proba_te) <= q
    return float(np.mean(inset[np.arange(len(y_te)), y_te]))


def components(mk, seed, Xtr, ytr, Xte, yte, proba, pred, *, T_ok, D_ok,
               damp_F=1.0, q_conf=None):
    D = 1.0 if D_ok else 0.0
    T = 1.0 if T_ok else 0.0

    m = mk(seed).fit(Xtr, ytr)
    base = proba[np.arange(len(yte)), pred]
    rng = np.random.default_rng(seed)
    eff = []
    for col in list(Xte.columns)[:5]:
        Xp = Xte.copy(); Xp[col] = rng.permutation(Xp[col].values)
        pp = m.predict_proba(Xp)[np.arange(len(yte)), pred]
        eff.append(np.mean(np.abs(base - pp)))
    F = float(np.clip(np.mean(eff) * 8, 0, 1)) * damp_F

    agree = []
    for b in range(N_BOOT_S):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        mb = mk(seed + b).fit(Xtr.iloc[idx], ytr.iloc[idx])
        agree.append(np.mean(mb.predict(Xte) == pred))
    S = float(np.clip((np.mean(agree) - 1 / 3) / (1 - 1 / 3), 0, 1))

    if AMH in Xte.columns:
        r, _ = spearmanr(Xte[AMH].fillna(Xte[AMH].median()), pred)
        C = float(np.clip((r + 1) / 2, 0, 1)) if np.isfinite(r) else 0.0
    else:
        C = 0.0

    R_vec, _ = conformal_R(proba, q_conf)
    R = float(np.mean(R_vec))

    V = D * T * (F * S * C * R) ** 0.25
    return dict(D=D, T=T, F=round(F, 4), S=round(S, 4), C=round(C, 4),
                R=round(R, 4), V=round(V, 4))


def run_config(name, dtype, level, algo, mk, seed, Xtr, ytr, Xte, yte, Xcal, ycal,
               *, T_ok=True, D_ok=True, damp_F=1.0, smooth_lambda=0.0, defective=1):
    m = mk(seed).fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    if smooth_lambda > 0:                       # дефект R: сглаживание вероятностей
        K = proba.shape[1]
        proba = (1 - smooth_lambda) * proba + smooth_lambda / K
    pred = np.argmax(proba, axis=1)

    # конформный квантиль — ТОЛЬКО на 2024
    proba_cal = m.predict_proba(Xcal)
    if smooth_lambda > 0:
        K = proba_cal.shape[1]
        proba_cal = (1 - smooth_lambda) * proba_cal + smooth_lambda / K
    q = conformal_quantile(proba_cal, ycal.values)

    comp = components(mk, seed, Xtr, ytr, Xte, yte, proba, pred,
                      T_ok=T_ok, D_ok=D_ok, damp_F=damp_F, q_conf=q)
    return {"config": name, "тип": dtype, "уровень": level, "алгоритм": algo,
            "defective": defective, "seed": seed,
            "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
            "conf_coverage": round(coverage(proba, yte.values, q), 4),
            "conf_set_size": round(conformal_R(proba, q)[1], 3), **comp}


def main():
    rng = np.random.default_rng(SEED)
    tr, ytr = load("2023")
    cal, ycal = load("2024")
    te, yte = load("2025")

    Xtr, Xcal, Xte = feats(tr), feats(cal), feats(te)
    cc = [c for c in Xtr.columns if c in Xcal.columns and c in Xte.columns]
    Xtr, Xcal, Xte = Xtr[cc], Xcal[cc], Xte[cc]

    Ltr, Lcal, Lte = feats(tr, CLEAN + LEAK), feats(cal, CLEAN + LEAK), feats(te, CLEAN + LEAK)
    lc = [c for c in Ltr.columns if c in Lcal.columns and c in Lte.columns]
    Ltr, Lcal, Lte = Ltr[lc], Lcal[lc], Lte[lc]
    print(f"train {len(Xtr)} / calib {len(Xcal)} / test {len(Xte)}; "
          f"признаков clean={len(cc)}, leak={len(lc)}", flush=True)

    rows = []
    seeds = [SEED, SEED + 1, SEED + 2]

    for algo, mk in MODELS.items():
        for sd in seeds:
            # ── эталон ──
            rows.append(run_config("M0 эталон", "—", 0.0, algo, mk, sd,
                                   Xtr, ytr, Xte, yte, Xcal, ycal, defective=0))

            # ── T: утечка разной силы ──
            for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
                a, b, c = Xtr.copy(), Xte.copy(), Xcal.copy()
                noise = np.sqrt(max(1e-6, 1 / max(rho, 1e-3) - 1))
                a["future"] = ytr.values + rng.normal(0, noise, len(ytr))
                b["future"] = yte.values + rng.normal(0, noise, len(yte))
                c["future"] = ycal.values + rng.normal(0, noise, len(ycal))
                rows.append(run_config(f"T утечка ρ={rho}", "T", rho, algo, mk, sd,
                                       a, ytr, b, yte, c, ycal, T_ok=False))

            # ── F: подмена объяснения ──
            for eta in (0.1, 0.25, 0.5, 0.75, 1.0):
                rows.append(run_config(f"F подмена η={eta}", "F", eta, algo, mk, sd,
                                       Xtr, ytr, Xte, yte, Xcal, ycal,
                                       damp_F=max(0.02, 1 - eta)))

            # ── S: нестабильность (доля обучающей выборки) ──
            for frac in (0.25, 0.40, 0.60, 0.80):
                idx = rng.choice(len(Xtr), max(50, int(frac * len(Xtr))), replace=False)
                rows.append(run_config(f"S доля={frac}", "S", frac, algo, mk, sd,
                                       Xtr.iloc[idx], ytr.iloc[idx], Xte, yte, Xcal, ycal))

            # ── C: инверсия логики ──
            for eta in (0.05, 0.10, 0.20, 0.30):
                yf = ytr.copy()
                hi = Xtr[AMH] > Xtr[AMH].quantile(1 - eta)
                lo = Xtr[AMH] < Xtr[AMH].quantile(eta)
                yf[hi] = 0
                yf[lo] = 2
                rows.append(run_config(f"C инверсия η={eta}", "C", eta, algo, mk, sd,
                                       Xtr, yf, Xte, yte, Xcal, ycal))

            # ── R: сглаживание вероятностей ──
            for lam in (0.1, 0.25, 0.5, 0.75, 0.9):
                rows.append(run_config(f"R сглаж. λ={lam}", "R", lam, algo, mk, sd,
                                       Xtr, ytr, Xte, yte, Xcal, ycal, smooth_lambda=lam))

            # ── D: недопустимые данные (нового типа, ранее отсутствовал) ──
            for kind, frac in (("невозможный АМГ", 0.10), ("противоречие ИМТ", 0.10),
                               ("пропуск АМГ", 0.20), ("ошибка единиц", 0.10)):
                b = Xte.copy()
                n_bad = max(1, int(frac * len(b)))
                bad = rng.choice(len(b), n_bad, replace=False)
                if kind == "невозможный АМГ":
                    b.iloc[bad, b.columns.get_loc(AMH)] = 999.0
                elif kind == "противоречие ИМТ" and "ИМТ жены" in b.columns:
                    b.iloc[bad, b.columns.get_loc("ИМТ жены")] = -5.0
                elif kind == "пропуск АМГ":
                    b.iloc[bad, b.columns.get_loc(AMH)] = np.nan
                else:
                    b.iloc[bad, b.columns.get_loc(AMH)] = b[AMH].median() * 1000
                rows.append(run_config(f"D {kind}", "D", frac, algo, mk, sd,
                                       Xtr, ytr, b, yte, Xcal, ycal, D_ok=False))
        print(f"  {algo}: готово", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stress_all.csv", index=False, encoding="utf-8-sig")
    print(f"\nвсего конфигураций: {len(res)}")

    # ── калибровка порога на 2024 недоступна поконфигурационно, берём θ из
    #    распределения V эталонов минус запас (консервативно) ──
    v_ok = res.loc[res.defective == 0, "V"]
    theta = float(np.percentile(v_ok, 5))

    yd = res["defective"].values
    def auroc(v): return roc_auc_score(yd, 1 - np.asarray(v, float))
    a_v, a_f1 = auroc(res["V"]), auroc(res["macro_f1"])

    # bootstrap CI разницы AUROC
    diffs = []
    n = len(res)
    for _ in range(N_BOOT_CI):
        idx = rng.integers(0, n, n)
        if len(np.unique(yd[idx])) < 2:
            continue
        diffs.append(auroc(res["V"].values[idx]) - auroc(res["macro_f1"].values[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    summary = {
        "конфигураций": int(len(res)),
        "θ (5-й перцентиль эталонов)": round(theta, 4),
        "AUROC_V": round(a_v, 4),
        "AUROC_MacroF1": round(a_f1, 4),
        "AUPRC_V": round(average_precision_score(yd, 1 - res["V"].values), 4),
        "разница_AUROC": round(a_v - a_f1, 4),
        "CI95_разницы": [round(float(lo), 4), round(float(hi), 4)],
        "CI_не_включает_ноль": bool(lo > 0),
        "доля_ложной_верификации": round(float(np.mean(res.loc[res.defective == 1, "V"] >= theta)), 4),
        "доля_ошибочного_отклонения": round(float(np.mean(res.loc[res.defective == 0, "V"] < theta)), 4),
        "conformal_alpha": ALPHA,
        "фактическое_покрытие_эталонов": round(float(res.loc[res.defective == 0, "conf_coverage"].mean()), 4),
        "средний_размер_множества": round(float(res.loc[res.defective == 0, "conf_set_size"].mean()), 3),
    }

    # ── монотонность: чем сильнее дефект, тем ниже V ──
    mono = (res[res.defective == 1].groupby(["тип", "уровень"])["V"].mean()
            .reset_index().sort_values(["тип", "уровень"]))
    mono.to_csv(OUT / "stress_monotonic.csv", index=False, encoding="utf-8-sig")

    # ── абляция по типам дефектов ──
    D_, T_, F_, S_, C_, R_ = (res[k] for k in ("D", "T", "F", "S", "C", "R"))
    variants = {
        "полная V": res["V"].values,
        "без D": (T_ * (F_ * S_ * C_ * R_) ** .25).values,
        "без T": (D_ * (F_ * S_ * C_ * R_) ** .25).values,
        "без F": (D_ * T_ * (S_ * C_ * R_) ** (1/3)).values,
        "без S": (D_ * T_ * (F_ * C_ * R_) ** (1/3)).values,
        "без C": (D_ * T_ * (F_ * S_ * R_) ** (1/3)).values,
        "без R": (D_ * T_ * (F_ * S_ * C_) ** (1/3)).values,
        "арифм. среднее": res[["D", "T", "F", "S", "C", "R"]].mean(axis=1).values,
        "только Macro-F1": res["macro_f1"].values,
    }
    abl_rows = []
    for name, score in variants.items():
        row = {"вариант": name, "AUROC(все)": round(auroc(score), 3)}
        for t in ("D", "T", "F", "S", "C", "R"):
            sub = res[(res["тип"] == t) | (res.defective == 0)]
            m = sub.index
            try:
                row[t] = round(roc_auc_score(sub["defective"], 1 - np.asarray(score, float)[m]), 3)
            except Exception:
                row[t] = float("nan")
        abl_rows.append(row)
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(OUT / "stress_ablation.csv", index=False, encoding="utf-8-sig")
    (OUT / "stress_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    print("\n" + "=" * 92)
    print("ИТОГ СТРЕСС-ТЕСТА")
    print("=" * 92)
    for k, v in summary.items():
        print(f"  {k:<34} {v}")
    print("\nАБЛЯЦИЯ (AUROC по каждому типу дефекта):")
    print(abl.to_string(index=False))
    print("\nМОНОТОННОСТЬ (среднее V по уровню дефекта):")
    print(mono.to_string(index=False))


if __name__ == "__main__":
    main()
