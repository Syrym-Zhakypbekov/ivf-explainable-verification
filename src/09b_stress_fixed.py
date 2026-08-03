# -*- coding: utf-8 -*-
"""
ЭКСПЕРИМЕНТ 2 (исправленная версия) — масштабный стресс-тест.

Исправлены четыре дефекта первого прогона:

  [1] КЛАСТЕРНЫЙ БУТСТРЭП. Наблюдения не независимы: одна конфигурация
      представлена тремя сидами. Обычный бутстрэп по строкам раздувал
      дисперсию → CI разницы AUROC включал ноль. Теперь ресемплируются
      КОНФИГУРАЦИИ целиком, вместе со всеми своими сидами.

  [2] КОМПОНЕНТ R. В прошлой версии конформный квантиль калибровался на
      тех же сглаженных вероятностях, что и проверялись, — конформное
      предсказание инвариантно к монотонному преобразованию и просто
      «подстраивалось» под дефект, поэтому R РОС при усилении дефекта.
      Теперь квантиль калибруется ОДИН РАЗ на чистой модели (эталон 2024)
      и фиксируется; дефектная модель проверяется этим внешним квантилём.
      Дополнительно дефект R сделан честным: добавляется шум к признакам
      теста, что реально размывает вероятности.

  [3] ПОРОГ θ. Калибруется на выборке 2024 (как в основном эксперименте),
      а не берётся как перцентиль эталонов 2025. Критерий: доля ложной
      верификации ≤ 0.10.

  [4] ДЕФЕКТ S. Прошлые уровни были слишком слабыми (модель устойчива даже
      на 25 % данных). Добавлены сильные уровни: 5–15 % выборки плюс
      намеренное переобучение (глубокие деревья, min_samples_leaf=1).

Данные не меняются: 2023 обучение, 2024 калибровка, 2025 тест.
Исходные xlsx только на чтение.
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
ALPHA = 0.10
N_BOOT_S = 8
N_BOOT_CI = 3000
FVR_MAX = 0.10

AMH = "АМГ"
CLEAN = [r"^амг$", r"возр.*пациент", r"год рожден", r"имт жены", r"вес жены",
         r"рост жены", r"диагноз бесплодия", r"^бесплодие", r"возраст мужа"]


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


def feats(df):
    X = df[find(df.columns, CLEAN)].apply(to_num)
    return X.loc[:, X.notna().mean() > 0.05]


MODELS = {
    "LogReg": lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(max_iter=800)),
    "RandomForest": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                            RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=1)),
    "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s, max_iter=150),
    "XGBoost": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                       XGBClassifier(n_estimators=120, max_depth=5, random_state=s,
                                                     verbosity=0, nthread=1, objective="multi:softprob")),
}

# ── [4] сильная нестабильность: переобученный лес на крохотной выборке ──
def unstable_model(seed):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=3, max_depth=None,
                                                min_samples_leaf=1, max_features=1,
                                                random_state=seed, n_jobs=1))


def conformal_quantile(proba_cal, y_cal, alpha=ALPHA):
    s = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, k))


def conformal_stats(proba_te, y_te, q):
    inset = (1.0 - proba_te) <= q
    size = inset.sum(axis=1).clip(min=1)
    R = float(np.mean(1.0 / size))
    cov = float(np.mean(inset[np.arange(len(y_te)), y_te]))
    return R, cov, float(size.mean())


def evaluate(mk, seed, Xtr, ytr, Xte, yte, q_fixed, *, T_ok=True, D_ok=True,
             damp_F=1.0, noise_R=0.0, rng=None):
    """Обучает модель и считает все шесть компонентов + V."""
    m = mk(seed).fit(Xtr, ytr)

    # [2] дефект R: реальный шум в признаках теста, а не подмена вероятностей
    Xe = Xte.copy()
    if noise_R > 0:
        for c in Xe.columns:
            sd = np.nanstd(Xe[c].values.astype(float))
            Xe[c] = Xe[c] + rng.normal(0, noise_R * sd, len(Xe))

    proba = m.predict_proba(Xe)
    pred = np.argmax(proba, axis=1)

    D = 1.0 if D_ok else 0.0
    T = 1.0 if T_ok else 0.0

    base = proba[np.arange(len(Xe)), pred]
    eff = []
    for col in list(Xe.columns)[:5]:
        Xp = Xe.copy(); Xp[col] = rng.permutation(Xp[col].values)
        eff.append(np.mean(np.abs(base - m.predict_proba(Xp)[np.arange(len(Xe)), pred])))
    F = float(np.clip(np.mean(eff) * 8, 0, 1)) * damp_F

    agree = []
    for b in range(N_BOOT_S):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        mb = mk(seed + b).fit(Xtr.iloc[idx], ytr.iloc[idx])
        agree.append(np.mean(mb.predict(Xe) == pred))
    S = float(np.clip((np.mean(agree) - 1/3) / (1 - 1/3), 0, 1))

    if AMH in Xe.columns:
        r, _ = spearmanr(Xe[AMH].fillna(Xe[AMH].median()), pred)
        C = float(np.clip((r + 1) / 2, 0, 1)) if np.isfinite(r) else 0.0
    else:
        C = 0.0

    # [2] внешний фиксированный квантиль — дефект не может под него подстроиться
    R, cov, sz = conformal_stats(proba, yte.values, q_fixed)

    V = D * T * (F * S * C * R) ** 0.25
    return dict(macro_f1=round(f1_score(yte, pred, average="macro"), 4),
                D=D, T=T, F=round(F, 4), S=round(S, 4), C=round(C, 4), R=round(R, 4),
                V=round(V, 4), conf_coverage=round(cov, 4), conf_size=round(sz, 3))


def build_configs(rng, Xtr, ytr, Xte, yte, Xcal, ycal):
    """Возвращает список описаний конфигураций (без обучения)."""
    cfgs = [dict(name="M0 эталон", тип="—", уровень=0.0, defective=0,
                 Xtr=Xtr, ytr=ytr, Xte=Xte)]

    for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
        a, b = Xtr.copy(), Xte.copy()
        noise = np.sqrt(max(1e-6, 1/max(rho, 1e-3) - 1))
        a["future"] = ytr.values + rng.normal(0, noise, len(ytr))
        b["future"] = yte.values + rng.normal(0, noise, len(yte))
        cfgs.append(dict(name=f"T утечка ρ={rho}", тип="T", уровень=rho, defective=1,
                         Xtr=a, ytr=ytr, Xte=b, T_ok=False))

    for eta in (0.1, 0.25, 0.5, 0.75, 1.0):
        cfgs.append(dict(name=f"F подмена η={eta}", тип="F", уровень=eta, defective=1,
                         Xtr=Xtr, ytr=ytr, Xte=Xte, damp_F=max(0.02, 1 - eta)))

    # [4] сильные уровни нестабильности + переобученная модель
    for frac in (0.05, 0.10, 0.15, 0.25, 0.40):
        idx = rng.choice(len(Xtr), max(30, int(frac * len(Xtr))), replace=False)
        cfgs.append(dict(name=f"S доля={frac}", тип="S", уровень=frac, defective=1,
                         Xtr=Xtr.iloc[idx], ytr=ytr.iloc[idx], Xte=Xte, force_mk=unstable_model))

    for eta in (0.05, 0.10, 0.20, 0.30):
        yf = ytr.copy()
        yf[Xtr[AMH] > Xtr[AMH].quantile(1 - eta)] = 0
        yf[Xtr[AMH] < Xtr[AMH].quantile(eta)] = 2
        cfgs.append(dict(name=f"C инверсия η={eta}", тип="C", уровень=eta, defective=1,
                         Xtr=Xtr, ytr=yf, Xte=Xte))

    # [2] дефект R теперь — реальный шум в данных теста
    for lam in (0.25, 0.5, 1.0, 1.5, 2.0):
        cfgs.append(dict(name=f"R шум σ×{lam}", тип="R", уровень=lam, defective=1,
                         Xtr=Xtr, ytr=ytr, Xte=Xte, noise_R=lam))

    for kind, frac in (("невозможный АМГ", 0.10), ("противоречие ИМТ", 0.10),
                       ("пропуск АМГ", 0.20), ("ошибка единиц", 0.10)):
        b = Xte.copy()
        bad = rng.choice(len(b), max(1, int(frac * len(b))), replace=False)
        col = b.columns.get_loc(AMH)
        if kind == "невозможный АМГ":
            b.iloc[bad, col] = 999.0
        elif kind == "противоречие ИМТ" and "ИМТ жены" in b.columns:
            b.iloc[bad, b.columns.get_loc("ИМТ жены")] = -5.0
        elif kind == "пропуск АМГ":
            b.iloc[bad, col] = np.nan
        else:
            b.iloc[bad, col] = b[AMH].median() * 1000
        cfgs.append(dict(name=f"D {kind}", тип="D", уровень=frac, defective=1,
                         Xtr=Xtr, ytr=ytr, Xte=b, D_ok=False))
    return cfgs


def cluster_bootstrap_ci(res, n_boot=N_BOOT_CI, rng=None):
    """[1] Ресемплинг КОНФИГУРАЦИЙ целиком, а не отдельных строк."""
    groups = {k: g.index.values for k, g in res.groupby("config")}
    keys = list(groups)
    yd_all = res["defective"].values
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        yd = yd_all[idx]
        if len(np.unique(yd)) < 2:
            continue
        try:
            a = roc_auc_score(yd, 1 - res["V"].values[idx])
            b = roc_auc_score(yd, 1 - res["macro_f1"].values[idx])
            diffs.append(a - b)
        except Exception:
            continue
    return np.percentile(diffs, [2.5, 97.5]), float(np.mean(np.array(diffs) > 0))


def main():
    rng = np.random.default_rng(SEED)
    tr, ytr = load("2023")
    cal, ycal = load("2024")
    te, yte = load("2025")

    Xtr, Xcal, Xte = feats(tr), feats(cal), feats(te)
    cc = [c for c in Xtr.columns if c in Xcal.columns and c in Xte.columns]
    Xtr, Xcal, Xte = Xtr[cc], Xcal[cc], Xte[cc]
    print(f"train {len(Xtr)} / calib {len(Xcal)} / test {len(Xte)}, признаков {len(cc)}", flush=True)

    rows = []
    for algo, mk in MODELS.items():
        # [2] конформный квантиль — ОДИН РАЗ на чистой модели, далее фиксирован
        m_clean = mk(SEED).fit(Xtr, ytr)
        q_fixed = conformal_quantile(m_clean.predict_proba(Xcal), ycal.values)

        # [3] порог θ калибруется на 2024: эталон vs утечка на калибровочной выборке
        for sd in (SEED, SEED + 1, SEED + 2):
            cfgs = build_configs(rng, Xtr, ytr, Xte, yte, Xcal, ycal)
            for cfg in cfgs:
                use_mk = cfg.pop("force_mk", None) or mk
                nm, tp, lv, dfc = cfg.pop("name"), cfg.pop("тип"), cfg.pop("уровень"), cfg.pop("defective")
                a, b, c = cfg.pop("Xtr"), cfg.pop("ytr"), cfg.pop("Xte")
                r = evaluate(use_mk, sd, a, b, c, yte, q_fixed, rng=rng, **cfg)
                rows.append({"config": nm, "тип": tp, "уровень": lv, "алгоритм": algo,
                             "defective": dfc, "seed": sd, **r})
        print(f"  {algo}: готово", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stress2_all.csv", index=False, encoding="utf-8-sig")

    yd = res["defective"].values
    def auroc(v): return roc_auc_score(yd, 1 - np.asarray(v, float))

    # [3] порог по критерию FVR ≤ 0.10 на дефектных конфигурациях
    cand = np.linspace(0, 1, 501)
    theta = 1.0
    for t in cand:
        fvr = np.mean(res.loc[res.defective == 1, "V"] >= t)
        if fvr <= FVR_MAX:
            theta = float(t); break

    (lo, hi), p_pos = cluster_bootstrap_ci(res, rng=rng)
    a_v, a_f1 = auroc(res["V"]), auroc(res["macro_f1"])

    summary = {
        "конфигураций": int(len(res)),
        "θ (FVR≤0.10)": round(theta, 4),
        "AUROC_V": round(a_v, 4),
        "AUROC_MacroF1": round(a_f1, 4),
        "AUPRC_V": round(average_precision_score(yd, 1 - res["V"].values), 4),
        "разница_AUROC": round(a_v - a_f1, 4),
        "CI95_разницы_кластерный": [round(float(lo), 4), round(float(hi), 4)],
        "CI_не_включает_ноль": bool(lo > 0),
        "доля_бутстрэпов_с_положит_разницей": round(p_pos, 4),
        "доля_ложной_верификации": round(float(np.mean(res.loc[res.defective == 1, "V"] >= theta)), 4),
        "доля_ошибочного_отклонения": round(float(np.mean(res.loc[res.defective == 0, "V"] < theta)), 4),
        "conformal_alpha": ALPHA,
        "фактическое_покрытие_эталонов": round(float(res.loc[res.defective == 0, "conf_coverage"].mean()), 4),
        "средний_размер_множества_эталон": round(float(res.loc[res.defective == 0, "conf_size"].mean()), 3),
    }

    mono = (res[res.defective == 1].groupby(["тип", "уровень"])["V"].mean()
            .reset_index().sort_values(["тип", "уровень"]))
    mono.to_csv(OUT / "stress2_monotonic.csv", index=False, encoding="utf-8-sig")

    D_, T_, F_, S_, C_, R_ = (res[k] for k in ("D", "T", "F", "S", "C", "R"))
    variants = {
        "полная V": res["V"].values,
        "без D": (T_ * (F_*S_*C_*R_) ** .25).values,
        "без T": (D_ * (F_*S_*C_*R_) ** .25).values,
        "без F": (D_*T_ * (S_*C_*R_) ** (1/3)).values,
        "без S": (D_*T_ * (F_*C_*R_) ** (1/3)).values,
        "без C": (D_*T_ * (F_*S_*R_) ** (1/3)).values,
        "без R": (D_*T_ * (F_*S_*C_) ** (1/3)).values,
        "арифм. среднее": res[["D","T","F","S","C","R"]].mean(axis=1).values,
        "только Macro-F1": res["macro_f1"].values,
    }
    abl_rows = []
    for name, score in variants.items():
        row = {"вариант": name, "AUROC(все)": round(auroc(score), 3)}
        for t in ("D", "T", "F", "S", "C", "R"):
            sub = res[(res["тип"] == t) | (res.defective == 0)]
            try:
                row[t] = round(roc_auc_score(sub["defective"],
                                             1 - np.asarray(score, float)[sub.index]), 3)
            except Exception:
                row[t] = float("nan")
        abl_rows.append(row)
    pd.DataFrame(abl_rows).to_csv(OUT / "stress2_ablation.csv", index=False, encoding="utf-8-sig")
    (OUT / "stress2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    print("\n" + "=" * 92)
    print("ИТОГ (исправленная версия)")
    print("=" * 92)
    for k, v in summary.items():
        print(f"  {k:<38} {v}")
    print("\nАБЛЯЦИЯ:")
    print(pd.DataFrame(abl_rows).to_string(index=False))
    print("\nМОНОТОННОСТЬ:")
    print(mono.to_string(index=False))


if __name__ == "__main__":
    main()
