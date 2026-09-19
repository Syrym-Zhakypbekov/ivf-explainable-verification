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

  [5] ПОРОГ θ ВНЕ ВЫБОРКИ (16.09.2026, замечание рецензента). Прежде θ выбирался
      как наименьшее t с FVR ≤ 0.10 по V тех же конфигураций, что оценивались на
      тесте 2025, — т.е. подгонялся in-sample, и FVR на 2025 получался по
      построению. Теперь каждая конфигурация (то же обучение на 2023, тот же
      seed) оценивается ДВАЖДЫ: на калибровочном 2024 (V_cal) и на тесте 2025 (V).
      θ = min t с FVR ≤ 0.10 по V_cal; на 2025 при этом θ считаются FVR, FRR,
      чувствительность, precision, balanced accuracy. Старый in-sample θ пишется
      в summary как θ_insample_2025 (Appendix B, анализ чувствительности).
      Розыгрыши для калибровочной оценки идут из ОТДЕЛЬНОГО генератора, поэтому
      строки 2025 бит-в-бит совпадают с прежним прогоном.
      env THETA_MODE=insample — прежнее поведение (одна оценка, θ по 2025).
"""
from __future__ import annotations

import json
import os
import sys
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

BASE = Path(os.environ.get("IVF_BASE", str(Path(__file__).resolve().parents[1])))
DATA = BASE / "data"
OUT = Path(os.environ.get("OUT_DIR", str(BASE / "out")))   # out_v2 для пересчёта 16.09.2026
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_v2 import load_year                             # лист по имени + когорта COHORT
SEED = 20260802
ALPHA = 0.10
N_BOOT_S = 8
N_BOOT_CI = 3000
FVR_MAX = 0.10
THETA_MODE = os.environ.get("THETA_MODE", "oos").strip().lower()   # oos | insample  [5]
OOS = THETA_MODE != "insample"

AMH = "АМГ"
CLEAN = [r"^амг$", r"возр.*пациент", r"год рожден", r"имт жены", r"вес жены",
         r"рост жены", r"диагноз бесплодия", r"^бесплодие", r"возраст мужа"]


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                         .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def find(cols, pats):
    return [c for c in cols if any(re.search(p, str(c).strip().lower()) for p in pats)]


def load(year):
    """Делегирует cohort_v2.load_year (лист по имени, COHORT=v2|base|legacy)."""
    df, y, _oo = load_year(year)
    return df, y


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


def proba3(m, X, K=3):
    """predict_proba, выровненный на K классов по m.classes_ (16.09.2026: у дефектной конфигурации в обучении
    может не остаться класса «высокий ответ» → 2 столбца → IndexError в conformal_stats). Недостающим классам — 0."""
    p = m.predict_proba(X)
    cls = np.asarray(getattr(m, "classes_", np.arange(p.shape[1]))).astype(int)
    if p.shape[1] == K and list(cls) == list(range(K)):
        return p
    out = np.zeros((p.shape[0], K)); out[:, cls] = p
    return out


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

    proba = proba3(m, Xe)
    pred = np.argmax(proba, axis=1)

    D = 1.0 if D_ok else 0.0
    T = 1.0 if T_ok else 0.0

    base = proba[np.arange(len(Xe)), pred]
    eff = []
    for col in list(Xe.columns)[:5]:
        Xp = Xe.copy(); Xp[col] = rng.permutation(Xp[col].values)
        eff.append(np.mean(np.abs(base - proba3(m, Xp)[np.arange(len(Xe)), pred])))
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


def corrupt_D(X, kind, frac, rng):
    """Дефект D: порча данных оцениваемой выборки (одинаково для 2024 и 2025)."""
    b = X.copy()
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
    return b


def build_configs(rng, Xtr, ytr, Xte, yte, Xcal, ycal, rng_cal=None):
    """Возвращает список описаний конфигураций (без обучения).

    [5] Если передан rng_cal, у каждой конфигурации появляется ключ Xca — та же
    порча, наложенная на калибровочную выборку 2024 (для утечки T и дефектов D);
    для остальных Xca = Xcal. Все розыгрыши для 2024 берутся из rng_cal, чтобы
    поток rng (и значит строки 2025) не отличался от режима insample."""
    cfgs = [dict(name="M0 эталон", тип="—", уровень=0.0, defective=0,
                 Xtr=Xtr, ytr=ytr, Xte=Xte)]

    for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
        a, b = Xtr.copy(), Xte.copy()
        noise = np.sqrt(max(1e-6, 1/max(rho, 1e-3) - 1))
        a["future"] = ytr.values + rng.normal(0, noise, len(ytr))
        b["future"] = yte.values + rng.normal(0, noise, len(yte))
        cfg = dict(name=f"T утечка ρ={rho}", тип="T", уровень=rho, defective=1,
                   Xtr=a, ytr=ytr, Xte=b, T_ok=False)
        if rng_cal is not None:
            c = Xcal.copy()
            c["future"] = ycal.values + rng_cal.normal(0, noise, len(ycal))
            cfg["Xca"] = c
        cfgs.append(cfg)

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
        cfg = dict(name=f"D {kind}", тип="D", уровень=frac, defective=1,
                   Xtr=Xtr, ytr=ytr, Xte=corrupt_D(Xte, kind, frac, rng), D_ok=False)
        if rng_cal is not None:
            cfg["Xca"] = corrupt_D(Xcal, kind, frac, rng_cal)
        cfgs.append(cfg)

    if rng_cal is not None:
        for cfg in cfgs:
            cfg.setdefault("Xca", Xcal)
    return cfgs


def pick_theta(V_defective, cand=None):
    """Наименьшее t (сетка 501 точка), при котором доля ложной верификации ≤ FVR_MAX."""
    cand = np.linspace(0, 1, 501) if cand is None else cand
    V_defective = np.asarray(V_defective, float)
    for t in cand:
        if np.mean(V_defective >= t) <= FVR_MAX:
            return float(t)
    return 1.0


def rates_at(res, theta, col="V"):
    """FVR, FRR, чувствительность, precision, balanced accuracy при пороге theta (Table 7)."""
    v = res[col].values.astype(float)
    d = res["defective"].values == 1
    fvr = float(np.mean(v[d] >= theta))            # дефект признан корректным
    frr = float(np.mean(v[~d] < theta))            # эталон отклонён
    sens = 1.0 - fvr                               # дефект отклонён
    spec = 1.0 - frr                               # эталон верифицирован
    rejected = v < theta
    prec = float(np.sum(rejected & d) / max(1, np.sum(rejected)))   # среди отклонённых — дефекты
    return dict(FVR=round(fvr, 4), FRR=round(frr, 4), sensitivity=round(sens, 4),
                precision=round(prec, 4), balanced_accuracy=round((sens + spec) / 2, 4))


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
    print(f"режим порога: {'out-of-sample (θ по 2024, оценка на 2025)' if OOS else 'insample (θ по 2025)'}",
          flush=True)
    # [5] отдельный генератор для оценки на 2024 — поток rng для 2025 не меняется
    rng_cal = np.random.default_rng(SEED + 2024) if OOS else None

    rows = []
    for algo, mk in MODELS.items():
        # [2] конформный квантиль — ОДИН РАЗ на чистой модели, далее фиксирован
        m_clean = mk(SEED).fit(Xtr, ytr)
        q_fixed = conformal_quantile(proba3(m_clean, Xcal), ycal.values)

        # [3]/[5] порог θ калибруется на 2024: те же конфигурации оцениваются на калибровочной выборке
        for sd in (SEED, SEED + 1, SEED + 2):
            cfgs = build_configs(rng, Xtr, ytr, Xte, yte, Xcal, ycal, rng_cal=rng_cal)
            for cfg in cfgs:
                use_mk = cfg.pop("force_mk", None) or mk
                nm, tp, lv, dfc = cfg.pop("name"), cfg.pop("тип"), cfg.pop("уровень"), cfg.pop("defective")
                a, b, c = cfg.pop("Xtr"), cfg.pop("ytr"), cfg.pop("Xte")
                c_cal = cfg.pop("Xca", None)
                r_cal = {}
                if OOS:
                    # [5] та же конфигурация (то же обучение на 2023, тот же seed) на 2024
                    rc = evaluate(use_mk, sd, a, b, c_cal, ycal, q_fixed, rng=rng_cal, **cfg)
                    r_cal = {"V_cal": rc["V"], "macro_f1_cal": rc["macro_f1"],
                             "conf_coverage_cal": rc["conf_coverage"]}
                r = evaluate(use_mk, sd, a, b, c, yte, q_fixed, rng=rng, **cfg)
                rows.append({"config": nm, "тип": tp, "уровень": lv, "алгоритм": algo,
                             "defective": dfc, "seed": sd, **r, **r_cal})
        print(f"  {algo}: готово", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stress2_all.csv", index=False, encoding="utf-8-sig")

    yd = res["defective"].values
    def auroc(v): return roc_auc_score(yd, 1 - np.asarray(v, float))

    # [3] порог по критерию FVR ≤ 0.10 на дефектных конфигурациях — in-sample (по 2025), для Appendix B
    theta_insample = pick_theta(res.loc[res.defective == 1, "V"])
    theta_block = {"θ_insample_2025": round(theta_insample, 4)}
    if OOS:
        # [5] out-of-sample: θ по V_cal (2024), применяется к V (2025)
        theta = pick_theta(res.loc[res.defective == 1, "V_cal"])
        cal_rates = rates_at(res, theta, col="V_cal")
        te_rates = rates_at(res, theta, col="V")
        theta_block.update({
            "θ_calib_2024": round(theta, 4),
            "FVR_calib_2024": cal_rates["FVR"],
            "FRR_calib_2024": cal_rates["FRR"],
            "FVR_2025_at_θ_calib": te_rates["FVR"],
            "FRR_2025_at_θ_calib": te_rates["FRR"],
            "sensitivity_2025_at_θ_calib": te_rates["sensitivity"],
            "precision_2025_at_θ_calib": te_rates["precision"],
            "balanced_accuracy_2025_at_θ_calib": te_rates["balanced_accuracy"],
            "FVR_2025_at_θ_insample": rates_at(res, theta_insample, col="V")["FVR"],
            "FRR_2025_at_θ_insample": rates_at(res, theta_insample, col="V")["FRR"],
            "AUROC_V_cal_2024": round(auroc(res["V_cal"]), 4),
            "AUROC_MacroF1_cal_2024": round(auroc(res["macro_f1_cal"]), 4),
        })
    else:
        theta = theta_insample

    (lo, hi), p_pos = cluster_bootstrap_ci(res, rng=rng)
    a_v, a_f1 = auroc(res["V"]), auroc(res["macro_f1"])

    summary = {
        "конфигураций": int(len(res)),
        "режим_порога": "oos" if OOS else "insample",
        "θ (FVR≤0.10)": round(theta, 4),
        **theta_block,
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
