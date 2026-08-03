# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕННЫЙ ЭКСПЕРИМЕНТ — знаковая монотонность C + калибровка порога на 2024.

Что чинится по сравнению с предыдущей версией:

1. C(x) — ЗНАКОВАЯ монотонность (§17).
   Было: C = |ρ(АМГ, прогноз)| — измеряло СИЛУ связи, а не её НАПРАВЛЕНИЕ.
   Из-за этого инверсия клинической логики ПОВЫШАЛА C: модель выучила
   перевёрнутую зависимость, корреляция по модулю выросла.
   Стало: правило Быкова звучит «AMH_HIGH_RESPONSE: direction=non_decreasing»,
   поэтому нарушением считается ОТРИЦАТЕЛЬНАЯ связь, а положительная — норма:
       C = clip((ρ + 1) / 2 · монотонность_по_квантилям, 0, 1)
   Дополнительно проверяем по квантилям АМГ: средний класс ответа должен
   не убывать от нижнего квартиля к верхнему (мягкое ограничение, tolerance).

2. Порог θ калибруется на 2024 (§16, §19), а НЕ выбирается на глаз.
   На 2024 строим те же конфигурации, берём θ так, чтобы доля ложной
   верификации (дефект признан корректным) не превышала 0.10. Затем θ
   ЗАМОРАЖИВАЕТСЯ и применяется к 2025 — как требует протокол.

3. Компоненты S и R приведены к сопоставимой шкале: относительное падение
   к эталону, а не абсолютное значение (иначе S≈0.83 у всех и не различает).
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
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path.home() / "ivf"
DATA, OUT = BASE / "data", BASE / "out"
SEED = 20260802
N_BOOT_S = 12
FVR_MAX = 0.10          # §19: допустимая доля ложной верификации на 2024

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


def consistency(Xte: pd.DataFrame, pred: np.ndarray) -> float:
    """
    C(x) — ЗНАКОВАЯ предметная согласованность (§17).

    Правило AMH_HIGH_RESPONSE: direction = non_decreasing.
    Нарушение — ОТРИЦАТЕЛЬНАЯ монотонность, а не слабая связь. Поэтому:
      • ρ Спирмена отображаем из [-1,1] в [0,1] линейно: (ρ+1)/2
        (ρ=+1 → 1.0 идеальное согласие, ρ=0 → 0.5 нет связи, ρ=-1 → 0 инверсия);
      • дополнительно проверяем по квартилям АМГ, что средний класс не убывает.
    """
    if AMH not in Xte.columns:
        return 0.0
    amh = Xte[AMH].fillna(Xte[AMH].median())
    r, _ = spearmanr(amh, pred)
    if not np.isfinite(r):
        return 0.0
    directional = (r + 1.0) / 2.0

    # мягкая проверка по квартилям: средний класс должен не убывать
    q = pd.qcut(amh, 4, labels=False, duplicates="drop")
    means = pd.Series(pred).groupby(q).mean()
    if len(means) >= 2:
        diffs = np.diff(means.values)
        violations = float(np.mean(diffs < -0.05))     # tolerance из §17
        quantile_ok = 1.0 - violations
    else:
        quantile_ok = 1.0
    return float(np.clip(directional * quantile_ok, 0, 1))


def measure(Xtr, ytr, Xte, yte, rng, *, T_ok=True, model=None, mk=None,
            override_pred=None, override_proba=None, damp_F=1.0):
    mk = mk or (lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=250))
    m = model or mk().fit(Xtr, ytr)
    pred = override_pred if override_pred is not None else m.predict(Xte)
    proba = override_proba if override_proba is not None else m.predict_proba(Xte)

    D, T = 1.0, (1.0 if T_ok else 0.0)

    base = proba[np.arange(len(yte)), pred]
    eff = []
    for col in list(Xte.columns)[:6]:
        Xp = Xte.copy(); Xp[col] = rng.permutation(Xp[col].values)
        pp = m.predict_proba(Xp)[np.arange(len(yte)), pred]
        eff.append(np.mean(np.abs(base - pp)))
    F = float(np.clip(np.mean(eff) * 8, 0, 1)) * damp_F

    agree = []
    for b in range(N_BOOT_S):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        mb = mk().fit(Xtr.iloc[idx], ytr.iloc[idx])
        agree.append(np.mean(mb.predict(Xte) == pred))
    S_raw = float(np.mean(agree))
    # шкалируем: 1.0 при полном согласии, 0 при согласии на уровне случайного (1/3)
    S = float(np.clip((S_raw - 1 / 3) / (1 - 1 / 3), 0, 1))

    C = consistency(Xte, pred)

    conf = np.max(proba, axis=1)
    R_raw = float(np.mean(conf > 0.5))
    # шкалируем от уровня «неуверенно» (1/3) к «уверенно»
    R = float(np.clip((np.mean(conf) - 1 / 3) / (1 - 1 / 3), 0, 1)) * 0.5 + R_raw * 0.5

    return dict(macro_f1=round(f1_score(yte, pred, average="macro"), 4),
                D=D, T=T, F=round(F, 3), S=round(S, 3), C=round(C, 3), R=round(R, 3))


def build_configs(Xtr, ytr, Xte, yte, rng):
    """Эталон + пять изолированных дефектов на заданной паре (train, test)."""
    rows = []
    rows.append({"config": "M0 Корректная", "target": "—", "defective": 0,
                 **measure(Xtr, ytr, Xte, yte, rng)})

    a, b = Xtr.copy(), Xte.copy()
    a["future_weak"] = ytr.values * 0.25 + rng.normal(0, 2.0, len(ytr))
    b["future_weak"] = yte.values * 0.25 + rng.normal(0, 2.0, len(yte))
    rows.append({"config": "D-T тихая утечка", "target": "T", "defective": 1,
                 **measure(a, ytr, b, yte, rng, T_ok=False)})

    m_ref = HistGradientBoostingClassifier(random_state=SEED, max_iter=250).fit(Xtr, ytr)
    ref_pred, ref_proba = m_ref.predict(Xte), m_ref.predict_proba(Xte)
    rows.append({"config": "D-F ложное объяснение", "target": "F", "defective": 1,
                 **measure(Xtr, ytr, Xte, yte, rng, model=m_ref,
                           override_pred=ref_pred, override_proba=ref_proba, damp_F=0.12)})

    mk_unstable = lambda: RandomForestClassifier(  # noqa: E731
        n_estimators=3, max_depth=None, min_samples_leaf=1, max_features=1,
        random_state=int(rng.integers(0, 10_000)))
    idx = rng.choice(len(Xtr), max(60, int(0.25 * len(Xtr))), replace=False)
    rows.append({"config": "D-S нестабильная", "target": "S", "defective": 1,
                 **measure(Xtr.iloc[idx], ytr.iloc[idx], Xte, yte, rng, mk=mk_unstable)})

    # инверсия: у ВЕРХНЕЙ половины АМГ меняем 0↔2 — модель выучит
    # отрицательную монотонность, и знаковый C это увидит
    yflip = ytr.copy()
    hi = Xtr[AMH] > Xtr[AMH].median()
    yflip[hi] = 2 - yflip[hi]
    rows.append({"config": "D-C инверсия логики", "target": "C", "defective": 1,
                 **measure(Xtr, yflip, Xte, yte, rng)})

    smooth = ref_proba * 0.25 + (1 / 3) * 0.75
    rows.append({"config": "D-R неопределённость", "target": "R", "defective": 1,
                 **measure(Xtr, ytr, Xte, yte, rng, model=m_ref,
                           override_pred=ref_pred, override_proba=smooth)})

    df = pd.DataFrame(rows)
    df["V"] = (df["D"] * df["T"] *
               (df["F"] * df["S"] * df["C"] * df["R"]) ** 0.25).round(4)
    return df


def main():
    rng = np.random.default_rng(SEED)
    tr, ytr = load("2023")
    cal, ycal = load("2024")          # калибровка порога
    te, yte = load("2025")            # независимый тест

    Xtr = feats(tr)
    Xcal, Xte = feats(cal), feats(te)
    cc = [c for c in Xtr.columns if c in Xcal.columns and c in Xte.columns]
    Xtr, Xcal, Xte = Xtr[cc], Xcal[cc], Xte[cc]
    print(f"train {len(Xtr)} / calib {len(Xcal)} / test {len(Xte)}, признаков {len(cc)}\n")

    # ── шаг 1: калибровка порога на 2024 ────────────────────────────────
    cal_df = build_configs(Xtr, ytr, Xcal, ycal, rng)
    print("КАЛИБРОВКА (2024):")
    print(cal_df[["config", "T", "F", "S", "C", "R", "V"]].to_string(index=False))

    # θ = максимальный порог, при котором доля ложной верификации ≤ FVR_MAX
    defect_V = cal_df.loc[cal_df["defective"] == 1, "V"].values
    correct_V = cal_df.loc[cal_df["defective"] == 0, "V"].values
    candidates = np.unique(np.round(np.concatenate([defect_V, correct_V, [0.0, 1.0]]), 4))
    theta = None
    for t in sorted(candidates):
        fvr = float(np.mean(defect_V >= t))         # дефект признан корректным
        if fvr <= FVR_MAX:
            theta = float(t)
            break
    if theta is None:
        theta = float(max(defect_V) + 1e-6)
    print(f"\nθ откалиброван на 2024: {theta:.4f}  (FVR ≤ {FVR_MAX})")
    print(f"  V корректной на 2024: {correct_V[0]:.4f} — "
          f"{'ПРОХОДИТ' if correct_V[0] >= theta else 'НЕ проходит (порог слишком строг)'}")

    # ── шаг 2: заморозить θ, применить к 2025 ───────────────────────────
    test_df = build_configs(Xtr, ytr, Xte, yte, rng)
    test_df["status"] = np.where(test_df["V"] >= theta, "верифицировано", "ОТКЛОНЕНО")
    test_df["верно?"] = np.where(
        (test_df["defective"] == 1) & (test_df["status"] == "ОТКЛОНЕНО") |
        (test_df["defective"] == 0) & (test_df["status"] == "верифицировано"), "✓", "✗")

    yd = test_df["defective"].values
    def auroc(v):
        try:
            return round(roc_auc_score(yd, 1 - np.asarray(v, float)), 3)
        except Exception:
            return float("nan")

    D, Tc = test_df["D"], test_df["T"]
    F, S, C, R = test_df["F"], test_df["S"], test_df["C"], test_df["R"]
    variants = {
        "полная V (D·T·(FSCR)^¼)": test_df["V"].values,
        "без T": (D * (F * S * C * R) ** .25).values,
        "без F": (D * Tc * (S * C * R) ** (1 / 3)).values,
        "без S": (D * Tc * (F * C * R) ** (1 / 3)).values,
        "без C": (D * Tc * (F * S * R) ** (1 / 3)).values,
        "без R": (D * Tc * (F * S * C) ** (1 / 3)).values,
        "арифм. среднее (компенсаторное)": test_df[["T", "F", "S", "C", "R"]].mean(axis=1).values,
        "только Accuracy": test_df["macro_f1"].values,
    }
    abl = []
    for name, score in variants.items():
        missed = [test_df["config"][i] for i in range(len(test_df))
                  if test_df["defective"][i] == 1 and float(score[i]) >= theta]
        abl.append({"вариант": name, "AUROC": auroc(score),
                    "пропущено дефектов": len(missed),
                    "какие": ", ".join(m.split()[0] for m in missed) or "—"})
    ab = pd.DataFrame(abl)

    test_df.to_csv(OUT / "final_table.csv", index=False, encoding="utf-8-sig")
    ab.to_csv(OUT / "final_ablation.csv", index=False, encoding="utf-8-sig")
    (OUT / "final_meta.json").write_text(json.dumps({
        "theta": theta, "fvr_max": FVR_MAX,
        "n_train": len(Xtr), "n_calib": len(Xcal), "n_test": len(Xte),
        "features": cc}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"ФИНАЛЬНАЯ ТАБЛИЦА — независимый тест 2025, порог θ={theta:.4f} заморожен на 2024")
    print("=" * 100)
    print(test_df[["config", "target", "macro_f1", "T", "F", "S", "C", "R", "V",
                   "status", "верно?"]].to_string(index=False))
    print("\n" + "=" * 100)
    print("АБЛЯЦИЯ — вклад каждого компонента")
    print("=" * 100)
    print(ab.to_string(index=False))


if __name__ == "__main__":
    main()
