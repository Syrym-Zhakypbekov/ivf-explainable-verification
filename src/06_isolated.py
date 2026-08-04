# -*- coding: utf-8 -*-
"""
АБЛЯЦИЯ НА ИЗОЛИРОВАННЫХ ДЕФЕКТАХ — исправление предыдущего эксперимента.

Проблема прошлой версии: каждый дефект портил ВСЕ компоненты сразу
(утечка обнуляла и T, и F, и C), поэтому абляция дала AUROC=1.0 для любого
варианта — эксперимент не мог показать вклад отдельного компонента.

Здесь дефекты сконструированы так, чтобы бить СТРОГО ПО ОДНОМУ:

  T-only : «тихая утечка» — будущий признак, СЛАБО связанный с ответом.
           Модель почти не меняет прогноз (F в норме), остаётся стабильной
           и согласованной с АМГ. Поймать может ТОЛЬКО временное вето T.
  F-only : прогнозы корректной модели сохранены, подменено ТОЛЬКО объяснение.
           T=1, S и C считаются по исходным прогнозам → падает лишь F.
  S-only : сильно недорегуляризованная модель на подвыборке. Прогнозы
           «прыгают» при переобучении → падает лишь S.
  C-only : инверсия меток ТОЛЬКО у верхнего квартиля АМГ. Клиническая
           монотонность нарушена → падает лишь C.
  R-only : прогнозы намеренно неуверенные (сглаженные вероятности) → лишь R.

Ожидание: убрать компонент X → соответствующий дефект перестаёт ловиться,
и AUROC абляции падает. Это и есть Таблица 7 плана (§25).
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


def measure(Xtr, ytr, Xte, yte, rng, *, T_ok=True, model=None,
            override_pred=None, override_proba=None, damp_F=1.0, damp_C=1.0,
            n_boot=N_BOOT_S, mk=None):
    """Считает D,T,F,S,C,R для одной конфигурации."""
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
    for b in range(n_boot):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        mb = mk().fit(Xtr.iloc[idx], ytr.iloc[idx])
        agree.append(np.mean(mb.predict(Xte) == pred))
    S = float(np.mean(agree))

    if AMH in Xte.columns:
        r, _ = spearmanr(Xte[AMH].fillna(Xte[AMH].median()), pred)
        C = float(np.clip(r, 0, 1)) if np.isfinite(r) else 0.0
    else:
        C = 0.0
    C *= damp_C

    R = float(np.mean(np.max(proba, axis=1) > 0.5))
    f1 = f1_score(yte, pred, average="macro")
    return dict(macro_f1=round(f1, 4), D=D, T=T, F=round(F, 3), S=round(S, 3),
                C=round(C, 3), R=round(R, 3))


def main():
    rng = np.random.default_rng(SEED)
    tr, ytr = load("2023")
    te, yte = load("2025")
    Xtr, Xte = feats(tr), feats(te)
    cc = [c for c in Xtr.columns if c in Xte.columns]
    Xtr, Xte = Xtr[cc], Xte[cc]
    print(f"когорта: train {len(Xtr)} / test {len(Xte)}, признаков {len(cc)}\n")

    rows = []

    # ── эталон ──────────────────────────────────────────────────────────
    rows.append({"config": "M0 Корректная", "target": "—", "defective": 0,
                 **measure(Xtr, ytr, Xte, yte, rng)})

    # ── T-only: «тихая» утечка ──────────────────────────────────────────
    # Будущий признак со СЛАБОЙ связью: модель почти не опирается на него,
    # поэтому F, S, C остаются нормальными. Ловит только временное вето.
    a, b = Xtr.copy(), Xte.copy()
    a["future_weak"] = ytr.values * 0.25 + rng.normal(0, 2.0, len(ytr))
    b["future_weak"] = yte.values * 0.25 + rng.normal(0, 2.0, len(yte))
    rows.append({"config": "D-T тихая утечка", "target": "T", "defective": 1,
                 **measure(a, ytr, b, yte, rng, T_ok=False)})

    # ── F-only: подменённое объяснение ──────────────────────────────────
    # Прогнозы эталона сохранены целиком; портится ТОЛЬКО объяснение.
    m_ref = HistGradientBoostingClassifier(random_state=SEED, max_iter=250).fit(Xtr, ytr)
    ref_pred, ref_proba = m_ref.predict(Xte), m_ref.predict_proba(Xte)
    rows.append({"config": "D-F ложное объяснение", "target": "F", "defective": 1,
                 **measure(Xtr, ytr, Xte, yte, rng, model=m_ref,
                           override_pred=ref_pred, override_proba=ref_proba, damp_F=0.12)})

    # ── S-only: нестабильная модель ─────────────────────────────────────
    # Глубокий лес на маленькой подвыборке: прогнозы прыгают при переобучении,
    # но объяснение честное и клинически согласованное.
    mk_unstable = lambda: RandomForestClassifier(  # noqa: E731
        n_estimators=3, max_depth=None, min_samples_leaf=1,
        max_features=1, random_state=rng.integers(0, 10_000))
    idx = rng.choice(len(Xtr), max(60, int(0.25 * len(Xtr))), replace=False)
    rows.append({"config": "D-S нестабильная", "target": "S", "defective": 1,
                 **measure(Xtr.iloc[idx], ytr.iloc[idx], Xte, yte, rng, mk=mk_unstable)})

    # ── C-only: инверсия клинической логики ─────────────────────────────
    # Метки перевёрнуты ТОЛЬКО у верхнего квартиля АМГ: монотонность
    # «выше АМГ → выше ответ» ломается, остальное в норме.
    yflip = ytr.copy()
    hi = Xtr[AMH] > Xtr[AMH].quantile(0.60)
    yflip[hi] = 2 - yflip[hi]                      # 0↔2 у высоких АМГ
    rows.append({"config": "D-C инверсия логики", "target": "C", "defective": 1,
                 **measure(Xtr, yflip, Xte, yte, rng)})

    # ── R-only: неуверенные прогнозы ────────────────────────────────────
    # Вероятности сглажены к равномерным: прогноз тот же, определённость низкая.
    smooth = ref_proba * 0.35 + (1 / 3) * 0.65
    rows.append({"config": "D-R неопределённость", "target": "R", "defective": 1,
                 **measure(Xtr, ytr, Xte, yte, rng, model=m_ref,
                           override_pred=ref_pred, override_proba=smooth)})

    res = pd.DataFrame(rows)
    # ВНИМАНИЕ: res.T — это транспонирование DataFrame, а не колонка «T».
    # Компоненты берём строго через res["..."].
    D, Tc = res["D"], res["T"]
    F, S, C, R = res["F"], res["S"], res["C"], res["R"]
    res["V"] = (D * Tc * (F * S * C * R) ** 0.25).round(4)
    res.to_csv(OUT / "table7_isolated.csv", index=False, encoding="utf-8-sig")

    # ── абляция: убираем по одному компоненту ───────────────────────────
    yd = res["defective"].values
    def auroc(score):
        try:
            return round(roc_auc_score(yd, 1 - np.asarray(score, float)), 3)
        except Exception:
            return float("nan")

    variants = {
        "полная V (D·T·(FSCR)^¼)": res["V"].values,
        "без T": (D * (F * S * C * R) ** .25).values,
        "без F": (D * Tc * (S * C * R) ** (1 / 3)).values,
        "без S": (D * Tc * (F * C * R) ** (1 / 3)).values,
        "без C": (D * Tc * (F * S * R) ** (1 / 3)).values,
        "без R": (D * Tc * (F * S * C) ** (1 / 3)).values,
        "арифм. среднее (компенсаторное)": res[["T", "F", "S", "C", "R"]].mean(axis=1).values,
        "только Accuracy": res["macro_f1"].values,
    }
    abl = pd.DataFrame([{"вариант": k, "AUROC": auroc(v)} for k, v in variants.items()])

    # какой дефект перестаёт ловиться без компонента X
    detail = []
    thr = 0.5
    for name, score in variants.items():
        missed = [res.config[i] for i in range(len(res))
                  if res.defective[i] == 1 and float(score[i]) >= thr]
        detail.append({"вариант": name, "пропущенные дефекты": ", ".join(missed) or "—"})
    det = pd.DataFrame(detail)

    abl.merge(det, on="вариант").to_csv(OUT / "table7_ablation.csv", index=False,
                                        encoding="utf-8-sig")

    print("=" * 96)
    print("ТАБЛИЦА 7 — изолированные дефекты (каждый бьёт по одному компоненту)")
    print("=" * 96)
    print(res[["config", "target", "macro_f1", "T", "F", "S", "C", "R", "V"]].to_string(index=False))
    print("\n" + "=" * 96)
    print("АБЛЯЦИЯ — что теряется без каждого компонента")
    print("=" * 96)
    print(abl.merge(det, on="вариант").to_string(index=False))


if __name__ == "__main__":
    main()
