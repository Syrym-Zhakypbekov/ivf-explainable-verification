# -*- coding: utf-8 -*-
"""
ПАКЕТ ДЛЯ ЭКСПЕРТНОЙ ОЦЕНКИ ВРАЧАМИ (по документу «Проверка врачами»).

Готовит 80 обезличенных случаев из независимой выборки 2025 года.
Для каждого случая: входные данные пациентки, прогноз системы,
3-5 объясняющих факторов, статус верификации.

ДЕИДЕНТИФИКАЦИЯ (жёстко):
  • берутся ТОЛЬКО клинические числовые признаки (АМГ, возраст, ИМТ, вес, рост);
  • ФИО, номера карт, даты, любые идентификаторы НЕ выгружаются в принципе —
    в выборку признаков они не попадают по построению (белый список CLEAN);
  • случай идентифицируется порядковым номером К-001 … К-080;
  • отдельным файлом сохраняется соответствие номер → индекс строки,
    чтобы при необходимости можно было вернуться к исходным данным.

Реальные метки (сколько ооцитов получено фактически) в анкету НЕ попадают —
иначе врач будет оценивать не объяснение, а совпадение с фактом.
Факт сохраняется отдельно, для последующего анализа согласия.
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
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

BASE = Path.home() / "ivf"
DATA = BASE / "data"
OUT = Path(os.environ.get("OUT_DIR", str(BASE / "out")))   # out_v2 для пересчёта 16.09.2026
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_v2 import load_year                             # лист по имени + когорта COHORT
SEED = 20260802
N_CASES = 80
ALPHA = 0.10
THETA = 0.778          # порог верификации, откалиброван на 2024 (см. final_meta.json)

AMH = "АМГ"
CLEAN = [r"^амг$", r"возр.*пациент", r"год рожден", r"имт жены", r"вес жены",
         r"рост жены", r"диагноз бесплодия", r"^бесплодие", r"возраст мужа"]

CLASS_NAME = {0: "бедный ответ (≤3 ооцитов)",
              1: "нормальный ответ (4–15 ооцитов)",
              2: "избыточный ответ (>15 ооцитов)"}


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                         .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def find(cols, pats):
    return [c for c in cols if any(re.search(p, str(c).strip().lower()) for p in pats)]


def load(year):
    """Делегирует cohort_v2.load_year (лист по имени, COHORT=v2|base|legacy)."""
    return load_year(year)


def feats(df):
    X = df[find(df.columns, CLEAN)].apply(to_num)
    return X.loc[:, X.notna().mean() > 0.05]


def conformal_quantile(proba_cal, y_cal, alpha=ALPHA):
    s = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, k))


def main():
    rng = np.random.default_rng(SEED)
    tr, ytr, _ = load("2023")
    cal, ycal, _ = load("2024")
    te, yte, oo_te = load("2025")

    Xtr, Xcal, Xte = feats(tr), feats(cal), feats(te)
    cc = [c for c in Xtr.columns if c in Xcal.columns and c in Xte.columns]
    Xtr, Xcal, Xte = Xtr[cc], Xcal[cc], Xte[cc]

    m = HistGradientBoostingClassifier(random_state=SEED, max_iter=180).fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    pred = np.argmax(proba, axis=1)
    q = conformal_quantile(m.predict_proba(Xcal), ycal.values)
    inset = (1.0 - proba) <= q
    setsize = inset.sum(axis=1).clip(min=1)

    # ── покейсовые объясняющие факторы: вклад признака через пермутацию ──
    base = proba[np.arange(len(Xte)), pred]
    contrib = {}
    for col in Xte.columns:
        Xp = Xte.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        contrib[col] = np.abs(base - m.predict_proba(Xp)[np.arange(len(Xte)), pred])
    contrib = pd.DataFrame(contrib)

    # ── глобальные компоненты (одинаковы для всей модели) ──
    F = float(np.clip(contrib.mean().mean() * 8, 0, 1))
    agree = []
    for b in range(8):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        mb = HistGradientBoostingClassifier(random_state=SEED + b, max_iter=180).fit(
            Xtr.iloc[idx], ytr.iloc[idx])
        agree.append(np.mean(mb.predict(Xte) == pred))
    S = float(np.clip((np.mean(agree) - 1 / 3) / (1 - 1 / 3), 0, 1))
    r, _ = spearmanr(Xte[AMH].fillna(Xte[AMH].median()), pred)
    C = float(np.clip((r + 1) / 2, 0, 1))

    # ── покейсовый V: D и T = 1 (данные валидны, утечки нет), R индивидуален ──
    R_case = 1.0 / setsize
    V_case = (F * S * C * R_case) ** 0.25

    def status(v, sz):
        if v >= THETA and sz == 1:
            return "верифицировано"
        if v >= THETA * 0.8:
            return "требует проверки"
        return "не верифицировано"

    # ── стратифицированная выборка: поровну по классам и по статусам ──
    df_all = pd.DataFrame({"idx": np.arange(len(Xte)), "pred": pred,
                           "V": V_case, "size": setsize})
    df_all["status"] = [status(v, s) for v, s in zip(df_all.V, df_all["size"])]
    picked = []
    for st in df_all.status.unique():
        for cl in sorted(df_all.pred.unique()):
            sub = df_all[(df_all.status == st) & (df_all.pred == cl)]
            n = min(len(sub), max(1, N_CASES // (df_all.status.nunique() * 3)))
            if n:
                picked.append(sub.sample(n, random_state=SEED))
    sel = pd.concat(picked).sample(frac=1, random_state=SEED).head(N_CASES).reset_index(drop=True)

    rows, key = [], []
    for n, r_ in sel.iterrows():
        i = int(r_.idx)
        case_id = f"К-{n + 1:03d}"
        top = contrib.iloc[i].sort_values(ascending=False).head(4)
        factors = "; ".join(
            f"{c} = {Xte.iloc[i][c]:.4g} (вклад {v:.3f})" if pd.notna(Xte.iloc[i][c])
            else f"{c} = нет данных (вклад {v:.3f})"
            for c, v in top.items())
        clinical = {c: (None if pd.isna(Xte.iloc[i][c]) else round(float(Xte.iloc[i][c]), 4))
                    for c in Xte.columns}
        rows.append({
            "Случай": case_id,
            **{f"Данные: {k}": v for k, v in clinical.items()},
            "Прогноз системы": CLASS_NAME[int(r_.pred)],
            "Уверенность прогноза": round(float(proba[i, int(r_.pred)]), 3),
            "Размер конформного множества": int(r_["size"]),
            "Объясняющие факторы": factors,
            "Показатель V": round(float(r_.V), 3),
            "Статус верификации": r_.status,
            "1. Прогноз соответствует данным? (да/нет/затрудняюсь)": "",
            "2. Факторы определены верно? (да/нет/частично)": "",
            "3. Есть клинически неверное утверждение? (да/нет)": "",
            "4. Упущен важный фактор? (да/нет; какой)": "",
            "5. Статус проверки выставлен верно? (да/нет)": "",
            "Оценка объяснения (1–5)": "",
            "Комментарий врача": "",
        })
        key.append({"Случай": case_id, "Индекс строки 2025": i,
                    "Факт: получено ооцитов": float(oo_te.iloc[i]),
                    "Факт: класс": CLASS_NAME[int(yte.iloc[i])],
                    "Прогноз": CLASS_NAME[int(r_.pred)],
                    "Совпало": bool(int(r_.pred) == int(yte.iloc[i]))})

    pd.DataFrame(rows).to_csv(OUT / "doctors_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(OUT / "doctors_key_CLOSED.csv", index=False, encoding="utf-8-sig")

    meta = {
        "случаев": len(rows),
        "источник": "2025.xlsx, независимая тестовая выборка",
        "порог θ": THETA,
        "conformal_alpha": ALPHA,
        "компоненты модели": {"F": round(F, 4), "S": round(S, 4), "C": round(C, 4)},
        "распределение статусов": sel.status.value_counts().to_dict(),
        "распределение прогнозов": {CLASS_NAME[k]: int(v)
                                    for k, v in sel.pred.value_counts().items()},
        "деидентификация": "выгружены только клинические числовые признаки из белого списка; "
                           "идентификаторы и даты отсутствуют в выборке по построению",
        "ВНИМАНИЕ": "doctors_key_CLOSED.csv содержит фактические исходы — врачам НЕ передавать",
    }
    (OUT / "doctors_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
