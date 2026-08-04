# -*- coding: utf-8 -*-
"""
Диагностика когорты — почему из ~94k записей осталось 377?

Для статьи нужна Таблица 1 (§30): сколько записей отсеивается на каждом
шаге и почему. Если потери непропорциональны, значит фильтр слишком строгий
и его надо ослабить, не нарушая методологию.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path.home() / "ivf"
DATA, OUT = BASE / "data", BASE / "out"


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.extract(r"(-?\d+\.?\d*)")[0],
        errors="coerce",
    )


def find(cols, *patterns):
    """Все колонки, чьё имя совпадает с любым из шаблонов."""
    out = []
    for c in cols:
        low = str(c).strip().lower()
        if any(re.search(p, low) for p in patterns):
            out.append(c)
    return out


for year in ("2023", "2024", "2025"):
    df = pd.read_excel(DATA / f"{year}.xlsx", sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    n0 = len(df)

    # все колонки, где может лежать число полученных ооцитов
    oo_cols = find(df.columns, r"получено ооцит")
    amh_cols = find(df.columns, r"^амг|^amh")

    print(f"\n=== {year} ===  строк: {n0}")
    print(f"колонки с 'получено ооцитов': {oo_cols}")
    print(f"колонки с АМГ:                {amh_cols}")

    # объединяем все источники ооцитов: пациентка могла идти по ЭКО, ИКСИ или ДО
    oo = pd.Series(np.nan, index=df.index)
    for c in oo_cols:
        v = to_num(df[c])
        oo = oo.fillna(v) if oo.isna().all() else oo.combine_first(v)
    amh = to_num(df[amh_cols[0]]) if amh_cols else pd.Series(np.nan, index=df.index)

    steps = [
        ("всего строк", n0),
        ("есть число ооцитов (любой источник)", int(oo.notna().sum())),
        ("  из них ооциты >= 0 (валидно)", int((oo >= 0).sum())),
        ("есть АМГ", int(amh.notna().sum())),
        ("  АМГ в диапазоне 0..30", int(((amh >= 0) & (amh <= 30)).sum())),
        ("ЕСТЬ И ТО, И ДРУГОЕ (когорта)", int((oo.notna() & amh.notna()).sum())),
        ("  + валидные диапазоны", int((oo.notna() & (oo >= 0) & amh.between(0, 30)).sum())),
    ]
    for label, n in steps:
        pct = 100 * n / n0 if n0 else 0
        print(f"  {label:<42} {n:>7}  ({pct:5.1f}%)")
