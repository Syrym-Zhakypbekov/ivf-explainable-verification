"""
cohort_v2 — единая загрузка регистра для всех скриптов конвейера.

Почему появился (16.09.2026):
  1. Все скрипты читали `sheet_name=0`. В 2025.xlsx первый лист — 'нов2024 (2)'
     (остаток 2024 года, 745 строк), а настоящий 'нов2025' (2 872 строки) — второй.
     «Тест 2025 = 592 цикла» в статье — это фрагмент 2024, а не 2025.
  2. Критерий «ооциты известны + АМГ 0–30» включал донорские (ДО), естественные /
     модифицированные циклы (ЕМЦ, ЕЦ, МОД …) и FET. Классы ответа искажены:
     2023 base = [1823, 766, 140], стимулированные аутологичные = [513, 660, 86].

Когорты (env COHORT, по умолчанию "v2"):
  legacy — sheet_name=0, старое правило (воспроизведение прежних чисел);
  base   — лист по имени `нов{year}`, старое правило (ооциты + АМГ);
  v2     — лист по имени + только стимулированные аутологичные циклы:
           `Категория программы` начинается с «ТВП» и не содержит «ДО»;
           `Протокол` не ∈ {ЕМЦ, ЕЦ, МОД, ЕЦ МОД, ЕЦ чистый, FET*} и не содержит «Дуо».

Выход: (df, y, oo) — как в 08/10; 05/07/09b берут первые два.
Таблица этапов отбора и распределение классов пишутся в OUT_DIR/cohort_v2_stages.csv.
Исходные xlsx открываются строго на чтение.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path.home() / "ivf"
DATA = BASE / "data"
OUT = Path(os.environ.get("OUT_DIR", str(BASE / "out")))

AMH = "АМГ"
CAT = "Категория программы"
PROT = "Протокол"
NATURAL = re.compile(r"^(ЕМЦ|ЕЦ|МОД|ЕЦ МОД|ЕЦ чистый|FET.*)$", re.I)
MIN_REAL_ROWS = 1000
N_HEAD_COLS = 12

STAGES = ["rows", "real_rows", "oocytes_known", "amh_ok", "own_tvp", "stimulated", "final"]


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                         .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def find(cols, pats):
    return [c for c in cols if any(re.search(p, str(c).strip().lower()) for p in pats)]


def real_rows_mask(df: pd.DataFrame) -> pd.Series:
    """Реальная строка — хоть что-то непустое в первых 12 колонках."""
    head = df.iloc[:, :N_HEAD_COLS]
    return head.notna().any(axis=1) & (head.astype(str).apply(lambda c: c.str.strip()) != "").any(axis=1)


def pick_sheet(xl: pd.ExcelFile, year: str) -> str:
    """Лист `нов{year}`; fallback — первый лист с ≥ 1 000 реальных строк; иначе первый."""
    want = f"нов{year}"
    for s in xl.sheet_names:
        if s.strip() == want:
            return s
    for s in xl.sheet_names:
        df = xl.parse(s)
        if int(real_rows_mask(df).sum()) >= MIN_REAL_ROWS:
            return s
    return xl.sheet_names[0]


def load_year(year, cohort: str | None = None, write_stages: bool = True):
    year = str(year)
    cohort = (cohort or os.environ.get("COHORT", "v2")).lower()
    if cohort not in ("legacy", "base", "v2"):
        raise ValueError(f"unknown cohort {cohort!r}; expected legacy|base|v2")

    xl = pd.ExcelFile(DATA / f"{year}.xlsx")
    sheet = xl.sheet_names[0] if cohort == "legacy" else pick_sheet(xl, year)
    df = xl.parse(sheet)
    df.columns = [str(c).strip() for c in df.columns]

    real = real_rows_mask(df)
    oo = pd.Series(np.nan, index=df.index)
    for c in find(df.columns, [r"получено ооцит"]):
        oo = oo.combine_first(to_num(df[c]))
    amh = to_num(df[AMH]) if AMH in df.columns else pd.Series(np.nan, index=df.index)

    m_oo = oo.notna()
    m_amh = m_oo & amh.between(0, 30)
    if cohort == "v2":
        cat = df[CAT].astype(str).str.strip() if CAT in df.columns else pd.Series("", index=df.index)
        prot = df[PROT].astype(str).str.strip() if PROT in df.columns else pd.Series("", index=df.index)
        own = cat.str.startswith("ТВП") & ~cat.str.contains("ДО", regex=False)
        stim = ~prot.str.match(NATURAL) & ~prot.str.contains("Дуо", case=False, regex=False)
        m_own = m_amh & own
        m_final = m_own & stim
    else:
        m_own = m_amh
        m_final = m_amh
    keep = m_final

    y = pd.Series(np.select([oo <= 3, oo <= 15], [0, 1], default=2), index=df.index)
    classes = np.bincount(y[keep].astype(int), minlength=3).tolist()

    if write_stages:
        counts = {"rows": int(len(df)), "real_rows": int(real.sum()),
                  "oocytes_known": int(m_oo.sum()), "amh_ok": int(m_amh.sum()),
                  "own_tvp": int(m_own.sum()), "stimulated": int(m_final.sum()),
                  "final": int(keep.sum())}
        row = {"year": year, "cohort": cohort, "sheet": sheet, **counts,
               "class0_poor": classes[0], "class1_normal": classes[1], "class2_high": classes[2]}
        _append_stage(row)

    return (df[keep].reset_index(drop=True),
            y[keep].reset_index(drop=True).astype(int),
            oo[keep].reset_index(drop=True))


def _append_stage(row: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cohort_v2_stages.csv"
    cols = ["year", "cohort", "sheet", *STAGES, "class0_poor", "class1_normal", "class2_high"]
    new = pd.DataFrame([row])[cols]
    if p.exists():
        old = pd.read_csv(p)
        old = old[~((old["year"].astype(str) == row["year"]) & (old["cohort"] == row["cohort"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(p, index=False, encoding="utf-8-sig")


def stages_table(cohorts=("legacy", "base", "v2"), years=("2023", "2024", "2025")) -> pd.DataFrame:
    for c in cohorts:
        for y in years:
            load_year(y, cohort=c)
    return pd.read_csv(OUT / "cohort_v2_stages.csv")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(stages_table().to_string(index=False))
