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
  v2pd   — v2 + patient-disjoint temporal split (sensitivity analysis, 16.09.2026):
           пациентка, встречающаяся более чем в одном году, остаётся только в самом
           раннем году (2023 > 2024 > 2025 по приоритету обучения); из последующих
           годов её циклы удаляются. Ключ пациентки — sha256(фамилия + инициалы +
           дата рождения), см. patient_key(); живёт только в памяти процесса и на диск
           не пишется. Строки без ключа (пустое ФИО) сопоставить нельзя — остаются.

Выход: (df, y, oo) — как в 08/10; 05/07/09b берут первые два.
Таблица этапов отбора и распределение классов пишутся в OUT_DIR/cohort_v2_stages.csv;
столбец removed_patient_overlap — сколько циклов снято по пересечению пациенток
(для когорт, кроме v2pd, всегда 0).
Исходные xlsx открываются строго на чтение.
"""
from __future__ import annotations

import hashlib
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

COHORTS = ("legacy", "base", "v2", "v2pd")
YEARS_ORDER = ("2023", "2024", "2025")          # приоритет: более ранний год держит пациентку

STAGES = ["rows", "real_rows", "oocytes_known", "amh_ok", "own_tvp", "stimulated",
          "removed_patient_overlap", "final"]

# Столбцы для ключа пациентки (первое совпадение по году; в 2025 ФИО записано полностью,
# в 2023/2024 — инициалами, поэтому ключ приводится к «Фамилия И О»).
FIO_PATS = [r"^фио$", r"^фио пациент", r"^ф\.?и\.?о\.? *пациент", r"^фио(?! мужа)"]
DOB_PATS = [r"^дата рожден", r"^год рождения$", r"^год рожд"]


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                         .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def find(cols, pats):
    return [c for c in cols if any(re.search(p, str(c).strip().lower()) for p in pats)]


def first_col(cols, pats, exact=()):
    for e in exact:  # точное совпадение с учётом регистра (пациентка vs партнёр)
        if e in cols:
            return e
    for p in pats:
        hit = [c for c in cols if re.search(p, str(c).strip().lower())]
        if hit:
            return hit[0]
    return None


def norm_text(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
            .str.replace("ё", "е").str.strip())


def name_key(s: pd.Series) -> pd.Series:
    """«Фамилия И О»: в 2023/2024 регистр хранит инициалы, в 2025 — полные имя и отчество,
    поэтому ключ сводится к фамилии и первым буквам остальных слов."""
    t = norm_text(s).str.replace(r"[.\-]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    def key(v: str) -> str:
        w = v.split(" ")
        if not w or w[0] in ("", "nan", "none"):
            return ""
        return " ".join([w[0]] + [x[0] for x in w[1:] if x])

    return t.map(key)


def patient_key(df: pd.DataFrame) -> pd.Series | None:
    """sha256(фамилия+инициалы + дата рождения) — только в памяти; None, если колонок нет.
    Строки без ФИО получают NaN. Наружу (csv/json/log) хэши никогда не пишутся."""
    fio_c = first_col(df.columns, FIO_PATS)
    dob_c = first_col(df.columns, DOB_PATS, exact=("Дата рождения", "Год рождения"))
    if fio_c is None:
        return None
    fio = name_key(df[fio_c])
    if dob_c is not None:
        dob = df[dob_c]
        if np.issubdtype(dob.dtype, np.datetime64):
            dob = dob.dt.strftime("%Y-%m-%d")
        dob = dob.astype(str)
        # полная дата, если она есть (в регистре — timestamp); иначе 4-значный год
        full = dob.str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        year = dob.str.extract(r"((?:19|20)\d{2})")[0]
        dob = full.fillna(year).fillna("")
    else:
        dob = pd.Series("", index=df.index)
    key = fio + "|" + dob
    ok = fio.ne("")
    h = key.map(lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    return h.where(ok)


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


_V2_KEYS: dict[str, set] = {}   # год → множество ключей пациенток финальной когорты v2 (только память)


def _v2_patient_keys(year: str) -> set:
    """Ключи пациенток финальной когорты v2 за год (кэш на процесс, без записи этапов)."""
    if year not in _V2_KEYS:
        df, _y, _oo = load_year(year, cohort="v2", write_stages=False)
        h = patient_key(df)
        _V2_KEYS[year] = set() if h is None else set(h.dropna())
    return _V2_KEYS[year]


def load_year(year, cohort: str | None = None, write_stages: bool = True):
    year = str(year)
    cohort = (cohort or os.environ.get("COHORT", "v2")).lower()
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; expected " + "|".join(COHORTS))

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
    if cohort in ("v2", "v2pd"):
        cat = df[CAT].astype(str).str.strip() if CAT in df.columns else pd.Series("", index=df.index)
        prot = df[PROT].astype(str).str.strip() if PROT in df.columns else pd.Series("", index=df.index)
        own = cat.str.startswith("ТВП") & ~cat.str.contains("ДО", regex=False)
        stim = ~prot.str.match(NATURAL) & ~prot.str.contains("Дуо", case=False, regex=False)
        m_own = m_amh & own
        m_stim = m_own & stim
    else:
        m_own = m_amh
        m_stim = m_amh

    n_removed = 0
    keep = m_stim
    if cohort == "v2pd":
        earlier = [y for y in YEARS_ORDER if y < year]
        if earlier:
            seen: set = set()
            for y in earlier:
                seen |= _v2_patient_keys(y)
            h = patient_key(df)
            if h is not None and seen:
                overlap = h.isin(seen) & m_stim
                n_removed = int(overlap.sum())
                keep = m_stim & ~overlap

    y = pd.Series(np.select([oo <= 3, oo <= 15], [0, 1], default=2), index=df.index)
    classes = np.bincount(y[keep].astype(int), minlength=3).tolist()

    if cohort == "v2pd":
        print(f"[cohort v2pd] {year}: stimulated={int(m_stim.sum())} "
              f"removed_patient_overlap={n_removed} final={int(keep.sum())} classes={classes}",
              flush=True)

    if write_stages:
        counts = {"rows": int(len(df)), "real_rows": int(real.sum()),
                  "oocytes_known": int(m_oo.sum()), "amh_ok": int(m_amh.sum()),
                  "own_tvp": int(m_own.sum()), "stimulated": int(m_stim.sum()),
                  "removed_patient_overlap": n_removed, "final": int(keep.sum())}
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
        if "removed_patient_overlap" not in old.columns:   # файл прежнего формата
            old["removed_patient_overlap"] = 0
        old = old[~((old["year"].astype(str) == row["year"]) & (old["cohort"] == row["cohort"]))]
        new = pd.concat([old, new], ignore_index=True)[cols]
    new.to_csv(p, index=False, encoding="utf-8-sig")


def stages_table(cohorts=("legacy", "base", "v2", "v2pd"), years=YEARS_ORDER) -> pd.DataFrame:
    for c in cohorts:
        for y in years:
            load_year(y, cohort=c)
    return pd.read_csv(OUT / "cohort_v2_stages.csv")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    only = os.environ.get("COHORT")
    cohorts = (only.lower(),) if only else ("legacy", "base", "v2", "v2pd")
    print(stages_table(cohorts).to_string(index=False))
