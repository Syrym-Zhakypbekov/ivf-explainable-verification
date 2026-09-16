# -*- coding: utf-8 -*-
"""
Этап 22 — характеристики когорты для TRIPOD+AI (Table 2b), только агрегаты.

Ответ на рецензию клинициста (round 2, § 4, Н-1/Н-2/Н-5/Н-6): состав когорты по годам —
циклы, уникальные пациентки, возраст/АМГ/ИМТ/длительность бесплодия (медиана [IQR]),
распределение протоколов (нормализованное), классы ответа, N = 0, пропуски по 11 эталонным
колонкам, отсев на шаге own-oocyte (донорские / FET / прочее), стимулированные циклы без
записи о пункции (вне когорты, для контекста).

Пациентка считается по sha256 от нормализованных фамилии + инициалов + даты рождения
(в 2025 ФИО записано полностью, в 2023/2024 — инициалами, поэтому ключ приведён к инициалам); хэши живут только
в памяти процесса и наружу не пишутся. В выходе — ТОЛЬКО числа. Ячейки с n < 5 не подавляются:
это агрегаты, а не индивидуальные записи.

Вход:  COHORT (env: v2|base|legacy), OUT_DIR (env); данные — cohort_v2.load_year.
Выход: OUT_DIR/cohort_characteristics.csv  (показатель × 2023/2024/2025/all)
       OUT_DIR/cohort_protocols.csv        (протокол × год: n и %)
       OUT_DIR/cohort_missing.csv          (эталонная колонка × год: % пропусков)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_v2 import (CAT, DATA, NATURAL, OUT, PROT, find, load_year, pick_sheet,
                       real_rows_mask, to_num)

YEARS = ("2023", "2024", "2025")
COHORT = os.environ.get("COHORT", "v2").lower()

# Столбцы, по которым ищем (первое совпадение по году). Имена берутся из словаря
# results/data_dictionary.csv; в 2025 часть полей записана иначе — поэтому шаблоны.
FIO_PATS = [r"^фио$", r"^фио пациент", r"^ф\.?и\.?о\.? *пациент", r"^фио(?! мужа)"]
DOB_PATS = [r"^дата рожден", r"^год рождения$", r"^год рожд"]
AGE_PATS = [r"^возр\.? *пациент", r"^возраст пациент", r"^возраст жен", r"^возраст$"]
AMH_PATS = [r"^амг$", r"^amh"]
BMI_PATS = [r"^имт жены", r"^имт$", r"^bmi"]
DUR_PATS = [r"продолж.*бесплод", r"длительн.*бесплод", r"бесплод.*лет"]

PROTOCOL_GROUPS = [
    ("antagonist", r"^ант|антаг|antag"),
    ("short", r"коротк|short"),
    ("long", r"длинн|long"),
    ("ppos_utrogestan", r"ppos|утрож|прогест|дюфаст"),
    ("mini", r"мини|mini|минимальн"),
]


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


def patient_hash(df: pd.DataFrame) -> pd.Series | None:
    """sha256(фамилия+инициалы + дата рождения) — только в памяти; None, если колонок нет."""
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


def med_iqr(x: pd.Series) -> str:
    x = x.dropna()
    if x.empty:
        return ""
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.1f} [{q1:.1f}-{q3:.1f}]"


def protocol_group(prot: pd.Series) -> pd.Series:
    p = norm_text(prot)
    out = pd.Series("other", index=prot.index)
    empty = p.eq("") | p.eq("nan") | p.eq("none")
    for name, pat in PROTOCOL_GROUPS:
        m = p.str.contains(pat, regex=True) & out.eq("other")
        out[m] = name
    out[empty] = "empty"
    return out


def cancelled_before_opu(year: str) -> tuple[int, int, int]:
    """Вне когорты: стимулированные аутологичные (ТВП без ДО, протокол не ЕЦ/FET/Дуо)
    строки без записи о числе ооцитов; плюс состав отсева own-oocyte (ДО / FET / прочее)
    среди строк с известными ооцитами и АМГ в норме."""
    xl = pd.ExcelFile(DATA / f"{year}.xlsx")
    df = xl.parse(pick_sheet(xl, year))
    df.columns = [str(c).strip() for c in df.columns]
    real = real_rows_mask(df)
    oo = pd.Series(np.nan, index=df.index)
    for c in find(df.columns, [r"получено ооцит"]):
        oo = oo.combine_first(to_num(df[c]))
    amh = to_num(df["АМГ"]) if "АМГ" in df.columns else pd.Series(np.nan, index=df.index)
    cat = df[CAT].astype(str).str.strip() if CAT in df.columns else pd.Series("", index=df.index)
    prot = df[PROT].astype(str).str.strip() if PROT in df.columns else pd.Series("", index=df.index)
    own = cat.str.startswith("ТВП") & ~cat.str.contains("ДО", regex=False)
    stim = ~prot.str.match(NATURAL) & ~prot.str.contains("Дуо", case=False, regex=False)
    m_amh = oo.notna() & amh.between(0, 30)
    dropped = m_amh & ~own
    n_cancel = int((real & own & stim & oo.isna()).sum())
    n_snyato = int((real & cat.str.contains("Снято ТВП", case=False, regex=False)).sum())
    n_beztvp = int((real & cat.str.contains("Без ТВП", case=False, regex=False)).sum())
    n_do = int((dropped & cat.str.contains("ДО", regex=False)).sum())
    not_do = dropped & ~cat.str.contains("ДО", regex=False)
    is_fet = cat.str.contains("FET|КРИО|разморо", case=False, regex=True) | prot.str.match(NATURAL)
    n_fet = int((not_do & is_fet).sum())
    is_cancel = cat.str.contains("Снято|Без ТВП", case=False, regex=True)
    n_notvp = int((not_do & ~is_fet & is_cancel).sum())
    n_other = int(dropped.sum()) - n_do - n_fet - n_notvp
    return n_cancel, n_snyato, n_beztvp, n_do, n_fet, n_notvp, n_other


def main() -> int:
    meta_p = OUT / "final_meta.json"
    features = json.loads(meta_p.read_text(encoding="utf-8"))["features"] if meta_p.exists() else []
    if not features:
        print(f"! {meta_p} не найден — пропуски по эталону не считаются", flush=True)

    rows: dict[str, dict] = {}
    prot_rows = []
    miss_rows = []
    hashes: dict[str, set] = {}
    all_parts = []

    def put(name, year, val):
        rows.setdefault(name, {})[year] = val

    for year in YEARS:
        df, y, oo = load_year(year, cohort=COHORT, write_stages=False)
        n = len(df)
        put("cycles", year, n)

        h = patient_hash(df)
        if h is not None:
            hv = h.dropna()
            per = hv.value_counts()
            hashes[year] = set(hv)
            put("patients_unique", year, int(per.size))
            put("cycles_with_patient_id", year, int(hv.size))
            put("patients_with_gt1_cycle", year, int((per > 1).sum()))
            put("cycles_of_patients_with_gt1_cycle", year, int(per[per > 1].sum()))
            put("cycles_per_patient_median", year, float(per.median()))
            put("cycles_per_patient_max", year, int(per.max()))
        else:
            put("patients_unique", year, "n/a (no name column)")

        age_c = first_col(df.columns, AGE_PATS)
        dob_c = first_col(df.columns, DOB_PATS, exact=("Год рождения",))
        age = to_num(df[age_c]) if age_c else pd.Series(np.nan, index=df.index)
        if age.notna().sum() < n * 0.5 and dob_c is not None:
            dob = to_num(df[dob_c]) if not np.issubdtype(df[dob_c].dtype, np.datetime64) \
                else pd.Series(df[dob_c].dt.year, index=df.index)
            age_alt = int(year) - dob.where(dob.between(1940, 2010))
            age = age.combine_first(age_alt)
        age = age.where(age.between(15, 60))
        put("age_median_iqr", year, med_iqr(age))
        put("age_source_column", year, age_c or (f"{year} - {dob_c}" if dob_c else ""))

        amh = to_num(df[first_col(df.columns, AMH_PATS)])
        put("amh_ng_ml_median_iqr", year, med_iqr(amh))

        bmi_c = first_col(df.columns, BMI_PATS)
        bmi = to_num(df[bmi_c]) if bmi_c else pd.Series(np.nan, index=df.index)
        bmi = bmi.where(bmi.between(12, 70))
        put("bmi_median_iqr", year, med_iqr(bmi))
        put("bmi_missing_pct", year, round(100 * bmi.isna().mean(), 1))

        dur_c = first_col(df.columns, DUR_PATS)
        dur = to_num(df[dur_c]) if dur_c else pd.Series(np.nan, index=df.index)
        dur = dur.where(dur.between(0, 40))
        put("infertility_duration_years_median_iqr", year, med_iqr(dur))
        put("infertility_duration_missing_pct", year, round(100 * dur.isna().mean(), 1))

        prot = df[PROT] if PROT in df.columns else pd.Series("", index=df.index)
        grp = protocol_group(prot)
        for g in [g for g, _ in PROTOCOL_GROUPS] + ["other", "empty"]:
            k = int((grp == g).sum())
            prot_rows.append({"protocol": g, "year": year, "n": k, "pct": round(100 * k / n, 1) if n else 0})
            put(f"protocol_{g}", year, k)

        cls = np.bincount(y.astype(int), minlength=3)
        put("class_low_0_3", year, int(cls[0]))
        put("class_normal_4_15", year, int(cls[1]))
        put("class_high_gt15", year, int(cls[2]))
        put("oocytes_zero", year, int((oo == 0).sum()))
        put("oocytes_median_iqr", year, med_iqr(oo))

        for f in features:
            col = f if f in df.columns else None
            miss = round(100 * df[col].isna().mean(), 1) if col else 100.0
            miss_rows.append({"feature": f, "year": year, "missing_pct": miss, "present": bool(col)})

        n_cancel, n_snyato, n_beztvp, n_do, n_fet, n_notvp, n_other = cancelled_before_opu(year)
        put("stimulated_autologous_without_retrieval_record", year, n_cancel)
        put("registry_rows_category_cancelled_tvp", year, n_snyato)
        put("registry_rows_category_no_tvp", year, n_beztvp)
        put("dropped_own_oocyte_step_donor", year, n_do)
        put("dropped_own_oocyte_step_fet_or_natural", year, n_fet)
        put("dropped_own_oocyte_step_cancelled_or_no_tvp", year, n_notvp)
        put("dropped_own_oocyte_step_other", year, n_other)

        all_parts.append(pd.DataFrame({"age": age, "amh": amh, "bmi": bmi, "dur": dur, "oo": oo,
                                       "y": y.values, "grp": grp.values}))

    # колонка all
    A = pd.concat(all_parts, ignore_index=True)
    put("cycles", "all", len(A))
    if hashes:
        union = set().union(*hashes.values())
        put("patients_unique", "all", len(union))
        overlap = set()
        ys = list(hashes)
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                overlap |= hashes[ys[i]] & hashes[ys[j]]
        put("patients_in_more_than_one_year", "all", len(overlap))
        for yr in ys:
            others = set().union(*(hashes[o] for o in ys if o != yr))
            put("patients_also_in_another_year", yr, len(hashes[yr] & others))
    put("age_median_iqr", "all", med_iqr(A["age"]))
    put("amh_ng_ml_median_iqr", "all", med_iqr(A["amh"]))
    put("bmi_median_iqr", "all", med_iqr(A["bmi"]))
    put("bmi_missing_pct", "all", round(100 * A["bmi"].isna().mean(), 1))
    put("infertility_duration_years_median_iqr", "all", med_iqr(A["dur"]))
    put("infertility_duration_missing_pct", "all", round(100 * A["dur"].isna().mean(), 1))
    put("oocytes_median_iqr", "all", med_iqr(A["oo"]))
    put("oocytes_zero", "all", int((A["oo"] == 0).sum()))
    for k, g in enumerate(["class_low_0_3", "class_normal_4_15", "class_high_gt15"]):
        put(g, "all", int((A["y"] == k).sum()))
    for g in [g for g, _ in PROTOCOL_GROUPS] + ["other", "empty"]:
        k = int((A["grp"] == g).sum())
        put(f"protocol_{g}", "all", k)
        prot_rows.append({"protocol": g, "year": "all", "n": k, "pct": round(100 * k / len(A), 1)})
    for name in ("stimulated_autologous_without_retrieval_record", "registry_rows_category_cancelled_tvp",
                 "registry_rows_category_no_tvp", "dropped_own_oocyte_step_donor",
                 "dropped_own_oocyte_step_fet_or_natural", "dropped_own_oocyte_step_cancelled_or_no_tvp",
                 "dropped_own_oocyte_step_other",
                 "patients_with_gt1_cycle", "cycles_of_patients_with_gt1_cycle", "cycles_with_patient_id"):
        vals = [rows.get(name, {}).get(yr) for yr in YEARS]
        if all(isinstance(v, (int, np.integer)) for v in vals):
            put(name, "all", int(sum(vals)))

    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame([{"indicator": k, **{yr: v.get(yr, "") for yr in (*YEARS, "all")}}
                        for k, v in rows.items()])
    tab.to_csv(OUT / "cohort_characteristics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prot_rows).to_csv(OUT / "cohort_protocols.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(miss_rows).to_csv(OUT / "cohort_missing.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(f"COHORT={COHORT} OUT={OUT}")
    print(tab.to_string(index=False))
    print(pd.DataFrame(prot_rows).pivot(index="protocol", columns="year", values="n").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
