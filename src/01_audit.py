# -*- coding: utf-8 -*-
"""
Этап 1 — аудит данных и ВРЕМЕННАЯ МАРКИРОВКА признаков.

Это фундамент всего метода Быкова: для каждого признака нужно знать, известен
ли он ДО решения (a_j = 1) или появляется ПОСЛЕ (a_j = 0). Один «будущий»
признак в модели → T(x) = 0 → V(x) = 0, какой бы ни была точность.

Скрипт ничего не выдумывает: он читает реальные заголовки трёх файлов,
сводит их в единый словарь и раскладывает по времени появления правилами,
которые видно и можно оспорить (их утверждает клинический соавтор).

Выход:
  OUT_DIR/data_dictionary.csv   — все колонки × 3 года, с меткой времени
  OUT_DIR/feature_timing.yaml   — a_j для модели (машинный вход Этапа 11)
  OUT_DIR/audit_summary.txt     — что нашли, для статьи

Лист выбирается по имени `нов{year}` (cohort_v2.pick_sheet); до 16.09.2026 читался
sheet_name=0, а в 2025.xlsx первый лист — остаток 2024 года.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

from cohort_v2 import pick_sheet

BASE = Path.home() / "ivf"
DATA = BASE / "data"
OUT = Path(os.environ.get("OUT_DIR", str(BASE / "out")))
OUT.mkdir(parents=True, exist_ok=True)
YEARS = {2023: "2023.xlsx", 2024: "2024.xlsx", 2025: "2025.xlsx"}

# ── Правила временной маркировки ────────────────────────────────────────
# Порядок важен: первое совпадение выигрывает. Каждое правило — это
# клиническое утверждение, которое соавтор может принять или отвергнуть.
RULES: list[tuple[str, str, str]] = [
    # (метка, регулярка по имени колонки, обоснование)
    ("service", r"^(кол-во|№|ф\.?и\.?о|фио|код|статус|врач|эмбриолог|дата)",
     "служебное/идентификатор — использовать запрещено (§17 TECHNICAL_ID)"),
    ("post_outcome", r"(получено ооцит|зрелых|mii|норм\.?л|аномал|атрез|дроблен|"
                     r"бласт|выход б/ц|пэ |заморож|эмбр|беременност|хгч|перенос|"
                     r"выжило|разморозк)",
     "результат пункции/эмбриологии — ЦЕЛЕВАЯ или пост-исход (утечка)"),
    ("after_stimulation", r"(триггер|день тригг|фолликул|суммарн|доза|длительност|"
                          r"стимуляц.*(день|доза)|э2|лг |фсг .*день)",
     "известно только после начала стимуляции (утечка для прогноза до неё)"),
    ("before_decision", r"(амг|amh|возраст|год рожден|имт|bmi|рост|вес|диагноз|"
                        r"анамнез|попытк|бесплод|afc|кол-во антральн)",
     "предлечебный признак — допустим (ESHRE: AMH/AFC — основные)"),
    ("protocol_choice", r"(протокол|вид стимуляц|программа|метод опл|схема)",
     "выбор врача: допустим ТОЛЬКО если прогноз строится ПОСЛЕ выбора"),
]


def classify(name: str) -> tuple[str, str]:
    low = str(name).strip().lower()
    for label, pattern, why in RULES:
        if re.search(pattern, low):
            return label, why
    return "unclassified", "требует ручного решения соавтора"


def main() -> int:
    frames: dict[int, pd.DataFrame] = {}
    for year, fname in YEARS.items():
        path = DATA / fname
        print(f"— читаю {fname} …", flush=True)
        xl = pd.ExcelFile(path)
        sheet = pick_sheet(xl, str(year))  # лист по имени `нов{year}`, как в cohort_v2
        df = xl.parse(sheet, nrows=0)  # только заголовки
        frames[year] = df
        print(f"  лист {sheet!r}: {len(df.columns)} колонок")

    # единый словарь: колонка × присутствие по годам
    rows = []
    seen: dict[str, dict] = {}
    for year, df in frames.items():
        for col in df.columns:
            key = str(col).strip()
            if not key or key.lower().startswith("unnamed"):
                continue
            rec = seen.setdefault(key, {"column": key, 2023: "", 2024: "", 2025: ""})
            rec[year] = "✓"
    for key, rec in seen.items():
        label, why = classify(key)
        rows.append({
            "column": key,
            "y2023": rec[2023], "y2024": rec[2024], "y2025": rec[2025],
            "in_all_years": "✓" if all(rec[y] for y in YEARS) else "",
            "timing": label,
            "a_j": 1 if label in ("before_decision",) else 0,
            "rationale": why,
        })

    dd = pd.DataFrame(rows).sort_values(["timing", "column"])
    dd.to_csv(OUT / "data_dictionary.csv", index=False, encoding="utf-8-sig")

    # машинный вход для Этапа 11
    with open(OUT / "feature_timing.yaml", "w", encoding="utf-8") as f:
        f.write("# a_j: 1 — признак известен ДО решения, 0 — появляется после.\n")
        f.write("# Утверждается клиническим соавтором. Один a_j=0 в модели → T=0 → V=0.\n")
        f.write("features:\n")
        for _, r in dd.iterrows():
            f.write(f'  - name: "{r["column"]}"\n')
            f.write(f'    timing: {r["timing"]}\n')
            f.write(f"    a_j: {r['a_j']}\n")

    # сводка для статьи
    counts = dd["timing"].value_counts()
    common = dd[dd["in_all_years"] == "✓"]
    lines = [
        "АУДИТ ДАННЫХ — Этап 1",
        "=" * 60,
        f"Всего уникальных колонок по трём годам: {len(dd)}",
        f"Присутствуют во ВСЕХ трёх годах:        {len(common)}",
        "",
        "Распределение по времени появления:",
    ]
    for label, n in counts.items():
        n_common = len(common[common["timing"] == label])
        lines.append(f"  {label:<20} {n:>4}   (из них общих для 3 лет: {n_common})")
    lines += [
        "",
        f"ДОПУСТИМЫХ для прогноза (a_j=1, общих для 3 лет): "
        f"{len(common[common['a_j'] == 1])}",
        "",
        "Примеры допустимых (before_decision):",
    ]
    for c in common[common["timing"] == "before_decision"]["column"].head(12):
        lines.append(f"  ✅ {c}")
    lines += ["", "Примеры УТЕЧКИ (post_outcome — то, что нельзя брать в признаки):"]
    for c in common[common["timing"] == "post_outcome"]["column"].head(12):
        lines.append(f"  ❌ {c}")
    lines += ["", "Не классифицировано автоматически (нужно решение соавтора):"]
    for c in dd[dd["timing"] == "unclassified"]["column"].head(25):
        lines.append(f"  ? {c}")

    text = "\n".join(lines)
    (OUT / "audit_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
