# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2 — глубокая разведка данных.

Задача этого прогона — не построить модель, а понять, что в выгрузке вообще
есть: какие признаки связаны с исходом, какие связаны подозрительно сильно
(признак утечки), где скрыт сигнал, которым мы ещё не пользуемся.

ЧТО ДЕЛАЕТСЯ

  1. Сплошной обход всех колонок трёх лет: тип, заполненность, число
     уникальных значений. Без этого невозможно понять, что мы не используем.

  2. Одномерная связь каждого числового признака с числом полученных
     ооцитов: корреляция Спирмена (устойчива к выбросам и нелинейности) +
     доля пропусков. Для категориальных — размах доли гиперответа между
     категориями.

  3. АВТОМАТИЧЕСКИЙ ДЕТЕКТОР УТЕЧЕК. Проверка на «слишком хорошо»:
     если одиночный признак предсказывает исход с AUROC > 0,85, он почти
     наверняка регистрируется одновременно с исходом или после него.
     Так были пойманы program_cat (категория программы) и sperm_qual
     (спермограмма сдаётся в день пункции).

  4. Нелинейности: сравнение линейной корреляции с взаимной информацией.
     Большой разрыв означает, что связь есть, но не линейная — такой признак
     бесполезен для регрессии, но ценен для деревьев.

  5. Пороговый анализ АМГ: где именно проходят реальные границы классов
     ответа по данным, а не по таблице. Проверяется, совпадают ли
     эмпирические границы с клиническими 0,6 / 2,0 нг/мл.

Все выводы сохраняются таблицами, чтобы попасть в статью как есть.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 20260802
BASE = Path.home() / "ivf"
OUT = Path.home() / "ivf2" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Признаки, регистрируемые в день триггера/пункции и позже. Их присутствие
# в модели прогноза — утечка по построению (компонент T статьи №1).
POST_HOC_MARKERS = (
    "получено", "зрелых", "mii", "норм.л", "аномал", "атрез", "дроблен",
    "бласт", "заморож", "пэ ", "выход б/ц", "эмбр", "хгч", "берем", "роды",
    "триггер", "тфс", "твп", "среда", "катетер", "качество на пе",
    "гонал", "менур", "диферелин", "цетротид", "овитрель", "прегнил",
    "дата", "фио", "телефон", "врач", "эмбриолог", "кто заполнял",
)


def is_post_hoc(col: str) -> bool:
    s = str(col).lower()
    return any(k in s for k in POST_HOC_MARKERS)


def load_year(year: int) -> pd.DataFrame:
    d = pd.read_excel(BASE / "data" / f"{year}.xlsx")
    d.columns = [str(c).strip() for c in d.columns]
    d["__year"] = year
    return d


def main() -> None:
    years = [load_year(y) for y in (2023, 2024, 2025)]
    common = set(years[0].columns)
    for d in years[1:]:
        common &= set(d.columns)
    df = pd.concat([d[list(common)] for d in years], ignore_index=True)

    oo_col = next(c for c in df.columns
                  if "Получено ооцитов" in c and c.strip().startswith("∑"))
    amh_col = next(c for c in df.columns if "АМГ" in c.upper())

    df["_oo"] = pd.to_numeric(df[oo_col], errors="coerce")
    df["_amh"] = pd.to_numeric(df[amh_col], errors="coerce")
    punct = df[(df["_oo"] > 0) & df["_amh"].between(0, 30)].copy()

    print(f"колонок, общих для трёх лет: {len(common)}")
    print(f"строк всего: {len(df)}   состоявшихся пункций: {len(punct)}")

    # ── 1. Сплошной обход колонок ────────────────────────────────────────
    inv = []
    for c in sorted(common):
        s = punct[c]
        num = pd.to_numeric(s, errors="coerce")
        inv.append({
            "колонка": c,
            "заполнено_%": round(100 * s.notna().mean(), 1),
            "уникальных": int(s.nunique(dropna=True)),
            "числовая": bool(num.notna().mean() > 0.5),
            "предлечебная": not is_post_hoc(c),
        })
    inv_df = pd.DataFrame(inv).sort_values("заполнено_%", ascending=False)
    inv_df.to_csv(OUT / "mining_inventory.csv", index=False)
    pre = inv_df[inv_df["предлечебная"] & (inv_df["заполнено_%"] > 40)]
    print(f"\nпредлечебных колонок с заполненностью > 40 %: {len(pre)}")

    # ── 2-4. Связь с исходом, детектор утечек, нелинейность ──────────────
    y_oo = punct["_oo"].values
    y_high = (punct["_oo"] > 20).astype(int).values
    rows = []
    for c in sorted(common):
        if c in (oo_col,):
            continue
        v = pd.to_numeric(punct[c], errors="coerce")
        if v.notna().sum() < 150 or v.nunique() < 3:
            continue
        m = v.notna().values
        rho, p = spearmanr(v[m], y_oo[m])
        # одномерный AUROC по гиперответу — детектор «слишком хорошо»
        try:
            auc = roc_auc_score(y_high[m], v[m].values)
            auc = max(auc, 1 - auc)
        except Exception:  # noqa: BLE001
            auc = np.nan
        mi = float(mutual_info_regression(
            v[m].values.reshape(-1, 1), y_oo[m], random_state=SEED)[0])
        rows.append({
            "признак": c,
            "заполнено_%": round(100 * v.notna().mean(), 1),
            "spearman_rho": round(float(rho), 4),
            "p": float(p),
            "abs_rho": abs(float(rho)),
            "AUROC_гиперответ": round(float(auc), 4) if auc == auc else np.nan,
            "взаимная_информация": round(mi, 4),
            "предлечебный": not is_post_hoc(c),
            "подозрение_на_утечку": bool(auc == auc and auc > 0.85 and is_post_hoc(c)),
        })

    corr = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    corr.to_csv(OUT / "mining_correlations.csv", index=False)

    print("\n" + "=" * 74)
    print("ТОП ПРЕДЛЕЧЕБНЫХ ПРИЗНАКОВ по связи с числом ооцитов")
    print("=" * 74)
    top_pre = corr[corr["предлечебный"]].head(18)
    print(top_pre[["признак", "заполнено_%", "spearman_rho",
                   "AUROC_гиперответ", "взаимная_информация"]].to_string(index=False))

    print("\n" + "=" * 74)
    print("ДЕТЕКТОР УТЕЧЕК: одиночный признак с AUROC > 0,85")
    print("=" * 74)
    leaks = corr[(corr["AUROC_гиперответ"] > 0.85)].head(20)
    print(leaks[["признак", "AUROC_гиперответ", "предлечебный"]].to_string(index=False))

    # нелинейные: слабая корреляция, но заметная взаимная информация
    nonlin = corr[corr["предлечебный"] & (corr["abs_rho"] < 0.15)
                  & (corr["взаимная_информация"] > 0.02)]
    print("\n" + "=" * 74)
    print("НЕЛИНЕЙНАЯ СВЯЗЬ (слабая корреляция, но есть информация)")
    print("=" * 74)
    print(nonlin[["признак", "spearman_rho", "взаимная_информация"]]
          .head(12).to_string(index=False) if len(nonlin) else "  не обнаружено")

    # ── 5. Эмпирические границы АМГ против клинических ───────────────────
    print("\n" + "=" * 74)
    print("АМГ: наблюдаемый исход по децилям (эмпирические границы)")
    print("=" * 74)
    q = punct.copy()
    q["дециль"] = pd.qcut(q["_amh"], 10, duplicates="drop")
    g = q.groupby("дециль", observed=True).agg(
        n=("_oo", "size"),
        амг_от=("_amh", "min"), амг_до=("_amh", "max"),
        ооцитов_медиана=("_oo", "median"),
        доля_бедный=("_oo", lambda s: round((s <= 3).mean(), 3)),
        доля_гипер=("_oo", lambda s: round((s > 20).mean(), 3)),
    ).round(2)
    print(g.to_string())
    g.to_csv(OUT / "mining_amh_deciles.csv")

    # проверка клинических порогов: значим ли скачок исхода на границе
    print("\nпроверка клинических порогов АМГ (Манна–Уитни):")
    thr_rows = []
    for thr in (0.6, 1.0, 1.2, 2.0, 3.0, 4.0):
        a = punct[punct["_amh"] <= thr]["_oo"]
        b = punct[punct["_amh"] > thr]["_oo"]
        if len(a) < 40 or len(b) < 40:
            continue
        u, p = mannwhitneyu(a, b)
        thr_rows.append({"порог": thr, "n_ниже": len(a), "n_выше": len(b),
                         "медиана_ниже": a.median(), "медиана_выше": b.median(),
                         "p": p})
        print(f"  АМГ ≤ {thr}: медиана {a.median():.0f} клеток (n={len(a)})  "
              f"против {b.median():.0f} (n={len(b)})   p = {p:.2e}")
    pd.DataFrame(thr_rows).to_csv(OUT / "mining_amh_thresholds.csv", index=False)

    (OUT / "mining_meta.json").write_text(json.dumps({
        "колонок_общих": len(common),
        "строк": int(len(df)),
        "пункций": int(len(punct)),
        "предлечебных_колонок_заполненных": int(len(pre)),
        "признаков_с_подозрением_на_утечку": int(corr["подозрение_на_утечку"].sum()),
        "seed": SEED,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено в {OUT}")


if __name__ == "__main__":
    main()
