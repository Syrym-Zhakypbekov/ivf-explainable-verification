# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2, этап 3 — очистка нулевого АМГ и переоценка моделей.

ПОВОД

Разведка по децилям АМГ выявила аномалию, противоречащую клинической логике:

    дециль АМГ 0,00–0,14 нг/мл: медиана 6 ооцитов, доля гиперответа 0,08
    дециль АМГ 0,14–0,40 нг/мл: медиана 5 ооцитов, доля гиперответа 0,02

У пациенток с практически нулевым овариальным резервом ответ оказался лучше,
чем у следующей группы. Физиологически это невозможно. Наиболее вероятное
объяснение — нулём закодирован непроведённый анализ, а не измеренное нулевое
значение. Это классическая ошибка кодирования пропуска.

ЧТО ПРОВЕРЯЕТСЯ

  1. Гипотеза «АМГ = 0 есть пропуск»: сравнение группы с нулём и группы с
     малыми положительными значениями по исходу. Если нули ведут себя как
     произвольная смесь, а не как крайне низкий резерв, гипотеза принимается.
  2. Влияние очистки на качество моделей: те же задачи пересчитываются на
     очищенной выборке.
  3. Дополнительные признаки, отобранные разведкой: номер попытки, ЛГ,
     возраст мужа, а также препараты как маркеры протокола.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 20260802
BASE = Path.home() / "ivf"
OUT = Path.home() / "ivf2" / "out"
OUT.mkdir(parents=True, exist_ok=True)

NUM_FEATURES = {
    "amh": ("амг",), "age": ("возр", "пациент"), "height": ("рост", "жены"),
    "weight": ("вес", "жены"), "bmi": ("имт", "жены"),
    "age_husb": ("возраст", "мужа"), "bmi_husb": ("имт", "мужа"),
    "fsh": ("фсг",), "lh": ("лг",), "tsh": ("ттг",), "prolactin": ("прл",),
    "infert_dur": ("продолж", "бесплод"), "attempt": ("номер", "попытк"),
    "marriages": ("кол-во", "браков"), "hystero_n": ("кол-во", "гистеро"),
    "mar_test": ("мар", "тест"), "morph_pct": ("морф",), "ab_pct": ("а+в",),
    "in_persona": ("в persone",),
}
CAT_FEATURES = {
    "infertility": ("бесплодие",), "diagnosis1": ("диагноз", "бесплодия", "1"),
    "karyotype_w": ("кариотип", "жены"), "funding": ("услуги",),
    "msg_result": ("результат", "мсг"), "hystero_diag": ("диагноз", "гистеро"),
    "nationality": ("нац", "жены"),
}
NUM, CAT = list(NUM_FEATURES), list(CAT_FEATURES)


def find(df: pd.DataFrame, *keys: str) -> str | None:
    for c in df.columns:
        s = str(c).lower()
        if all(k.lower() in s for k in keys):
            return c
    return None


def load_raw(year: int) -> pd.DataFrame:
    d = pd.read_excel(BASE / "data" / f"{year}.xlsx")
    oo = [c for c in d.columns
          if "Получено ооцитов" in str(c) and str(c).strip().startswith("∑")]
    out = pd.DataFrame({
        "oocytes": pd.to_numeric(d[oo[0]], errors="coerce") if oo else np.nan})
    for name, keys in NUM_FEATURES.items():
        c = find(d, *keys)
        out[name] = pd.to_numeric(d[c], errors="coerce") if c else np.nan
    for name, keys in CAT_FEATURES.items():
        c = find(d, *keys)
        out[name] = d[c].astype(str).str.strip().str.lower() if c else "неизвестно"
    src = find(d, "источник", "клеток")
    out["_src"] = d[src].astype(str).str.lower() if src else ""
    out["year"] = year
    out = out[out["amh"].notna() & (out["amh"] <= 30) & (out["amh"] >= 0)]
    out = out[~out["_src"].str.contains("до|донор", na=False)]
    return out.drop(columns=["_src"])


def preproc(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=15,
                              sparse_output=False), cat_cols)])


def cv_binary(X, y, label: str, rows: list) -> None:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    zoo = {
        "бейзлайн": DummyClassifier(strategy="most_frequent"),
        "случайный лес": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=4, random_state=SEED,
            n_jobs=3, class_weight="balanced"),
        "градиентный бустинг": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=400, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=1.0),
    }
    print(f"\n{label}   n = {len(y)}, положительных {int(y.sum())}")
    for name, clf in zoo.items():
        pipe = Pipeline([("pre", preproc(NUM, CAT)), ("clf", clf)])
        pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
        try:
            proba = cross_val_predict(pipe, X, y, cv=cv,
                                      method="predict_proba", n_jobs=1)[:, 1]
            auc = roc_auc_score(y, proba)
        except Exception:  # noqa: BLE001
            auc = np.nan
        rows.append({"задача": label, "модель": name,
                     "accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"),
                     "auroc": auc})
        print(f"  {name:22s} acc {rows[-1]['accuracy']:.4f}"
              f"  macroF1 {rows[-1]['macro_f1']:.4f}  AUROC {auc:.4f}")


def main() -> None:
    raw = pd.concat([load_raw(y) for y in (2023, 2024, 2025)], ignore_index=True)
    punct = raw[raw["oocytes"] > 0].copy()

    # ── 1. Проверка гипотезы «нулевой АМГ есть пропуск» ──────────────────
    zero = punct[punct["amh"] == 0]
    tiny = punct[(punct["amh"] > 0) & (punct["amh"] <= 0.30)]
    low = punct[(punct["amh"] > 0.30) & (punct["amh"] <= 0.60)]

    print("=" * 74)
    print("ПРОВЕРКА: чем является АМГ = 0 — измерением или пропуском")
    print("=" * 74)
    for name, grp in (("АМГ = 0 ровно", zero),
                      ("АМГ 0,01–0,30", tiny),
                      ("АМГ 0,31–0,60", low)):
        if len(grp):
            print(f"  {name:16s} n = {len(grp):4d}   медиана {grp['oocytes'].median():5.1f}"
                  f"   доля бедных {(grp['oocytes'] <= 3).mean():.3f}"
                  f"   доля гипер {(grp['oocytes'] > 20).mean():.3f}")

    verdict = {}
    if len(zero) > 30 and len(tiny) > 30:
        u, p = mannwhitneyu(zero["oocytes"], tiny["oocytes"])
        print(f"\n  Манна–Уитни «АМГ = 0» против «0,01–0,30»: p = {p:.4f}")
        # если нули дают ответ ЛУЧШЕ, чем крайне низкий резерв — это пропуск
        higher = zero["oocytes"].median() > tiny["oocytes"].median()
        verdict = {"p": float(p), "медиана_нулей": float(zero["oocytes"].median()),
                   "медиана_малых": float(tiny["oocytes"].median()),
                   "нули_дают_ответ_выше": bool(higher)}
        print("  ВЫВОД: " + ("нули ведут себя не как крайне низкий резерв — "
                             "с высокой вероятностью это незаполненный анализ"
                             if higher else
                             "нули согласуются с крайне низким резервом"))

    # ── 2. Модели на очищенной выборке ───────────────────────────────────
    clean = punct[punct["amh"] > 0].copy()
    print(f"\nисходно пункций: {len(punct)}   после удаления АМГ = 0: {len(clean)}"
          f"   удалено: {len(punct) - len(clean)}")

    rows: list[dict] = []
    for tag, data in (("с нулями", punct), ("без нулей", clean)):
        X = data[NUM + CAT]
        cv_binary(X, (data["oocytes"] > 20).astype(int).values,
                  f"Риск гиперответа (>20), {tag}", rows)
        cv_binary(X, (data["oocytes"] <= 3).astype(int).values,
                  f"Риск бедного ответа (≤3), {tag}", rows)

    # регрессия числа ооцитов на очищенных данных
    print("\nЧисло ооцитов (регрессия), без нулей АМГ")
    Xc, yc = clean[NUM + CAT], clean["oocytes"].values
    pipe = Pipeline([("pre", preproc(NUM, CAT)),
                     ("reg", HistGradientBoostingRegressor(
                         random_state=SEED, max_iter=400, learning_rate=0.05,
                         max_leaf_nodes=15, l2_regularization=1.0))])
    pred = cross_val_predict(pipe, Xc, yc, cv=KFold(5, shuffle=True,
                                                    random_state=SEED), n_jobs=1)
    mae, r2 = mean_absolute_error(yc, pred), r2_score(yc, pred)
    rows.append({"задача": "Число ооцитов, без нулей", "модель": "градиентный бустинг",
                 "MAE": mae, "R2": r2})
    print(f"  градиентный бустинг     MAE {mae:.2f} клеток   R² {r2:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "clean_comparison.csv", index=False)
    (OUT / "clean_meta.json").write_text(json.dumps({
        "пункций_всего": int(len(punct)),
        "пункций_без_нулевого_АМГ": int(len(clean)),
        "удалено_нулей": int(len(punct) - len(clean)),
        "проверка_нулей": verdict,
        "seed": SEED,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено в {OUT}")


if __name__ == "__main__":
    main()
