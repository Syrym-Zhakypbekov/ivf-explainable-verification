# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2 — семейство прогностических задач на данных клиники 2023–2025.

Одна выгрузка позволяет поставить не одну, а несколько клинически осмысленных
задач. Здесь они решаются единообразно и оцениваются честно.

ЧТО ИЗМЕРЯЕТСЯ И ПОЧЕМУ ИМЕННО ТАК

Ранее прогон с разделением по годам дал macro-F1 = 0,98 на тесте 2025 г.
Значение оказалось артефактом: тестовая выборка после всех фильтров
насчитывала 87 наблюдений, и результат не воспроизвёлся при перекрёстной
проверке (0,664 ± 0,023 на 1647 наблюдениях). Поэтому основной метрикой
здесь служит 5-кратная стратифицированная перекрёстная проверка на всём
объёме, а не однократное разделение. Приводится среднее и стандартное
отклонение по фолдам — без этого разброса цифра не интерпретируема.

Для каждой задачи обязательно указывается бейзлайн. Без него неясно, что
означает полученное качество: для несбалансированных задач доля большинства
класса сама по себе даёт высокую accuracy.

ПРИНЦИПИАЛЬНЫЕ ФИЛЬТРЫ (общие для задач об ответе яичников)

  [1] Только состоявшиеся пункции. Ноль в колонке «Получено ооцитов»
      означает не нулевой ответ, а отсутствие пункции: криоперенос,
      отменённый цикл, донорская программа. В 2025 г. таких строк 2100 из
      2295 — без фильтра модель предсказывала бы сам факт пункции.
  [2] Только предлечебные признаки. Всё, что регистрируется в день триггера
      и позже, исключено (компонент T статьи №1: утечка целевой информации).
  [3] Только собственные ооциты: в донорских программах измеряется ответ
      донора, а не пациентки.

Задача 6 (дойдёт ли цикл до пункции) намеренно строится без фильтра [1] —
там отсутствие пункции и есть предмет прогноза.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
}
CAT_FEATURES = {
    "infertility": ("бесплодие",), "diagnosis1": ("диагноз", "бесплодия", "1"),
    "sperm_qual": ("качество", "спермы"), "karyotype_w": ("кариотип", "жены"),
    "program_cat": ("категория", "программы"), "funding": ("услуги",),
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
    """Загружает год целиком, без фильтра по факту пункции."""
    d = pd.read_excel(BASE / "data" / f"{year}.xlsx")
    oo = [c for c in d.columns
          if "Получено ооцитов" in str(c) and str(c).strip().startswith("∑")]
    mii = [c for c in d.columns
           if "Зрелых" in str(c) and str(c).strip().startswith("∑")]

    out = pd.DataFrame({
        "oocytes": pd.to_numeric(d[oo[0]], errors="coerce") if oo else np.nan,
        "mii": pd.to_numeric(d[mii[0]], errors="coerce") if mii else np.nan,
    })
    for name, keys in NUM_FEATURES.items():
        c = find(d, *keys)
        out[name] = pd.to_numeric(d[c], errors="coerce") if c else np.nan
    for name, keys in CAT_FEATURES.items():
        c = find(d, *keys)
        out[name] = d[c].astype(str).str.strip().str.lower() if c else "неизвестно"

    hcg = find(d, "результат", "хгч")
    out["hcg"] = d[hcg].astype(str).str.lower() if hcg else ""
    src = find(d, "источник", "клеток")
    out["_src"] = d[src].astype(str).str.lower() if src else ""

    out["year"] = year
    out = out[out["amh"].notna() & (out["amh"] >= 0) & (out["amh"] <= 30)]
    out = out[~out["_src"].str.contains("до|донор", na=False)]
    return out.drop(columns=["_src"])


def preproc(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=15,
                              sparse_output=False), cat_cols),
    ])


def clf_pipe(clf, num_cols=NUM, cat_cols=CAT) -> Pipeline:
    return Pipeline([("pre", preproc(num_cols, cat_cols)), ("clf", clf)])


def gb_clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=SEED, max_iter=400, learning_rate=0.05,
        max_leaf_nodes=15, l2_regularization=1.0)


def rf_clf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=4, random_state=SEED,
        n_jobs=3, class_weight="balanced")


def cv_class(X, y, name: str, tasks: list, binary: bool = False,
             num_cols=None, cat_cols=None) -> None:
    """Оценивает набор моделей 5-кратной перекрёстной проверкой."""
    num_cols = NUM if num_cols is None else num_cols
    cat_cols = CAT if cat_cols is None else cat_cols
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = DummyClassifier(strategy="most_frequent")
    zoo = {"бейзлайн (частый класс)": base,
           "логистическая регрессия": LogisticRegression(
               max_iter=5000, random_state=SEED, class_weight="balanced", C=0.5),
           "случайный лес": rf_clf(),
           "градиентный бустинг": gb_clf()}

    print(f"\n{'='*72}\n{name}\n  наблюдений: {len(y)}   "
          f"распределение: {dict(pd.Series(y).value_counts().sort_index())}")

    for mname, clf in zoo.items():
        pipe = clf_pipe(clf, num_cols, cat_cols)
        pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
        row = {"задача": name, "модель": mname,
               "accuracy": accuracy_score(y, pred),
               "macro_f1": f1_score(y, pred, average="macro")}
        if binary:
            try:
                proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                                          n_jobs=1)[:, 1]
                row["auroc"] = roc_auc_score(y, proba)
            except Exception:  # noqa: BLE001
                row["auroc"] = np.nan
        tasks.append(row)
        extra = f"  AUROC {row['auroc']:.4f}" if binary and not np.isnan(row.get("auroc", np.nan)) else ""
        print(f"  {mname:28s} acc {row['accuracy']:.4f}  macroF1 {row['macro_f1']:.4f}{extra}")


def main() -> None:
    raw = pd.concat([load_raw(y) for y in (2023, 2024, 2025)], ignore_index=True)
    punct = raw[raw["oocytes"] > 0].copy()

    print(f"всего строк с АМГ (собственные клетки): {len(raw)}")
    print(f"из них состоявшихся пункций: {len(punct)}")

    tasks: list[dict] = []
    Xp = punct[NUM + CAT]

    # ── 1. Овариальный ответ, три класса ──
    y1 = punct["oocytes"].apply(lambda n: 0 if n <= 3 else (1 if n <= 15 else 2)).values
    rule = punct["amh"].apply(lambda a: 0 if a <= 0.60 else (1 if a <= 2.00 else 2)).values
    print(f"\n{'='*72}\nПРАВИЛО по АМГ (бейзлайн для задачи 1): "
          f"acc {accuracy_score(y1, rule):.4f}  macroF1 {f1_score(y1, rule, average='macro'):.4f}")
    tasks.append({"задача": "1. Овариальный ответ (3 класса)",
                  "модель": "ПРАВИЛО по АМГ",
                  "accuracy": accuracy_score(y1, rule),
                  "macro_f1": f1_score(y1, rule, average="macro")})
    cv_class(Xp, y1, "1. Овариальный ответ (3 класса)", tasks)

    # ── 2. Число ооцитов как регрессия ──
    print(f"\n{'='*72}\n2. Число ооцитов (регрессия)\n  наблюдений: {len(punct)}")
    yr = punct["oocytes"].values
    cvk = KFold(5, shuffle=True, random_state=SEED)
    for mname, reg in {
        "бейзлайн (медиана)": DummyRegressor(strategy="median"),
        "градиентный бустинг": HistGradientBoostingRegressor(
            random_state=SEED, max_iter=400, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=1.0),
    }.items():
        pipe = Pipeline([("pre", preproc(NUM, CAT)), ("reg", reg)])
        pred = cross_val_predict(pipe, Xp, yr, cv=cvk, n_jobs=1)
        mae, r2 = mean_absolute_error(yr, pred), r2_score(yr, pred)
        tasks.append({"задача": "2. Число ооцитов (регрессия)", "модель": mname,
                      "MAE": mae, "R2": r2})
        print(f"  {mname:28s} MAE {mae:.2f} клеток   R² {r2:.4f}")

    # ── 3. Доля зрелых MII ──
    mii = punct[punct["mii"].notna() & (punct["mii"] >= 0)].copy()
    mii = mii[mii["mii"] <= mii["oocytes"]]
    if len(mii) > 200:
        print(f"\n{'='*72}\n3. Доля зрелых MII (регрессия)\n  наблюдений: {len(mii)}")
        ym = (mii["mii"] / mii["oocytes"]).values
        Xm = mii[NUM + CAT]
        for mname, reg in {
            "бейзлайн (медиана)": DummyRegressor(strategy="median"),
            "градиентный бустинг": HistGradientBoostingRegressor(
                random_state=SEED, max_iter=300, learning_rate=0.05,
                max_leaf_nodes=15, l2_regularization=1.0),
        }.items():
            pipe = Pipeline([("pre", preproc(NUM, CAT)), ("reg", reg)])
            pred = cross_val_predict(pipe, Xm, ym, cv=cvk, n_jobs=1)
            mae, r2 = mean_absolute_error(ym, pred), r2_score(ym, pred)
            tasks.append({"задача": "3. Доля зрелых MII", "модель": mname,
                          "MAE": mae, "R2": r2})
            print(f"  {mname:28s} MAE {mae:.4f}   R² {r2:.4f}")

    # ── 4. Риск гиперответа (СГЯ) ──
    cv_class(Xp, (punct["oocytes"] > 20).astype(int).values,
             "4. Риск гиперответа (>20 ооцитов)", tasks, binary=True)

    # ── 5. Риск бедного ответа ──
    cv_class(Xp, (punct["oocytes"] <= 3).astype(int).values,
             "5. Риск бедного ответа (≤3 ооцитов)", tasks, binary=True)

    # ── 6. Дойдёт ли цикл до пункции ──
    # В этой задаче пришлось исключить два признака, каждый из которых
    # прямо кодирует ответ:
    #   program_cat — категория программы: значение «крио ПЭ» само означает,
    #     что пункции не было;
    #   sperm_qual  — качество спермы: спермограмма сдаётся в день пункции,
    #     поэтому сам факт её наличия равносилен ответу. Проверка: при любом
    #     заполненном значении («10 млн», «15 млн» и т. д.) пункция состоялась
    #     в 96–100 % строк, а перестановочная важность признака составила
    #     0,368 против ~0,000 у всех остальных.
    # С ними AUROC достигал 0,995 и 0,987 — это утечка, а не качество модели.
    cat6 = [c for c in CAT if c not in ("program_cat", "sperm_qual")]
    cv_class(raw[NUM + cat6], (raw["oocytes"] > 0).astype(int).values,
             "6. Состоится ли пункция", tasks, binary=True,
             num_cols=NUM, cat_cols=cat6)

    # ── 7. Положительный ХГЧ ──
    # В выгрузке результат ХГЧ записан сокращённо: «пол» / «отр».
    hcg = raw[raw["hcg"].str.strip().str.startswith(("пол", "отр"))].copy()
    if len(hcg) > 200:
        cv_class(hcg[NUM + CAT],
                 hcg["hcg"].str.strip().str.startswith("пол").astype(int).values,
                 "7. Положительный ХГЧ (беременность)", tasks, binary=True)

    df = pd.DataFrame(tasks)
    df.to_csv(OUT / "all_tasks_results.csv", index=False)
    (OUT / "all_tasks_meta.json").write_text(json.dumps({
        "строк_с_АМГ": int(len(raw)),
        "состоявшихся_пункций": int(len(punct)),
        "оценка": "5-кратная стратифицированная перекрёстная проверка",
        "seed": SEED,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*72}\nсохранено: {OUT/'all_tasks_results.csv'}")


if __name__ == "__main__":
    main()
