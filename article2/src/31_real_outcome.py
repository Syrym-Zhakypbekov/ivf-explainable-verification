# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2 — модель прогноза ФАКТИЧЕСКОГО овариального ответа.

Постановка задачи изменена по результатам проверки разметки
(см. docs/ANNOTATION-CRITIQUE.md):

    Экспертные метки оказались детерминированной функцией АМГ — совпадение
    с правилом classifyByAmh составило 3056 из 3056 при медианном времени
    решения 0,93 с. Обучение на них воспроизводило бы известную формулу.

Поэтому целевая переменная здесь — не метка, а РЕАЛЬНЫЙ ИСХОД: число
полученных ооцитов из выгрузок клиники (колонка «∑ Получено ооцитов»).
Предсказание делается по признакам, известным ДО начала стимуляции.

Вопрос исследования:
    даёт ли модель на предлечебных признаках прогноз лучше, чем
    клиническое правило по одному АМГ?

Дизайн:
  [1] Разделение по годам: 2023 обучение, 2024 калибровка/валидация,
      2025 независимый тест. Так же, как в статье №1 — сценарий реального
      применения, когда модель обучена на прошлом.
  [2] Бейзлайн — правило по АМГ (таблица Ш. К.), а не случайное угадывание.
  [3] Только предлечебные признаки. Всё, что регистрируется на пункции и
      позже, исключено как утечка (компонент T статьи №1).
  [4] Отчёт поклассово: классы неравны.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 20260802
BASE = Path.home() / "ivf"
OUT = Path.home() / "ivf2" / "out"
OUT.mkdir(parents=True, exist_ok=True)

OOCYTES = "∑ Получено ооцитов"
CLASSES = ["низкий (≤3)", "нормальный (4–15)", "высокий (>15)"]


def to_class(n: float) -> int:
    """Трёхклассовый овариальный ответ — как в статье №1."""
    if n <= 3:
        return 0
    if n <= 15:
        return 1
    return 2


def rule_amh(amh: float) -> int:
    """Клиническое правило: категория по АМГ → ожидаемый ответ.

    Клиническая таблица разметки даёт пять категорий; сводим к трём классам исхода:
    крайне низкий/низкий → низкий ответ, средний → нормальный,
    высокий/чрезмерно высокий → высокий.
    """
    if amh <= 0.60:
        return 0
    if amh <= 2.00:
        return 1
    return 2


def find(df: pd.DataFrame, *keys: str) -> str | None:
    for c in df.columns:
        s = str(c).lower()
        if all(k.lower() in s for k in keys):
            return c
    return None


def load_year(year: int) -> pd.DataFrame:
    d = pd.read_excel(BASE / "data" / f"{year}.xlsx")
    oo = [c for c in d.columns if "Получено ооцитов" in str(c) and str(c).strip().startswith("∑")]
    amh = find(d, "амг")
    if not oo or amh is None:
        raise RuntimeError(f"{year}: не найдены колонки исхода/АМГ")

    out = pd.DataFrame({
        "oocytes": pd.to_numeric(d[oo[0]], errors="coerce"),
        "amh": pd.to_numeric(d[amh], errors="coerce"),
    })
    # предлечебные признаки — только то, что известно ДО стимуляции
    for name, keys in {
        "age": ("возраст", "пациент"),
        "height": ("рост",),
        "weight": ("вес",),
        "bmi": ("имт",),
        "age_husband": ("возраст", "супруг"),
        "fsh": ("фсг",),
        "lh": ("лг",),
        "tsh": ("ттг",),
        "prolactin": ("пролактин",),
    }.items():
        c = find(d, *keys)
        out[name] = pd.to_numeric(d[c], errors="coerce") if c else np.nan

    out["year"] = year
    out = out[out["oocytes"].notna() & out["amh"].notna()]
    # физиологический фильтр — как в статье №1
    out = out[(out["amh"] >= 0) & (out["amh"] <= 30)]
    # ВАЖНО: нулевое число ооцитов в выгрузке означает не «пункция дала ноль
    # клеток», а «пункции в этой строке не было» — криоперенос, отменённый
    # цикл, донорская программа. В 2025 г. таких строк 2100 из 2295. Обучение
    # на них означало бы предсказание факта проведения пункции, а не
    # овариального ответа. Оставляем только состоявшиеся пункции.
    out = out[out["oocytes"] > 0]
    return out


FEATURES = ["amh", "age", "height", "weight", "bmi", "age_husband",
            "fsh", "lh", "tsh", "prolactin"]


def evaluate(name, y_true, y_pred) -> dict:
    return {
        "модель": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "kappa_quad": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def main() -> None:
    tr = pd.concat([load_year(2023)], ignore_index=True)
    va = load_year(2024)
    te = load_year(2025)

    for d in (tr, va, te):
        d["y"] = d["oocytes"].apply(to_class)
        d["y_rule"] = d["amh"].apply(rule_amh)

    print(f"обучение 2023: {len(tr)}   валидация 2024: {len(va)}   тест 2025: {len(te)}")
    print("распределение классов (тест 2025):",
          {CLASSES[i]: int((te['y'] == i).sum()) for i in range(3)})
    print(f"медиана полученных ооцитов: {te['oocytes'].median():.0f}")

    Xtr, ytr = tr[FEATURES].values, tr["y"].values
    Xte, yte = te[FEATURES].values, te["y"].values

    rows = [evaluate("ПРАВИЛО по АМГ", yte, te["y_rule"].values)]
    print(f"\nПРАВИЛО по АМГ (тест 2025):"
          f"  acc {rows[0]['accuracy']:.4f}"
          f"  macroF1 {rows[0]['macro_f1']:.4f}"
          f"  κ {rows[0]['kappa_quad']:.4f}")

    mdls = {
        "логистическая регрессия": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=SEED,
                               class_weight="balanced")),
        "случайный лес": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=3,
                                   class_weight="balanced")),
        "градиентный бустинг": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=400, learning_rate=0.06),
    }

    preds = {}
    for name, m in mdls.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        rows.append(evaluate(name, yte, p))
        print(f"  {name:24s} acc {rows[-1]['accuracy']:.4f}"
              f"  macroF1 {rows[-1]['macro_f1']:.4f}"
              f"  κ {rows[-1]['kappa_quad']:.4f}")

    # только АМГ — проверка, добавляют ли остальные признаки хоть что-то
    m_amh = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, random_state=SEED,
                                             class_weight="balanced"))
    m_amh.fit(tr[["amh"]].values, ytr)
    p_amh = m_amh.predict(te[["amh"]].values)
    rows.append(evaluate("модель только на АМГ", yte, p_amh))
    print(f"  {'модель только на АМГ':24s} acc {rows[-1]['accuracy']:.4f}"
          f"  macroF1 {rows[-1]['macro_f1']:.4f}"
          f"  κ {rows[-1]['kappa_quad']:.4f}")

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(OUT / "real_outcome_comparison.csv", index=False)

    best = res[~res["модель"].str.startswith("ПРАВИЛО")].iloc[0]["модель"]
    bp = preds.get(best, p_amh)
    pd.DataFrame(classification_report(yte, bp, target_names=CLASSES,
                                       output_dict=True, zero_division=0)).T \
        .to_csv(OUT / "real_outcome_report.csv")
    np.savetxt(OUT / "real_outcome_confusion.csv",
               confusion_matrix(yte, bp, labels=range(3)), fmt="%d", delimiter=",")
    np.savetxt(OUT / "real_outcome_confusion_rule.csv",
               confusion_matrix(yte, te["y_rule"].values, labels=range(3)),
               fmt="%d", delimiter=",")

    rule_f1 = rows[0]["macro_f1"]
    best_f1 = float(res[res["модель"] == best]["macro_f1"].iloc[0])
    meta = {
        "обучение_2023": int(len(tr)),
        "валидация_2024": int(len(va)),
        "тест_2025": int(len(te)),
        "признаков": len(FEATURES),
        "целевая": "фактическое число полученных ооцитов, 3 класса",
        "правило_macro_f1": rule_f1,
        "лучшая_модель": best,
        "лучшая_macro_f1": best_f1,
        "выигрыш_над_правилом": best_f1 - rule_f1,
        "seed": SEED,
    }
    (OUT / "real_outcome_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nлучшая модель: {best}")
    print(f"выигрыш над правилом по macro-F1: {best_f1 - rule_f1:+.4f}")
    print(f"сохранено в {OUT}")


if __name__ == "__main__":
    main()
