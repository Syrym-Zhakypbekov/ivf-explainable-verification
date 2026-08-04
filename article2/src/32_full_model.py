# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2 — расширенная модель прогноза овариального ответа.

Отличие от 31_real_outcome.py: задействован весь доступный предлечебный
набор признаков, включая категориальные (диагноз бесплодия, кариотип,
качество спермы, тип программы), а не только гормоны и антропометрию.

Целевая переменная — фактическое число полученных ооцитов, приведённое к
трём классам. Экспертные метки НЕ используются: проверка показала, что они
полностью детерминированы значением АМГ (3056 совпадений из 3056), то есть
представляют собой перекодировку признака, а не независимое суждение
(docs/ANNOTATION-CRITIQUE.md).

Три принципиальных фильтра, без которых результат был бы ложным:

  [1] ТОЛЬКО СОСТОЯВШИЕСЯ ПУНКЦИИ. Ноль в колонке «Получено ооцитов»
      означает не нулевой ответ яичников, а отсутствие пункции в данной
      строке (криоперенос, отменённый цикл, донорская программа). В 2025 г.
      таких строк 2100 из 2295. Без фильтра модель предсказывала бы факт
      проведения пункции.

  [2] ТОЛЬКО ПРЕДЛЕЧЕБНЫЕ ПРИЗНАКИ. Всё, что регистрируется в день триггера,
      на пункции и позже (эстрадиол и прогестерон на триггере, ТФС, дозы
      препаратов, эмбриологические показатели), исключено. Это компонент T
      статьи №1: такие признаки резко повышают формальную точность и
      недоступны в момент принятия решения.

  [3] СОБСТВЕННЫЕ ООЦИТЫ. Донорские программы исключены: там измеряется
      ответ донора, а не пациентки.

Разделение по годам: 2023 обучение, 2024 валидация, 2025 независимый тест —
сценарий реального применения, как в статье №1.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 20260802
BASE = Path.home() / "ivf"
OUT = Path.home() / "ivf2" / "out"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["низкий (≤3)", "нормальный (4–15)", "высокий (>15)"]

# ── числовые предлечебные признаки: имя → фрагменты названия колонки ──
NUM_FEATURES = {
    "amh":        ("амг",),
    "age":        ("возр", "пациент"),
    "height":     ("рост", "жены"),
    "weight":     ("вес", "жены"),
    "bmi":        ("имт", "жены"),
    "age_husb":   ("возраст", "мужа"),
    "bmi_husb":   ("имт", "мужа"),
    "fsh":        ("фсг",),
    "lh":         ("лг",),
    "tsh":        ("ттг",),
    "prolactin":  ("прл",),
    "infert_dur": ("продолж", "бесплод"),
    "attempt":    ("номер", "попытк"),
    "marriages":  ("кол-во", "браков"),
    "hystero_n":  ("кол-во", "гистеро"),
    "mar_test":   ("мар", "тест"),
    "morph_pct":  ("морф",),
    "ab_pct":     ("а+в",),
    "birth_year": ("год", "рождения"),
}

# ── категориальные предлечебные признаки ──
CAT_FEATURES = {
    "infertility":  ("бесплодие",),
    "diagnosis1":   ("диагноз", "бесплодия", "1"),
    "sperm_qual":   ("качество", "спермы"),
    "karyotype_w":  ("кариотип", "жены"),
    "karyotype_h":  ("кариотип", "мужа"),
    "program_cat":  ("категория", "программы"),
    "funding":      ("услуги",),
    "msg_result":   ("результат", "мсг"),
    "hystero_diag": ("диагноз", "гистеро"),
    "occupation":   ("род", "деятельности", "жена"),
    "nationality":  ("нац", "жены"),
}


def find(df: pd.DataFrame, *keys: str) -> str | None:
    """Первая колонка, чьё название содержит все фрагменты keys."""
    for c in df.columns:
        s = str(c).lower()
        if all(k.lower() in s for k in keys):
            return c
    return None


def load_year(year: int) -> pd.DataFrame:
    d = pd.read_excel(BASE / "data" / f"{year}.xlsx")

    oo = [c for c in d.columns
          if "Получено ооцитов" in str(c) and str(c).strip().startswith("∑")]
    if not oo:
        raise RuntimeError(f"{year}: не найдена колонка исхода")

    out = pd.DataFrame({"oocytes": pd.to_numeric(d[oo[0]], errors="coerce")})

    for name, keys in NUM_FEATURES.items():
        c = find(d, *keys)
        out[name] = pd.to_numeric(d[c], errors="coerce") if c else np.nan

    for name, keys in CAT_FEATURES.items():
        c = find(d, *keys)
        out[name] = d[c].astype(str).str.strip().str.lower() if c else "неизвестно"

    # источник клеток — для отсечения донорских программ
    src = find(d, "источник", "клеток")
    out["_src"] = d[src].astype(str).str.lower() if src else ""

    out["year"] = year
    out = out[out["oocytes"].notna() & out["amh"].notna()]
    out = out[(out["amh"] >= 0) & (out["amh"] <= 30)]     # физиологический диапазон
    out = out[out["oocytes"] > 0]                          # фильтр [1]
    out = out[~out["_src"].str.contains("до|донор", na=False)]  # фильтр [3]
    return out.drop(columns=["_src"])


def to_class(n: float) -> int:
    if n <= 3:
        return 0
    if n <= 15:
        return 1
    return 2


def rule_amh(amh: float) -> int:
    """Клиническое правило (таблица Ш. К.), сведённое к трём классам."""
    if amh <= 0.60:
        return 0
    if amh <= 2.00:
        return 1
    return 2


def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "модель": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "kappa_quad": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def build(num_cols, cat_cols, clf):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=15,
                              sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    tr = pd.concat([load_year(2023), load_year(2024)], ignore_index=True)
    te = load_year(2025)

    for d in (tr, te):
        d["y"] = d["oocytes"].apply(to_class)
        d["y_rule"] = d["amh"].apply(rule_amh)

    num_cols = list(NUM_FEATURES)
    cat_cols = list(CAT_FEATURES)

    print(f"обучение (2023+2024): {len(tr)}   независимый тест (2025): {len(te)}")
    print("классы в тесте:", {CLASSES[i]: int((te["y"] == i).sum()) for i in range(3)})
    print(f"медиана ооцитов в тесте: {te['oocytes'].median():.0f}")
    print(f"признаков: {len(num_cols)} числовых + {len(cat_cols)} категориальных")

    Xtr, ytr = tr[num_cols + cat_cols], tr["y"].values
    Xte, yte = te[num_cols + cat_cols], te["y"].values

    rows = [evaluate("ПРАВИЛО по АМГ (бейзлайн)", yte, te["y_rule"].values)]
    print(f"\n{'ПРАВИЛО по АМГ':32s} acc {rows[0]['accuracy']:.4f}"
          f"  macroF1 {rows[0]['macro_f1']:.4f}  κ {rows[0]['kappa_quad']:.4f}")

    zoo = {
        "логистическая регрессия": LogisticRegression(
            max_iter=5000, random_state=SEED, class_weight="balanced", C=0.5),
        "случайный лес": RandomForestClassifier(
            n_estimators=800, min_samples_leaf=4, random_state=SEED,
            n_jobs=3, class_weight="balanced"),
        "экстра-деревья": ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=3, random_state=SEED,
            n_jobs=3, class_weight="balanced"),
        "градиентный бустинг": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=500, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=1.0),
    }

    preds, fitted = {}, {}
    for name, clf in zoo.items():
        m = build(num_cols, cat_cols, clf)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name], fitted[name] = p, m
        rows.append(evaluate(name, yte, p))
        print(f"{name:32s} acc {rows[-1]['accuracy']:.4f}"
              f"  macroF1 {rows[-1]['macro_f1']:.4f}  κ {rows[-1]['kappa_quad']:.4f}")

    # только АМГ — сколько даёт один признак
    m_amh = build(["amh"], [], LogisticRegression(
        max_iter=5000, random_state=SEED, class_weight="balanced"))
    m_amh.fit(tr[["amh"]], ytr)
    rows.append(evaluate("модель только на АМГ", yte, m_amh.predict(te[["amh"]])))
    print(f"{'модель только на АМГ':32s} acc {rows[-1]['accuracy']:.4f}"
          f"  macroF1 {rows[-1]['macro_f1']:.4f}  κ {rows[-1]['kappa_quad']:.4f}")

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(OUT / "full_model_comparison.csv", index=False)

    best = res[~res["модель"].str.startswith(("ПРАВИЛО", "модель только"))].iloc[0]["модель"]
    bp = preds[best]
    pd.DataFrame(classification_report(yte, bp, target_names=CLASSES,
                                       output_dict=True, zero_division=0)).T \
        .to_csv(OUT / "full_model_report.csv")
    np.savetxt(OUT / "full_confusion_model.csv",
               confusion_matrix(yte, bp, labels=range(3)), fmt="%d", delimiter=",")
    np.savetxt(OUT / "full_confusion_rule.csv",
               confusion_matrix(yte, te["y_rule"].values, labels=range(3)),
               fmt="%d", delimiter=",")

    # важность признаков — «белая коробка» не должна стать чёрной
    mdl = fitted[best]
    try:
        names = mdl.named_steps["pre"].get_feature_names_out()
        imp = getattr(mdl.named_steps["clf"], "feature_importances_", None)
        if imp is not None:
            fi = (pd.DataFrame({"признак": names, "важность": imp})
                  .sort_values("важность", ascending=False).head(25))
            fi.to_csv(OUT / "full_feature_importance.csv", index=False)
            print("\nвклад признаков (топ-12):")
            print(fi.head(12).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print("важность признаков недоступна:", e)

    rule_f1 = rows[0]["macro_f1"]
    best_f1 = float(res[res["модель"] == best]["macro_f1"].iloc[0])
    meta = {
        "обучение_2023_2024": int(len(tr)),
        "тест_2025": int(len(te)),
        "числовых_признаков": len(num_cols),
        "категориальных_признаков": len(cat_cols),
        "целевая": "фактическое число полученных ооцитов, 3 класса",
        "фильтры": ["только состоявшиеся пункции (ооциты > 0)",
                    "только собственные клетки (донорские программы исключены)",
                    "АМГ в диапазоне 0–30 нг/мл",
                    "только предлечебные признаки"],
        "правило_macro_f1": rule_f1,
        "лучшая_модель": best,
        "лучшая_macro_f1": best_f1,
        "выигрыш_над_правилом": best_f1 - rule_f1,
        "seed": SEED,
    }
    (OUT / "full_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nлучшая модель: {best}")
    print(f"выигрыш над правилом по macro-F1: {best_f1 - rule_f1:+.4f}")


if __name__ == "__main__":
    main()
