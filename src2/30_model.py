# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2, эксперимент 1 — модель категории овариального ответа
на экспертной разметке 3056 циклов.

Вопрос исследования формулируется узко и честно:
    даёт ли обучаемая модель что-либо СВЕРХ клинического правила по АМГ,
    которое эксперт подтвердил в 99,9 % случаев?

Отсюда весь дизайн:

  [1] БЕЙЗЛАЙН — не случайное угадывание, а само правило classifyByAmh
      (таблица Ш. К.). Сравнение с константой или со случайностью здесь
      бессмысленно: правило уже работает и уже подтверждено экспертом.

  [2] ГРУППОВОЕ РАЗДЕЛЕНИЕ. Поле person_key в выгрузке пустое, то есть
      связать программы одной пациентки штатным способом нельзя. При этом
      807 групп записей полностью совпадают по (АМГ, возраст, ИМТ) — почти
      наверняка это повторные программы одних и тех же женщин. Разделение
      по строкам развело бы такие записи между train и test и завысило бы
      качество. Поэтому ключом группы служит кортеж (АМГ, возраст, ИМТ),
      а разделение выполняется StratifiedGroupKFold.

  [3] БЕЗ КАФ. Признак заполнен у 15 записей из 3056 (0,5 %) — включать
      его нельзя, хотя клинически он ценен.

  [4] ОТЧЁТ ПОКЛАССОВО. Классы неравны (336–1016), поэтому accuracy
      скрывает поведение на редких категориях.

  [5] ГРАНИЦЫ ПРАВИЛА. Отдельно измеряется качество на пограничных зонах
      АМГ (окрестности 0,6 и 2,0 нг/мл): именно там правило может ошибаться
      и именно там модель имеет шанс дать выигрыш.

Зерно фиксировано и совпадает с зерном статьи №1.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 20260802
BASE = Path.home() / "ivf2"
OUT = BASE / "out"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["extremely_low", "low", "medium", "high", "extremely_high"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── клиническое правило (таблица Ш. К.), перенесено из src/lib/engine/amh.ts ──
def rule_by_amh(amh: float) -> int:
    """Категория ответа по уровню АМГ. Границы — из рукописной таблицы."""
    if amh <= 0.30:
        return 0   # крайне низкий
    if amh <= 0.60:
        return 1   # низкий
    if amh <= 2.00:
        return 2   # средний
    if amh <= 3.00:
        return 3   # высокий
    return 4       # чрезмерно высокий


def load() -> pd.DataFrame:
    df = pd.read_csv(BASE / "labeled.csv")
    df = df[df["amh"].notna() & df["responder"].notna()].copy()
    df["y"] = df["responder"].map(CLS_IDX)
    df["y_rule"] = df["amh"].apply(rule_by_amh)
    # ключ группы: прокси пациентки (см. [2] в шапке)
    df["grp"] = (df["amh"].round(3).astype(str) + "|"
                 + df["age"].fillna(-1).astype(int).astype(str) + "|"
                 + df["bmi"].fillna(-1).round(1).astype(str))
    return df


FEATURES = ["amh", "fsh", "lh", "tsh", "prolactin", "testosterone",
            "age", "bmi", "duration_infertility", "attempt_number",
            "icsi", "tubal_factor", "male_factor", "unexplained",
            "anovulatory", "previous_pregnancy"]


def models():
    return {
        "логистическая регрессия": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED)),
        "случайный лес": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=3)),
        "градиентный бустинг": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=300),
    }


def main() -> None:
    df = load()
    X = df[FEATURES].values
    y = df["y"].values
    g = df["grp"].values

    print(f"записей: {len(df)}   групп (прокси пациенток): {df['grp'].nunique()}")
    print("распределение классов:",
          {CLASSES[i]: int((y == i).sum()) for i in range(5)})

    # ── качество правила на всей выборке — это и есть планка ──
    rule_acc = accuracy_score(y, df["y_rule"])
    rule_f1 = f1_score(y, df["y_rule"], average="macro")
    rule_kappa = cohen_kappa_score(y, df["y_rule"], weights="quadratic")
    print(f"\nПРАВИЛО по АМГ:  accuracy {rule_acc:.4f}   macro-F1 {rule_f1:.4f}"
          f"   κ(quad) {rule_kappa:.4f}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, oof_store = [], {}

    # базовая планка: самый частый класс
    for name, mdl in {"мажоритарный класс":
                      DummyClassifier(strategy="most_frequent"), **models()}.items():
        oof = np.full(len(y), -1)
        for tr, te in cv.split(X, y, groups=g):
            m = mdl.__class__(**mdl.get_params()) if hasattr(mdl, "get_params") else mdl
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        oof_store[name] = oof
        rows.append({
            "модель": name,
            "accuracy": accuracy_score(y, oof),
            "macro_f1": f1_score(y, oof, average="macro"),
            "kappa_quad": cohen_kappa_score(y, oof, weights="quadratic"),
            "согласие_с_правилом": float((oof == df["y_rule"].values).mean()),
        })
        print(f"  {name:24s} acc {rows[-1]['accuracy']:.4f}"
              f"  macroF1 {rows[-1]['macro_f1']:.4f}"
              f"  κ {rows[-1]['kappa_quad']:.4f}")

    rows.append({"модель": "ПРАВИЛО по АМГ", "accuracy": rule_acc,
                 "macro_f1": rule_f1, "kappa_quad": rule_kappa,
                 "согласие_с_правилом": 1.0})
    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(OUT / "model_comparison.csv", index=False)

    # ── лучшая модель: подробный отчёт ──
    best = res[res["модель"] != "ПРАВИЛО по АМГ"].iloc[0]["модель"]
    oof = oof_store[best]
    rep = classification_report(y, oof, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(OUT / "best_model_report.csv")
    np.savetxt(OUT / "confusion_model.csv",
               confusion_matrix(y, oof, labels=range(5)), fmt="%d", delimiter=",")
    np.savetxt(OUT / "confusion_rule.csv",
               confusion_matrix(y, df["y_rule"], labels=range(5)), fmt="%d", delimiter=",")

    # ── пограничные зоны АМГ: там, где правило рискует ошибаться ──
    zones = {
        "граница 0,30 (±0,10)": (0.20, 0.40),
        "граница 0,60 (±0,15)": (0.45, 0.75),
        "граница 2,00 (±0,40)": (1.60, 2.40),
        "граница 3,00 (±0,50)": (2.50, 3.50),
    }
    zrows = []
    for zname, (lo, hi) in zones.items():
        m = (df["amh"] >= lo) & (df["amh"] <= hi)
        if m.sum() < 20:
            continue
        zrows.append({
            "зона": zname, "n": int(m.sum()),
            "правило_acc": accuracy_score(y[m.values], df["y_rule"].values[m.values]),
            "модель_acc": accuracy_score(y[m.values], oof[m.values]),
        })
    zdf = pd.DataFrame(zrows)
    zdf["выигрыш"] = zdf["модель_acc"] - zdf["правило_acc"]
    zdf.to_csv(OUT / "boundary_zones.csv", index=False)

    print(f"\nлучшая модель: {best}")
    print("\nпограничные зоны АМГ:")
    print(zdf.to_string(index=False))

    disagree = int((df["y"].values != df["y_rule"].values).sum())
    meta = {
        "записей": int(len(df)),
        "групп_прокси_пациенток": int(df["grp"].nunique()),
        "признаков": len(FEATURES),
        "исключён_КАФ_заполнен": 15,
        "правило_accuracy": rule_acc,
        "правило_macro_f1": rule_f1,
        "лучшая_модель": best,
        "лучшая_accuracy": float(res.iloc[0]["accuracy"]) if res.iloc[0]["модель"] != "ПРАВИЛО по АМГ" else float(best_acc := res[res["модель"] == best]["accuracy"].iloc[0]),
        "расхождений_эксперта_с_правилом": disagree,
        "seed": SEED,
        "разделение": "StratifiedGroupKFold(5) по ключу (АМГ, возраст, ИМТ)",
    }
    (OUT / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nрасхождений эксперта с правилом: {disagree} из {len(df)}")
    print(f"результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()
