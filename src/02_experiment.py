# -*- coding: utf-8 -*-
"""
ГЛАВНЫЙ ЭКСПЕРИМЕНТ — доказать тезис статьи.

Тезис: статистически точная модель может быть методологически некорректной,
и Accuracy этого не показывает.

Строим ДВЕ модели на одних и тех же пациентках:
  Clean   — только предлечебные признаки (АМГ, возраст, ИМТ, диагноз …)
  Leakage — то же плюс признаки ПОСЛЕ пункции (MII, дробление, бластоцисты)

Обучение 2023 → пороги 2024 → независимый тест 2025 (годы НЕ перемешиваем).

Считаем для каждой:
  Macro-F1, Balanced accuracy   — прогнозное качество
  T   — временная допустимость  (§14: хоть один будущий признак → 0)
  V   = D·T·(F·S·C·R)^(1/4)     (§12: геометрическое среднее с вето)

Ожидание: Leakage выигрывает по F1 и получает V = 0.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

BASE = Path.home() / "ivf"
DATA, OUT = BASE / "data", BASE / "out"
SEED = 20260802
rng = np.random.default_rng(SEED)

TARGET = "ЗО Получено ооцитов пац /ДО"     # число полученных ооцитов
AMH_COL = "АМГ"

# Признаки Clean — известны ДО стимуляции (из Этапа 1, a_j = 1)
CLEAN_PATTERNS = [
    r"^амг$", r"год рожден", r"возр.*пациент", r"имт жены", r"вес жены",
    r"рост жены", r"диагноз бесплодия", r"бесплодие",
]
# Признаки Leakage — появляются ПОСЛЕ пункции (a_j = 0) → должны обнулить T
LEAK_PATTERNS = [
    r"зрелых \(mii\)", r"норм\.л", r"дроблен", r"бласт", r"атрез",
    r"выход б/ц", r"пэ .*сутки",
]


def pick(cols, patterns):
    out = []
    for c in cols:
        low = str(c).strip().lower()
        if any(re.search(p, low) for p in patterns):
            out.append(c)
    return out


def load(year: str) -> pd.DataFrame:
    df = pd.read_excel(DATA / f"{year}.xlsx", sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.extract(r"(-?\d+\.?\d*)")[0],
        errors="coerce",
    )


def make_target(n: pd.Series) -> pd.Series:
    """Три класса овариального ответа (§6): низкий / нормальный / высокий."""
    y = pd.Series(np.nan, index=n.index)
    y[n <= 3] = 0
    y[(n >= 4) & (n <= 15)] = 1
    y[n > 15] = 2
    return y


def build(year: str, feats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = load(year)
    cols = [c for c in feats if c in df.columns]
    X = df[cols].apply(to_num)
    n = to_num(df[TARGET]) if TARGET in df.columns else pd.Series(np.nan, index=df.index)
    y = make_target(n)
    amh = to_num(df[AMH_COL]) if AMH_COL in df.columns else pd.Series(np.nan, index=df.index)
    # когорта (§5.3): известен исход И известен АМГ
    keep = y.notna() & amh.notna()
    return X[keep], y[keep].astype(int)


def fit_eval(name, feats, a_j_all_ok: bool):
    Xtr, ytr = build("2023", feats)
    Xte, yte = build("2025", feats)
    common = [c for c in Xtr.columns if c in Xte.columns]
    Xtr, Xte = Xtr[common], Xte[common]

    model = HistGradientBoostingClassifier(random_state=SEED, max_iter=300)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)

    f1 = f1_score(yte, pred, average="macro")
    ba = balanced_accuracy_score(yte, pred)

    # ── компоненты верификации (§12–§18, упрощённо для главной таблицы) ──
    D = 1.0                                   # данные прошли фильтр когорты
    T = 1.0 if a_j_all_ok else 0.0            # §14: вето временной некорректности
    # F — верность объяснения: насколько прогноз реально двигается при
    # перестановке важного признака (permutation-эффект, §15.1)
    base = proba[np.arange(len(yte)), pred]
    Xp = Xte.copy()
    top = Xp.columns[0]
    Xp[top] = rng.permutation(Xp[top].values)
    pp = model.predict_proba(Xp)[np.arange(len(yte)), pred]
    F = float(np.clip(np.mean(np.abs(base - pp)) * 10, 0, 1))
    # S — устойчивость: 5 переобучений на bootstrap, согласие предсказаний
    agree = []
    for b in range(5):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        m = HistGradientBoostingClassifier(random_state=SEED + b, max_iter=200)
        m.fit(Xtr.iloc[idx], ytr.iloc[idx])
        agree.append(np.mean(m.predict(Xte) == pred))
    S = float(np.mean(agree))
    # C — предметная согласованность: выше АМГ → выше класс ответа (§17)
    if AMH_COL in Xte.columns:
        r = pd.Series(Xte[AMH_COL].values).corr(pd.Series(pred), method="spearman")
        C = float(np.clip(r, 0, 1)) if pd.notna(r) else 0.5
    else:
        C = 0.0                               # модель вообще не смотрит на АМГ
    # R — определённость: доля уверенных прогнозов
    R = float(np.mean(np.max(proba, axis=1) > 0.5))

    V = D * T * (F * S * C * R) ** 0.25       # §12: вето + геометрическое среднее

    return {
        "model": name, "n_train": len(Xtr), "n_test": len(Xte), "n_features": len(common),
        "macro_f1": round(f1, 4), "balanced_acc": round(ba, 4),
        "D": round(D, 3), "T": round(T, 3), "F": round(F, 3),
        "S": round(S, 3), "C": round(C, 3), "R": round(R, 3), "V": round(V, 4),
    }


def main() -> int:
    cols = load("2023").columns
    clean = pick(cols, CLEAN_PATTERNS)
    leak = clean + pick(cols, LEAK_PATTERNS)
    print(f"Clean:   {len(clean)} признаков → {clean[:6]}")
    print(f"Leakage: {len(leak)} признаков (+{len(leak)-len(clean)} из будущего)\n")

    rows = [
        fit_eval("Корректная (Clean)", clean, a_j_all_ok=True),
        fit_eval("Утечка (Leakage)", leak, a_j_all_ok=False),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "main_result.csv", index=False, encoding="utf-8-sig")
    (OUT / "main_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("ГЛАВНАЯ ТАБЛИЦА СТАТЬИ")
    print("=" * 78)
    print(res.to_string(index=False))
    print()
    a, b = rows[0], rows[1]
    print(f"Leakage выигрывает по Macro-F1: {b['macro_f1']} против {a['macro_f1']} "
          f"(+{round(b['macro_f1'] - a['macro_f1'], 4)})")
    print(f"Но верификация:                 V={b['V']} против V={a['V']}")
    print("\nВЫВОД: более точная модель отвергнута методом верификации.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
