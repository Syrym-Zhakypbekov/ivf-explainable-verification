# -*- coding: utf-8 -*-
"""
ПОЛУСИНТЕТИЧЕСКИЙ ЭКСПЕРИМЕНТ (по документу А.А. Быкова «Эксперимент 1»).

БЕЗОПАСНОСТЬ ДАННЫХ — читать перед запуском:
  • Исходные xlsx НЕ изменяются: скрипт открывает их только на чтение.
  • Реальная метка овариального ответа сохраняется в колонке y_real и
    НИКОГДА не перезаписывается.
  • Синтетическая метка живёт в ОТДЕЛЬНОЙ колонке y_synth.
  • Обе колонки выгружаются рядом в labels_audit.csv — в любой момент видно,
    какая метка настоящая, какая сгенерирована, и совпадают ли они.
  • Модели полусинтетического контура обучаются ТОЛЬКО на y_synth; ни одна
    строка результатов основного эксперимента не перезаписывается — вывод
    идёт в отдельные файлы semisynth_*.

Зачем эксперимент: на реальных данных неизвестно, какое объяснение
ПРАВИЛЬНОЕ. Здесь механизм задан формулой, поэтому эталонное объяснение
известно заранее, и верность объяснения можно измерить численно.

Формула (из документа):
    h(x) = 1.5·z_AMH − 0.6·z_Age + 0.4·z_BMI + 0.7·z_AMH·I_PCOS + ε

Заранее известно:
    S* = {АМГ (+), возраст (−), ИМТ (+), взаимодействие АМГ×СПКЯ}
    остальные признаки — шумовые.
"""
from __future__ import annotations

import json
import os
import sys
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("IVF_BASE", str(Path(__file__).resolve().parents[1])))
DATA = BASE / "data"
OUT = Path(os.environ.get("OUT_DIR", str(BASE / "out")))   # out_v2 для пересчёта 16.09.2026
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_v2 import load_year                             # лист по имени + когорта COHORT
SEED = 20260802
rng = np.random.default_rng(SEED)

AMH = "АМГ"
CLEAN = [r"^амг$", r"возр.*пациент", r"год рожден", r"имт жены", r"вес жены",
         r"рост жены", r"диагноз бесплодия", r"^бесплодие", r"возраст мужа"]

# Коэффициенты формулы — ровно из документа Быкова
BETA = {"amh": 1.5, "age": -0.6, "bmi": 0.4, "amh_pcos": 0.7}
NOISE_SD = 0.5


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False)
                         .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def find(cols, pats):
    return [c for c in cols if any(re.search(p, str(c).strip().lower()) for p in pats)]


def load(year):
    """Делегирует cohort_v2.load_year (лист по имени, COHORT=v2|base|legacy)."""
    return load_year(year)


def feats(df):
    X = df[find(df.columns, CLEAN)].apply(to_num)
    return X.loc[:, X.notna().mean() > 0.05]


def z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    m, sd = s.median(), s.std(ddof=0)
    return ((s - m) / (sd if sd and np.isfinite(sd) else 1.0)).fillna(0.0)


def pick(cols, pat):
    m = find(cols, [pat])
    return m[0] if m else None


def make_synthetic(X: pd.DataFrame, df_raw: pd.DataFrame, y_real: pd.Series):
    """
    Строит СИНТЕТИЧЕСКУЮ метку по формуле документа.
    Возвращает (y_synth, служебная информация) — реальная метка не трогается.
    """
    c_amh = pick(X.columns, r"^амг$")
    c_age = pick(X.columns, r"возр.*пациент") or pick(X.columns, r"год рожден")
    c_bmi = pick(X.columns, r"имт жены")
    if not (c_amh and c_age and c_bmi):
        raise SystemExit(f"не найдены базовые признаки: amh={c_amh} age={c_age} bmi={c_bmi}")

    z_amh, z_age, z_bmi = z(X[c_amh]), z(X[c_age]), z(X[c_bmi])

    # СПКЯ: ищем в текстовых полях диагнозов
    diag_cols = [c for c in df_raw.columns
                 if re.search(r"диагноз|бесплодие", str(c).lower())]
    pcos = pd.Series(False, index=X.index)
    for c in diag_cols:
        txt = df_raw[c].astype(str).str.lower()
        pcos |= txt.str.contains(r"спкя|поликистоз|pcos", regex=True, na=False)
    pcos_share = float(pcos.mean())
    if pcos_share < 0.02:
        # СПКЯ в данных почти нет — берём верхний квинтиль АМГ как прокси,
        # это ЧЕСТНО фиксируется в отчёте
        pcos = X[c_amh] > X[c_amh].quantile(0.80)
        pcos_source = "прокси: верхний квинтиль АМГ (диагноз СПКЯ в данных не найден)"
    else:
        pcos_source = f"текст диагнозов, доля {pcos_share:.3f}"
    I_pcos = pcos.astype(float)

    h = (BETA["amh"] * z_amh + BETA["age"] * z_age + BETA["bmi"] * z_bmi
         + BETA["amh_pcos"] * z_amh * I_pcos
         + rng.normal(0, NOISE_SD, len(X)))

    # пороги подбираем так, чтобы РАСПРЕДЕЛЕНИЕ КЛАССОВ совпало с реальным
    p0 = float((y_real == 0).mean())
    p01 = float((y_real <= 1).mean())
    t1, t2 = np.quantile(h, [p0, p01])
    y_synth = pd.Series(np.select([h < t1, h <= t2], [0, 1], default=2), index=X.index).astype(int)

    info = {
        "формула": "h = 1.5*z_AMH - 0.6*z_Age + 0.4*z_BMI + 0.7*z_AMH*I_PCOS + N(0,0.5)",
        "признак_АМГ": c_amh, "признак_возраст": c_age, "признак_ИМТ": c_bmi,
        "СПКЯ_источник": pcos_source, "СПКЯ_доля": float(I_pcos.mean()),
        "пороги": {"t1": round(float(t1), 4), "t2": round(float(t2), 4)},
        "распределение_реальной": {str(k): int(v) for k, v in y_real.value_counts().sort_index().items()},
        "распределение_синтетической": {str(k): int(v) for k, v in y_synth.value_counts().sort_index().items()},
        "совпадение_меток_доля": round(float((y_synth == y_real).mean()), 4),
        "истинно_значимые_S*": [c_amh, c_age, c_bmi],
        "истинные_знаки": {c_amh: "+", c_age: "-", c_bmi: "+"},
    }
    return y_synth, info, (c_amh, c_age, c_bmi)


MODELS = {
    "LogReg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(max_iter=1000)),
    "RandomForest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                          RandomForestClassifier(n_estimators=200, random_state=SEED)),
    "HistGB": lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=250),
}


def explain(mk, X, y, Xte, yte):
    """Локальная важность признаков через permutation importance."""
    m = mk().fit(X, y)
    r = permutation_importance(m, Xte, yte, n_repeats=8, random_state=SEED,
                               scoring="f1_macro")
    imp = pd.Series(r.importances_mean, index=Xte.columns)
    return m, imp


def signed_direction(m, X, col, base_pred):
    """Знак влияния: растёт ли предсказанный класс при увеличении признака."""
    Xp = X.copy()
    Xp[col] = Xp[col] + Xp[col].std(ddof=0)
    try:
        newp = m.predict(Xp)
    except Exception:
        return 0
    d = float(np.mean(newp) - np.mean(base_pred))
    return int(np.sign(d))


def main():
    tr, y_real_tr, oo_tr = load("2023")
    te, y_real_te, oo_te = load("2025")
    Xtr, Xte = feats(tr), feats(te)
    cc = [c for c in Xtr.columns if c in Xte.columns]
    Xtr, Xte = Xtr[cc], Xte[cc]
    print(f"когорта: train {len(Xtr)} / test {len(Xte)}, признаков {len(cc)}")

    y_syn_tr, info_tr, key_tr = make_synthetic(Xtr, tr, y_real_tr)
    y_syn_te, info_te, key_te = make_synthetic(Xte, te, y_real_te)
    c_amh, c_age, c_bmi = key_tr
    S_true = [c_amh, c_age, c_bmi]
    sign_true = {c_amh: +1, c_age: -1, c_bmi: +1}

    print("\n" + "=" * 78)
    print("АУДИТ МЕТОК — что заменено синтетикой")
    print("=" * 78)
    print(f"формула:        {info_tr['формула']}")
    print(f"СПКЯ:           {info_tr['СПКЯ_источник']}")
    print(f"реальная (2023):      {info_tr['распределение_реальной']}")
    print(f"синтетическая (2023): {info_tr['распределение_синтетической']}")
    print(f"совпадение меток:     {info_tr['совпадение_меток_доля']:.1%} "
          f"(низкое совпадение — норма: механизмы разные)")
    print(f"истинно значимые S*:  {S_true}")

    # ── файл аудита: реальная и синтетическая метки рядом ───────────────
    audit = pd.DataFrame({
        "год": 2025,
        "ооцитов_реально": oo_te.values,
        "y_real": y_real_te.values,
        "y_synth": y_syn_te.values,
        "совпало": (y_real_te.values == y_syn_te.values),
        c_amh: Xte[c_amh].values, c_age: Xte[c_age].values, c_bmi: Xte[c_bmi].values,
    })
    audit.to_csv(OUT / "semisynth_labels_audit.csv", index=False, encoding="utf-8-sig")

    # ── обучение на СИНТЕТИЧЕСКОЙ метке ─────────────────────────────────
    rows = []
    for mname, mk in MODELS.items():
        m, imp = explain(mk, Xtr, y_syn_tr, Xte, y_syn_te)
        pred = m.predict(Xte)
        f1 = f1_score(y_syn_te, pred, average="macro")

        order = imp.sort_values(ascending=False)
        for k in (3, 5):
            topk = list(order.index[:k])
            prec = len(set(topk) & set(S_true)) / k
            rec = len(set(topk) & set(S_true)) / len(S_true)
            if k == 3:
                p3, r3 = prec, rec
            else:
                p5, r5 = prec, rec

        # правильность знака влияния
        signs_ok = []
        for c in S_true:
            s = signed_direction(m, Xte, c, pred)
            signs_ok.append(int(s == sign_true[c]) if s != 0 else 0)
        sign_acc = float(np.mean(signs_ok))

        rows.append({"алгоритм": mname, "macro_f1_synth": round(f1, 4),
                     "Precision@3": round(p3, 3), "Recall@3": round(r3, 3),
                     "Precision@5": round(p5, 3), "Recall@5": round(r5, 3),
                     "SignAccuracy": round(sign_acc, 3),
                     "топ-3": ", ".join(order.index[:3])})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "semisynth_recovery.csv", index=False, encoding="utf-8-sig")
    (OUT / "semisynth_meta.json").write_text(
        json.dumps({"train": info_tr, "test": info_te,
                    "коэффициенты": BETA, "шум_sd": NOISE_SD, "seed": SEED},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("ВОССТАНОВЛЕНИЕ ЭТАЛОННОГО ОБЪЯСНЕНИЯ")
    print("=" * 78)
    print(res.to_string(index=False))
    print("\nФайлы: semisynth_recovery.csv, semisynth_labels_audit.csv, semisynth_meta.json")
    print("Реальные метки сохранены в y_real; исходные xlsx не изменялись.")


if __name__ == "__main__":
    main()
