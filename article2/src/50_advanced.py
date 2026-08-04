# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2, углублённый анализ — шесть методов сверх базовых моделей.

Каждый раздел отвечает на отдельный клинический или методологический вопрос,
которого не покрывает обычная классификация.

  1. АНАЛИЗ ВЫЖИВАЕМОСТИ (Каплан — Мейер, регрессия Кокса)
     Вопрос: сколько попыток требуется до наступления беременности и какие
     предлечебные факторы это число меняют. Обычная классификация здесь
     неприменима: часть пациенток прекращает лечение, не достигнув исхода,
     то есть наблюдения цензурированы справа. Игнорировать цензурирование
     значит систематически занижать вероятность успеха.

  2. ПРИЧИННЫЙ АНАЛИЗ (сопоставление по склонности)
     Вопрос: влияет ли выбор протокола на исход у сопоставимых пациенток.
     Прямое сравнение групп бессмысленно: протокол назначается не случайно,
     а по состоянию пациентки (confounding by indication). Сопоставление по
     склонности уравнивает группы по наблюдаемым признакам и позволяет
     оценить эффект протокола при прочих равных.

  3. КЛАСТЕРИЗАЦИЯ (UMAP + HDBSCAN)
     Вопрос: существуют ли фенотипы пациенток, не сводимые к градациям АМГ.
     Если обнаруженные кластеры различаются по исходу сильнее, чем группы по
     АМГ, значит одномерная классификация теряет информацию.

  4. SHAP — вклад признаков в отдельное предсказание
     Прямое продолжение статьи №1: показатель верифицированности требует
     объяснения, соответствующего фактическому поведению модели. SHAP даёт
     аддитивное разложение каждого прогноза с теоретико-игровым обоснованием
     (значения Шепли), а не эвристическую важность.

  5. КАЛИБРОВКА
     AUROC измеряет только упорядочивание. Для клиники важнее, соответствует
     ли заявленная вероятность фактической частоте: если модель говорит
     «риск 30 %», гиперответ должен наступать примерно в 30 % таких случаев.
     Оцениваются кривая надёжности, оценка Брайера и ожидаемая ошибка
     калибровки; сравниваются методы Платта и изотонический.

  6. АНСАМБЛЬ СО СТЕКИНГОМ
     Проверка, добавляет ли объединение разнородных моделей что-либо сверх
     лучшей одиночной. Мета-модель обучается на внекратных предсказаниях,
     иначе оценка будет завышена.

Все оценки — пятикратная стратифицированная перекрёстная проверка, зерно
20260802 (совпадает со статьёй №1).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
    for name, keys in {"protocol": ("протокол",), "stim": ("вид", "стимул")}.items():
        c = find(d, *keys)
        out[name] = d[c].astype(str).str.strip().str.lower() if c else "неизвестно"
    hcg = find(d, "результат", "хгч")
    out["hcg"] = d[hcg].astype(str).str.strip().str.lower() if hcg else ""
    src = find(d, "источник", "клеток")
    out["_src"] = d[src].astype(str).str.lower() if src else ""
    out["year"] = year
    out = out[out["amh"].notna() & out["amh"].between(0, 30)]
    out = out[~out["_src"].str.contains("до|донор", na=False)]
    return out.drop(columns=["_src"])


def build_matrix(df: pd.DataFrame, num=NUM, cat=CAT):
    """Числовая матрица: медианное восполнение + прямое кодирование."""
    X = df[num].copy()
    X = X.fillna(X.median(numeric_only=True))
    for c in cat:
        vc = df[c].value_counts()
        keep = vc[vc >= 30].index
        s = df[c].where(df[c].isin(keep), "прочее")
        X = pd.concat([X, pd.get_dummies(s, prefix=c, drop_first=True)], axis=1)
    return X.astype(float)


# ═══════════════════ 1. Анализ выживаемости ═══════════════════
def survival(raw: pd.DataFrame, report: dict) -> None:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    print("\n" + "=" * 74)
    print("1. АНАЛИЗ ВЫЖИВАЕМОСТИ: число попыток до наступления беременности")
    print("=" * 74)

    d = raw[raw["attempt"].notna() & (raw["attempt"] >= 1) & (raw["attempt"] <= 12)].copy()
    d["event"] = d["hcg"].str.startswith("пол").astype(int)
    d["T"] = d["attempt"].clip(1, 12)
    if len(d) < 200:
        print("  недостаточно данных"); return

    km = KaplanMeierFitter()
    km.fit(d["T"], d["event"], label="вся когорта")
    print(f"  наблюдений {len(d)}, событий {int(d['event'].sum())}, "
          f"цензурировано {int((1 - d['event']).sum())}")

    curve = km.survival_function_.reset_index()
    curve.columns = ["attempt", "S"]
    curve["cumulative_pregnancy"] = 1 - curve["S"]
    curve.to_csv(OUT / "surv_km_overall.csv", index=False)
    for a in (1, 2, 3, 4):
        row = curve[curve["attempt"] <= a].tail(1)
        if len(row):
            print(f"  кумулятивная вероятность беременности к попытке {a}: "
                  f"{float(row['cumulative_pregnancy'].iloc[0]):.3f}")

    # стратификация по овариальному резерву
    d["strata"] = pd.cut(d["amh"], [-.01, .6, 2.0, 30],
                         labels=["АМГ ≤ 0,6", "АМГ 0,61–2,0", "АМГ > 2,0"])
    rows = []
    for name, g in d.groupby("strata", observed=True):
        if len(g) < 50:
            continue
        k = KaplanMeierFitter().fit(g["T"], g["event"], label=str(name))
        cur = k.survival_function_.reset_index()
        cur.columns = ["attempt", "S"]
        cur["strata"] = str(name)
        cur["cumulative_pregnancy"] = 1 - cur["S"]
        rows.append(cur)
        print(f"  {name}: n = {len(g)}, событий {int(g['event'].sum())}")
    if rows:
        pd.concat(rows).to_csv(OUT / "surv_km_strata.csv", index=False)

    lo = d[d["amh"] <= 0.6]
    hi = d[d["amh"] > 2.0]
    if len(lo) > 40 and len(hi) > 40:
        lr = logrank_test(lo["T"], hi["T"], lo["event"], hi["event"])
        print(f"  лог-ранговый критерий «АМГ ≤ 0,6» против «АМГ > 2,0»: "
              f"p = {lr.p_value:.3e}")
        report["logrank_p"] = float(lr.p_value)

    # регрессия Кокса на предлечебных признаках
    cols = ["amh", "age", "bmi", "fsh", "infert_dur"]
    cx = d[cols + ["T", "event"]].dropna()
    if len(cx) > 200:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cx, duration_col="T", event_col="event")
        summ = cph.summary[["coef", "exp(coef)", "p",
                            "exp(coef) lower 95%", "exp(coef) upper 95%"]]
        summ.to_csv(OUT / "surv_cox.csv")
        print("\n  регрессия Кокса (отношение рисков наступления беременности):")
        for name, r in summ.iterrows():
            star = "*" if r["p"] < 0.05 else " "
            print(f"   {star} {name:12s} HR {r['exp(coef)']:.3f} "
                  f"[{r['exp(coef) lower 95%']:.3f}; {r['exp(coef) upper 95%']:.3f}]  "
                  f"p = {r['p']:.4f}")
        report["cox_concordance"] = float(cph.concordance_index_)
        print(f"  индекс конкордации: {cph.concordance_index_:.4f}")


# ═══════════════ 2. Причинный анализ протокола ═══════════════
def causal(punct: pd.DataFrame, report: dict) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from scipy.stats import mannwhitneyu

    print("\n" + "=" * 74)
    print("2. ПРИЧИННЫЙ АНАЛИЗ: влияние протокола при сопоставимых пациентках")
    print("=" * 74)

    top = punct["protocol"].value_counts()
    top = top[top >= 120]
    if len(top) < 2:
        print("  недостаточно протоколов с достаточным числом наблюдений"); return
    a, b = top.index[0], top.index[1]
    d = punct[punct["protocol"].isin([a, b])].copy()
    d["treat"] = (d["protocol"] == a).astype(int)
    print(f"  сравниваются: «{a}» (n = {int(d['treat'].sum())}) и "
          f"«{b}» (n = {int((1 - d['treat']).sum())})")

    X = build_matrix(d)
    ps_model = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=4000, random_state=SEED))
    ps_model.fit(X, d["treat"])
    d["ps"] = ps_model.predict_proba(X)[:, 1]

    # наивное сравнение — заведомо смещённое
    naive = d[d["treat"] == 1]["oocytes"].mean() - d[d["treat"] == 0]["oocytes"].mean()

    # сопоставление ближайшего соседа по склонности с ограничением расстояния
    tr = d[d["treat"] == 1].sort_values("ps").reset_index(drop=True)
    ct = d[d["treat"] == 0].sort_values("ps").reset_index(drop=True)
    caliper = 0.2 * d["ps"].std()
    used, pairs = set(), []
    ct_ps = ct["ps"].values
    for _, r in tr.iterrows():
        j = int(np.argmin(np.abs(ct_ps - r["ps"])))
        for cand in np.argsort(np.abs(ct_ps - r["ps"]))[:50]:
            if cand not in used and abs(ct_ps[cand] - r["ps"]) <= caliper:
                j = int(cand); break
        else:
            continue
        used.add(j)
        pairs.append((r["oocytes"], ct.iloc[j]["oocytes"]))

    if len(pairs) < 40:
        print("  сопоставлено слишком мало пар"); return
    t_out = np.array([p[0] for p in pairs])
    c_out = np.array([p[1] for p in pairs])
    att = float(t_out.mean() - c_out.mean())
    u, p = mannwhitneyu(t_out, c_out)

    print(f"  наивная разница средних:              {naive:+.2f} ооцитов")
    print(f"  после сопоставления по склонности:    {att:+.2f} ооцитов "
          f"(пар {len(pairs)}, p = {p:.4f})")
    print(f"  смещение, устранённое сопоставлением: {naive - att:+.2f}")
    if p >= 0.05:
        print("  вывод: при сопоставимых пациентках различие между протоколами\n"
              "         статистически не подтверждается")
    report["causal"] = {"протокол_A": a, "протокол_B": b, "наивная_разница": naive,
                        "эффект_после_сопоставления": att, "p": float(p),
                        "пар": len(pairs)}
    pd.DataFrame({"treated": t_out, "control": c_out}).to_csv(
        OUT / "causal_matched_pairs.csv", index=False)


# ═══════════════ 3. Кластеризация фенотипов ═══════════════
def clustering(punct: pd.DataFrame, report: dict) -> None:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import kruskal

    print("\n" + "=" * 74)
    print("3. КЛАСТЕРИЗАЦИЯ: существуют ли фенотипы сверх градаций АМГ")
    print("=" * 74)

    X = StandardScaler().fit_transform(build_matrix(punct))
    emb = umap.UMAP(n_neighbors=25, min_dist=0.05, n_components=2,
                    random_state=SEED).fit_transform(X)
    lab = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=10).fit_predict(emb)

    punct = punct.copy()
    punct["cluster"] = lab
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"  найдено кластеров: {n_cl}, шумовых точек: {int((lab == -1).sum())}")

    rows = []
    for c in sorted(set(lab)):
        g = punct[punct["cluster"] == c]
        rows.append({"кластер": int(c), "n": len(g),
                     "АМГ_медиана": round(float(g["amh"].median()), 2),
                     "возраст_медиана": round(float(g["age"].median()), 1),
                     "ооцитов_медиана": round(float(g["oocytes"].median()), 1),
                     "доля_бедный": round(float((g["oocytes"] <= 3).mean()), 3),
                     "доля_гипер": round(float((g["oocytes"] > 20).mean()), 3)})
    cl = pd.DataFrame(rows)
    print(cl.to_string(index=False))
    cl.to_csv(OUT / "cluster_profile.csv", index=False)
    pd.DataFrame({"umap1": emb[:, 0], "umap2": emb[:, 1], "cluster": lab,
                  "oocytes": punct["oocytes"].values,
                  "amh": punct["amh"].values}).to_csv(
        OUT / "cluster_embedding.csv", index=False)

    real = [g["oocytes"].values for c, g in punct.groupby("cluster") if c != -1 and len(g) > 30]
    if len(real) >= 2:
        h, p = kruskal(*real)
        print(f"  критерий Краскела — Уоллиса по числу ооцитов между кластерами: p = {p:.3e}")
        report["cluster_kruskal_p"] = float(p)
    # сравнение с разбиением по АМГ
    amh_groups = [g["oocytes"].values for _, g in
                  punct.groupby(pd.cut(punct["amh"], [-.01, .6, 2.0, 30]), observed=True)]
    if len(amh_groups) >= 2:
        h2, p2 = kruskal(*[a for a in amh_groups if len(a) > 30])
        print(f"  то же для групп по АМГ:                                    p = {p2:.3e}")
        report["amh_groups_kruskal_p"] = float(p2)
    report["n_clusters"] = int(n_cl)


# ═══════════════ 4. SHAP ═══════════════
def shap_analysis(punct: pd.DataFrame, report: dict) -> None:
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 74)
    print("4. SHAP: вклад признаков в предсказание риска гиперответа")
    print("=" * 74)

    X = build_matrix(punct)
    y = (punct["oocytes"] > 20).astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                               random_state=SEED, n_jobs=3,
                               class_weight="balanced").fit(Xtr, ytr)

    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(Xte)
    vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    imp = pd.DataFrame({"признак": X.columns,
                        "средний_модуль_SHAP": np.abs(vals).mean(0)}) \
        .sort_values("средний_модуль_SHAP", ascending=False)
    imp.to_csv(OUT / "shap_importance.csv", index=False)
    print(imp.head(12).to_string(index=False))

    # сохраняем матрицу для построения summary plot
    pd.DataFrame(vals, columns=X.columns).to_csv(OUT / "shap_values.csv", index=False)
    Xte.reset_index(drop=True).to_csv(OUT / "shap_features.csv", index=False)
    report["shap_top"] = imp.head(5)["признак"].tolist()


# ═══════════════ 5. Калибровка ═══════════════
def calibration(punct: pd.DataFrame, report: dict) -> None:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import brier_score_loss, roc_auc_score

    print("\n" + "=" * 74)
    print("5. КАЛИБРОВКА: соответствует ли заявленная вероятность частоте")
    print("=" * 74)

    X = build_matrix(punct)
    y = (punct["oocytes"] > 20).astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                                  random_state=SEED, n_jobs=3,
                                  class_weight="balanced")

    rows = []
    for name, mdl in {
        "без калибровки": base,
        "Платт (сигмоида)": CalibratedClassifierCV(base, method="sigmoid", cv=3),
        "изотоническая": CalibratedClassifierCV(base, method="isotonic", cv=3),
    }.items():
        p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        brier = brier_score_loss(y, p)
        auc = roc_auc_score(y, p)
        # ожидаемая ошибка калибровки: средний по бинам модуль расхождения
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).sum()
                  for b in range(10) if (idx == b).sum() > 0) / len(y)
        rows.append({"метод": name, "Брайер": brier, "ECE": ece, "AUROC": auc})
        print(f"  {name:20s} Брайер {brier:.4f}   ECE {ece:.4f}   AUROC {auc:.4f}")

        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        pd.DataFrame({"средняя_вероятность": mean_pred,
                      "фактическая_частота": frac}).to_csv(
            OUT / f"calib_{name.split()[0]}.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "calibration_summary.csv", index=False)
    best = min(rows, key=lambda r: r["Брайер"])
    report["calibration_best"] = best["метод"]
    print(f"  наименьшая оценка Брайера: {best['метод']}")


# ═══════════════ 6. Ансамбль со стекингом ═══════════════
def stacking(punct: pd.DataFrame, report: dict) -> None:
    from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                                  HistGradientBoostingClassifier,
                                  ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, f1_score

    print("\n" + "=" * 74)
    print("6. АНСАМБЛЬ СО СТЕКИНГОМ")
    print("=" * 74)

    X = build_matrix(punct)
    y = (punct["oocytes"] > 20).astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    singles = {
        "случайный лес": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=4, random_state=SEED,
            n_jobs=3, class_weight="balanced"),
        "экстра-деревья": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=SEED,
            n_jobs=3, class_weight="balanced"),
        "градиентный бустинг": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=400, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=1.0),
    }
    rows = []
    for name, m in singles.items():
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        rows.append({"модель": name, "AUROC": roc_auc_score(y, p),
                     "macro_f1": f1_score(y, (p > 0.5).astype(int), average="macro")})
        print(f"  {name:24s} AUROC {rows[-1]['AUROC']:.4f}")

    stack = StackingClassifier(
        estimators=list(singles.items()),
        final_estimator=LogisticRegression(max_iter=3000, random_state=SEED),
        cv=3, n_jobs=1)
    p = cross_val_predict(stack, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    rows.append({"модель": "стекинг", "AUROC": roc_auc_score(y, p),
                 "macro_f1": f1_score(y, (p > 0.5).astype(int), average="macro")})
    print(f"  {'стекинг':24s} AUROC {rows[-1]['AUROC']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stacking_results.csv", index=False)
    best_single = max(r["AUROC"] for r in rows[:-1])
    gain = rows[-1]["AUROC"] - best_single
    print(f"  прирост стекинга над лучшей одиночной моделью: {gain:+.4f}")
    report["stacking_gain"] = float(gain)


def main() -> None:
    raw = pd.concat([load_raw(y) for y in (2023, 2024, 2025)], ignore_index=True)
    punct = raw[raw["oocytes"] > 0].copy()
    print(f"записей с АМГ: {len(raw)}   состоявшихся пункций: {len(punct)}")

    report: dict = {"записей": int(len(raw)), "пункций": int(len(punct)),
                    "seed": SEED}
    for fn in (survival, causal, clustering, shap_analysis, calibration, stacking):
        try:
            fn(raw if fn is survival else punct, report)
        except Exception as e:  # noqa: BLE001
            print(f"\n  [{fn.__name__}] не выполнен: {type(e).__name__}: {e}")
            report[f"{fn.__name__}_error"] = str(e)

    (OUT / "advanced_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 74}\nсохранено в {OUT}")


if __name__ == "__main__":
    main()
