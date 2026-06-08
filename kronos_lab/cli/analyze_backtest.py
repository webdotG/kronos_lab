"""
kronos_lab/cli/analyze_backtest.py

Анализ ГОТОВОГО backtest_results.csv (исторический прогон Kronos по BTC).
Воспроизводит и ужесточает проверки из KRONOS_RESEARCH:
  - направление (hit rate, экстремумы)
  - волатильность (vol_hit, навык, переходит ли вола в движение цены)
  - величина хода как сигнал
  - комбо-сигнал upside<=15 + рост волы -> лонг, с ЧЕСТНЫМ OOS на непересекающихся окнах

Чистый pandas/scipy, секунды, без GPU.

Запуск:
    python cli/analyze_backtest.py /home/grant/aProject/trading/scripts/backtest_results.csv

Ожидаемые колонки: date, price, pos_pct, ctx_trend, pred_chg, upside,
                    real_chg, hit, pred_vol, hist_vol, real_vol, vol_hit
"""

import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest, mannwhitneyu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def binom_p(k, n, p=0.5):
    if not HAVE_SCIPY or n == 0:
        return float("nan")
    return binomtest(int(k), int(n), p).pvalue


def nonoverlap_subset(df, hours=24):
    """Берёт непересекающиеся точки: каждая >= hours после предыдущей выбранной (честный OOS)."""
    if "date" not in df.columns:
        return df
    d = df.copy()
    d["_dt"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["_dt"]).sort_values("_dt")
    keep, last = [], None
    for idx, t in zip(d.index, d["_dt"]):
        if last is None or (t - last) >= pd.Timedelta(hours=hours):
            keep.append(idx)
            last = t
    return df.loc[keep]


def main():
    if len(sys.argv) != 2:
        print("использование: python cli/analyze_backtest.py <backtest_results.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    n = len(df)
    print(f"\n=== АНАЛИЗ БЭКТЕСТА: {n} точек ===")
    print(f"колонки: {list(df.columns)}")
    if "date" in df.columns:
        print(f"период: {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")

    # приведение типов
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 1. направление ----
    print("\n--- 1. НАПРАВЛЕНИЕ ---")
    if "hit" in df.columns:
        hit = df["hit"].mean()
        print(f"hit rate общий: {hit*100:.1f}%  (p vs 0.5 = {binom_p(df['hit'].sum(), n):.4f})")
    if {"upside", "real_chg"}.issubset(df.columns):
        hi = df[df["upside"] >= 85]
        lo = df[df["upside"] <= 15]
        if len(hi):
            up_hit = (hi["real_chg"] > 0).mean()
            print(f"upside>=85 (ждёт рост): реально вверх {up_hit*100:.1f}% (n={len(hi)}, p={binom_p((hi['real_chg']>0).sum(), len(hi)):.4f})")
        if len(lo):
            dn_hit = (lo["real_chg"] < 0).mean()
            print(f"upside<=15 (ждёт падение): реально вниз {dn_hit*100:.1f}% (n={len(lo)}, p={binom_p((lo['real_chg']<0).sum(), len(lo)):.4f})")

    # ---- 2. волатильность ----
    print("\n--- 2. ВОЛАТИЛЬНОСТЬ ---")
    if "vol_hit" in df.columns:
        print(f"vol_hit общий: {df['vol_hit'].mean()*100:.1f}%")
    if {"pred_vol", "hist_vol", "real_vol", "real_chg"}.issubset(df.columns):
        vol_up = df[df["pred_vol"] > df["hist_vol"]]
        if len(vol_up):
            real_vol_up = (vol_up["real_vol"] > vol_up["hist_vol"]).mean()
            base = (df["real_vol"] > df["hist_vol"]).mean()
            print(f"когда модель ждёт рост волы: реально выросла {real_vol_up*100:.1f}% (n={len(vol_up)}), базовая частота {base*100:.1f}%")
            print(f"  p (навык vs база) = {binom_p((vol_up['real_vol']>vol_up['hist_vol']).sum(), len(vol_up), base):.4f}")
            # переходит ли вола в ДВИЖЕНИЕ цены
            move_volup = vol_up["real_chg"].abs().median()
            move_rest = df[df["pred_vol"] <= df["hist_vol"]]["real_chg"].abs().median()
            print(f"медиана |хода| в дни рост-волы {move_volup:.2f}% vs остальные {move_rest:.2f}%")
            if HAVE_SCIPY:
                u = mannwhitneyu(vol_up["real_chg"].abs().dropna(),
                                 df[df["pred_vol"] <= df["hist_vol"]]["real_chg"].abs().dropna())
                print(f"  p (движение крупнее?) = {u.pvalue:.4f}  ('трясёт' != 'уйдёт далеко' если не значимо)")

    # ---- 3. величина хода как сигнал ----
    print("\n--- 3. ВЕЛИЧИНА ХОДА ---")
    if {"pred_chg", "real_chg"}.issubset(df.columns):
        d = df.dropna(subset=["pred_chg", "real_chg"])
        q = pd.qcut(d["pred_chg"].abs(), 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        by_q = d.groupby(q, observed=True)["real_chg"].apply(lambda x: x.abs().median())
        print("медиана |реального хода| по квартилям |pred_chg|:")
        for k, v in by_q.items():
            print(f"  {k}: {v:.2f}%")
        corr = d["pred_chg"].abs().corr(d["real_chg"].abs())
        print(f"корреляция |pred_chg| ~ |real_chg|: {corr:.3f}")

    # ---- 4. КОМБО-СИГНАЛ (главный кандидат) ----
    print("\n--- 4. КОМБО-СИГНАЛ: upside<=15 + рост волы -> лонг ---")
    if {"upside", "pred_vol", "hist_vol", "real_chg"}.issubset(df.columns):
        def combo_report(sub, label):
            if len(sub) == 0:
                print(f"  [{label}] нет точек")
                return
            up = (sub["real_chg"] > 0).mean()
            print(f"  [{label}] цена вверх {up*100:.1f}% (n={len(sub)}, p vs 0.5 = {binom_p((sub['real_chg']>0).sum(), len(sub)):.4f})")

        combo = df[(df["upside"] <= 15) & (df["pred_vol"] > df["hist_vol"])]
        combo_report(combo, "все точки (перекрытые)")

        # честный OOS на НЕПЕРЕСЕКАЮЩИХСЯ окнах
        indep = nonoverlap_subset(df, hours=24)
        combo_indep = indep[(indep["upside"] <= 15) & (indep["pred_vol"] > indep["hist_vol"])]
        combo_report(combo_indep, "НЕЗАВИСИМЫЕ окна (этому верь)")

        # OOS по половинам времени
        if "date" in df.columns and len(combo) > 4:
            c = combo.copy()
            c["_dt"] = pd.to_datetime(c["date"], errors="coerce")
            c = c.dropna(subset=["_dt"]).sort_values("_dt")
            mid = c["_dt"].iloc[len(c) // 2]
            combo_report(c[c["_dt"] < mid], "1-я половина времени")
            combo_report(c[c["_dt"] >= mid], "2-я половина времени")

        # зеркало (должно НЕ работать, если сигнал односторонний)
        mirror = df[(df["upside"] >= 85) & (df["pred_vol"] > df["hist_vol"])]
        if len(mirror):
            dn = (mirror["real_chg"] < 0).mean()
            print(f"  зеркало upside>=85+вола -> вниз {dn*100:.1f}% (n={len(mirror)}) - ожидаем ~50% (несимметрично)")

    print("\n[!] шаг 1 день = окна 24ч перекрываются -> автокорреляция, общий p оптимистичен")
    print("    верь строке 'НЕЗАВИСИМЫЕ окна' и согласованности половин, не общему p")
    if not HAVE_SCIPY:
        print("[warn] scipy не установлен -> p-values пропущены. pip install scipy")


if __name__ == "__main__":
    main()
