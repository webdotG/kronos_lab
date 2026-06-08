"""
kronos_lab/cli/whatif_trade.py

Моделирование "что если бы торговали" по закрытым прогнозам журнала.
Сравнивает несколько стратегий, все net (комиссия из config, mixed 0.136%):

  buy_hold         - всегда лонг (база: что делает наивное "всегда вверх")
  follow           - по сигналу модели, фикс размер
  fade             - против сигнала, фикс размер
  follow_cone      - по сигналу, размер ~ 1/ширина_конуса (узкий конус = больше)
  fade_cone        - против сигнала, размер по конусу
  follow_narrow    - по сигналу, ВХОДИМ только когда конус уже медианного (модель уверена)

Чистый pandas, CPU, карту не трогает.

Запуск:
    python cli/whatif_trade.py            # берёт config.JOURNAL_PATH
    python cli/whatif_trade.py <path.csv>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from foundation.journal import load_journal, ROUND_TURN_FEE_PCT


def cone_width_pct(row):
    """Ширина конуса p5..p95 в процентах от last_close - мера неуверенности модели."""
    return (row["p95"] - row["p5"]) / row["last_close"] * 100


def run(df: pd.DataFrame) -> pd.DataFrame:
    done = df[df["outcome_filled"] == 1].copy()
    n = len(done)
    if n == 0:
        return pd.DataFrame()

    for c in ["last_close", "actual_close", "p5", "p95", "upside_prob"]:
        done[c] = pd.to_numeric(done[c], errors="coerce")
    done = done.dropna(subset=["last_close", "actual_close", "p5", "p95"])

    # реальная доходность инструмента за 24ч, %
    ret = (done["actual_close"] / done["last_close"] - 1.0) * 100
    # сторона по сигналу: long если upside>=50 (модель почти всегда long)
    side = np.where(done["upside_prob"] >= 50, 1.0, -1.0)
    cone = done.apply(cone_width_pct, axis=1).values
    fee = ROUND_TURN_FEE_PCT

    # сайзинг по конусу: нормируем так, чтобы СРЕДНИЙ размер был 1 (сопоставимо с фикс)
    inv = 1.0 / cone
    size_cone = inv / inv.mean()
    # фильтр узкого конуса: входим только если конус уже медианы
    narrow = cone <= np.median(cone)

    r = ret.values
    out = {}
    # buy-hold: всегда long, фикс размер, платим комиссию (раз за сделку 24ч)
    out["buy_hold"] = r - fee
    out["follow"] = side * r - fee
    out["fade"] = -side * r - fee
    # сайзинг: комиссия тоже масштабируется размером (больше размер - больше оборот)
    out["follow_cone"] = size_cone * side * r - fee * size_cone
    out["fade_cone"] = size_cone * (-side) * r - fee * size_cone
    # фильтр: вне узкого конуса размер 0 (не торгуем, комиссии нет)
    fr = np.where(narrow, side * r - fee, 0.0)
    out["follow_narrow"] = fr

    rows = []
    for name, pnl in out.items():
        pnl = np.asarray(pnl, dtype=float)
        traded = np.count_nonzero(pnl) if name == "follow_narrow" else n
        total = pnl.sum()
        mean = pnl.mean()
        win = (pnl > 0).mean() * 100
        sharpe = mean / (pnl.std() + 1e-9) if pnl.std() > 0 else 0.0
        rows.append({"стратегия": name, "сделок": traded, "итог_%": round(total, 2),
                     "сред_%": round(mean, 4), "винрейт_%": round(win, 1),
                     "sharpe_за_сделку": round(sharpe, 3)})
    return pd.DataFrame(rows)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.JOURNAL_PATH
    if not os.path.exists(path):
        print("журнала нет:", path)
        return
    df = load_journal(path)
    done = int((df["outcome_filled"] == 1).sum())
    print(f"всего прогнозов: {len(df)}, закрыто: {done}")
    if done == 0:
        print("нечего считать - ни одного закрытого исхода. крон копит, жди суток")
        return

    res = run(df)
    print(f"\n=== ЧТО ЕСЛИ БЫ ТОРГОВАЛИ (net комиссия {round(ROUND_TURN_FEE_PCT,3)}%) ===")
    print(res.to_string(index=False))

    if done < 20:
        print(f"\n[!] закрытых исходов {done} - это ШУМ, не результат. сайзинг усиливает разброс.")
        print("    верить цифрам можно от ~20-30 НЕЗАВИСИМЫХ суточных точек (3+ недели сбора)")
    print("\n[i] sharpe за сделку - сырой, не годовой. база сравнения - buy_hold")


if __name__ == "__main__":
    main()

