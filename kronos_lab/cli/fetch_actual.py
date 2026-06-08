"""
kronos_lab/cli/fetch_actual.py

Закрывает созревшие прогнозы: реальный исход с Bybit + расчёт P&L.
Запуск: python cli/fetch_actual.py
"""

import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from foundation.journal import load_journal, COLUMNS, _parse_dt, settle_pnl
from foundation.bybit_data import fetch_window_ohlc


def close_matured(journal_path: str = None) -> int:
    journal_path = journal_path or config.JOURNAL_PATH
    if not os.path.exists(journal_path):
        return 0
    df = load_journal(journal_path)
    now = pd.Timestamp.now("UTC").to_pydatetime()
    filled = 0
    for idx, row in df.iterrows():
        if int(row.get("outcome_filled", 0)) == 1:
            continue
        target = _parse_dt(str(row["target_at"]))
        if target > now:
            continue
        gen = _parse_dt(str(row["generated_at"]))
        start = gen + timedelta(hours=1)
        print(f"[{row['forecast_id']}] тяну окно {start} .. {target}")
        res = fetch_window_ohlc(str(row["symbol"]), start, target)
        if res is None:
            print("    не удалось получить свечи, пропуск")
            continue
        close, hi, lo, n = res
        if n < config.HORIZON_HOURS:
            print(f"    внимание: {n}/{config.HORIZON_HOURS} свечей, пишу что есть")
        pnl = settle_pnl(float(row["last_close"]), close, str(row["signal_side"]))
        df.at[idx, "actual_close"] = round(close, 4)
        df.at[idx, "actual_high"] = round(hi, 4)
        df.at[idx, "actual_low"] = round(lo, 4)
        for k, v in pnl.items():
            df.at[idx, k] = v
        df.at[idx, "outcome_filled"] = 1
        filled += 1
    df[COLUMNS].to_csv(journal_path, index=False)
    return filled


if __name__ == "__main__":
    n = close_matured()
    print(f"\nзаполнено исходов: {n}")
