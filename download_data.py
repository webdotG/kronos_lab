"""
download_data.py - качает свежую часовую историю BTCUSDT с Bybit (linear)
в формат, который ест backtest: data/BTCUSDT_1h.csv с колонками
dt,open,high,low,close,volume,amount (amount = turnover из API).

Идёт назад от сейчас до START, пагинация по 1000 свечей, дедуп.
"""
import os, time
import pandas as pd
import requests
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
START = os.environ.get("HIST_START", "2023-01-01")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_lab", "data", "BTCUSDT_1h.csv")
URL = "https://api.bybit.com/v5/market/kline"
STEP_MS = 3_600_000  # 1 час

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    start_ms = int(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)
    cursor = int(time.time()*1000)
    rows = {}
    req = 0
    print(f"качаю {SYMBOL} 1h назад от сейчас до {START}...")
    while cursor > start_ms:
        params = {"category":"linear","symbol":SYMBOL,"interval":"60","end":cursor,"limit":1000}
        try:
            data = requests.get(URL, params=params, timeout=15).json()
        except Exception as e:
            print(f"  ошибка {e}, повтор через 3с"); time.sleep(3); continue
        if data.get("retCode") != 0:
            msg = data.get("retMsg","")
            if "rate" in msg.lower() or "many" in msg.lower():
                time.sleep(10); continue
            print(f"  API error: {msg}"); break
        lst = data["result"]["list"]
        if not lst: break
        for c in lst:
            ts = int(c[0])
            if ts < start_ms: continue
            dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            rows[ts] = (dt, c[1], c[2], c[3], c[4], c[5], c[6])  # turnover = amount
        earliest = min(int(c[0]) for c in lst)
        new_cursor = earliest - STEP_MS
        if new_cursor >= cursor: break
        cursor = new_cursor
        req += 1
        if req % 20 == 0:
            print(f"  запросов {req}, свечей {len(rows)}, дошли до {datetime.fromtimestamp(earliest/1000, tz=timezone.utc)}")
        time.sleep(0.3)
    ordered = [rows[k] for k in sorted(rows)]
    df = pd.DataFrame(ordered, columns=["dt","open","high","low","close","volume","amount"])
    df.to_csv(OUT, index=False)
    print(f"ГОТОВО: {len(df)} часовых свечей -> {OUT}")
    if len(df):
        print(f"период: {df['dt'].iloc[0]} .. {df['dt'].iloc[-1]}")

if __name__ == "__main__":
    main()
