"""
kronos_lab/foundation/bybit_data.py

Вся работа с Bybit в одном месте: свежий контекст для модели и окно
реального исхода. Источник - linear perpetual (usdt маржа).

Bybit kline массив свечи: [startTime, open, high, low, close, volume, turnover]
turnover (c[6]) - оборот в usdt, это и есть amount для Kronos (не приближение).
"""

import time

import pandas as pd
import requests

URL = "https://api.bybit.com/v5/market/kline"
CATEGORY = "linear"
INTERVAL = "60"  # часовые свечи


def _get(params, retries=5):
    for _ in range(retries):
        try:
            data = requests.get(URL, params=params, timeout=10).json()
        except Exception as e:
            print(f"  ошибка запроса: {e}, повтор через 2с")
            time.sleep(2)
            continue
        if data.get("retCode") != 0:
            msg = data.get("retMsg", "")
            if "rate" in msg.lower() or "many" in msg.lower():
                print("  rate-limit, жду 10с")
                time.sleep(10)
                continue
            raise RuntimeError(f"Bybit API error: {msg}")
        return data["result"]["list"]
    return None


def fetch_recent_context(symbol, n_ctx=360):
    """
    Последние n_ctx ЗАКРЫТЫХ часовых свечей. Незакрытую текущую отбрасываем.
    Возвращает (x_df[open,high,low,close,volume,amount], x_timestamp).
    """
    lst = _get({"category": CATEGORY, "symbol": symbol, "interval": INTERVAL,
                "limit": min(n_ctx + 1, 1000)})
    if not lst:
        raise RuntimeError(f"пустой ответ контекста для {symbol}")
    rows = [{
        "ts": pd.to_datetime(int(c[0]), unit="ms", utc=True),
        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
        "close": float(c[4]), "volume": float(c[5]), "amount": float(c[6]),
    } for c in lst]
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df = df.iloc[:-1].tail(n_ctx).reset_index(drop=True)  # отбросить незакрытую
    x_timestamp = df["ts"].reset_index(drop=True)
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    return x_df, x_timestamp


def fetch_window_ohlc(symbol, start_dt, end_dt):
    """
    Часовые свечи на окне [start_dt, end_dt]. Возвращает (last_close, max_high, min_low, n) или None.
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    lst = _get({"category": CATEGORY, "symbol": symbol, "interval": INTERVAL,
                "start": start_ms, "end": end_ms, "limit": 1000})
    if not lst:
        return None
    highs, lows, closes = [], [], []
    for c in lst:
        ts = int(c[0])
        if ts < start_ms or ts > end_ms:
            continue
        highs.append(float(c[2]))
        lows.append(float(c[3]))
        closes.append((ts, float(c[4])))
    if not closes:
        return None
    closes.sort(key=lambda x: x[0])
    return closes[-1][1], max(highs), min(lows), len(closes)
