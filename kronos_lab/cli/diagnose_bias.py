"""
kronos_lab/cli/diagnose_bias.py

РЕШАЮЩИЙ ТЕСТ: есть ли у модели направленный перекос и какой природы.

Кормим модель тремя искусственными контекстами одинаковой волатильности:
  - явный аптренд   (цена росла)
  - явный даунтренд (цена падала)
  - боковик         (без тренда)
и смотрим upside и ожидаемый ход прогноза.

Интерпретация:
  - upside высокий ВЕЗДЕ (и на аптренде, и на даунтренде) -> жёсткий бычий уклон / баг
  - upside высокий на даунтренде, низкий на аптренде       -> откат к среднему (mean reversion)
  - upside высокий на аптренде, низкий на даунтренде       -> следование тренду (momentum)

Запуск: python cli/diagnose_bias.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import config
from foundation.generation import generate_ensemble

sys.path.append(config.KRONOS_MODEL_PATH)
from model import Kronos, KronosTokenizer, KronosPredictor


def make_context(regime, n=360, start_price=65000.0, vol=0.004, seed=0):
    """Синтетический OHLCV-контекст заданного режима. drift на бар."""
    rng = np.random.default_rng(seed)
    drift = {"uptrend": 0.0015, "downtrend": -0.0015, "flat": 0.0}[regime]
    price = start_price
    rows = []
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(n):
        r = rng.normal(drift, vol)
        o = price
        price = price * (1 + r)
        c = price
        hi = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
        v = abs(rng.normal(1000, 200))
        rows.append({"ts": t0 + pd.Timedelta(hours=i), "open": o, "high": hi,
                     "low": lo, "close": c, "volume": v, "amount": c * v})
    df = pd.DataFrame(rows)
    x_timestamp = df["ts"].reset_index(drop=True)
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    return x_df, x_timestamp


def main():
    print("гружу модель...", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(config.TOKENIZER_REPO)
    model = Kronos.from_pretrained(config.MODEL_REPO)
    if torch.cuda.is_available():
        model = model.cuda()
    predictor = KronosPredictor(model, tokenizer, max_context=config.MAX_CONTEXT)

    print(f"\n{'режим':<12}{'контекст ход':>14}{'upside %':>12}{'ожид.ход %':>14}{'конус p5p95':>14}", flush=True)
    print("-" * 66)
    for regime in ["uptrend", "downtrend", "flat"]:
        x_df, x_ts = make_context(regime, n=config.N_CONTEXT)
        ctx_move = (x_df["close"].iloc[-1] / x_df["close"].iloc[0] - 1) * 100
        y_ts = pd.date_range(start=x_ts.iloc[-1] + pd.Timedelta(hours=1),
                             periods=config.HORIZON_HOURS, freq="1h").to_series().reset_index(drop=True)
        ens = generate_ensemble(
            predictor=predictor, df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=config.HORIZON_HOURS, n_paths=config.N_PATHS, chunk=config.CHUNK,
            T=config.T, top_k=config.TOP_K, top_p=config.TOP_P, seed=config.SEED,
            symbol=f"SYN_{regime}", model_id=config.MODEL_ID,
        )
        term = ens.terminal_close()
        upside = float((term > ens.last_close).mean()) * 100
        exp_move = (term.mean() / ens.last_close - 1) * 100
        cp = ens.close_paths()
        cone = float(np.percentile(cp[:, -1], 95) - np.percentile(cp[:, -1], 5))
        print(f"{regime:<12}{ctx_move:>13.1f}%{upside:>11.1f}%{exp_move:>13.2f}%{cone:>14.1f}", flush=True)

    print("\nинтерпретация:", flush=True)
    print("  upside высокий и на up, и на down -> бычий уклон/баг (направление бесполезно)", flush=True)
    print("  up<50 на аптренде, up>50 на даунтренде -> откат к среднему", flush=True)
    print("  up>50 на аптренде, up<50 на даунтренде -> следование тренду", flush=True)


if __name__ == "__main__":
    main()
