"""
kronos_lab/cli/backtest.py

Тяжёлый прогон по истории на ПРАВИЛЬНОЙ версии. Один проход -> сохраняем
сырьё, потом любые стратегии считаются бесплатно на CPU.

Контракт данных (сохраняем в каждой точке T):
  1. полный ансамбль .npz (paths [n_paths,24,6])           -> распределение, форма, Kelly, skew
  2. контекст-фичи в строке журнала (vol/trend/zscore/range) -> условные/режимные стратегии
  3. реальный почасовой путь будущего .realized.npz [24,4]   -> barrier, стоп/тейк, MFE/MAE, тайминг

Надёжность для unattended-прогона на чужой машине:
  - РЕЗЮМ: уже посчитанные точки пропускаются (читает out-журнал), можно прерывать/перезапускать
  - АВТО-CHUNK: при CUDA OOM сам уполовинивает батч и продолжает, а не падает
  - чекпоинт-лог с ETA, flush в файл

Шаг 24ч = независимые непересекающиеся точки = честная статистика.

Запуск (в докере через run_backtest.sh):
    python cli/backtest.py --n_points 1200 --step 24 --n_paths 500 --chunk 128
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import config
from foundation.generation import generate_ensemble
from foundation.ensemble import save_ensemble, validate
from foundation.analytics import context_features
from foundation.journal import signal_from_ensemble, settle_pnl, append_row_cols, BACKTEST_COLUMNS

sys.path.append(config.KRONOS_MODEL_PATH)
from model import Kronos, KronosTokenizer, KronosPredictor

DEFAULT_HIST = os.path.join(config.DATA_DIR, "BTCUSDT_1h.csv")


def log(msg):
    print(f"[{pd.Timestamp.now('UTC').strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_hourly(path):
    """Читает подготовленный часовой csv с заголовком dt,open,high,low,close,volume,amount."""
    df = pd.read_csv(path, parse_dates=["dt"]).set_index("dt").sort_index()
    need = ["open", "high", "low", "close", "volume", "amount"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"в {path} нет колонок {miss}. ожидается dt,open,high,low,close,volume,amount")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default=DEFAULT_HIST)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n_paths", type=int, default=500)
    ap.add_argument("--step", type=int, default=24)
    ap.add_argument("--n_points", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--n_ctx", type=int, default=config.N_CONTEXT)
    ap.add_argument("--horizon", type=int, default=config.HORIZON_HOURS)
    ap.add_argument("--out", default=os.path.join(config.DATA_DIR, "backtest_journal.csv"))
    ap.add_argument("--ens_dir", default=os.path.join(config.DATA_DIR, "backtest_ensembles"))
    args = ap.parse_args()

    os.makedirs(args.ens_dir, exist_ok=True)
    h = load_hourly(args.hist)
    n = len(h)
    first_T = args.n_ctx
    last_T = n - args.horizon - 1
    all_T = list(range(first_T, last_T, args.step))
    points = all_T[-args.n_points:]
    log(f"часов истории: {n}, валидных точек (шаг {args.step}ч): {len(all_T)}, берём {len(points)}")
    log(f"период: {h.index[points[0]]} .. {h.index[points[-1]]}")

    # РЕЗЮМ: что уже посчитано
    done_ts = set()
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out)
            done_ts = set(prev["generated_at"].astype(str))
            log(f"резюм: уже посчитано {len(done_ts)} точек, продолжаю")
        except Exception:
            pass

    log("гружу токенизатор и модель...")
    tokenizer = KronosTokenizer.from_pretrained(config.TOKENIZER_REPO)
    model = Kronos.from_pretrained(config.MODEL_REPO)
    device = "cpu"
    if torch.cuda.is_available():
        try:
            model = model.cuda(); torch.zeros(1).cuda(); device = "cuda:0"
            log(f"GPU: {torch.cuda.get_device_name(0)}")
        except torch.OutOfMemoryError:
            log("GPU занята/мала -> CPU"); model = model.cpu(); torch.cuda.empty_cache()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=config.MAX_CONTEXT)

    cur_chunk = args.chunk
    t_start = time.time()
    computed = 0

    for i, T in enumerate(points):
        ctx = h.iloc[T - args.n_ctx:T]
        last_date = ctx.index[-1]
        if last_date.isoformat() in done_ts:
            continue  # резюм

        x_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = ctx.index.to_series().reset_index(drop=True)
        y_ts = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                             periods=args.horizon, freq="1h").to_series().reset_index(drop=True)

        # генерация с авто-откатом chunk при OOM
        ens = None
        while ens is None:
            try:
                ens = generate_ensemble(
                    predictor=predictor, df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                    pred_len=args.horizon, n_paths=args.n_paths, chunk=cur_chunk,
                    T=config.T, top_k=config.TOP_K, top_p=config.TOP_P, seed=config.SEED,
                    symbol=args.symbol, model_id=config.MODEL_ID,
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_chunk <= 1:
                    log("OOM даже на chunk=1, пропуск точки"); break
                cur_chunk = max(1, cur_chunk // 2)
                log(f"OOM -> уменьшаю chunk до {cur_chunk}")
        if ens is None:
            continue

        try:
            validate(ens)
            prefix = os.path.join(args.ens_dir, f"{args.symbol}_{last_date.strftime('%Y%m%dT%H%M%S')}")
            ens.meta["ensemble_prefix"] = prefix
            save_ensemble(ens, prefix)

            # реальный почасовой путь будущего из истории [horizon, 4] OHLC
            future = h.iloc[T:T + args.horizon]
            realized = future[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)
            np.savez_compressed(f"{prefix}.realized.npz", realized=realized,
                                realized_ts=np.array([str(t) for t in future.index]))

            row = signal_from_ensemble(ens)
            row.update(context_features(ctx))
            actual_close = float(future["close"].iloc[-1])
            row["actual_close"] = round(actual_close, 4)
            row["actual_high"] = round(float(future["high"].max()), 4)
            row["actual_low"] = round(float(future["low"].min()), 4)
            row.update(settle_pnl(float(row["last_close"]), actual_close, row["signal_side"]))
            row["outcome_filled"] = 1
            append_row_cols(row, args.out, BACKTEST_COLUMNS)

            computed += 1
            el = time.time() - t_start
            eta = el / computed * (len(points) - i - 1) / 3600
            log(f"[{i+1}/{len(points)}] {last_date} up={row['upside_prob']}% "
                f"ход={round((ens.terminal_close().mean()/ens.last_close-1)*100,2)}% "
                f"реал={round((actual_close/float(row['last_close'])-1)*100,2)}% "
                f"chunk={cur_chunk} ETA~{eta:.1f}ч")
        except Exception as e:
            log(f"[{i+1}/{len(points)}] {last_date} ОШИБКА записи: {e}")

    log(f"готово. посчитано в этом запуске: {computed}. журнал: {args.out}")
    log(f"анализ: python cli/whatif_trade.py {args.out}")


if __name__ == "__main__":
    main()
