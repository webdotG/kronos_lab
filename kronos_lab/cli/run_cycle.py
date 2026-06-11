"""
kronos_lab/cli/run_cycle.py

Автоматический цикл сбора статистики по всем монетам из config.
ДЕНЕГ НЕ ТРОГАЕТ - только бумажная торговля и калибровка.

Один проход:
  модель грузится ОДИН раз, затем по каждой монете:
    свежий контекст с Bybit -> ансамбль -> сохранение -> запись прогноза + paper-сделки
  затем: закрыть созревшие исходы -> отчёт (калибровка + P&L + по монетам)

Запуск:  python cli/run_cycle.py
Под cron (каждый час в :05):
  5 * * * * cd ~/aProject/trading/kronos_lab && /path/venv_kronos/bin/python cli/run_cycle.py >> data/cycle.log 2>&1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import config
from foundation.bybit_data import fetch_recent_context
from foundation.generation import generate_ensemble
from foundation.ensemble import save_ensemble, validate
from foundation.journal import signal_from_ensemble, append_row, load_journal, evaluate, format_report
from cli.fetch_actual import close_matured

sys.path.append(config.KRONOS_MODEL_PATH)
from model import Kronos, KronosTokenizer, KronosPredictor


def main():
    print(f"=== CYCLE {pd.Timestamp.now('UTC')} model={config.MODEL_ID} coins={config.COINS} ===")

    tokenizer = KronosTokenizer.from_pretrained(config.TOKENIZER_REPO)
    model = Kronos.from_pretrained(config.MODEL_REPO)
    if torch.cuda.is_available():
        model = model.cuda()
    predictor = KronosPredictor(model, tokenizer, max_context=config.MAX_CONTEXT)

    for symbol in config.COINS:
        try:
            x_df, x_timestamp = fetch_recent_context(symbol, config.N_CONTEXT)
            last_date = x_timestamp.iloc[-1]
            y_timestamp = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                        periods=config.HORIZON_HOURS, freq="1h").to_series().reset_index(drop=True)
            ens = generate_ensemble(
                predictor=predictor, df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=config.HORIZON_HOURS, n_paths=config.N_PATHS, chunk=config.CHUNK,
                T=config.T, top_k=config.TOP_K, top_p=config.TOP_P, seed=config.SEED,
                symbol=symbol, model_id=config.MODEL_ID,
            )
            validate(ens)
            prefix = os.path.join(config.ENSEMBLE_DIR,
                                  f"{symbol}_{str(last_date).replace(':', '-').replace(' ', '_')}")
            ens.meta["ensemble_prefix"] = prefix
            save_ensemble(ens, prefix)
            row = signal_from_ensemble(ens)
            append_row(row, config.JOURNAL_PATH)
            up = round(float((ens.terminal_close() > ens.last_close).mean() * 100), 1)
            print(f"  {symbol}: ансамбль ок, last={last_date}, side={row['signal_side']}, upside={up}%")
        except Exception as e:
            print(f"  {symbol}: ОШИБКА {e}")

    n_closed = close_matured(config.JOURNAL_PATH)
    print(f"закрыто созревших: {n_closed}")

    if os.path.exists(config.JOURNAL_PATH):
        out = evaluate(load_journal(config.JOURNAL_PATH))
        print("\n" + format_report(out))


if __name__ == "__main__":
    main()
