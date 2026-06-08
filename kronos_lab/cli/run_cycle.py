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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch


def log(msg):
    """Лог с таймстампом и немедленным сбросом в файл/экран (для tail -f)."""
    print(f"[{pd.Timestamp.now('UTC').strftime('%H:%M:%S')}] {msg}", flush=True)

import config
from foundation.bybit_data import fetch_recent_context
from foundation.generation import generate_ensemble
from foundation.ensemble import save_ensemble, validate
from foundation.journal import signal_from_ensemble, append_row, load_journal, evaluate, format_report
from cli.fetch_actual import close_matured

sys.path.append(config.KRONOS_MODEL_PATH)
from model import Kronos, KronosTokenizer, KronosPredictor


def main():
    log(f"=== CYCLE model={config.MODEL_ID} coins={config.COINS} ===")

    t0 = time.time()
    log("гружу токенизатор и модель...")
    tokenizer = KronosTokenizer.from_pretrained(config.TOKENIZER_REPO)
    model = Kronos.from_pretrained(config.MODEL_REPO)
    if torch.cuda.is_available():
        model = model.cuda()
        log(f"модель на GPU за {time.time()-t0:.0f}с")
    else:
        log(f"модель на CPU за {time.time()-t0:.0f}с (GPU не найдена)")
    predictor = KronosPredictor(model, tokenizer, max_context=config.MAX_CONTEXT)

    for symbol in config.COINS:
        try:
            tc = time.time()
            log(f"{symbol}: тяну контекст...")
            x_df, x_timestamp = fetch_recent_context(symbol, config.N_CONTEXT)
            last_date = x_timestamp.iloc[-1]
            log(f"{symbol}: контекст ок ({len(x_df)} свечей, last={last_date}) за {time.time()-tc:.0f}с, генерю ансамбль...")
            tg = time.time()
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
            exp_ret = round((ens.close_paths()[:, -1].mean() / ens.last_close - 1) * 100, 2)
            log(f"{symbol}: ГОТОВО за {time.time()-tg:.0f}с  side={row['signal_side']} upside={up}% ожид.ход={exp_ret}%")
        except Exception as e:
            log(f"{symbol}: ОШИБКА {e}")

    log("закрываю созревшие исходы...")
    n_closed = close_matured(config.JOURNAL_PATH)
    log(f"закрыто созревших: {n_closed}")

    if os.path.exists(config.JOURNAL_PATH):
        out = evaluate(load_journal(config.JOURNAL_PATH))
        print("\n" + format_report(out), flush=True)
    log(f"=== цикл завершён за {time.time()-t0:.0f}с ===")


if __name__ == "__main__":
    main()
