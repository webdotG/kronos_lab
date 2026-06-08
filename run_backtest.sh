#!/bin/bash
set -e
cd /app/kronos_lab

# 1. данные: качаем свежую историю, если ещё нет
if [ ! -f /app/kronos_lab/data/BTCUSDT_1h.csv ]; then
  echo "=== качаю историю BTC с Bybit (разово) ==="
  python /app/download_data.py
fi

# 2. backtest. параметры можно переопределить через docker run ... -- args
echo "=== старт backtest ==="
exec python cli/backtest.py "$@"
