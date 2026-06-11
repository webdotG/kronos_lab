#!/bin/bash
# entrypoint-диспетчер: gpu_info | backtest | merge
set -e
cd /app/kronos_lab
CMD="${1:-backtest}"
case "$CMD" in
  gpu_info)
    python /app/gpu_info.py
    ;;
  merge)
    python /app/merge_results.py
    ;;
  backtest)
    shift
    if [ ! -f /app/kronos_lab/data/BTCUSDT_1h.csv ]; then
      echo "данные не найдены, качаю..."; python /app/download_data.py
    fi
    exec python cli/backtest.py "$@"
    ;;
  *)
    # без подкоманды - считаем backtest со всеми аргументами
    if [ ! -f /app/kronos_lab/data/BTCUSDT_1h.csv ]; then python /app/download_data.py; fi
    exec python cli/backtest.py "$@"
    ;;
esac
