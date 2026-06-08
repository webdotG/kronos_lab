"""
kronos_lab/cli/record_signal.py
Записать сохранённый ансамбль в журнал как прогноз (ручной режим).
Запуск: python cli/record_signal.py <ensemble_prefix>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from foundation.ensemble import load_ensemble
from foundation.journal import signal_from_ensemble, append_row

def main():
    if len(sys.argv) != 2:
        print("использование: python cli/record_signal.py <ensemble_prefix>")
        sys.exit(1)
    ens = load_ensemble(sys.argv[1])
    ens.meta["ensemble_prefix"] = sys.argv[1]
    row = signal_from_ensemble(ens)
    append_row(row, config.JOURNAL_PATH)
    print(f"записан {row['forecast_id']} side={row['signal_side']} upside={row['upside_prob']}%")

if __name__ == "__main__":
    main()
