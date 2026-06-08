"""
kronos_lab/cli/evaluate_journal.py

Отчёт по журналу: калибровка, P&L (все строки и независимый суточный срез), по монетам.
Запуск: python cli/evaluate_journal.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from foundation.journal import load_journal, evaluate, format_report


def main():
    if not os.path.exists(config.JOURNAL_PATH):
        print("журнала пока нет:", config.JOURNAL_PATH)
        return
    df = load_journal(config.JOURNAL_PATH)
    done = int((df["outcome_filled"] == 1).sum())
    print(f"всего прогнозов: {len(df)}, закрыто: {done}\n")
    print(format_report(evaluate(df)))


if __name__ == "__main__":
    main()
