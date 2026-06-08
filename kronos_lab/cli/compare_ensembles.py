"""
kronos_lab/cli/compare_ensembles.py
Сравнить два сохранённых ансамбля бок о бок.
Запуск: python cli/compare_ensembles.py <prefixA> <prefixB>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundation.ensemble import load_ensemble
from foundation.analytics import summarize

ROWS = [
    ("upside_prob_%", "upside %", "{:.1f}"),
    ("skew", "асимметрия", "{:.3f}"),
    ("excess_kurtosis", "эксцесс", "{:.3f}"),
    ("VaR_5%_(loss%)", "VaR 5%", "{:.2f}"),
    ("CVaR_5%_(loss%)", "CVaR 5%", "{:.2f}"),
    ("left_tail_%", "левый хвост %", "{:.2f}"),
    ("right_tail_%", "правый хвост %", "{:.2f}"),
    ("cone_p5p95_step_last", "конус финал", "{:.1f}"),
]

def main():
    if len(sys.argv) != 3:
        print("использование: python cli/compare_ensembles.py <prefixA> <prefixB>")
        sys.exit(1)
    a = summarize(load_ensemble(sys.argv[1]))
    b = summarize(load_ensemble(sys.argv[2]))
    print(f"A: {sys.argv[1]}  ({a['meta'].get('model_id')}, top_p={a['meta'].get('top_p')})")
    print(f"B: {sys.argv[2]}  ({b['meta'].get('model_id')}, top_p={b['meta'].get('top_p')})")
    print(f"\n{'метрика':<18}{'A':>12}{'B':>12}{'B-A':>12}")
    print("-" * 54)
    for key, label, f in ROWS:
        va, vb = a[key], b[key]
        print(f"{label:<18}{f.format(va):>12}{f.format(vb):>12}{f.format(vb-va):>12}")

if __name__ == "__main__":
    main()
