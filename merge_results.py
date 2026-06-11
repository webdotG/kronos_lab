"""
merge_results.py - ШАГ 3. Склеивает журналы воркеров в один backtest_journal.csv.
Ансамбли остаются в своих ens_shard* папках (анализ читает все).
"""
import os, glob
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_lab", "data")


def main():
    shards = sorted(glob.glob(os.path.join(DATA, "journal_shard*.csv")))
    if not shards:
        print("нет журналов воркеров journal_shard*.csv"); return
    dfs = [pd.read_csv(s) for s in shards]
    merged = (pd.concat(dfs, ignore_index=True)
              .drop_duplicates("generated_at").sort_values("generated_at"))
    out = os.path.join(DATA, "backtest_journal.csv")
    merged.to_csv(out, index=False)
    print(f"склеено {len(shards)} шардов -> {len(merged)} строк -> {out}")
    for s in shards:
        print(f"  {os.path.basename(s)}: {len(pd.read_csv(s))} строк")


if __name__ == "__main__":
    main()
