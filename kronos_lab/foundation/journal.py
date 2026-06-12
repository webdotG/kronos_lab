"""
kronos_lab/foundation/journal.py

Ядро стенда: калибровка + бумажная торговля follow/fade.
Комиссия берётся из config (по умолчанию реалистичный mixed 0.136%).

Две частоты:
- генерация ежечасно -> богатая калибровка (все строки)
- P&L по НЕЗАВИСИМОМУ суточному срезу -> честная доходность без перекрытий
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from foundation.analytics import summarize
from foundation.ensemble import ForecastEnsemble

ROUND_TURN_FEE_PCT = config.FEE_DEFAULT * 100

COLUMNS = [
    "forecast_id", "generated_at", "target_at", "symbol", "model_id",
    "last_close",
    "upside_prob", "median_terminal", "skew", "excess_kurtosis", "var5", "cvar5",
    "p5", "p25", "p50", "p75", "p95",
    "n_paths", "T", "top_k", "top_p", "seed", "ensemble_prefix",
    "signal_side", "follow_pnl", "fade_pnl", "follow_pnl_net", "fade_pnl_net",
    "actual_close", "actual_high", "actual_low", "outcome_filled",
]


def _parse_dt(s: str) -> datetime:
    dt = pd.to_datetime(s)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime()


def signal_from_ensemble(ens: ForecastEnsemble, horizon_hours: int = None) -> dict:
    horizon_hours = horizon_hours or config.HORIZON_HOURS
    s = summarize(ens)
    pq = s["price_quantiles"]
    gen_raw = ens.context_timestamp[-1] if ens.context_timestamp else ens.meta.get("generated_at_utc")
    gen_dt = _parse_dt(str(gen_raw))
    target_dt = gen_dt + timedelta(hours=horizon_hours)
    m = ens.meta
    side = "long" if s["upside_prob_%"] >= 50.0 else "short"
    return {
        "forecast_id": f"{m.get('symbol', 'NA')}_{gen_dt.strftime('%Y%m%dT%H%M%S')}",
        "generated_at": gen_dt.isoformat(),
        "target_at": target_dt.isoformat(),
        "symbol": m.get("symbol", "NA"),
        "model_id": m.get("model_id", "NA"),
        "last_close": round(s["last_close"], 4),
        "upside_prob": round(s["upside_prob_%"], 3),
        "median_terminal": round(s["median_terminal"], 4),
        "skew": round(s["skew"], 5),
        "excess_kurtosis": round(s["excess_kurtosis"], 5),
        "var5": round(s["VaR_5%_(loss%)"], 4),
        "cvar5": round(s["CVaR_5%_(loss%)"], 4),
        "p5": round(pq["p5"], 4), "p25": round(pq["p25"], 4), "p50": round(pq["p50"], 4),
        "p75": round(pq["p75"], 4), "p95": round(pq["p95"], 4),
        "n_paths": m.get("n_paths"), "T": m.get("T"), "top_k": m.get("top_k"),
        "top_p": m.get("top_p"), "seed": m.get("seed"),
        "ensemble_prefix": m.get("ensemble_prefix", ""),
        "signal_side": side,
        "follow_pnl": "", "fade_pnl": "", "follow_pnl_net": "", "fade_pnl_net": "",
        "actual_close": "", "actual_high": "", "actual_low": "", "outcome_filled": 0,
    }


def settle_pnl(last_close: float, actual_close: float, side: str) -> dict:
    raw_pct = (actual_close / last_close - 1.0) * 100
    follow = raw_pct if side == "long" else -raw_pct
    fade = -follow
    return {
        "follow_pnl": round(follow, 5), "fade_pnl": round(fade, 5),
        "follow_pnl_net": round(follow - ROUND_TURN_FEE_PCT, 5),
        "fade_pnl_net": round(fade - ROUND_TURN_FEE_PCT, 5),
    }


def append_row(row: dict, journal_path: str = None) -> None:
    journal_path = journal_path or config.JOURNAL_PATH
    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    new_file = not os.path.exists(journal_path)
    with open(journal_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def load_journal(journal_path: str = None) -> pd.DataFrame:
    return pd.read_csv(journal_path or config.JOURNAL_PATH)


def select_independent(df: pd.DataFrame, gap_hours: int = None) -> pd.DataFrame:
    """Жадно отбирает непересекающиеся сделки: каждая >= gap_hours после предыдущей выбранной."""
    gap_hours = gap_hours or config.INDEP_GAP_HOURS
    d = df.copy()
    d["_gen"] = d["generated_at"].apply(_parse_dt)
    d = d.sort_values("_gen")
    keep, last = [], None
    for _, r in d.iterrows():
        if last is None or (r["_gen"] - last) >= timedelta(hours=gap_hours):
            keep.append(r.name)
            last = r["_gen"]
    return df.loc[keep]


def _calib(done: pd.DataFrame) -> dict:
    for c in ["last_close", "upside_prob", "p5", "p95", "actual_close", "actual_high", "actual_low"]:
        done[c] = pd.to_numeric(done[c], errors="coerce")
    up_hit = (done["actual_close"] > done["last_close"]).astype(int)
    in_cone = ((done["actual_close"] >= done["p5"]) & (done["actual_close"] <= done["p95"])).astype(int)
    return {
        "n": int(len(done)),
        "claimed_upside_%": round(float(done["upside_prob"].mean()), 1),
        "realized_upside_%": round(float(up_hit.mean() * 100), 1),
        "cone_coverage_%": round(float(in_cone.mean() * 100), 1),
    }


def _paper(done: pd.DataFrame) -> dict:
    d = done.copy()
    for c in ["follow_pnl_net", "fade_pnl_net"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["follow_pnl_net"])
    if len(d) == 0:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "follow_total_%": round(float(d["follow_pnl_net"].sum()), 2),
        "follow_win_%": round(float((d["follow_pnl_net"] > 0).mean() * 100), 1),
        "fade_total_%": round(float(d["fade_pnl_net"].sum()), 2),
        "fade_win_%": round(float((d["fade_pnl_net"] > 0).mean() * 100), 1),
    }


def evaluate(df: pd.DataFrame) -> dict:
    done = df[df["outcome_filled"] == 1].copy()
    if len(done) == 0:
        return {"n": 0}
    out = {"overall_calib": _calib(done.copy()), "overall_paper": _paper(done)}
    indep = select_independent(done)
    out["indep_paper"] = _paper(indep)
    out["indep_n"] = int(len(indep))
    # разбивка по монетам
    out["by_coin"] = {}
    for sym, g in done.groupby("symbol"):
        out["by_coin"][sym] = {"calib": _calib(g.copy()), "paper": _paper(g)}
    return out


def format_report(out: dict) -> str:
    if out.get("n", -1) == 0:
        return "нет закрытых прогнозов - исходы ещё не наступили"
    c = out["overall_calib"]
    op = out["overall_paper"]
    ip = out["indep_paper"]
    L = [
        f"закрытых прогнозов: {c['n']}  (независимых суточных: {out['indep_n']})",
        "",
        "-- калибровка (все строки) --",
        f"  upside заявлено/реально: {c['claimed_upside_%']}% / {c['realized_upside_%']}%   разрыв {round(c['claimed_upside_%']-c['realized_upside_%'],1)}%",
        f"  покрытие конуса p5..p95: {c['cone_coverage_%']}%  (ожидание ~90%)",
        "",
        "-- P&L бумажный, net после комиссии " + f"{round(ROUND_TURN_FEE_PCT,3)}% --",
        f"  ВСЕ строки (перекрытые, доверять осторожно):",
        f"    follow итог {op.get('follow_total_%','-')}%  винрейт {op.get('follow_win_%','-')}%",
        f"    fade   итог {op.get('fade_total_%','-')}%  винрейт {op.get('fade_win_%','-')}%",
        f"  НЕЗАВИСИМЫЕ суточные (этому верь):",
        f"    follow итог {ip.get('follow_total_%','-')}%  винрейт {ip.get('follow_win_%','-')}%",
        f"    fade   итог {ip.get('fade_total_%','-')}%  винрейт {ip.get('fade_win_%','-')}%",
        "",
        "-- по монетам (калибровка | P&L follow/fade, все строки) --",
    ]
    for sym, s in out["by_coin"].items():
        ca, pa = s["calib"], s["paper"]
        L.append(f"  {sym}: upside {ca['claimed_upside_%']}/{ca['realized_upside_%']}%  конус {ca['cone_coverage_%']}%"
                 f"  | {pa.get('follow_total_%','-')}% / {pa.get('fade_total_%','-')}%  (n={ca['n']})")
    if out["indep_n"] < 20:
        L += ["", f"[warn] независимых наблюдений {out['indep_n']} - анекдот, не статистика. нужно >= ~20-30"]
    return "\n".join(L)


if __name__ == "__main__":
    from foundation.ensemble import make_synthetic_ensemble
    jp = "/tmp/journal_lab_selftest.csv"
    if os.path.exists(jp):
        os.remove(jp)
    rng = np.random.default_rng(0)
    coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    base = {"BTCUSDT": 77000, "ETHUSDT": 4000, "SOLUSDT": 180}
    for i in range(30):
        sym = coins[i % 3]
        ens = make_synthetic_ensemble(n_paths=500, last_close=base[sym], up_drift=0.003,
                                      base_vol=0.004, crash_prob=0.0, seed=i)
        # генерим ежечасно
        ens.context_timestamp = [f"2026-06-{1+i//3:02d} {(i*1)%24:02d}:00:00"]
        ens.meta.update({"symbol": sym, "model_id": "synthetic", "n_paths": 500,
                         "T": 1.0, "top_k": 0, "top_p": 1.0, "seed": i, "ensemble_prefix": f"/tmp/e{i}"})
        row = signal_from_ensemble(ens)
        real = ens.last_close * (1 + rng.normal(-0.01, 0.008))  # рынок чаще вниз
        row.update(settle_pnl(ens.last_close, real, row["signal_side"]))
        row["actual_close"] = round(real, 2)
        row["actual_high"] = round(real * 1.01, 2)
        row["actual_low"] = round(real * 0.99, 2)
        row["outcome_filled"] = 1
        append_row(row, jp)

    out = evaluate(load_journal(jp))
    print(format_report(out))
    assert out["indep_n"] < out["overall_calib"]["n"], "независимых должно быть меньше, чем всех"
    assert len(out["by_coin"]) == 3, "должны быть все три монеты"
    print("\njournal kronos_lab: независимый срез и разбивка по монетам работают")


# --- контекстные колонки для backtest (живой журнал их не использует, чтоб не ломать формат) ---
CONTEXT_COLUMNS = ["ctx_realized_vol", "ctx_trend", "ctx_zscore", "ctx_range_pct"]
BACKTEST_COLUMNS = COLUMNS + CONTEXT_COLUMNS


def append_row_cols(row: dict, path: str, columns: list) -> None:
    """Универсальный writer под заданный набор колонок (для backtest с контекст-фичами)."""
    import csv as _csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=columns)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in columns})
