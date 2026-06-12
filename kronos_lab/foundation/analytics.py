"""
foundation/analytics.py

Чистый numpy. Превращает сырой ForecastEnsemble в ФОРМУ РАСПРЕДЕЛЕНИЯ.
Это слой математики, на котором стоят инсайты 3 (skew/хвосты),
4 (калибровка) и 5 (ширина конуса -> сайзинг).

Принцип: средняя линия игнорируется, работаем с полным набором путей.
Все функции детерминированы и тестируются на синтетике без GPU.
"""

from __future__ import annotations

import numpy as np

try:
    from foundation.ensemble import ForecastEnsemble
except ImportError:
    from ensemble import ForecastEnsemble


def terminal_returns(ens: ForecastEnsemble) -> np.ndarray:
    """[n_paths] лог-доходность конца горизонта относительно last_close."""
    return np.log(ens.terminal_close() / ens.last_close)


def _skew(x: np.ndarray) -> float:
    """Популяционная асимметрия. Отрицательная = длинный левый хвост (риск обвала)."""
    z = x - x.mean()
    sd = x.std()
    return float((z ** 3).mean() / (sd ** 3 + 1e-12))


def _excess_kurtosis(x: np.ndarray) -> float:
    """Избыточный эксцесс. > 0 = хвосты толще нормального распределения."""
    z = x - x.mean()
    sd = x.std()
    return float((z ** 4).mean() / (sd ** 4 + 1e-12) - 3.0)


def summarize(ens: ForecastEnsemble) -> dict:
    """
    Полная сводка формы распределения по терминальной цене и доходности.
    VaR/CVaR считаются на уровне 5% (хвост худших 5% исходов).
    """
    r = terminal_returns(ens)                 # доходности
    pr = ens.terminal_close()                 # цены
    q_levels = [1, 5, 25, 50, 75, 95, 99]

    price_q = {f"p{q}": float(v) for q, v in zip(q_levels, np.percentile(pr, q_levels))}
    ret_q = {f"p{q}": float(v) for q, v in zip(q_levels, np.percentile(r, q_levels))}

    # VaR 5% и CVaR (expected shortfall) 5% по доходности, в процентах
    cut = np.percentile(r, 5)
    var5 = -float(cut) * 100
    tail = r[r <= cut]
    cvar5 = -float(tail.mean()) * 100 if tail.size else float("nan")

    # асимметрия хвостов вокруг медианы цены, в процентах от last_close
    med = np.percentile(pr, 50)
    left = (med - np.percentile(pr, 5)) / ens.last_close * 100
    right = (np.percentile(pr, 95) - med) / ens.last_close * 100

    # конус по шагам: ширина интервала p5..p95 (робастнее, чем min..max)
    cp = ens.close_paths()
    band = np.percentile(cp, 95, axis=0) - np.percentile(cp, 5, axis=0)
    minmax = cp.max(axis=0) - cp.min(axis=0)

    return {
        "n_paths": ens.n_paths,
        "horizon": ens.horizon,
        "last_close": float(ens.last_close),
        "upside_prob_%": float((pr > ens.last_close).mean()) * 100,
        "mean_terminal": float(pr.mean()),
        "median_terminal": float(med),
        "skew": _skew(r),
        "excess_kurtosis": _excess_kurtosis(r),
        "VaR_5%_(loss%)": var5,
        "CVaR_5%_(loss%)": cvar5,
        "left_tail_%": float(left),
        "right_tail_%": float(right),
        "tail_asymmetry_(L-R)%": float(left - right),
        "cone_p5p95_step1": float(band[0]),
        "cone_p5p95_step_last": float(band[-1]),
        "cone_minmax_step1": float(minmax[0]),
        "cone_minmax_step_last": float(minmax[-1]),
        "price_quantiles": price_q,
        "return_quantiles_%": {k: v * 100 for k, v in ret_q.items()},
        "meta": ens.meta,
    }

def context_features(df) -> dict:
    """Признаки контекста в момент прогноза (для условных/режимных стратегий)."""
    import numpy as np
    close = df["close"].to_numpy(dtype=float)
    rets = np.diff(np.log(close))
    return {
        "ctx_realized_vol": round(float(np.std(rets) * 100), 5),
        "ctx_trend": round(float((close[-1] / close[0] - 1) * 100), 4),
        "ctx_zscore": round(float((close[-1] - close.mean()) / (close.std() + 1e-9)), 4),
        "ctx_range_pct": round(float((df["high"].max() - df["low"].min()) / close[-1] * 100), 4),
    }

if __name__ == "__main__":
    # тест математики на синтетике с ИЗВЕСТНОЙ формой
    try:
        from foundation.ensemble import make_synthetic_ensemble
    except ImportError:
        from ensemble import make_synthetic_ensemble

    # узкий ансамбль почти без хвоста
    tight = make_synthetic_ensemble(n_paths=1000, crash_prob=0.0, base_vol=0.003, seed=1)
    # широкий с жирным левым хвостом (частые обвалы)
    fat = make_synthetic_ensemble(n_paths=1000, crash_prob=0.10, crash_mag=0.07, seed=1)

    st = summarize(tight)
    sf = summarize(fat)
    print(f"{'метрика':<26}{'tight':>14}{'fat-tail':>14}")
    for key in ["upside_prob_%", "skew", "excess_kurtosis", "VaR_5%_(loss%)",
                "CVaR_5%_(loss%)", "left_tail_%", "right_tail_%", "tail_asymmetry_(L-R)%"]:
        print(f"{key:<26}{st[key]:>14.3f}{sf[key]:>14.3f}")
    # проверки: fat должен иметь более отрицательный skew, толще хвост, больше CVaR
    assert sf["skew"] < st["skew"], "fat-tail должен быть скошен влево сильнее"
    assert sf["CVaR_5%_(loss%)"] > st["CVaR_5%_(loss%)"], "fat-tail должен иметь больший CVaR"
    assert sf["tail_asymmetry_(L-R)%"] > st["tail_asymmetry_(L-R)%"], "fat-tail левее асимметричнее"
    print("\nматематика формы распределения: проверки пройдены")
