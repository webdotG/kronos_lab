"""
foundation/ensemble.py

Чистый numpy-слой. Никакого torch, никакой модели, никакого GPU.
Здесь живёт КОНТРАКТ ДАННЫХ, на котором стоят все 6 инсайтов:
сырой ансамбль путей Монте-Карло, а не схлопнутая средняя линия.

Всё, что считают инсайты (skew, хвосты, ширина конуса, калибровка,
кросс-секция, синтетика), читает именно ForecastEnsemble и ничего больше.
Это позволяет писать и гонять тесты математики без запуска модели.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np


# индексы фичей в осях paths[..., k]
OHLCV_FEATURES = ["open", "high", "low", "close", "volume", "amount"]


@dataclass
class ForecastEnsemble:
    """
    Единственный объект, который генерация отдаёт, а все инсайты потребляют.

    paths: np.ndarray формы [n_paths, horizon, n_features]
        n_paths    - число траекторий Монте-Карло (сотни-тысячи, не 30)
        horizon    - число шагов прогноза (например 24 часа)
        n_features - порядок как в feature_names (обычно OHLCV + amount)
    feature_names: имена фичей в порядке последней оси paths
    last_close: цена закрытия последней известной свечи (точка отсчёта прогноза)
    y_timestamp: таймстампы прогнозных шагов (iso-строки), длина horizon
    context_close: исторические close, поданные в модель как контекст
    context_timestamp: таймстампы контекста (iso-строки)
    meta: параметры генерации - n_paths, horizon, T, top_k, top_p, seed,
          symbol, model_id, max_context, generated_at_utc и т.д.
          Это критично для инсайта 4 (калибровка): без зафиксированных
          параметров сэмплирования прогнозы несопоставимы между собой.
    """

    paths: np.ndarray
    feature_names: list
    last_close: float
    y_timestamp: list
    context_close: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    context_timestamp: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- удобные проекции, чтобы инсайты не лезли в индексы руками ---

    def feature(self, name: str) -> np.ndarray:
        """Вернуть срез [n_paths, horizon] по одной фиче."""
        if name not in self.feature_names:
            raise KeyError(f"нет фичи {name!r}, есть: {self.feature_names}")
        k = self.feature_names.index(name)
        return self.paths[:, :, k]

    def close_paths(self) -> np.ndarray:
        """[n_paths, horizon] цен закрытия - основной материал для большинства инсайтов."""
        return self.feature("close")

    def terminal_close(self) -> np.ndarray:
        """[n_paths] цена закрытия в конце горизонта по каждому пути."""
        return self.close_paths()[:, -1]

    def log_returns_terminal(self) -> np.ndarray:
        """[n_paths] лог-доходность конца горизонта относительно last_close."""
        return np.log(self.terminal_close() / self.last_close)

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def horizon(self) -> int:
        return self.paths.shape[1]


# --------------------------------------------------------------------------
# сохранение / загрузка: дорогую генерацию делаем ОДИН раз, анализ - сколько угодно
# --------------------------------------------------------------------------

def save_ensemble(ens: ForecastEnsemble, path_prefix: str) -> None:
    """
    Пишет {prefix}.npz (тяжёлые массивы) + {prefix}.meta.json (всё остальное).
    Разделение нужно, чтобы быстро читать meta без распаковки массивов.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
    np.savez_compressed(
        f"{path_prefix}.npz",
        paths=ens.paths,
        context_close=ens.context_close,
    )
    side = {
        "feature_names": ens.feature_names,
        "last_close": ens.last_close,
        "y_timestamp": ens.y_timestamp,
        "context_timestamp": ens.context_timestamp,
        "meta": ens.meta,
    }
    with open(f"{path_prefix}.meta.json", "w") as f:
        json.dump(side, f, indent=2, default=str)


def load_ensemble(path_prefix: str) -> ForecastEnsemble:
    arr = np.load(f"{path_prefix}.npz")
    with open(f"{path_prefix}.meta.json") as f:
        side = json.load(f)
    return ForecastEnsemble(
        paths=arr["paths"],
        feature_names=side["feature_names"],
        last_close=side["last_close"],
        y_timestamp=side["y_timestamp"],
        context_close=arr["context_close"],
        context_timestamp=side["context_timestamp"],
        meta=side["meta"],
    )


# --------------------------------------------------------------------------
# валидация контракта: ловим битый ансамбль до того, как он отравит инсайт
# --------------------------------------------------------------------------

def validate(ens: ForecastEnsemble) -> None:
    p = ens.paths
    assert p.ndim == 3, f"paths должен быть 3D [n_paths, horizon, feat], получено {p.shape}"
    assert p.shape[2] == len(ens.feature_names), "число фичей не совпадает с feature_names"
    assert len(ens.y_timestamp) == p.shape[1], "длина y_timestamp != horizon"
    assert np.isfinite(p).all(), "в paths есть NaN/inf - модель или денормализация сломаны"
    assert ens.last_close > 0, "last_close должен быть положительным"
    # хвост из 30 путей статистически бесполезен для skew/хвостов - предупреждаем
    if p.shape[0] < 100:
        print(f"[warn] всего {p.shape[0]} путей: для оценки хвостов и skew мало, нужно >= ~200")


# --------------------------------------------------------------------------
# синтетический ансамбль: фикстур с ИЗВЕСТНЫМ ответом для тестов математики.
# Намеренно делаем отрицательно скошенное распределение (тот самый паттерн
# с графика: средняя плоская, левый хвост проваливается, но upside высокий),
# чтобы тесты инсайта 3 проверялись на данных, где ответ заранее понятен.
# --------------------------------------------------------------------------

def make_synthetic_ensemble(
    n_paths: int = 500,
    horizon: int = 24,
    last_close: float = 60000.0,
    up_drift: float = 0.001,
    base_vol: float = 0.004,
    crash_prob: float = 0.03,
    crash_mag: float = 0.06,
    seed: int = 0,
) -> ForecastEnsemble:
    rng = np.random.default_rng(seed)
    feats = list(OHLCV_FEATURES)
    paths = np.zeros((n_paths, horizon, len(feats)), dtype=np.float32)

    for i in range(n_paths):
        price = last_close
        for t in range(horizon):
            r = rng.normal(up_drift, base_vol)
            if rng.random() < crash_prob:               # редкий, но сильный обвал -> левый хвост
                r -= abs(rng.normal(crash_mag, crash_mag * 0.3))
            o = price
            price = price * (1.0 + r)
            c = price
            hi = max(o, c) * (1.0 + abs(rng.normal(0, base_vol / 2)))
            lo = min(o, c) * (1.0 - abs(rng.normal(0, base_vol / 2)))
            v = abs(rng.normal(1000.0, 300.0))
            paths[i, t] = [o, hi, lo, c, v, c * v]

    return ForecastEnsemble(
        paths=paths,
        feature_names=feats,
        last_close=float(last_close),
        y_timestamp=[f"step_{t}" for t in range(horizon)],
        context_close=np.full(48, last_close, dtype=np.float32),
        context_timestamp=[f"ctx_{t}" for t in range(48)],
        meta={
            "synthetic": True,
            "n_paths": n_paths,
            "horizon": horizon,
            "up_drift": up_drift,
            "base_vol": base_vol,
            "crash_prob": crash_prob,
            "crash_mag": crash_mag,
            "seed": seed,
        },
    )


if __name__ == "__main__":
    # быстрый self-check контракта (гоняется без GPU/torch)
    ens = make_synthetic_ensemble(n_paths=500, seed=42)
    validate(ens)
    term = ens.terminal_close()
    upside = float((term > ens.last_close).mean())
    print("paths shape       :", ens.paths.shape)
    print("last_close        :", ens.last_close)
    print("terminal mean     :", round(float(term.mean()), 2))
    print("terminal min/max  :", round(float(term.min()), 2), "/", round(float(term.max()), 2))
    print("upside probability:", round(upside * 100, 1), "%")
    save_ensemble(ens, "/tmp/ens_selftest")
    back = load_ensemble("/tmp/ens_selftest")
    assert np.allclose(back.paths, ens.paths)
    print("save/load roundtrip: OK")
