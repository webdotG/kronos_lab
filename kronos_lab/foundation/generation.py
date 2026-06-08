"""
foundation/generation.py

Torch-слой. Запускает модель Kronos и собирает СЫРОЙ ансамбль путей.

Ключевое отличие от штатного kronos.py: функция _infer_raw - это форк
auto_regressive_inference, из которого убрана строка np.mean(preds, axis=1).
Штатная либа генерит sample_count путей за один проход и тут же усредняет их
в среднюю линию. Мы возвращаем все пути, потому что вся ценность по плану -
в форме распределения, а не в средней.

Что добавлено сверху оригинала:
- сид -> воспроизводимость (для тестов и для калибровки инсайта 4)
- чанкинг по числу путей -> не упираемся в 2 ГБ VRAM на GT 1030
- денормализация всех путей (а не усреднённого), с теми же x_mean/x_std

Размещение: положить рядом с ensemble.py. Перед импортом model.kronos
нужен тот же sys.path.append на каталог kronos_model, что и в твоём скрипте.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import torch

# из библиотеки Kronos (kronos_model/model/kronos.py)
import config
import sys, os as _os
sys.path.append(config.KRONOS_MODEL_PATH)
from model.kronos import sample_from_logits, calc_time_stamps

from foundation.ensemble import ForecastEnsemble


@torch.no_grad()
def _infer_raw(predictor, x_norm, x_stamp, y_stamp, pred_len, sample_count, T, top_k, top_p):
    """
    Форк auto_regressive_inference для ОДНОГО ряда (B=1).
    Единственное смысловое отличие от оригинала: НЕ усредняем по sample_count,
    возвращаем все пути.

    Вход (нормализованный, numpy):
        x_norm  [seq, feat]
        x_stamp [seq, 5]
        y_stamp [pred_len, 5]
    Выход:
        np.ndarray [sample_count, win, feat] в нормализованном пространстве,
        где win = min(seq + pred_len, max_context)
    """
    tokenizer = predictor.tokenizer
    model = predictor.model
    device = predictor.device
    max_context = predictor.max_context
    clip = predictor.clip

    x = torch.from_numpy(np.asarray(x_norm, np.float32)).to(device).unsqueeze(0)        # [1, seq, feat]
    x_stamp_t = torch.from_numpy(np.asarray(x_stamp, np.float32)).to(device).unsqueeze(0)
    y_stamp_t = torch.from_numpy(np.asarray(y_stamp, np.float32)).to(device).unsqueeze(0)

    x = torch.clip(x, -clip, clip)

    # реплицируем B=1 -> sample_count независимых путей в одном батче
    x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
    x_stamp_t = x_stamp_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp_t.size(1), x_stamp_t.size(2))
    y_stamp_t = y_stamp_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp_t.size(1), y_stamp_t.size(2))

    x_token = tokenizer.encode(x, half=True)

    initial_seq_len = x.size(1)
    batch_size = x_token[0].size(0)               # == sample_count
    total_seq_len = initial_seq_len + pred_len
    full_stamp = torch.cat([x_stamp_t, y_stamp_t], dim=1)

    generated_pre = x_token[0].new_empty(batch_size, pred_len)
    generated_post = x_token[1].new_empty(batch_size, pred_len)

    pre_buffer = x_token[0].new_zeros(batch_size, max_context)
    post_buffer = x_token[1].new_zeros(batch_size, max_context)
    buffer_len = min(initial_seq_len, max_context)
    if buffer_len > 0:
        start_idx = max(0, initial_seq_len - max_context)
        pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
        post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

    for i in range(pred_len):
        current_seq_len = initial_seq_len + i
        window_len = min(current_seq_len, max_context)

        if current_seq_len <= max_context:
            input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
        else:
            input_tokens = [pre_buffer, post_buffer]

        context_end = current_seq_len
        context_start = max(0, context_end - max_context)
        current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

        s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
        s1_logits = s1_logits[:, -1, :]
        sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

        s2_logits = model.decode_s2(context, sample_pre)
        s2_logits = s2_logits[:, -1, :]
        sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

        generated_pre[:, i] = sample_pre.squeeze(-1)
        generated_post[:, i] = sample_post.squeeze(-1)

        if current_seq_len < max_context:
            pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
            post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
        else:
            pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
            post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
            pre_buffer[:, -1] = sample_pre.squeeze(-1)
            post_buffer[:, -1] = sample_post.squeeze(-1)

    full_pre = torch.cat([x_token[0], generated_pre], dim=1)
    full_post = torch.cat([x_token[1], generated_post], dim=1)

    context_start = max(0, total_seq_len - max_context)
    input_tokens = [
        full_pre[:, context_start:total_seq_len].contiguous(),
        full_post[:, context_start:total_seq_len].contiguous(),
    ]
    z = tokenizer.decode(input_tokens, half=True)        # [sample_count, win, feat]
    # ОРИГИНАЛ ТУТ ДЕЛАЛ np.mean(..., axis=sample) - МЫ НЕ ДЕЛАЕМ
    return z.cpu().numpy()


def generate_ensemble(
    predictor,
    df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int = 24,
    n_paths: int = 200,
    chunk: int = 32,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int = 0,
    symbol: str = "BTCUSDT",
    model_id: str = "Kronos-small",
) -> ForecastEnsemble:
    """
    Собрать сырой ансамбль из n_paths путей и вернуть ForecastEnsemble.

    df: контекстные свечи с колонками open, high, low, close, volume (+ amount опц.)
    chunk: сколько путей гнать за один проход. Уменьшить, если ловишь OOM на GT 1030;
           увеличить ради скорости, если памяти хватает. На результат (распределение)
           chunk не влияет, влияет только на скорость и точную раскладку RNG.

    Нормализация/денормализация повторяют логику KronosPredictor.predict,
    чтобы числа были в той же шкале, что и у штатного прогноза.
    """
    price_cols = predictor.price_cols                 # ['open','high','low','close']
    vol_col = predictor.vol_col                       # 'volume'
    amt_col = predictor.amt_vol                        # 'amount'
    cols = list(price_cols) + [vol_col, amt_col]

    df = df.copy()
    if amt_col not in df.columns:
        df[amt_col] = df[vol_col] * df[price_cols].mean(axis=1)
    if df[cols].isnull().values.any():
        raise ValueError("в контексте есть NaN в цене/объёме")

    x_raw = df[cols].values.astype(np.float32)        # [seq, feat]
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_norm = np.clip((x_raw - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

    # воспроизводимость: один сид на весь прогон, чанки идут по одному RNG-потоку
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    collected = []
    done = 0
    while done < n_paths:
        c = min(chunk, n_paths - done)
        raw = _infer_raw(predictor, x_norm, x_stamp, y_stamp, pred_len, c, T, top_k, top_p)
        raw = raw[:, -pred_len:, :]                   # [c, pred_len, feat], нормализованное
        collected.append(raw)
        done += c

    paths_norm = np.concatenate(collected, axis=0)    # [n_paths, pred_len, feat]
    paths = paths_norm * (x_std + 1e-5) + x_mean      # денорм, broadcast по фиче
    paths = paths.astype(np.float32)

    last_close = float(df[price_cols[3]].iloc[-1])

    return ForecastEnsemble(
        paths=paths,
        feature_names=cols,
        last_close=last_close,
        y_timestamp=[str(t) for t in pd.Index(y_timestamp)],
        context_close=df[price_cols[3]].values.astype(np.float32),
        context_timestamp=[str(t) for t in pd.Index(x_timestamp)],
        meta={
            "n_paths": n_paths,
            "horizon": pred_len,
            "T": T,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "chunk": chunk,
            "symbol": symbol,
            "model_id": model_id,
            "max_context": predictor.max_context,
            "generated_at_utc": _dt.datetime.utcnow().isoformat(),
        },
    )
