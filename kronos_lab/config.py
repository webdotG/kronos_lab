"""
kronos_lab/config.py

Единственное место для всех настроек стенда.
Меняешь монеты/комиссию/параметры здесь, скрипты не трогаешь.
"""

import os

# --- пути (вычисляются от расположения этого файла) ---
LAB_DIR = os.path.dirname(os.path.abspath(__file__))          # .../kronos_lab
REPO_ROOT = os.path.dirname(LAB_DIR)                          # .../trading
KRONOS_MODEL_PATH = os.path.join(REPO_ROOT, "kronos_model")   # библиотека модели

DATA_DIR = os.path.join(LAB_DIR, "data")
ENSEMBLE_DIR = os.path.join(DATA_DIR, "ensembles")
JOURNAL_PATH = os.path.join(DATA_DIR, "journal.csv")

# --- монеты стенда ---
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# --- модель ---
MODEL_REPO = "NeoQuasar/Kronos-base"
MODEL_ID = "Kronos-base"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512

# --- параметры генерации (канон, фиксируем ради сопоставимости) ---
N_PATHS = 500
CHUNK = 0           # 0 = авто-подбор по свободной памяти карты (грузим по максимуму)
GPU_RESERVE_GB = 1.5  # сколько ГБ оставить свободными при авто-подборе
HORIZON_HOURS = 24
N_CONTEXT = 360
T = 1.0
TOP_K = 0
TOP_P = 1.0         # хвост не режем
SEED = 0

# --- комиссии Bybit non-vip linear perp (реальные данные, май 2026) ---
FEE_MAKER = 0.00036                       # 0.036%
FEE_TAKER = 0.00100                        # 0.100%
FEE_ROUND_TRIP_TAKER = FEE_TAKER + FEE_TAKER    # 0.200% консервативный
FEE_ROUND_TRIP_MIXED = FEE_MAKER + FEE_TAKER    # 0.136% реалистичный
FEE_ROUND_TRIP_MAKER = FEE_MAKER + FEE_MAKER    # 0.072% оптимистичный
# по умолчанию реалистичный: maker-вход (conditional limit) + taker-выход (market)
FEE_DEFAULT = FEE_ROUND_TRIP_MIXED              # 0.136%

# --- оценка ---
INDEP_GAP_HOURS = 24    # минимальный разрыв между независимыми сделками для честного P&L
