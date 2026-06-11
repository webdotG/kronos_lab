# образ с torch+CUDA, совместим с драйвером 580 (RTX 4000 Ada / 4060)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# зависимости анализа и работы
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# библиотека модели Kronos (даёт пакет model/) - клонируем публичный репозиторий
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /app/kronos_model

# наш стенд
COPY kronos_lab /app/kronos_lab
COPY download_data.py /app/download_data.py
COPY gpu_info.py /app/gpu_info.py
COPY merge_results.py /app/merge_results.py
COPY run_backtest.sh /app/run_backtest.sh
RUN chmod +x /app/run_backtest.sh

# предзагрузка весов в образ (чтобы прогон не зависел от HF в рантайме)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('NeoQuasar/Kronos-Tokenizer-base'); \
    snapshot_download('NeoQuasar/Kronos-base')" || echo "предзагрузка весов пропущена, скачаются в рантайме"

ENTRYPOINT ["/app/run_backtest.sh"]
