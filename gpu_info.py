"""
gpu_info.py - ШАГ 1. Запускается внутри контейнера с --gpus all.
Смотрит все видимые карты, считает, как поделить точки пропорционально
свободной памяти, и ГЕНЕРИРУЕТ готовый скрипт запуска воркеров (по одному на карту).

Также разово качает историю BTC (чтобы воркеры её не качали наперегонки).

Печатает план + пишет в смонтированную папку:
  - gpu_plan.json       - машинно-читаемый план
  - run_workers.sh      - готовый скрипт: запустить и поедет на всех картах
"""
import os, json, sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_lab", "data")


def enumerate_gpus():
    import torch
    devs = []
    if not torch.cuda.is_available():
        return devs
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        devs.append({"index": i, "name": torch.cuda.get_device_name(i),
                     "free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2)})
    return devs


def plan_split(devices, n_points):
    """Пропорционально свободной памяти, непересекающиеся срезы. Большой карте больше точек."""
    w = [max(d["free_gb"], 0.1) for d in devices]
    tot = sum(w)
    bounds, acc = [0], 0.0
    for x in w[:-1]:
        acc += x / tot
        bounds.append(int(round(acc * n_points)))
    bounds.append(n_points)
    shards = []
    for i, d in enumerate(devices):
        shards.append({"device": d["index"], "name": d["name"], "free_gb": d["free_gb"],
                       "slice_start": bounds[i], "slice_end": bounds[i + 1],
                       "n": bounds[i + 1] - bounds[i]})
    return shards


def write_worker_script(shards, n_points, n_paths, path):
    lines = ['#!/bin/bash',
             '# АВТО-СГЕНЕРИРОВАНО gpu_info. Запускать из папки проекта: bash results/run_workers.sh',
             'set -e', '']
    for s in shards:
        name = f"kronos-bt-{s['device']}"
        lines.append(f"# карта {s['device']} ({s['name']}, {s['free_gb']}ГБ) -> {s['n']} точек")
        lines.append(f"docker rm -f {name} 2>/dev/null || true")
        lines.append(
            f"docker run -d --name {name} --gpus '\"device={s['device']}\"' "
            f"-v \"$PWD/results:/app/kronos_lab/data\" kronos-bt backtest "
            f"--n_points {n_points} --slice_start {s['slice_start']} --slice_end {s['slice_end']} "
            f"--n_paths {n_paths} "
            f"--out /app/kronos_lab/data/journal_shard{s['device']}.csv "
            f"--ens_dir /app/kronos_lab/data/ens_shard{s['device']}")
        lines.append('')
    lines.append('echo "запущено воркеров: %d. логи: docker logs -f kronos-bt-<N>"' % len(shards))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    n_points = int(os.environ.get("N_POINTS", "1200"))
    n_paths = int(os.environ.get("N_PATHS", "500"))
    os.makedirs(DATA, exist_ok=True)

    # разовая закачка данных (пока один контейнер, без гонки воркеров)
    hist = os.path.join(DATA, "BTCUSDT_1h.csv")
    if not os.path.exists(hist):
        print("=== качаю историю BTC (разово) ===", flush=True)
        os.system(f"python {os.path.join(os.path.dirname(os.path.abspath(__file__)),'download_data.py')}")

    devs = enumerate_gpus()
    print("\n=== ВИДИМЫЕ КАРТЫ ===")
    if not devs:
        print("CUDA не видна! проверь docker run --gpus all и nvidia-container-toolkit")
        sys.exit(1)
    for d in devs:
        print(f"  [{d['index']}] {d['name']}: свободно {d['free_gb']} / всего {d['total_gb']} ГБ")

    shards = plan_split(devs, n_points)
    print(f"\n=== ПЛАН (всего {n_points} точек, {n_paths} путей) ===")
    for s in shards:
        print(f"  карта {s['device']} ({s['free_gb']}ГБ): точки {s['slice_start']}..{s['slice_end']} = {s['n']} шт")

    json.dump({"devices": devs, "shards": shards, "n_points": n_points, "n_paths": n_paths},
              open(os.path.join(DATA, "gpu_plan.json"), "w"), indent=2)
    ws = os.path.join(DATA, "run_workers.sh")
    write_worker_script(shards, n_points, n_paths, ws)
    print(f"\nплан сохранён. ТЕПЕРЬ ЗАПУСТИ:  bash results/run_workers.sh")


if __name__ == "__main__":
    main()
