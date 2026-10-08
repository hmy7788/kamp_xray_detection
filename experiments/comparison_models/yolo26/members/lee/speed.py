"""추론 속도 측정 (현장 활용 근거). entry.py 가 학습·예측 뒤에 부르고, 지난 실험은 measure_speed.py 로 따로 잰다.

잰 것 (runs/<exp_id>/speed_seed<seed>.json):
  params_m        파라미터 수 (백만)
  gflops          FLOPs (G, 설정 imgsz 기준. ultralytics 가 640 기준으로 돌려주는 값을 (imgsz/640)^2 로 환산)
  weight_mb       best.pt 파일 크기 (MB)
  gpu / cpu       한 장씩(batch 1) 넣었을 때 사진 한 장당 ms 와 FPS. 전처리·추론·후처리 전부 포함한 벽시계 시간.
                  GPU 는 val 전부, CPU 는 앞 n_cpu 장. 각각 처음 몇 장은 예열로 빼고 잰다.

현장(컨베이어)은 사진이 한 장씩 들어오므로 batch 1 이 맞는 조건이다.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path


def _timeit(model, paths: list[str], imgsz: int, device: str, warmup: int = 5) -> dict:
    import torch

    def _one(p):
        model.predict(p, imgsz=imgsz, conf=0.001, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()

    for p in paths[:warmup]:
        _one(p)
    ts = []
    for p in paths[warmup:]:
        t0 = time.perf_counter()
        _one(p)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    n = len(ts)
    mean = sum(ts) / n
    return {"n_images": n, "ms_mean": round(mean, 2), "ms_median": round(ts[n // 2], 2),
            "ms_p95": round(ts[int(n * 0.95)], 2), "fps": round(1000 / mean, 1)}


def measure(best_pt: Path, paths: list[str], imgsz: int, n_cpu: int = 50) -> dict:
    import torch
    from ultralytics import YOLO

    from ultralytics.utils.torch_utils import get_flops, get_num_params

    gpu = YOLO(str(best_pt))
    out = {
        "imgsz": imgsz, "weight": str(best_pt.name),
        "params_m": round(get_num_params(gpu.model) / 1e6, 3), "layers": len(list(gpu.model.modules())),
        "gflops": round(get_flops(gpu.model, imgsz), 2),
        "weight_mb": round(best_pt.stat().st_size / 1e6, 2),
    }
    if torch.cuda.is_available():
        out["gpu"] = {"device": torch.cuda.get_device_name(0), **_timeit(gpu, paths, imgsz, "0")}
    del gpu
    cpu = YOLO(str(best_pt))
    out["cpu"] = {"device": platform.processor() or "cpu", "threads": torch.get_num_threads(),
                  **_timeit(cpu, paths[:n_cpu + 5], imgsz, "cpu")}
    return out


def measure_and_save(best_pt: Path, paths: list[str], imgsz: int, out_json: Path, n_cpu: int = 50) -> dict:
    r = measure(best_pt, paths, imgsz, n_cpu=n_cpu)
    out_json.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    g = r.get("gpu", {})
    c = r["cpu"]
    print(f"[speed] {r['params_m']}M params, {r['gflops']} GFLOPs@{imgsz}, {r['weight_mb']}MB | "
          f"GPU {g.get('ms_mean', '-')}ms/장 ({g.get('fps', '-')} FPS) | CPU {c['ms_mean']}ms/장 ({c['fps']} FPS) → {out_json}")
    return r
