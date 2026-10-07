"""팀 공통 속도 표용 측정: GPU 평균, CPU 1·4·16스레드 (PyTorch 와 ONNX Runtime), 한 장씩(batch 1), fp32.

팀 experiments.md "추론 속도" 표와 같은 조건(사진 1장, 배치 1, 모델 연산+후처리)으로 잰다. 사진은 CPU 가 val 앞 55장(5장 예열), GPU 가 val 전부(30장 예열).
스레드 설정마다 별도 프로세스로 돌린다 (torch 스레드 수는 프로세스 안에서 한 번만 바꾸는 게 안전하고, ONNX 세션도 새로 만들어야 하므로).
1·4스레드는 P코어(논리 0~11)에 고정한다. 하이브리드 CPU 에서 E코어에 배정되면 3~4배 느려져 값이 흔들리기 때문이다. 16스레드는 전 코어.

사용:  KAMP_MEMBER=lee PYTHONUTF8=1 python members/lee/speed_threads.py lee_003 [--onnx runs/synth_band/exports/lee003_640.onnx]
결과:  runs/<exp_id>/speed_threads.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import kx  # noqa: E402

P_CORES = list(range(12))  # i7-13620H: P코어 6개 × 2 = 논리 0~11


def _worker(a):
    import psutil
    if a.threads < 16:
        psutil.Process().cpu_affinity(P_CORES)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)  # torch 를 들여오기 전에 정해야 MKL/OpenMP 풀에 적용된다
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    import torch
    from ultralytics import YOLO
    import ultralytics.utils.torch_utils as tu
    tu.NUM_THREADS = a.threads  # ultralytics 가 CPU 추론 시작 때 torch 스레드를 min(8, 코어-1) 로 바꿔 버린다. 그 값을 우리 설정으로.
    torch.set_num_threads(a.threads)
    if a.backend == "onnx":
        import onnxruntime as ort
        _Orig = ort.InferenceSession

        class _Sess(_Orig):
            def __init__(self, path, sess_options=None, providers=None, **kw):
                so = sess_options or ort.SessionOptions()
                so.intra_op_num_threads = a.threads
                so.inter_op_num_threads = 1
                super().__init__(path, so, providers=providers, **kw)
        ort.InferenceSession = _Sess
    model = YOLO(a.weights, task="detect")
    paths = a.paths.split(";")

    def one(p):
        model.predict(p, imgsz=a.imgsz, conf=0.001, max_det=50, device=a.device, verbose=False)
        if a.device != "cpu":
            torch.cuda.synchronize()
    warm = 5 if a.device == "cpu" else 30  # GPU 는 클럭이 올라오고 커널이 준비되는 데 더 걸린다 (5장이면 첫 측정이 4~5배 느리게 나옴)
    for p in paths[:warm]:
        one(p)
    torch.set_num_threads(a.threads)  # 예열 중 ultralytics 가 바꿨을 수 있으니 한 번 더
    ts = []
    for p in paths[warm:]:
        t0 = time.perf_counter(); one(p); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); n = len(ts); mean = sum(ts) / n
    print("RESULT " + json.dumps({"torch_threads": torch.get_num_threads(), "n_images": n, "ms_mean": round(mean, 2), "ms_median": round(ts[n // 2], 2),
                                  "ms_p95": round(ts[int(n * 0.95)], 2), "fps": round(1000 / mean, 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id", nargs="?")
    ap.add_argument("--onnx", help="같은 가중치의 ONNX 파일 (없으면 PyTorch 만)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--backend"); ap.add_argument("--threads", type=int)
    ap.add_argument("--weights"); ap.add_argument("--imgsz", type=int); ap.add_argument("--device"); ap.add_argument("--paths")
    a = ap.parse_args()
    if a.worker:
        return _worker(a)
    cfg = kx.read_yaml(kx.ROOT / "configs" / "lee" / f"{a.exp_id}.yaml") if hasattr(kx, "read_yaml") else None
    imgsz = int(cfg["model"]["imgsz"]) if cfg else int(json.loads((kx.RUNS / a.exp_id / "speed_seed42.json").read_text(encoding="utf-8"))["imgsz"])
    best = kx.RUNS / a.exp_id / "work" / "seed42" / "train" / "weights" / "best.pt"
    split = kx.load_split(); imgs, _ = kx.path_maps(split["val"])
    paths_cpu = ";".join(str(imgs[i]) for i in split["val"][: a.n + 5])
    paths_gpu = ";".join(str(imgs[i]) for i in split["val"])  # GPU 는 val 전부 (speed.py 와 같은 조건)
    import platform, torch
    out = {"exp_id": a.exp_id, "imgsz": imgsz, "weights": str(best.relative_to(kx.ROOT)), "onnx": a.onnx,
           "cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "images": "CPU: val 앞 %d장 (5장 예열 제외) / GPU: val 전부 (30장 예열 제외)" % a.n, "runs": []}
    jobs = [("torch", "0", 0)] + [("torch", "cpu", t) for t in (1, 4, 16)]
    if a.onnx:
        jobs += [("onnx", "cpu", t) for t in (1, 4, 16)]
    for backend, device, threads in jobs:
        w = str(best) if backend == "torch" else str(kx.ROOT / a.onnx)
        cmd = [sys.executable, __file__, "--worker", "--backend", backend, "--threads", str(threads or 16), "--weights", w,
               "--imgsz", str(imgsz), "--device", device, "--paths", paths_cpu if device == "cpu" else paths_gpu]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"[{backend} {device} t{threads}] 실패:\n{r.stderr[-800:]}"); continue
        res = json.loads(line[0][7:])
        tag = f"{backend}_gpu" if device != "cpu" else f"{backend}_cpu_t{threads}"
        out["runs"].append({"tag": tag, "backend": backend, "device": device, "threads": threads or None,
                            "pinned_p_cores": device == "cpu" and threads < 16, **res})
        print(f"[{tag}] {res['ms_mean']} ms (p50 {res['ms_median']}, p95 {res['ms_p95']}) → {res['fps']} FPS", flush=True)
    kx.write_json(kx.RUNS / a.exp_id / "speed_threads.json", out)


if __name__ == "__main__":
    main()
