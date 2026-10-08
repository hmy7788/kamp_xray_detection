"""5단계: 처리 속도 측정 (배치 1). 보고서 2.1절·4.2절의 FPS 측정 조건을 따른다.

    python scripts/speed.py --model dfine_n                       # CPU 4스레드 (기본)
    python scripts/speed.py --model dfine_n --configs cpu4,cpu1,gpu
    python scripts/speed.py --model dfine_n --affinity 0,2,4,6    # 하이브리드 CPU에서 고성능 코어에 고정 (보고서 측정 조건)

- 이미지: val 앞 100장(이름순). 파일 읽기는 미리 끝내고, 전처리(크기 조정·텐서 변환) + 모델 추론 + 후처리 시간을 잰다.
- 예열 10장 후 측정. 설정마다 별도 프로세스로 3회 반복하고, 회차별 평균 지연의 중앙값으로 FPS(= 1000 / ms)를 낸다.
- 노트북은 AC 전원, 전원 모드 '최고 성능'에서 재야 한다 (배터리면 크게 느려진다).
결과: outputs/<model>/speed.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from kamp_xray.common import CONFIGS, DATA, OUTPUTS, default_weights, imread, load_yaml, read_manifest, save_json, split_ids


def run_once(a):
    """한 설정을 한 번 측정해 RESULT 줄로 출력한다 (내부용 하위 프로세스)."""
    import torch
    if a.affinity:
        import psutil
        psutil.Process().cpu_affinity([int(c) for c in a.affinity.split(",")])
    dev = torch.device("cuda:0" if a.one == "gpu" else "cpu")
    if a.one.startswith("cpu"):
        torch.set_num_threads(int(a.one[3:]))
        torch.set_num_interop_threads(1)
    hyp = load_yaml(CONFIGS / f"{a.model}.yaml")
    weights = Path(a.weights) if a.weights else default_weights(a.model)
    man = read_manifest()
    imgs = [imread(DATA / man[i]["image_path"]) for i in split_ids(man, "val")[:a.n]]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    if a.model == "dfine_n":
        from kamp_xray import dfine
        model, processor = dfine.load_model(weights, dev)
        size = hyp["imgsz"]

        def step(img):
            t0 = time.perf_counter()
            x = dfine.to_tensor(img, size).unsqueeze(0).to(dev)
            sync(); t1 = time.perf_counter()
            with torch.inference_mode():
                o = model(pixel_values=x)
            sync(); t2 = time.perf_counter()
            processor.post_process_object_detection(o, threshold=0.001, target_sizes=[img.shape[:2]])
            sync(); t3 = time.perf_counter()
            return t1 - t0, t2 - t1, t3 - t2
    else:
        from kamp_xray import yolov3_tiny
        from kamp_xray.common import YOLOV3_DIR  # noqa: F401
        _, _, nms, scale_coords = yolov3_tiny._import_yolov3()
        from utils.datasets import letterbox
        model = yolov3_tiny.load_model(weights, dev, hyp["cfg"], hyp["img_size"])
        size = hyp["img_size"]

        def step(img):
            t0 = time.perf_counter()
            x = letterbox(img, new_shape=size)[0][:, :, ::-1].transpose(2, 0, 1)
            x = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float().div(255).unsqueeze(0)
            sync(); t1 = time.perf_counter()
            with torch.inference_mode():
                pred = model(x)[0]
            sync(); t2 = time.perf_counter()
            det = nms(pred, 0.001, hyp["nms_iou"], multi_label=False)[0]
            if det is not None:
                scale_coords(x.shape[2:], det[:, :4], img.shape).round()
            sync(); t3 = time.perf_counter()
            return t1 - t0, t2 - t1, t3 - t2

    for k in range(a.warmup):
        step(imgs[k % len(imgs)])
    rec = np.array([step(im) for im in imgs]) * 1000
    tot = rec.sum(1)
    print("RESULT " + json.dumps({"pre_ms": float(rec[:, 0].mean()), "infer_ms": float(rec[:, 1].mean()),
                                  "post_ms": float(rec[:, 2].mean()), "mean_ms": float(tot.mean()),
                                  "median_ms": float(np.median(tot)), "p95_ms": float(np.percentile(tot, 95)),
                                  "torch_threads": torch.get_num_threads()}))


def power_source():
    if sys.platform != "win32":
        return "unknown"
    try:
        s = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        return {"1": "battery", "2": "AC", "": "AC(데스크톱)"}.get(s, f"unknown({s})")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dfine_n", choices=["dfine_n", "yolov3_tiny"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--configs", default="cpu4", help="쉼표로 구분: cpu4, cpu1, gpu")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--affinity", default="", help="고정할 논리 CPU 번호 (예: 0,2,4,6)")
    ap.add_argument("--out", default=None, help="기본: outputs/<model>")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        return run_once(a)
    power = power_source()
    if power == "battery":
        print("경고: 배터리 전원입니다. AC 전원을 연결하고 다시 재세요.")
    results = {}
    for cfg in a.configs.split(","):
        runs = []
        for rep in range(a.repeats):
            cmd = [sys.executable, "-u", __file__, "--model", a.model, "--one", cfg, "--n", str(a.n),
                   "--warmup", str(a.warmup)] + (["--weights", a.weights] if a.weights else []) + \
                  (["--affinity", a.affinity] if a.affinity and cfg.startswith("cpu") else [])
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print(f"{cfg} {rep + 1}회차 실패:\n{p.stderr[-2000:]}")
                continue
            r = json.loads(line[0][7:])
            runs.append(r)
            print(f"[{a.model}] {cfg} {rep + 1}/{a.repeats}: 평균 {r['mean_ms']:.1f} ms (전처리 {r['pre_ms']:.1f} / "
                  f"추론 {r['infer_ms']:.1f} / 후처리 {r['post_ms']:.1f})", flush=True)
        if runs:
            med = float(np.median([r["mean_ms"] for r in runs]))
            results[cfg] = {"ms_per_image": round(med, 2), "fps": round(1000 / med, 1), "runs": runs}
    import torch
    out = Path(a.out) if a.out else OUTPUTS / a.model
    save_json(out / "speed.json", {"model": a.model, "batch": 1, "images": f"val 앞 {a.n}장", "warmup": a.warmup,
                                   "repeats": a.repeats, "affinity": a.affinity or None, "power": power,
                                   "cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                   "torch": torch.__version__, "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "timing": "전처리+추론+후처리, 파일 읽기 제외, 회차별 평균의 중앙값", "results": results})
    for cfg, r in results.items():
        print(f"[{a.model}] {cfg}: {r['ms_per_image']} ms/장 → {r['fps']} FPS")
    print("  →", out / "speed.json")


if __name__ == "__main__":
    main()
