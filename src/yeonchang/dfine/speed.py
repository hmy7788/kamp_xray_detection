"""D-FINE 추론 속도 측정 (batch 1). 설정마다 별도 프로세스로 실행해 스레드 수가 섞이지 않게 한다.

    python src/yeonchang/dfine/speed.py --exp 03_dfinen_img640_e50              # GPU FP32/FP16, CPU 4/1스레드 전부
    python src/yeonchang/dfine/speed.py --exp ... --one cpu --threads 1         # 한 설정만 (내부용)

구간: 전처리(리사이즈·텐서·장치 이동) / 추론(모델 forward) / 후처리(점수·박스 변환). 정확도는 보지 않으므로 val 이미지를 쓴다.
결과: runs/yeonchang/<exp>/speed_benchmark.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtdetr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RUNS, read_manifest  # noqa: E402

CONFIGS = [("cuda", "fp32", 0), ("cuda", "fp16", 0), ("cpu", "fp32", 4), ("cpu", "fp32", 1)]


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def power_source():
    """노트북 전원: AC / battery / unknown. 배터리면 GPU·CPU가 제한되어 속도가 크게 느려진다."""
    try:
        s = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        return {"1": "battery", "2": "AC"}.get(s, "AC" if s in {"3", "6", "7", "8", "9"} else f"unknown({s})")
    except Exception:
        return "unknown"


def run_one_ultralytics(args, exp_dir, size):
    """ultralytics(RT-DETR) 모델. 구간은 ultralytics 내부 측정(Results.speed), 합계는 predict 호출 전체의 바깥 측정."""
    import torch
    from ultralytics import RTDETR
    from train import imread
    if args.one == "cpu":  # ultralytics import가 스레드 설정을 바꿀 수 있어 다시 지정
        torch.set_num_threads(args.threads)
    model = RTDETR(str(exp_dir / "weights" / "best.pt"))
    device = "0" if args.one == "cuda" else "cpu"
    half = args.precision == "fp16"
    manifest = read_manifest()
    ids = sorted(i for i, r in manifest.items() if r["split"] == "val")[:args.n]
    imgs = [imread(DATA / manifest[i]["image_path"]) for i in ids]

    def step(img):
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=size, conf=0.001, device=device, half=half, verbose=False)[0]
        t1 = time.perf_counter()  # ultralytics는 후처리 끝에 결과를 CPU로 옮기므로 GPU 작업이 끝난 뒤다
        s = r.speed
        return s["preprocess"], s["inference"], s["postprocess"], (t1 - t0) * 1000

    step(imgs[0])  # 첫 호출에서 ultralytics가 장치를 고르며 스레드 수를 자체 값(최대 8)으로 바꾼다
    if args.one == "cpu":
        torch.set_num_threads(args.threads)
    for k in range(1, args.warmup):
        step(imgs[k % len(imgs)])
    return np.array([step(im) for im in imgs])


def run_one(args):
    import torch
    if args.affinity:  # 하이브리드 CPU(i7-13620H: 0~11 고성능, 12~15 저전력)에서 코어 배정에 따른 흔들림을 막는다
        import psutil
        psutil.Process().cpu_affinity([int(c) for c in args.affinity.split(",")])
    if args.one == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    exp_dir = RUNS / args.exp
    size = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))["hyp"]["imgsz"]
    if (exp_dir / "weights" / "best.pt").exists():
        rec = run_one_ultralytics(args, exp_dir, size)
        tot = rec[:, 3]
        report(args, rec[:, :3], tot, "ultralytics Results.speed (구간) + perf_counter (합계, predict 호출 전체)")
        return
    from transformers import AutoImageProcessor, DFineForObjectDetection
    from train import imread, to_tensor

    device = torch.device("cuda:0" if args.one == "cuda" else "cpu")
    processor = AutoImageProcessor.from_pretrained(exp_dir / "weights" / "best")
    model = DFineForObjectDetection.from_pretrained(exp_dir / "weights" / "best").to(device).eval()
    manifest = read_manifest()
    ids = sorted(i for i, r in manifest.items() if r["split"] == "val")[:args.n]
    imgs = [imread(DATA / manifest[i]["image_path"]) for i in ids]
    half = args.precision == "fp16"
    if half:  # ultralytics와 같은 방식: 가중치·입력을 미리 FP16으로 (autocast 아님)
        model.half()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def step(img):
        t0 = time.perf_counter()
        x = to_tensor(img, size).unsqueeze(0).to(device)
        if half:
            x = x.half()
        sync(); t1 = time.perf_counter()
        with torch.inference_mode():
            o = model(pixel_values=x)
        sync(); t2 = time.perf_counter()
        o.logits, o.pred_boxes = o.logits.float(), o.pred_boxes.float()
        processor.post_process_object_detection(o, threshold=0.001, target_sizes=[img.shape[:2]])
        sync(); t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    for k in range(args.warmup):
        step(imgs[k % len(imgs)])
    rec = np.array([step(im) for im in imgs])
    report(args, rec, rec.sum(1), "perf_counter, GPU는 구간마다 synchronize")


def report(args, rec, tot, timing):
    import torch
    res = {"device": args.one, "precision": args.precision, "threads": args.threads if args.one == "cpu" else None,
           "torch_threads_actual": torch.get_num_threads(), "affinity": args.affinity or None,
           "n_images": len(rec), "warmup": args.warmup, "timing": timing,
           "pre_ms": round(float(rec[:, 0].mean()), 2), "infer_ms": round(float(rec[:, 1].mean()), 2),
           "post_ms": round(float(rec[:, 2].mean()), 2), "total_ms_mean": round(float(tot.mean()), 2),
           "total_ms_median": round(float(np.median(tot)), 2), "total_ms_p95": round(float(np.percentile(tot, 95)), 2),
           "fps": round(1000 / float(tot.mean()), 1)}
    print("RESULT " + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--one", choices=["cuda", "cpu"])
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--affinity", default="", help="내부용: 고정할 논리 CPU 번호 (예: 0,2,4,6)")
    ap.add_argument("--gpu-repeat", type=int, default=0,
                    help="GPU FP32/FP16만 다시 재기: --exp에 여러 실험을 쉼표로 주면 모델을 번갈아 N회 반복")
    ap.add_argument("--cpu-repeat", type=int, default=0,
                    help="CPU만 다시 재기: --exp에 여러 실험을 쉼표로 주면 모델을 번갈아 N회 반복, 고성능 코어에 고정")
    args = ap.parse_args()
    if args.one:
        return run_one(args)
    if args.cpu_repeat:
        return cpu_repeat(args)
    if args.gpu_repeat:
        return gpu_repeat(args)

    import torch
    power = power_source()
    if power != "AC":
        print(f"경고: 전원이 {power}입니다. 배터리면 속도가 제한되니 AC 전원을 연결하고 다시 재세요.")
    results = []
    for dev, prec, th in CONFIGS:
        cmd = [sys.executable, "-u", __file__, "--exp", args.exp, "--n", str(args.n), "--warmup", str(args.warmup),
               "--one", dev, "--precision", prec, "--threads", str(th)]
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"{dev} {prec} {th}: 실패\n{out.stderr[-2000:]}")
            continue
        r = json.loads(line[0][7:])
        results.append(r)
        name = f"{dev}-{prec}" + (f"-{th}thr" if dev == "cpu" else "")
        print(f"{name:16} pre {r['pre_ms']:7.2f}  infer {r['infer_ms']:8.2f}  post {r['post_ms']:6.2f}  "
              f"total mean {r['total_ms_mean']:8.2f} (median {r['total_ms_median']:.2f}, p95 {r['total_ms_p95']:.2f}) ms  {r['fps']} img/s")
    cfg = json.loads((RUNS / args.exp / "config.json").read_text(encoding="utf-8"))
    out = {"exp": args.exp, "model": cfg.get("model"), "params": cfg.get("params"), "imgsz": cfg["hyp"]["imgsz"],
           "batch": 1, "images": f"val 앞 {args.n}장 (이름순)",
           "power_at_start": power, "power_at_end": power_source(),
           "cpu": cpu_name(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "measured": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}
    (RUNS / args.exp / "speed_benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("cpu:", out["cpu"], "| gpu:", out["gpu"])


CPU_PIN = {1: "0", 4: "0,2,4,6"}  # 고성능 코어. 4스레드는 하이퍼스레딩 짝을 피해 물리 코어 하나씩


def cpu_repeat(args):
    """CPU 1/4스레드를 고성능 코어에 고정하고, 실험들을 번갈아 N회 반복. 회차별 중앙값의 중앙값을 낸다."""
    exps = args.exp.split(",")
    power = power_source()
    if power != "AC":
        print(f"경고: 전원이 {power}입니다.")
    runs = {(e, th): [] for e in exps for th in CPU_PIN}
    for rep in range(1, args.cpu_repeat + 1):
        for e in exps:
            for th, pin in CPU_PIN.items():
                cmd = [sys.executable, "-u", __file__, "--exp", e, "--n", str(args.n), "--warmup", str(args.warmup),
                       "--one", "cpu", "--precision", "fp32", "--threads", str(th), "--affinity", pin]
                out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:
                    print(f"{e} {th}thr rep{rep}: 실패\n{out.stderr[-1500:]}")
                    continue
                r = json.loads(line[0][7:])
                runs[(e, th)].append(r)
                print(f"rep{rep} {e:24} cpu-{th}thr (pin {pin}): median {r['total_ms_median']:8.2f}  mean {r['total_ms_mean']:8.2f}  p95 {r['total_ms_p95']:8.2f} ms", flush=True)
    for e in exps:
        res = []
        for th, pin in CPU_PIN.items():
            rs = runs[(e, th)]
            if not rs:
                continue
            med = float(np.median([r["total_ms_median"] for r in rs]))
            res.append({"threads": th, "affinity": pin, "repeats": len(rs), "n_images_each": args.n,
                        "total_ms_median_of_medians": round(med, 2),
                        "total_ms_median_each": [r["total_ms_median"] for r in rs],
                        "total_ms_mean_each": [r["total_ms_mean"] for r in rs],
                        "total_ms_p95_each": [r["total_ms_p95"] for r in rs],
                        "infer_ms_mean_each": [r["infer_ms"] for r in rs],
                        "torch_threads_actual": [r["torch_threads_actual"] for r in rs],
                        "fps_from_median": round(1000 / med, 2)})
        out = {"exp": e, "power_at_start": power, "power_at_end": power_source(), "power_mode": "최고 성능 (AC overlay ded574b5)",
               "cpu": cpu_name(), "batch": 1, "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "note": "실험들을 번갈아 반복 측정. CPU 1스레드는 논리 CPU 0, 4스레드는 0,2,4,6에 고정", "results": res}
        (RUNS / e / "speed_cpu_pinned.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in res:
            print(f"{e:24} cpu-{r['threads']}thr: median of medians {r['total_ms_median_of_medians']} ms ({r['fps_from_median']} img/s)")


def gpu_repeat(args):
    """GPU FP32/FP16(model.half)을 실험들을 번갈아 N회 반복. 회차별 중앙값의 중앙값을 낸다."""
    import torch
    exps = args.exp.split(",")
    power = power_source()
    if power != "AC":
        print(f"경고: 전원이 {power}입니다.")
    runs = {(e, p): [] for e in exps for p in ("fp32", "fp16")}
    for rep in range(1, args.gpu_repeat + 1):
        for e in exps:
            for prec in ("fp32", "fp16"):
                cmd = [sys.executable, "-u", __file__, "--exp", e, "--n", str(args.n), "--warmup", str(args.warmup),
                       "--one", "cuda", "--precision", prec]
                out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:
                    print(f"{e} {prec} rep{rep}: 실패\n{out.stderr[-1500:]}")
                    continue
                r = json.loads(line[0][7:])
                runs[(e, prec)].append(r)
                print(f"rep{rep} {e:24} cuda-{prec}: median {r['total_ms_median']:8.2f}  mean {r['total_ms_mean']:8.2f}  p95 {r['total_ms_p95']:8.2f} ms", flush=True)
    for e in exps:
        res = []
        for prec in ("fp32", "fp16"):
            rs = runs[(e, prec)]
            if not rs:
                continue
            med = float(np.median([r["total_ms_median"] for r in rs]))
            res.append({"precision": prec, "fp16_method": "model.half() + 입력 half (autocast 아님)" if prec == "fp16" else None,
                        "repeats": len(rs), "n_images_each": args.n, "total_ms_median_of_medians": round(med, 2),
                        "total_ms_median_each": [r["total_ms_median"] for r in rs],
                        "total_ms_mean_each": [r["total_ms_mean"] for r in rs],
                        "total_ms_p95_each": [r["total_ms_p95"] for r in rs],
                        "pre_infer_post_ms_mean_each": [[r["pre_ms"], r["infer_ms"], r["post_ms"]] for r in rs],
                        "fps_from_median": round(1000 / med, 2)})
        out = {"exp": e, "power_at_start": power, "power_at_end": power_source(), "power_mode": "최고 성능 (AC overlay ded574b5)",
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "batch": 1,
               "measured": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "실험들을 번갈아 반복 측정", "results": res}
        (RUNS / e / "speed_gpu_repeat.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in res:
            print(f"{e:24} cuda-{r['precision']}: median of medians {r['total_ms_median_of_medians']} ms ({r['fps_from_median']} img/s)")


if __name__ == "__main__":
    main()
