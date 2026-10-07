"""Measure a standalone CPU runtime in a Python environment without torch.

The input list is explicitly exported from the frozen validation manifest by
the main evaluation. No dataset search or split selection occurs here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import psutil
import onnxruntime

from onnx_cpu import OnnxCPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--roi", action="store_true", help="Exploratory foreground crop before one model call")
    a = ap.parse_args()
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("Use a separate CPU runtime environment without PyTorch")
    inputs = json.loads(a.inputs.read_text(encoding="utf-8"))
    if inputs["split"] != "val" or len(inputs["images"]) != 369:
        raise ValueError("Expected the full, explicit validation list")
    model = OnnxCPU(a.model, threads=a.threads)
    predict = model.predict
    if a.roi:
        from roi import predict_roi
        predict = lambda im: predict_roi(model, im, {}, lambda m, x, s: m.predict(x))
    read = lambda p: np.asarray(Image.open(p).convert("L"))
    for row in inputs["images"][:16]:
        predict(read(row["path"]))
    durations, rss, predictions = [], [], {}
    process = psutil.Process()
    for row in inputs["images"]:
        start = time.perf_counter()
        result, _ = predict(read(row["path"]))
        durations.append((time.perf_counter()-start)*1000)
        rss.append(process.memory_info().rss)
        predictions[row["id"]] = result
    report = {"model": a.model.name, "model_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
              "threads": a.threads, "roi": a.roi, "python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
              "torch_installed": False, "n_images": len(durations), "batch": 1, "warmup_images": 16,
              "mean_ms": float(np.mean(durations)), "p50_ms": float(np.percentile(durations, 50)),
              "p95_ms": float(np.percentile(durations, 95)), "p99_ms": float(np.percentile(durations, 99)),
              "serial_images_per_second": float(1000/np.mean(durations)), "peak_sampled_rss_mib": max(rss)/2**20,
              "note": "Same PC, no PyTorch installed. PNG read-to-result; no camera/PLC/actuator time."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (a.out.parent / f"predictions_{a.out.stem}.json").write_text(json.dumps(predictions), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
