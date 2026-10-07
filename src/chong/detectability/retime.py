"""Remeasure latency without deprecated-argument logging; verify predictions.

The first run used a deprecated precision alias that printed a warning on every
image. This repair reuses unchanged map scores, checks all original predictions,
and replaces only timing/memory measurements. Run after other inference ends.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

from core import Dataset, REPO, digest, json_write
from evaluate import array_predict, latency
from onnx_cpu import OnnxCPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    record = json.loads((a.input / "record.json").read_text(encoding="utf-8"))
    ds = Dataset()
    weights = REPO / "runs/chong" / record["training_run"] / "work/seed42/train/weights/best.pt"
    if digest(weights) != record["weights_sha256"]:
        raise ValueError("Changed weights")
    ids = list(ds.splits["val"])
    np.random.default_rng(record["config"]["seed"]).shuffle(ids)
    cv2.setNumThreads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    for setting in record["config"]["settings"]:
        if a.only and setting["id"] not in a.only:
            continue
        path = a.input / f"{setting['id']}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reference = json.loads((a.input / f"predictions_{setting['id']}.json").read_text(encoding="utf-8"))
        if setting.get("backend") == "onnx":
            model_path = a.input / ("exports/int8/baseline_640_int8.onnx" if setting.get("int8") else "exports/fp32/baseline_640.onnx")
            if digest(model_path) != report["onnx_sha256"]:
                raise ValueError("Changed ONNX model")
            model = OnnxCPU(model_path, threads=setting["threads"])
        else:
            model = YOLO(str(weights))
        torch.set_num_threads(setting["threads"])
        array_predict(model, ds.image(ids[0]), setting)
        torch.set_num_threads(setting["threads"])
        for iid in ids[:record["config"]["warmup"]]:
            array_predict(model, ds.image(iid), setting)
        gpu = setting["device"] != "cpu"
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        durations, times, rss = [], [], []
        for iid in ids:
            if gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            predictions, internal = array_predict(model, ds.image(iid), setting)
            if gpu:
                torch.cuda.synchronize()
            durations.append((time.perf_counter()-start)*1000)
            times.append(sum(internal.values()))
            rss.append(process.memory_info().rss)
            np.testing.assert_allclose(np.asarray(predictions), np.asarray(reference[iid]), rtol=1e-4, atol=2e-4)
        report["full_api_latency"] = latency(durations)
        report["backend_pre_infer_post_latency"] = latency(times)
        report["peak_sampled_rss_mib"] = max(rss)/2**20
        report["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20 if gpu else None
        report["retiming"] = {"reason": "Use quantize parameter to exclude deprecated half-alias warnings from timed loop.",
                              "prediction_equivalence_images": len(ids), "source_sha256": digest(Path(__file__)),
                              "inference_source_sha256": digest(Path(__file__).with_name("evaluate.py"))}
        json_write(path, report)
        print(f"{setting['id']}: identical predictions, p95={report['full_api_latency']['p95_ms']:.2f}ms", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
