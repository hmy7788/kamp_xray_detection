"""Exploratory follow-up: same 640 model, label-free foreground crop.

Prompted by the first fixed-map results. This is validation-driven model
selection, not an independent test result. No training or test access.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

from core import Dataset, REPO, digest, hit_at, inject, json_write
from evaluate import array_predict, latency, metric_library, score, summarize_probes
from roi import crop_box, predict_roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["gpu", "onnx"], default="gpu")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    base = json.loads((a.comparison / "record.json").read_text(encoding="utf-8"))
    ds = Dataset()
    ids = list(ds.splits["val"])
    np.random.default_rng(42).shuffle(ids)
    weights = REPO / "runs/chong" / base["training_run"] / "work/seed42/train/weights/best.pt"
    assert digest(weights) == base["weights_sha256"]
    setting = {"id": "gpu640_roi_fp32", "device": "0", "imgsz": 640, "half": False, "threads": 4}
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if a.backend == "onnx":
        from onnx_cpu import OnnxCPU
        onnx = a.comparison / "exports/fp32/baseline_640.onnx"
        assert digest(onnx) == base["onnx_sha256"]
        setting = {"id": f"onnx640_roi_t{a.threads}", "device": "cpu", "imgsz": 640, "half": False, "threads": a.threads}
        model = OnnxCPU(onnx, threads=a.threads)
        predict = lambda m, x, s: m.predict(x)
        sync = lambda: None
    else:
        model = YOLO(str(weights))
        predict = array_predict
        sync = torch.cuda.synchronize
    for iid in ids[:16]:
        predict_roi(model, ds.image(iid), setting, predict)
    sync()
    if a.backend == "gpu":
        torch.cuda.reset_peak_memory_stats()
    gt, pred, raw, rois = {}, {}, {}, {}
    durations, rss, missed_roi = [], [], []
    process = psutil.Process()
    for iid in ids:
        sync()
        start = time.perf_counter()
        im = ds.image(iid)
        roi = crop_box(im)
        boxes, _ = predict_roi(model, im, setting, predict, roi)
        sync()
        durations.append((time.perf_counter()-start)*1000)
        rss.append(process.memory_info().rss)
        h, w = im.shape
        rois[iid], raw[iid] = roi, boxes
        gt[iid] = ds.boxes(iid)
        pred[iid] = [((x0+x1)/2/w, (y0+y1)/2/h, (x1-x0)/w, (y1-y0)/h, c) for x0,y0,x1,y1,c in boxes]
        missed_roi.extend((iid, j) for j,(cx,cy,bw,bh) in enumerate(gt[iid])
                          if not (roi[0] <= (cx-bw/2)*w and (cx+bw/2)*w <= roi[2]
                                  and roi[1] <= (cy-bh/2)*h and (cy+bh/2)*h <= roi[3]))
    print(f"Original validation complete; ROI excludes {len(missed_roi)} labelled boxes", flush=True)
    probe_record = json.loads((a.probes / "record.json").read_text(encoding="utf-8"))
    assert digest(a.probes / "samples.jsonl") == probe_record["samples_sha256"]
    signals = {}
    for t in probe_record["templates"]:
        path = a.probes / t["file"]
        assert digest(path) == t["sha256"]
        with np.load(path) as data:
            signals[t["id"]] = {k: data[k].copy() for k in ["signal", "sham"]}
    samples = [json.loads(s) for s in (a.probes / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    cache, rows = {}, []
    for i, s in enumerate(samples):
        iid = s["background_id"]
        if iid not in cache:
            cache[iid] = ds.image(iid)
        im = cache[iid]
        # Freeze the crop on the original background for all three paired cases.
        after, _ = inject(im, signals[s["template_id"]]["signal"], s["x"], s["y"])
        sham, _ = inject(im, signals[s["template_id"]]["sham"], s["x"], s["y"])
        p, _ = predict_roi(model, after, setting, predict, rois[iid])
        c, _ = predict_roi(model, sham, setting, predict, rois[iid])
        rows.append({**s, "before_conf": hit_at(raw[iid], s["target_box"]),
                     "sham_conf": hit_at(c, s["target_box"]), "after_conf": hit_at(p, s["target_box"])})
        if (i+1) % 200 == 0:
            print(f"Paired ROI probes {i+1}/{len(samples)}", flush=True)
    lib = metric_library()
    report = {"setting": setting, "weights_sha256": digest(weights), "probes_sha256": probe_record["samples_sha256"],
              "reason": "After the initial map, test whether cropping excess background improves effective target scale at the same 640 model input.",
              "exploratory": True, "source_sha256": {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('roi.py')]},
              "roi_rule": "largest dark foreground component, 12% of longest product side padding, minimum 12 pixels; fallback full frame",
              "paired_roi": "frozen from original background; insertion cannot alter crop",
              "n_images": len(ids), "n_gt": sum(map(len, gt.values())), "roi_excluded_gt": len(missed_roi),
              "ap50": lib.map50(gt, pred), "fixed_thresholds": [score(gt,pred,t,lib) for t in [.42,.1]],
              "full_api_latency": latency(durations), "peak_sampled_rss_mib": max(rss)/2**20,
              "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if a.backend == "gpu" else None,
              "probe_maps": [summarize_probes(rows,t,5) for t in [.42,.1]],
              "limits": ["Validation-driven exploratory comparison; test untouched.", "No physical synthesis validation or production false-alarm estimate."]}
    suffix = "" if a.backend == "gpu" else f"_cpu_t{a.threads}"
    if a.backend == "onnx":
        report["onnx_sha256"] = digest(onnx)
    json_write(a.out / f"roi{suffix}.json", report)
    json_write(a.out / f"predictions_val{suffix}.json", raw)
    json_write(a.out / f"predictions_probes{suffix}.json", rows)
    print(json.dumps({"AP50": report["ap50"], "fixed": report["fixed_thresholds"], "map": report["probe_maps"][0]["overall"], "p95": report["full_api_latency"]["p95_ms"]}), flush=True)


if __name__ == "__main__":
    main()
