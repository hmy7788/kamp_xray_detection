"""Same-checkpoint real validation, batch-one latency, and paired probe maps.

Use only after common/run.py has completed. Does not train or access test data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import psutil
import cv2

from core import Dataset, REPO, digest, hit_at, inject, json_write


def metric_library():
    path = REPO / "src/chong/harness/common/metrics/__init__.py"
    spec = importlib.util.spec_from_file_location("kamp_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(gt, pred, threshold, lib):
    tp, fp, fn, _, _ = lib.match(gt, pred, threshold)
    precision, recall, f1 = lib.prf(tp, fp, fn)
    ctp, _, cfn, _, _ = lib.match(gt, pred, threshold, mode="center")
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "center_tp": ctp, "center_fn": cfn, "center_recall": ctp/(ctp+cfn) if ctp+cfn else None,
            "positive_images_without_any_detection": sum(bool(g) and not any(p[4] >= threshold for p in pred[i]) for i, g in gt.items()),
            "fp_per_provided_image": fp/len(gt),
            "note": "Provided NG/partial-product set; not a production false-alarm rate."}


def latency(values):
    values = np.asarray(values, float)
    return {"n": len(values), "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)), "max_ms": float(values.max()),
            "serial_images_per_second": float(1000/values.mean())}


def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor()


def array_predict(model, image, setting):
    if setting.get("backend") == "onnx":
        return model.predict(image)
    result = model.predict(np.repeat(image[:, :, None], 3, axis=2), imgsz=setting["imgsz"],
                           device=setting["device"], quantize=16 if setting["half"] else None, conf=.001, max_det=50,
                           batch=1, rect=False, nms=True, iou=.7, verbose=False, save=False)[0]
    if result.boxes is None or not len(result.boxes):
        return [], result.speed
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    return [[*map(float, b), float(c)] for b, c in zip(boxes, confidences)], result.speed


def summarize_probes(rows, threshold, grid):
    cells, groups, templates = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        before, sham, after = [row[k] >= threshold for k in ["before_conf", "sham_conf", "after_conf"]]
        item = {"before": before, "sham": sham, "after": after,
                "eligible": not before and not sham, "new_hit": after and not before and not sham}
        cells[(row["machine"], row["grid_y"], row["grid_x"])].append(item)
        groups[row["background_burst"]].append(item)
        templates[row["template_id"]].append(item)

    def reduce(items):
        n = len(items)
        s = {k: sum(int(x[k]) for x in items) for k in ["before", "sham", "after", "eligible", "new_hit"]}
        return {"n": n, **s, "new_hit_fraction_all": s["new_hit"]/n if n else None,
                "hit_fraction_unconfounded": s["new_hit"]/s["eligible"] if s["eligible"] else None}

    all_items = [i for v in groups.values() for i in v]
    result = {"threshold": threshold, "overall": reduce(all_items),
              "n_background_bursts": len(groups),
              "cells": [{"machine": m, "row": y, "col": x, **reduce(cells[(m, y, x)])}
                        for m in ["1", "2", "3"] for y in range(grid) for x in range(grid)],
              "background_bursts": {k: reduce(v) for k, v in groups.items()},
              "templates": {tid: reduce(items) for tid, items in sorted(templates.items())}}
    # Keep burst-level descriptives; 762 synthetic placements are NOT 762
    # independent real-world tests, so no binomial confidence interval is used.
    result["burst_macro_new_hit_fraction"] = float(np.mean([reduce(v)["new_hit_fraction_all"] for v in groups.values()]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-run", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--int8", type=Path)
    a = ap.parse_args()
    training = json.loads((a.training_run / "record.json").read_text(encoding="utf-8"))
    if not training.get("finished_at") or training["runs"].get("42", {}).get("status") != "ok":
        raise ValueError("Training must finish successfully before evaluation/timing")
    weights = a.training_run / "work/seed42/train/weights/best.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    if cfg.get("rect") is not False:
        raise ValueError("This comparison requires square letterboxing across backends")
    probe_record = json.loads((a.probes / "record.json").read_text(encoding="utf-8"))
    qa = json.loads((a.probes / "qa_review.json").read_text(encoding="utf-8"))
    if (not qa.get("accepted") or qa.get("samples_sha256") != digest(a.probes / "samples.jsonl")
            or qa.get("record_sha256") != digest(a.probes / "record.json")):
        raise ValueError("Frozen probe set needs a matching visual QA review")
    samples = [json.loads(line) for line in (a.probes / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    signals = {}
    for t in probe_record["templates"]:
        path = a.probes / t["file"]
        if digest(path) != t["sha256"]:
            raise ValueError("Changed probe template")
        with np.load(path) as arrays:
            signals[t["id"]] = {"signal": arrays["signal"].copy(), "sham": arrays["sham"].copy()}
    ds, lib = Dataset(), metric_library()
    ids = list(ds.splits["val"])
    np.random.default_rng(cfg["seed"]).shuffle(ids)
    gt = {iid: ds.boxes(iid) for iid in ids}
    for s in probe_record["sources"]:
        if ds.rows[s["image_id"]].split != "val" or digest(ds.images[s["image_id"]]) != s["image_sha256"]:
            raise ValueError("Probe sources must be unchanged validation images")
    import torch
    import ultralytics
    import onnxruntime
    import onnx
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    torch.set_num_interop_threads(1)
    reference_model = YOLO(str(weights))
    parameter_count = sum(p.numel() for p in reference_model.model.parameters())
    del reference_model

    a.out.mkdir(parents=True, exist_ok=True)
    json_write(a.out / "work/standalone_inputs.json", {"split": "val", "manifest_sha256": training["manifest_sha256"],
               "images": [{"id": i, "path": str(ds.images[i])} for i in ids]})
    process = psutil.Process()
    base = {"training_run": a.training_run.name, "weights_sha256": digest(weights),
            "weights_bytes": weights.stat().st_size, "manifest_sha256": training["manifest_sha256"],
            "probes_sha256": probe_record["samples_sha256"], "config": cfg, "config_sha256": digest(a.config),
            "n_probes": len(samples), "n_templates": len(probe_record["templates"]),
            "n_background_bursts": len({s["background_burst"] for s in samples}),
            "onnx_sha256": digest(a.onnx) if a.onnx else None,
            "int8_sha256": digest(a.int8) if a.int8 else None,
            "qa_sha256": digest(a.probes / "qa_review.json"),
            "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            "env": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__, "opencv_threads": cv2.getNumThreads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "os": platform.platform()},
            "limits": ["All accuracy checks use validation; test remains unopened.",
                       "Synthetic rates describe fixed templates/positions only.",
                       "Current-PC serial API throughput excludes camera, PLC, conveyor, and reject actuator.",
                       "Peak RSS includes Python/runtime/data; CUDA allocated peak is not whole-system GPU usage."]}
    json_write(a.out / "record.json", base)
    for setting in cfg["settings"]:
        if a.only and setting["id"] not in a.only:
            continue
        is_onnx = setting.get("backend") == "onnx"
        onnx_path = a.int8 if setting.get("int8") else a.onnx
        outfile = a.out / f"{setting['id']}.json"
        if outfile.exists():
            saved = json.loads(outfile.read_text(encoding="utf-8"))
            if (saved["weights_sha256"] != base["weights_sha256"] or saved["setting"] != setting
                    or saved.get("probes_sha256") != base["probes_sha256"]
                    or saved.get("config_sha256") != base["config_sha256"]
                    or (is_onnx and (not onnx_path or saved.get("onnx_sha256") != digest(onnx_path)))):
                raise ValueError("Existing output belongs to a different checkpoint/setting/probe set")
            print(f"Already completed: {setting['id']}", flush=True)
            continue
        print(f"Starting {setting['id']}: {len(ids)} real images", flush=True)
        if is_onnx:
            if not onnx_path:
                raise ValueError("ONNX settings require matching --onnx/--int8 export")
            export = json.loads((onnx_path.parent / "export.json").read_text(encoding="utf-8"))
            if export["source_weights_sha256"] != base["weights_sha256"] or export["onnx_sha256"] != digest(onnx_path):
                raise ValueError("ONNX checkpoint mismatch")
            from onnx_cpu import OnnxCPU
            model = OnnxCPU(onnx_path, size=setting["imgsz"], threads=setting["threads"])
        else:
            model = YOLO(str(weights))
        gpu = setting["device"] != "cpu"
        if gpu and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device unavailable; refusing a silent CPU fallback")
        torch.set_num_threads(setting["threads"])
        array_predict(model, ds.image(ids[0]), setting)  # initialize backend
        # Backend setup may reset CPU threads. Apply the cap after setup.
        torch.set_num_threads(setting["threads"])
        for iid in ids[:cfg["warmup"]]:
            array_predict(model, ds.image(iid), setting)
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        durations, backend_ms, pred, raw, rss = [], [], {}, {}, []
        for index, iid in enumerate(ids):
            if gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            im = ds.image(iid)
            boxes, speed = array_predict(model, im, setting)
            if gpu:
                torch.cuda.synchronize()
            durations.append((time.perf_counter()-start)*1000)
            backend_ms.append(sum(speed.values()))
            rss.append(process.memory_info().rss)
            h, w = im.shape
            pred[iid] = [((x0+x1)/2/w, (y0+y1)/2/h, (x1-x0)/w, (y1-y0)/h, c) for x0,y0,x1,y1,c in boxes]
            raw[iid] = boxes
            if (index+1) % 100 == 0:
                print(f"  Real images {index+1}/{len(ids)}", flush=True)
        report = {"setting": setting, "weights_sha256": base["weights_sha256"],
                  "detection_head": "one-to-many with NMS, iou=0.7, conf_floor=0.001, max_det=50",
                  "config_sha256": base["config_sha256"],
                  "probes_sha256": base["probes_sha256"], "n_images": len(ids), "n_gt": sum(map(len, gt.values())),
                  "ap50": lib.map50(gt, pred), "fixed_thresholds": [score(gt, pred, t, lib) for t in cfg["thresholds"]],
                  "by_machine": {m: [score({i:g for i,g in gt.items() if ds.rows[i].machine==m},
                                          {i:p for i,p in pred.items() if ds.rows[i].machine==m}, t, lib) for t in cfg["thresholds"]]
                                 for m in ["1", "2", "3"]},
                  "full_api_latency": latency(durations), "backend_pre_infer_post_latency": latency(backend_ms),
                  "actual_torch_threads": torch.get_num_threads(), "peak_sampled_rss_mib": max(rss)/2**20,
                  "actual_parameter_dtype": "ONNX tensor(float)" if is_onnx else str(next(model.model.parameters()).dtype),
                  "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if gpu else None,
                  "parameters": parameter_count,
                  "onnx_sha256": digest(onnx_path) if is_onnx else None,
                  "model_bytes": onnx_path.stat().st_size if is_onnx else weights.stat().st_size,
                  "precision_description": export["precision"] if is_onnx else ("FP16" if setting["half"] else "FP32"),
                  "onnx_intra_op_threads": setting["threads"] if is_onnx else None}
        json_write(a.out / f"predictions_{setting['id']}.json", raw)
        if setting["map"]:
            print(f"  Paired probes {len(samples)}", flush=True)
            evaluated, background_cache = [], {}
            for index, sample in enumerate(samples):
                iid = sample["background_id"]
                if iid not in background_cache:
                    background_cache[iid] = ds.image(iid)
                im = background_cache[iid]
                template = signals[sample["template_id"]]
                inserted, _ = inject(im, template["signal"], sample["x"], sample["y"])
                sham, _ = inject(im, template["sham"], sample["x"], sample["y"])
                after, _ = array_predict(model, inserted, setting)
                control, _ = array_predict(model, sham, setting)
                evaluated.append({**sample, "before_conf": hit_at(raw[iid], sample["target_box"]),
                                  "sham_conf": hit_at(control, sample["target_box"]),
                                  "after_conf": hit_at(after, sample["target_box"])})
                if (index+1) % 200 == 0:
                    print(f"  Probes {index+1}/{len(samples)}", flush=True)
            json_write(a.out / f"predictions_probes_{setting['id']}.json", evaluated)
            report["probe_maps"] = [summarize_probes(evaluated, t, probe_record["config"]["grid_size"]) for t in cfg["thresholds"]]
        json_write(outfile, report)
        print(f"Completed {setting['id']}: AP50={report['ap50']:.5f}, API p95={report['full_api_latency']['p95_ms']:.2f}ms", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Evaluation complete", flush=True)


if __name__ == "__main__":
    main()
