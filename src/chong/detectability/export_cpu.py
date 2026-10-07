"""Export a completed baseline to a standalone CPU ONNX model."""
import argparse
import json
from pathlib import Path
import shutil

from core import digest, json_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    record = json.loads((a.training_run / "record.json").read_text(encoding="utf-8"))
    if not record.get("finished_at") or record["runs"].get("42", {}).get("status") != "ok":
        raise ValueError("Training has not completed successfully")
    weights = a.training_run / "work/seed42/train/weights/best.pt"
    a.out.mkdir(parents=True, exist_ok=True)
    copy = a.out / "baseline_640.pt"
    shutil.copy2(weights, copy)
    from ultralytics import YOLO
    model = YOLO(str(copy))
    result = Path(model.export(format="onnx", imgsz=640, batch=1, device="cpu", dynamic=False,
                               simplify=False, opset=18, quantize=None, nms=True, conf=.001, iou=.7, max_det=50))
    import onnx
    onnx.checker.check_model(str(result))
    json_write(a.out / "export.json", {"source_weights_sha256": digest(weights), "onnx_sha256": digest(result),
                                      "file": result.name, "onnx_bytes": result.stat().st_size,
                                      "batch": 1, "imgsz": 640, "opset": 18, "dynamic": False, "precision": "FP32",
                                      "detection_head": "one-to-many with embedded NMS, iou=0.7, conf_floor=0.001, max_det=50",
                                      "note": "Format conversion; no quantization, pruning, or accuracy claim before validation."})
    print(result)


if __name__ == "__main__":
    main()
