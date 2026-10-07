"""One predeclared INT8 candidate, calibrated on train images only."""
import argparse
import hashlib
import json
from pathlib import Path

from core import Dataset, digest, json_write
from onnx_cpu import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnx

    source = json.loads((a.onnx.parent / "export.json").read_text(encoding="utf-8"))
    if source["onnx_sha256"] != digest(a.onnx):
        raise ValueError("Source ONNX file changed")
    ds = Dataset()
    selected = []
    for machine in ["1", "2", "3"]:
        rows = [r for r in ds.rows.values() if r.split == "train" and r.machine == machine]
        rows.sort(key=lambda r: hashlib.sha256(f"42/calibrate/{r.image_id}".encode()).hexdigest())
        bursts = set()
        for r in rows:
            if r.burst_id in bursts:
                continue
            selected.append(r.image_id)
            bursts.add(r.burst_id)
            if len(bursts) == 16:
                break
    if len(selected) != 48:
        raise ValueError("Expected 16 train bursts per machine")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.ids = iter(selected)

        def get_next(self):
            iid = next(self.ids, None)
            return None if iid is None else {"images": preprocess(ds.image(iid), 640)[0]}

    a.out.mkdir(parents=True, exist_ok=True)
    prepared = a.out / "prepared.onnx"
    quant_pre_process(a.onnx, prepared, skip_optimization=True, skip_symbolic_shape=True)
    result = a.out / "baseline_640_int8.onnx"
    quantize_static(prepared, result, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"], per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax,
                    calibration_providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(result))
    json_write(a.out / "export.json", {
        "source_weights_sha256": source["source_weights_sha256"], "source_onnx_sha256": source["onnx_sha256"],
        "onnx_sha256": digest(result), "onnx_bytes": result.stat().st_size, "file": result.name,
        "precision": "Conv INT8 QDQ S8S8 per-channel; other ops and external input/output FP32",
        "calibration": "MinMax, 48 train images / distinct bursts, 16 per machine, seed 42 hash order",
        "calibration_images": [{"id": i, "burst": ds.rows[i].burst_id, "sha256": digest(ds.images[i])} for i in selected],
        "batch": 1, "imgsz": 640, "note": "Single predeclared candidate; must validate accuracy and speed before selection."})
    print(result)


if __name__ == "__main__":
    main()
