"""Check checkpoint/export identity and standalone-runtime prediction parity."""
import argparse
import json
from pathlib import Path

import numpy as np

from core import digest, json_write


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(left, right):
    a, b = read(left), read(right)
    if set(a) != set(b) or len(a) != 369:
        raise ValueError("Validation image IDs differ")
    maximum = 0.0
    for iid in a:
        x, y = np.asarray(a[iid]), np.asarray(b[iid])
        if x.shape != y.shape:
            raise ValueError(f"Prediction shape differs: {iid}")
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-5, err_msg=iid)
        if x.size:
            maximum = max(maximum, float(np.abs(x-y).max()))
    return {"left": left.name, "right": right.name, "images": len(a),
            "maximum_absolute_error": maximum, "atol": 1e-5, "passed": True}


def paired(left, right, threshold=.42):
    a, b = read(left), read(right)
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("Probe positions or order differ")
    hit = lambda r: r["after_conf"] >= threshold and r["before_conf"] < threshold and r["sham_conf"] < threshold
    both = sum(hit(x) and hit(y) for x,y in zip(a,b))
    lost = sum(hit(x) and not hit(y) for x,y in zip(a,b))
    gained = sum(not hit(x) and hit(y) for x,y in zip(a,b))
    neither = len(a)-both-lost-gained
    return {"n": len(a), "threshold": threshold, "both": both,
            "lost": lost, "gained": gained, "neither": neither}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--roi", type=Path, required=True)
    a = ap.parse_args()
    root, roi = a.comparison, a.roi
    record = read(root / "record.json")
    for key, relative in [("onnx_sha256", "exports/fp32/baseline_640.onnx"),
                          ("int8_sha256", "exports/int8/baseline_640_int8.onnx")]:
        if digest(root / relative) != record[key]:
            raise ValueError(f"Export changed: {relative}")
    pairs = [("onnx640_t1", "standalone_fp32_t1"),
             ("onnx640_t4", "standalone_fp32_t4"),
             ("int8_640_t1", "standalone_int8_t1")]
    checks = [compare(root/f"predictions_{x}.json", root/f"predictions_{y}.json") for x,y in pairs]
    checks.append(compare(roi/"predictions_val_cpu_t1.json", roi/"predictions_standalone_roi_t1.json"))
    result = {"status": "passed", "standalone_prediction_parity": checks,
              "roi_paired_vs_full640_cpu": paired(root/"predictions_probes_onnx640_t1.json", roi/"predictions_probes_cpu_t1.json"),
              "roi_paired_vs_full1024_gpu": paired(root/"predictions_probes_gpu1024_fp32.json", roi/"predictions_probes_cpu_t1.json"),
              "final_source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "note": "Parity checks reuse saved predictions; no new inference, labels, or test data opened. Source hashes in prior run records refer to their execution-time versions."}
    json_write(root/"verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
