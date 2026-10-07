"""Exploratory CPU ROI candidate: one crop, one model inference."""
import argparse
import json

import numpy as np
from PIL import Image

from onnx_cpu import OnnxCPU
from roi import crop_box, predict_roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=.42)
    a = ap.parse_args()
    model = OnnxCPU(a.model, threads=a.threads)
    image = np.asarray(Image.open(a.image).convert("L"))
    box = crop_box(image)
    predictions, _ = predict_roi(model, image, {}, lambda m, x, s: m.predict(x, conf=a.threshold), box)
    print(json.dumps({"roi_xyxy": box, "detections_xyxy_conf": predictions,
                      "note": "Validation-driven exploratory crop, one inference; no timing guarantee."}))


if __name__ == "__main__":
    main()
