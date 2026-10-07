"""CPU inference adapter: one model, one forward pass, no torch dependency.

Requires a static square model with decoded [xyxy, confidence, class] output.
The verified export uses the baseline one-to-many head with embedded NMS.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def preprocess(image, size):
    h, w = image.shape
    gain = min(size/h, size/w)
    nw, nh = round(w*gain), round(h*gain)
    pad_x, pad_y = (size-nw)/2, (size-nh)/2
    left, top = round(pad_x-.1), round(pad_y-.1)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, round(pad_y+.1), left, round(pad_x+.1), cv2.BORDER_CONSTANT, value=114)
    tensor = np.repeat(padded[None, None, :, :], 3, axis=1).astype(np.float32)/255.0
    return tensor, gain, left, top


class OnnxCPU:
    def __init__(self, path, size=640, threads=1):
        import onnxruntime as ort
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.size = size
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, size, size] or inputs[0].type != "tensor(float)":
            raise ValueError("Expected a static batch-one FP32 square YOLO26 model")
        self.input_name = inputs[0].name

    def predict(self, image, conf=.001, max_det=50):
        start = time.perf_counter()
        h, w = image.shape
        tensor, gain, left, top = preprocess(image, self.size)
        pre_end = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        infer_end = time.perf_counter()
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
            raise ValueError("Expected decoded detection output [1, N, 6]")
        selected = output[0][output[0, :, 4] >= conf][:max_det].copy()
        if len(selected) and not np.all(selected[:, 5] == 0):
            raise ValueError("Expected the single defect class")
        selected[:, [0, 2]] = np.clip((selected[:, [0, 2]]-left)/gain, 0, w)
        selected[:, [1, 3]] = np.clip((selected[:, [1, 3]]-top)/gain, 0, h)
        result = selected[:, :5].astype(float).tolist()
        end = time.perf_counter()
        return result, {"preprocess": (pre_end-start)*1000, "inference": (infer_end-pre_end)*1000, "postprocess": (end-infer_end)*1000}


if __name__ == "__main__":
    import argparse
    import json
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=.42)
    args = ap.parse_args()
    model = OnnxCPU(args.model, threads=args.threads)
    image = np.asarray(Image.open(args.image).convert("L"))
    predictions, _ = model.predict(image, conf=args.threshold)
    print(json.dumps({"detections_xyxy_conf": predictions, "threshold": args.threshold,
                      "note": "Single-image result, not a warmed-up latency benchmark."}))
