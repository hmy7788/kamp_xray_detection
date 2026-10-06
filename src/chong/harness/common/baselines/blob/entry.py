"""학습 없는 기준선: 밝기 점(blob) 검출. 색상 박스 영향을 전혀 받지 않는 바닥 점수.

이물질은 제품 안의 작은 어두운 점이다. 사진을 뒤집어(밝은 점으로) LoG blob 검출을 돌리고,
점 둘레와의 밝기 차이를 확신도로 쓴다. 학습 단계가 없으므로 train_ids 는 쓰지 않는다. 진입점은 run_split(cfg, data, seed, out_dir).

model 영역 (configs):
  min_sigma: 1.5   max_sigma: 5   threshold: 0.01   max_dets: 30   box_scale: 1.4
  (LoG 에서 점의 반지름 ≈ sigma*1.41 이므로 box_scale 1.4 면 상자가 점 크기와 같다. 밝기 차이가 몇 단계뿐이라 threshold 는 낮게)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import blob_log


def run_split(cfg: dict, data: dict, seed: int, out_dir: Path) -> Path:
    m = cfg.get("model", {})
    min_sigma = float(m.get("min_sigma", 1.5))
    max_sigma = float(m.get("max_sigma", 5))
    thr = float(m.get("threshold", 0.01))
    max_dets = int(m.get("max_dets", 30))
    box_scale = float(m.get("box_scale", 1.4))
    rows = []
    for iid in data["val_ids"]:
        g = np.asarray(Image.open(data["images"][iid]).convert("L")).astype(float)
        H, W = g.shape
        inv = (255.0 - g) / 255.0
        blobs = blob_log(inv, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=6, threshold=thr)
        dets = []
        for y, x, s in blobs:
            r = max(2, int(round(s * box_scale)))
            y0, y1, x0, x1 = int(max(0, y - r)), int(min(H, y + r + 1)), int(max(0, x - r)), int(min(W, x + r + 1))
            inner = g[y0:y1, x0:x1]
            Y0, Y1, X0, X1 = max(0, y0 - 3), min(H, y1 + 3), max(0, x0 - 3), min(W, x1 + 3)
            outer = g[Y0:Y1, X0:X1]
            ring_mean = (outer.sum() - inner.sum()) / max(1, outer.size - inner.size)
            conf = float(np.clip((ring_mean - inner.mean()) / 255.0, 0, 1))
            if conf <= 0:
                continue
            dets.append((x / W, y / H, 2 * r / W, 2 * r / H, conf))
        dets = sorted(dets, key=lambda d: -d[4])[:max_dets]
        if dets:
            for cx, cy, w, h, c in dets:
                rows.append(dict(image_id=iid, cx=cx, cy=cy, w=w, h=h, conf=c))
        else:
            rows.append(dict(image_id=iid, cx=None, cy=None, w=None, h=None, conf=None))
    out = out_dir / "preds.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    return out
