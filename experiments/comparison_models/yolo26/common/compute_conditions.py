"""라벨 상자마다 조건 축 값을 계산해 data/conditions.csv 를 만든다.

축 정의는 common/conditions.yaml 한 곳에만 있다. 축을 추가하려면 yaml 에 줄을 더하고,
새 feature 가 필요하면 아래 FEATURES 에 함수를 더한다.
결과 열: image_id, box_idx, cx, cy, w, h, <feature 들>, <축 이름들>
경계값은 data/conditions_thresholds.json 에 남긴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def _box_px(r, W, H):
    x0 = int(round((r.cx - r.w / 2) * W)); x1 = int(round((r.cx + r.w / 2) * W))
    y0 = int(round((r.cy - r.h / 2) * H)); y1 = int(round((r.cy + r.h / 2) * H))
    return max(0, x0), max(0, y0), min(W, max(x1, x0 + 1)), min(H, max(y1, y0 + 1))


def f_box_side_px(r, img):
    H, W = img.shape
    return float(np.sqrt(r.w * W * r.h * H))


def f_edge_dist_norm(r, img):
    return float(min(r.cx, 1 - r.cx, r.cy, 1 - r.cy))


def _ring(r, img, pad=3):
    H, W = img.shape
    x0, y0, x1, y1 = _box_px(r, W, H)
    inner = img[y0:y1, x0:x1].astype(float)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    outer = img[Y0:Y1, X0:X1].astype(float)
    m = np.ones_like(outer, dtype=bool)
    m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    ring = outer[m]
    return inner, ring


def f_local_contrast(r, img):
    inner, ring = _ring(r, img)
    return float(abs(inner.mean() - ring.mean())) if inner.size and ring.size else 0.0


def f_ring_mean(r, img):
    _, ring = _ring(r, img)
    return float(ring.mean()) if ring.size else 0.0


FEATURES = {
    "box_side_px": f_box_side_px,
    "edge_dist_norm": f_edge_dist_norm,
    "local_contrast": f_local_contrast,
    "ring_mean": f_ring_mean,
}


def main() -> None:
    spec = kx.read_yaml(kx.COMMON / "conditions.yaml")
    df = kx.load_manifest()
    df["resolution"] = df["width"].astype(str) + "x" + df["height"].astype(str)
    lab = df[[(kx.DATA / p).exists() for p in df.label_path]]   # 라벨이 있는 사진만 (라벨링 중이면 일부)
    df["n_boxes"] = [len(kx.read_yolo_labels(kx.DATA / p)) if (kx.DATA / p).exists() else 0 for p in df.label_path]

    rows = []
    for r in lab.itertuples(index=False):
        boxes = kx.read_yolo_labels(kx.DATA / r.label_path)
        img = np.asarray(Image.open(kx.DATA / r.image_path).convert("L"))
        for i, (cx, cy, w, h) in enumerate(boxes):
            b = pd.Series(dict(cx=cx, cy=cy, w=w, h=h))
            row = dict(image_id=r.image_id, box_idx=i, cx=cx, cy=cy, w=w, h=h)
            for name, fn in FEATURES.items():
                row[name] = fn(b, img)
            rows.append(row)
    out = pd.DataFrame(rows)
    out = out.merge(df[["image_id", "machine", "resolution", "month", "source", "labeler", "n_boxes"]], on="image_id", how="left")

    thresholds = {}
    for axis, a in spec["axes"].items():
        t = a["type"]
        if t == "manifest":
            out[axis] = out[a["column"]].astype(str)
        elif t == "quantile":
            q = np.quantile(out[a["feature"]], np.linspace(0, 1, a["bins"] + 1)[1:-1])
            thresholds[axis] = [float(x) for x in q]
            out[axis] = pd.cut(out[a["feature"]], [-np.inf, *q, np.inf], labels=a["labels"]).astype(str)
        elif t == "fixed":
            e = a["edges"]
            thresholds[axis] = list(e)
            out[axis] = pd.cut(out[a["feature"]], [-np.inf, *e, np.inf], labels=a["labels"]).astype(str)
        else:
            raise ValueError(f"모르는 type: {t} ({axis})")

    out.to_csv(kx.CONDITIONS_CSV, index=False, lineterminator="\n")
    kx.write_json(kx.DATA / "conditions_thresholds.json",
                  {"manifest_sha256": kx.manifest_sha(), "n_images": int(lab.shape[0]), "thresholds": thresholds, "created_at": kx.now_iso()})
    print(f"사진 {len(lab)}장, 상자 {len(out)}개, 축 {list(spec['axes'])}")
    for axis in spec["axes"]:
        print(f"  {axis}: {out[axis].value_counts().sort_index().to_dict()}")
    print("경계값:", thresholds)
    print(f"저장: {kx.CONDITIONS_CSV}")


if __name__ == "__main__":
    main()
