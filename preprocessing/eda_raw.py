"""EDA 숫자를 계산해 reports/eda_stats.json 과 표(마크다운)를 출력한다. 원본을 읽는 허용 스크립트다.

profiler 에이전트는 이 스크립트의 출력만 읽고 reports/eda.md 를 쓴다. 계산을 직접 하지 않는다.
사용: python common/eda_raw.py [--sample 300]
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def color_rects(rgb: np.ndarray):
    sat = rgb.max(2).astype(int) - rgb.min(2).astype(int)
    mask = sat > 0
    lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=2))
    rects = []
    for sl in ndimage.find_objects(lab):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            rects.append((x0, y0, x1, y1))
    return mask, rects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300, help="연속 프레임·밝기 통계용 표본 수")
    a = ap.parse_args()
    random.seed(0)
    raw = kx.raw_root()
    full = kx.load_manifest(valid_only=False)
    df = full[full.dup_of == ""]
    lab = df[df.labeled == 1]
    S: dict = {}

    S["counts"] = {"files": int(len(full)), "valid": int(len(df)), "duplicates": int((full.dup_of != "").sum()),
                   "labeled": int(len(lab)), "unlabeled": int(len(df) - len(lab)),
                   "by_machine": df.groupby("machine").size().to_dict(),
                   "labeled_by_machine": lab.groupby("machine").size().to_dict(),
                   "labeled_by_month": lab.groupby("month").size().to_dict(),
                   "bursts": int(df.burst_id.nunique()), "labeled_bursts": int(lab.burst_id.nunique())}
    S["resolution"] = {f"m{m}_{w}x{h}": int(n) for (m, w, h), n in df.groupby(["machine", "width", "height"]).size().items()}

    # 라벨 상자 통계
    rows = []
    for r in lab.itertuples(index=False):
        for cx, cy, w, h in kx.read_yolo_labels(raw / r.label_path):
            rows.append(dict(image_id=r.image_id, machine=r.machine, wpx=w * r.width, hpx=h * r.height, cx=cx, cy=cy))
    b = pd.DataFrame(rows)
    side = np.sqrt(b.wpx * b.hpx)
    S["boxes"] = {"n": int(len(b)), "per_image": lab.n_boxes.value_counts().sort_index().to_dict(),
                  "w_px_pct": np.percentile(b.wpx, [0, 10, 50, 90, 100]).round(1).tolist(),
                  "h_px_pct": np.percentile(b.hpx, [0, 10, 50, 90, 100]).round(1).tolist(),
                  "side_px_pct": np.percentile(side, [0, 10, 50, 90, 100]).round(1).tolist(),
                  "frac_side_lt_16": float((side < 16).mean()), "frac_side_lt_32": float((side < 32).mean()),
                  "cx_pct": np.percentile(b.cx, [5, 50, 95]).round(3).tolist(),
                  "cy_pct": np.percentile(b.cy, [5, 50, 95]).round(3).tolist(),
                  "frac_edge_lt_0.05": float((np.minimum.reduce([b.cx, 1 - b.cx, b.cy, 1 - b.cy]) < 0.05).mean())}

    # 색상 박스와 라벨 (라벨 사진 전체)
    inside = total = 0
    rect_cnt, lab_cnt = [], []
    palette = collections.Counter()
    for r in lab.itertuples(index=False):
        rgb = np.asarray(Image.open(raw / r.src_path).convert("RGB"))
        mask, rects = color_rects(rgb)
        for c, n in collections.Counter(map(tuple, rgb[mask])).most_common(3):
            palette[str(c)] += n
        boxes = kx.read_yolo_labels(raw / r.label_path)
        rect_cnt.append(len(rects)); lab_cnt.append(len(boxes))
        for cx, cy, w, h in boxes:
            total += 1
            X, Y = cx * r.width, cy * r.height
            if any(x0 <= X <= x1 and y0 <= Y <= y1 for x0, y0, x1, y1 in rects):
                inside += 1
    S["color_box"] = {"label_center_inside_rate": inside / total if total else 0.0,
                      "n_labels": total, "palette_top": palette.most_common(6),
                      "images_rects_eq_labels": int(sum(1 for x, y in zip(rect_cnt, lab_cnt) if x == y)),
                      "images_rects_lt_labels": int(sum(1 for x, y in zip(rect_cnt, lab_cnt) if x < y))}

    # 색 픽셀 비율·밝기·연속 프레임 (표본)
    samp = df.sample(min(a.sample, len(df)), random_state=0)
    frac, inten = [], collections.defaultdict(list)
    for r in samp.itertuples(index=False):
        rgb = np.asarray(Image.open(raw / r.src_path).convert("RGB"))
        mask, _ = color_rects(rgb)
        frac.append(mask.mean())
        g = rgb[:, :, 1].astype(float)
        inten[int(r.machine)].append((g.mean(), g.std()))
    S["color_pixel_fraction_pct"] = np.percentile(frac, [0, 50, 100]).round(5).tolist()
    S["intensity_by_machine"] = {m: {"mean": float(np.mean([x for x, _ in v])), "std": float(np.mean([s for _, s in v]))}
                                 for m, v in inten.items()}
    cors = []
    for bid, g in df.groupby("burst_id"):
        if len(g) < 2 or len(cors) > a.sample:
            continue
        g = g.sort_values("ts")
        prev = None
        for r in g.itertuples(index=False):
            arr = np.asarray(Image.open(raw / r.src_path).convert("L")).astype(float)
            if prev is not None and prev.shape == arr.shape:
                cors.append(np.corrcoef(prev.ravel(), arr.ravel())[0, 1])
            prev = arr
    S["consecutive_frame_corr_pct"] = np.percentile(cors, [10, 50, 90]).round(3).tolist() if cors else []
    S["burst_size_dist"] = df.groupby("burst_id").size().value_counts().sort_index().head(12).to_dict()
    S["suffix_n_vs_nboxes_match_rate"] = float((lab.suffix_n == lab.n_boxes).mean())

    kx.REPORTS.mkdir(exist_ok=True)
    kx.write_json(kx.REPORTS / "eda_stats.json", S)

    def md(title, d):
        print(f"\n### {title}\n")
        print("| 항목 | 값 |\n|---|---|")
        for k, v in d.items():
            print(f"| {k} | {v} |")

    md("개수", S["counts"]); md("해상도", S["resolution"]); md("라벨 상자", S["boxes"]); md("색상 박스", S["color_box"])
    md("기타", {"color_pixel_fraction_pct(min/med/max)": S["color_pixel_fraction_pct"],
               "intensity_by_machine": S["intensity_by_machine"],
               "consecutive_frame_corr(p10/50/90)": S["consecutive_frame_corr_pct"],
               "burst_size_dist": S["burst_size_dist"], "suffix_n==n_boxes": S["suffix_n_vs_nboxes_match_rate"]})
    print(f"\n저장: {kx.REPORTS / 'eda_stats.json'}")


if __name__ == "__main__":
    main()
