"""README 그림을 만든다 (팀 내부용 도구). 결과: docs/figures/*.png

    python tools/make_readme_figures.py [--raw "<KAMP 원본>/dataset/test1/yolov3"]

- detection_examples.png : 호기별 test 영상 1장씩, 최종 모델 검출 결과와 확대 그림
- model_comparison.png    : 6개 모델의 test mAP50-95
- preprocessing.png       : 원본 영상의 장비 색상 표시와 전처리 결과 (--raw 를 줄 때만)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kamp_xray.common import DATA, imread, read_manifest  # noqa: E402

OUT = ROOT / "docs" / "figures"
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
GT_C, PR_C = "#2ca02c", "#d62728"


def pick_examples(man, recs):
    by_id = {r["image_id"]: r for r in recs}
    out = []
    for m in "123":
        cands = sorted(i for i, r in man.items() if r["split"] == "test" and r["machine"] == m and r["source"] == "official"
                       and len(by_id[i]["gt"]) >= 3)
        out.append(cands[len(cands) // 2])
    return out


def crop_box(gt, w, h, pad=22):
    g = np.array(gt)
    x1, y1 = max(0, int(g[:, 0].min()) - pad), max(0, int(g[:, 1].min()) - pad)
    x2, y2 = min(w, int(g[:, 2].max()) + pad), min(h, int(g[:, 3].max()) + pad)
    return x1, y1, x2, y2


def detection_examples(man, recs, thr):
    by_id = {r["image_id"]: r for r in recs}
    ids = pick_examples(man, recs)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), gridspec_kw={"height_ratios": [1.15, 1]})
    for col, i in enumerate(ids):
        r = by_id[i]
        img = imread(DATA / man[i]["image_path"], cv2.IMREAD_GRAYSCALE)
        dets = [d for d in r["dets"] if d[4] >= thr]
        x1, y1, x2, y2 = crop_box(r["gt"], r["w"], r["h"])
        for row, (ax, view) in enumerate(zip(axes[:, col], ["full", "zoom"])):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            for g in r["gt"]:
                ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=GT_C, lw=1.2 if row == 0 else 2))
            for d in dets:
                ax.add_patch(Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, ec=PR_C, lw=1.2 if row == 0 else 2,
                                       ls="--"))
                if row == 1:
                    ax.text(d[2] + 1, d[1], f"{d[4]:.2f}", color=PR_C, fontsize=10, va="bottom", fontweight="bold")
            if row == 0:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#ffbf00", lw=1.5))
                ax.set_title(f"{man[i]['machine']}호기  {r['w']}x{r['h']}", fontsize=12)
            else:
                ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
                ax.set_title("노란 영역 확대", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec=GT_C, lw=2, label="정답 라벨"),
                        Rectangle((0, 0), 1, 1, fill=False, ec=PR_C, lw=2, ls="--", label=f"D-FINE-N 검출 (신뢰도 {thr:.2f} 이상)")],
               loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "detection_examples.png", dpi=110)
    plt.close(fig)
    print("detection_examples.png:", ids)


def model_comparison():
    rows = []
    for m, name in (("yolov3_tiny", "YOLOv3-tiny\n베이스라인"), ("dfine_n", "D-FINE-N\n최종 모델")):
        rows.append((name, json.loads((ROOT / "results" / m / "eval_report_test.json").read_text(encoding="utf-8"))["overall"]["map50_95"]))
    six = list(json.loads((ROOT / "results/comparison_models/six_models/summary.json").read_text(encoding="utf-8")).values())[0]
    rows.append(("Faster R-CNN\nResNet-50", six["Faster R-CNN R50-FPN"]["mAP50-95"]))
    rows.append(("Faster R-CNN\nMobileNetV3", six["Faster R-CNN MobileNetV3-FPN"]["mAP50-95"]))
    y26 = json.loads((ROOT / "results/comparison_models/yolo26n_img640/common_eval_report_test.json").read_text(encoding="utf-8"))
    rows.append(("YOLO26n", y26["overall"]["map50_95"]))
    rows.append(("RT-DETR-l", six["RT-DETR-l"]["mAP50-95"]))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#d62728" if "D-FINE" in n else "#7f7f7f" if "베이스라인" in n else "#4c72b0" for n, _ in rows]
    bars = ax.barh([n for n, _ in rows], [v for _, v in rows], color=colors)
    for b, (_, v) in zip(bars, rows):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=11)
    ax.set_xlim(0.5, 0.68)
    ax.set_xlabel("test mAP50-95")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=110)
    plt.close(fig)
    print("model_comparison.png")


def preprocessing(man, raw):
    index = {}
    for dp, _, fn in os.walk(raw):
        for f in fn:
            if f.lower().endswith(".bmp"):
                index.setdefault(f[:-4], []).append(Path(dp) / f)
    i = next(i for i in sorted(man) if man[i]["split"] == "test" and man[i]["machine"] == "3" and man[i]["source"] == "official"
             and i in index)
    proc = imread(DATA / man[i]["image_path"], cv2.IMREAD_GRAYSCALE)
    rgb = next(cv2.cvtColor(imread(p), cv2.COLOR_BGR2RGB) for p in index[i] if imread(p).shape[:2] == proc.shape)
    color = (rgb.max(2).astype(int) - rgb.min(2).astype(int)) > 0
    ys, xs = np.where(color)
    x1, y1, x2, y2 = max(0, xs.min() - 25), max(0, ys.min() - 25), min(proc.shape[1], xs.max() + 25), min(proc.shape[0], ys.max() + 25)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(rgb[y1:y2, x1:x2]); axes[0].set_title("원본: 장비가 그린 색상 표시", fontsize=12)
    axes[1].imshow(proc[y1:y2, x1:x2], cmap="gray", vmin=0, vmax=255); axes[1].set_title("전처리 후: 표시 제거, 이물질 유지", fontsize=12)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "preprocessing.png", dpi=110)
    plt.close(fig)
    print("preprocessing.png:", i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = read_manifest()
    rep = json.loads((ROOT / "results/dfine_n/eval_report_test.json").read_text(encoding="utf-8"))
    recs = json.loads((ROOT / "results/dfine_n/preds_test.json").read_text(encoding="utf-8"))["records"]
    detection_examples(man, recs, rep["threshold"])
    model_comparison()
    if a.raw:
        preprocessing(man, Path(a.raw))


if __name__ == "__main__":
    main()
