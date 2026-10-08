"""09_extra_models_v1/summary.json 으로 6개 모델 비교 그림 1장을 만든다.

  python src/minyeop/extra_models/figures.py --run runs/minyeop/09_extra_models_v1
(a) 대비별 합성 점 검출률(모델마다 val 임계값)  (b) 임계값과 무관한 비교: 합성 점 AP, 오경보 예산이 같을 때의 검출률  (c) 대비 14 이상 합성 점의 호기별 검출률
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
COL = {"Faster R-CNN R50-FPN": "#3b6fd4", "YOLOv3-tiny": "#e8912d", "Faster R-CNN MobileNetV3-FPN": "#4aa564",
       "YOLO26n": "#d6604d", "RT-DETR-l": "#8e6bbf", "D-FINE-N": "#2aa7b8"}
SHORT = {"Faster R-CNN R50-FPN": "R50-FPN", "Faster R-CNN MobileNetV3-FPN": "MobileNetV3-FPN"}
BINS = ["<6", "6~10", "10~14", "14~20", "≥20"]
BIN_LABELS = ["<6", "6~10", "10~14", "14~20", "20 이상"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/minyeop/09_extra_models_v1")
    a = ap.parse_args()
    run = ROOT / a.run
    for f in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    S = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    syn, cmp_, models = S["합성 점"], S["합성 점: 임계값 무관 비교"], list(S["합성 점"])
    nm = lambda m: SHORT.get(m, m)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.1, 1.05, 1]})
    x = np.arange(5)
    for m in models:
        v = [syn[m]["R5"]["대비"][b]["검출률"] * 100 for b in BINS]
        ax[0].plot(x, v, marker="o", color=COL[m], label=nm(m), lw=2)
    n_per = [syn[models[0]]["R5"]["대비"][b]["점 수"] for b in BINS]
    ax[0].set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(BIN_LABELS, n_per)])
    ax[0].set_xlabel("합성 점의 대비")
    ax[0].set_ylabel("검출률 (%)")
    ax[0].set_ylim(-3, 103)
    ax[0].set_title("(a) 모델마다 val에서 고른 임계값으로", fontsize=12)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[0].grid(alpha=0.25)

    order = sorted(models, key=lambda m: -cmp_[m]["AP(R=5)"])
    y = np.arange(len(order))
    ax[1].barh(y, [cmp_[m]["AP(R=5)"] for m in order], color=[COL[m] for m in order], height=0.6)
    for i, m in enumerate(order):
        ax[1].text(cmp_[m]["AP(R=5)"] + 0.01, i, f"AP {cmp_[m]['AP(R=5)']:.2f} | 오경보 이미지당 0.03 이하에서 검출 {cmp_[m]['오경보 예산별']['0.03']['검출률'] * 100:.0f}%", va="center", fontsize=9)
    ax[1].set_yticks(y, [nm(m) for m in order])
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1.18)
    ax[1].set_xlabel("합성 점 AP (중심 5px, 임계값 전체를 훑음)")
    ax[1].set_title("(b) 임계값과 무관하게 비교하면 차이가 줄어든다", fontsize=12)
    ax[1].grid(axis="x", alpha=0.25)

    w = 0.13
    for i, m in enumerate(models):
        v = [syn[m]["R5"]["호기별 대비 14 이상"][k]["검출률"] * 100 for k in "123"]
        ax[2].bar(np.arange(3) + (i - 2.5) * w, v, w, color=COL[m], label=nm(m))
    n_k = [syn[models[0]]["R5"]["호기별 대비 14 이상"][k]["점 수"] for k in "123"]
    ax[2].set_xticks(range(3), [f"{k}호기\n(n={n})" for k, n in zip("123", n_k)])
    ax[2].set_ylim(0, 105)
    ax[2].set_ylabel("검출률 (%)")
    ax[2].set_title("(c) 대비 14 이상 점의 호기별 검출률", fontsize=12)
    ax[2].grid(axis="y", alpha=0.25)
    fig.suptitle("합성 점(작대기 안, 시드 42) 평가: 6개 모델", fontsize=13, y=1.02)
    fig.tight_layout()
    (run / "figures").mkdir(exist_ok=True)
    fig.savefig(run / "figures" / "compare6.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("저장:", run / "figures" / "compare6.png")


if __name__ == "__main__":
    main()
