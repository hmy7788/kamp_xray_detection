"""09_extra_models_v1/robust_summary.json 으로 점 제거·가짜 정상 그림 1장을 만든다.

  python src/minyeop/extra_models/figures_robust.py --run runs/minyeop/09_extra_models_v1
(a) 점을 지운 뒤에도 정답 자리에서 검출이 남는 비율(설정별)  (b) 가짜 정상 이미지의 오경보율  (c) 원본 대 가짜 정상의 이미지 점수 AUC(임계값과 무관한 분리도)
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
MODELS = ["Faster R-CNN R50-FPN", "YOLOv3-tiny", "Faster R-CNN MobileNetV3-FPN", "YOLO26n", "RT-DETR-l", "D-FINE-N"]
SHORT = {"Faster R-CNN R50-FPN": "R50-FPN", "Faster R-CNN MobileNetV3-FPN": "MobileNetV3-FPN"}
REMOVE = [("mean_half4", "평균 4"), ("mean_half6", "평균 6"), ("mean_half8", "평균 8"), ("meannoise_half6", "평균+노이즈 6"),
          ("ns_half4", "NS+노이즈 4"), ("ns_half5", "NS+노이즈 5"), ("ns_half6", "NS+노이즈 6"), ("ns_half8", "NS+노이즈 8")]
FAKE = [("mean_half6", "평균 6"), ("mean_half8", "평균 8"), ("ns_half6", "NS+노이즈 6"), ("ns_half8", "NS+노이즈 8")]


def heat(ax, M, rows, cols, title, fmt, vmax, cmap):
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=9, color="white" if M[i, j] > vmax * 0.55 else "black")
    ax.set_title(title, fontsize=11.5)


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
    R = json.loads((run / "robust_summary.json").read_text(encoding="utf-8"))
    rows = [SHORT.get(m, m) for m in MODELS]
    A = np.array([[R["점 제거"][t]["모델"][m]["지운 뒤 정답 자리 검출"] / R["점 제거"][t]["모델"][m]["정답 수"] * 100 for t, _ in REMOVE] for m in MODELS])
    B = np.array([[R["가짜 정상"][t][m]["오경보율"] * 100 for t, _ in FAKE] for m in MODELS])
    C = np.array([[R["가짜 정상"][t][m]["AUC(원본 대 가짜 정상)"] for t, _ in FAKE] for m in MODELS])
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.9), gridspec_kw={"width_ratios": [1.9, 1, 1]})
    heat(ax[0], A, rows, [l for _, l in REMOVE], "(a) 점을 지운 뒤에도 정답 자리에서 남은 검출 (%)", lambda v: f"{v:.0f}", 100, "Reds")
    heat(ax[1], B, rows, [l for _, l in FAKE], "(b) 가짜 정상 이미지 오경보율 (%)", lambda v: f"{v:.1f}", 70, "Reds")
    heat(ax[2], C, rows, [l for _, l in FAKE], "(c) 원본 대 가짜 정상 분리도 (AUC, 1에 가까울수록 좋음)", lambda v: f"{v:.3f}", 1.0, "Blues")
    ax[2].images[0].set_clim(0.8, 1.0)
    for k in (1, 2):
        ax[k].set_yticks([])
    ax[0].set_xlabel("지운 방법과 반변(px): 평균 = 주변 평균색, NS = Navier-Stokes 보간, +노이즈 = 주변과 같은 세기의 노이즈를 더함", fontsize=9)
    fig.suptitle("점 제거와 가짜 정상 시험: 6개 모델 (test 결함 369장, 점 663개, 모델마다 val 임계값)", fontsize=13, y=1.02)
    fig.tight_layout()
    (run / "figures").mkdir(exist_ok=True)
    fig.savefig(run / "figures" / "robust6.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("저장:", run / "figures" / "robust6.png")


if __name__ == "__main__":
    main()
