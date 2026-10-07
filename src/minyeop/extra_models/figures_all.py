"""6개 모델 전부를 담은 그림: 중심 거리 R 별 F1·못 찾은 개수, 점 제거 시험.

  python src/minyeop/extra_models/figures_all.py --run runs/minyeop/09_extra_models_v1
입력: v2_all_R.json(eval_v2_all.py), robust_summary.json(robust_extra.py)
출력: figures/v2_sweep_6models.png, figures/dot_removal_6models.png
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
COL = ["#3b6fd4", "#e8912d", "#4aa564", "#d6604d", "#8e6bbf", "#2aa7b8"]
MARK = ["o", "s", "^", "D", "v", "P"]
REMOVE = [("mean_half4", "평균 4"), ("mean_half6", "평균 6"), ("mean_half8", "평균 8"), ("meannoise_half6", "평균+노이즈 6"),
          ("ns_half4", "NS+노이즈 4"), ("ns_half5", "NS+노이즈 5"), ("ns_half6", "NS+노이즈 6"), ("ns_half8", "NS+노이즈 8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/minyeop/09_extra_models_v1")
    a = ap.parse_args()
    run = ROOT / a.run
    for f in ("C:/Windows/Fonts/malgun.ttf",):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    (run / "figures").mkdir(exist_ok=True)

    V = json.loads((run / "v2_all_R.json").read_text(encoding="utf-8"))
    Rs = V["R"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, m in enumerate(MODELS):
        f1 = [V["결과"][str(r)][m]["F1"] for r in Rs]
        fn = [V["결과"][str(r)][m]["FN"] for r in Rs]
        ax[0].plot(Rs, f1, marker=MARK[i], color=COL[i], lw=2, label=SHORT.get(m, m))
        ax[1].plot(Rs, fn, marker=MARK[i], color=COL[i], lw=2, label=SHORT.get(m, m))
    ax[0].axvspan(3, 5.2, color="#2b8a3e", alpha=0.08)
    ax[0].text(4.1, 0.93, "R ≥ 3px: 거의 포화(0.997 이상)", ha="center", fontsize=11, color="#2b6a2b")
    ax[0].set_xlabel("중심 거리 허용 R (px)")
    ax[0].set_ylabel("F1 (R마다 val에서 임계값 재선정)")
    ax[0].set_title("(a) F1: R=3px부터 거의 만점", fontsize=12.5)
    ax[0].set_xticks(Rs)
    ax[0].set_xlim(0.8, 5.2)
    ax[0].legend(frameon=False, fontsize=9.5, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax[0].grid(alpha=0.3)
    ax[1].set_yscale("symlog", linthresh=5)
    ax[1].set_xlabel("중심 거리 허용 R (px)")
    ax[1].set_ylabel("못 찾은 정답 수 (663개 중, 로그 눈금)")
    ax[1].set_title("(b) 못 찾은 개수: R=4~5px에서 0~1개", fontsize=12.5)
    ax[1].set_xticks(Rs)
    ax[1].set_xlim(0.8, 5.2)
    ax[1].set_yticks([0, 1, 2, 5, 10, 25, 50, 100, 150])
    ax[1].set_yticklabels(["0", "1", "2", "5", "10", "25", "50", "100", "150"])
    ax[1].grid(alpha=0.3)
    fig.suptitle("평가 v2(중심 거리 기준), 6개 모델 전부, test 정답 663개", fontsize=13.5, y=1.0)
    fig.tight_layout()
    fig.savefig(run / "figures" / "v2_sweep_6models.png", dpi=150, facecolor="white", bbox_inches="tight")

    R = json.loads((run / "robust_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(REMOVE))
    for i, m in enumerate(MODELS):
        v = [R["점 제거"][t]["모델"][m]["지운 뒤 정답 자리 검출"] / R["점 제거"][t]["모델"][m]["정답 수"] * 100 for t, _ in REMOVE]
        ax.plot(x, v, marker=MARK[i], color=COL[i], lw=2, label=SHORT.get(m, m))
    ax.set_xticks(x, [l for _, l in REMOVE], fontsize=10.5)
    ax.set_ylabel("지운 뒤에도 정답 자리에서 남은 검출 (%)")
    ax.set_xlabel("지운 방법과 반변(px): 평균 = 주변 평균색, NS = Navier-Stokes 보간, +노이즈 = 주변과 같은 세기의 노이즈를 더함")
    ax.set_title("점을 지우면 검출이 사라진다: 노이즈 없는 평균 보간은 6개 모델 모두 0~3개, 노이즈를 더하면 모델마다 다르다", fontsize=12.5)
    ax.axvspan(-0.4, 2.4, color="#2b8a3e", alpha=0.07)
    ax.text(1, 62, "노이즈 없는 평균 보간: 반변 6px 이상은 0.5% 이하", ha="center", fontsize=10.5, color="#2b6a2b")
    ax.legend(frameon=False, ncol=3, fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 75)
    fig.tight_layout()
    fig.savefig(run / "figures" / "dot_removal_6models.png", dpi=150, facecolor="white")
    print("저장:", run / "figures")


if __name__ == "__main__":
    main()
