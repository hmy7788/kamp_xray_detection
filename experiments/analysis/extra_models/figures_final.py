"""최종 모델(D-FINE-N)의 운영 근거 그림 1장: 실제 test 이미지 최고 신뢰도 분포와 임계값별 검출·오경보.

  python src/minyeop/extra_models/figures_final.py --run runs/minyeop/09_extra_models_v1
입력: final_model.py 가 만든 final_model_dfine.json, preds_test.json
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
    R = json.loads((run / "final_model_dfine.json").read_text(encoding="utf-8"))
    te = json.loads((run / "dfine_n" / "preds_test.json").read_text(encoding="utf-8"))
    mx = lambda r: max([d[4] for d in r["dets"]], default=0.0)
    defect = [mx(r) for r in te if r["gt"]]
    empty = [mx(r) for r in te if not r["gt"]]
    thr = R["val 임계값"]
    ts = [float(t) for t in R["임계값 목록"]]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(defect, bins=bins, color="#2aa7b8", alpha=0.9, label="결함 있는 이미지 (369장)")
    ax[0].hist(empty, bins=bins, color="#d6604d", alpha=0.9, label="빈 라벨 이미지 (27장)")
    ax[0].axvline(thr, color="black", ls="--", lw=1.3)
    ax[0].text(thr + 0.01, ax[0].get_ylim()[1] * 0.62, f"val 임계값 {thr}", fontsize=10)
    ax[0].axvspan(0.3, thr, color="#f2c14e", alpha=0.25)
    ax[0].text(0.31, ax[0].get_ylim()[1] * 0.75, "재검사 구간\n(제안 0.3~0.66)", fontsize=10)
    ax[0].set_xlabel("이미지의 최고 신뢰도 (D-FINE-N)")
    ax[0].set_ylabel("이미지 수")
    ax[0].set_title("(a) 실제 test: 결함 이미지는 전부 0.74 이상, 빈 라벨은 0.04 이하", fontsize=11.5)
    ax[0].legend(frameon=False, loc="center", bbox_to_anchor=(0.43, 0.32), fontsize=10)

    faint = [R["합성 점"][str(t)]["옅은 점(대비 9~18) 검출률"] * 100 for t in R["임계값 목록"]]
    fa_mean = [R["가짜 정상"]["mean_half6"][str(t)]["비율"] * 100 for t in R["임계값 목록"]]
    fa_ns = [R["가짜 정상"]["ns_half6"][str(t)]["비율"] * 100 for t in R["임계값 목록"]]
    ax[1].plot(ts, faint, marker="o", color="#2aa7b8", lw=2.2, label="옅은 합성 점(대비 9~18) 검출률")
    ax[1].plot(ts, fa_mean, marker="s", color="#e8912d", lw=2, label="가짜 정상 오경보율: 평균 보간")
    ax[1].plot(ts, fa_ns, marker="^", color="#d6604d", lw=2, label="가짜 정상 오경보율: NS+노이즈 보간")
    ax[1].axvline(thr, color="black", ls="--", lw=1.3)
    ax[1].set_xlabel("임계값")
    ax[1].set_ylabel("%")
    ax[1].set_ylim(-3, 135)
    ax[1].set_title("(b) 임계값을 낮추면 옅은 점은 더 찾지만 오경보도 는다", fontsize=11.5)
    ax[1].legend(frameon=False, fontsize=9.5, loc="upper center", ncol=1)
    ax[1].grid(alpha=0.25)
    ax[1].annotate("0.9에서는 옅은 점도\n거의 못 찾음(신뢰도 포화)", xy=(0.9, 2), xytext=(0.36, 28), fontsize=9.5, arrowprops=dict(arrowstyle="->", color="gray"))
    fig.suptitle("최종 모델 D-FINE-N의 임계값 운영 근거 (합성·가짜 정상은 참고용)", fontsize=13, y=1.02)
    fig.tight_layout()
    (run / "figures").mkdir(exist_ok=True)
    fig.savefig(run / "figures" / "final_dfine.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("저장:", run / "figures" / "final_dfine.png")


if __name__ == "__main__":
    main()
