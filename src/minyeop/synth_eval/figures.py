"""08_synth_eval_v1 의 summary.json, per_dot.csv 로 그림 1장을 만든다.

  python src/minyeop/synth_eval/figures.py --run runs/minyeop/08_synth_eval_v1
(a) 합성 점의 대비 구간별 검출률  (b) 실제 세기·크기(s=1.0, f=1.0)에서 호기별 검출률  (c) 대비 14 이상에서 호기별 검출률
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
COLORS = {"Faster R-CNN R50-FPN": "#3b6fd4", "YOLOv3-tiny": "#e8912d", "Faster R-CNN MobileNetV3-FPN": "#4aa564"}
SHORT = {"Faster R-CNN R50-FPN": "R50-FPN", "YOLOv3-tiny": "YOLOv3-tiny", "Faster R-CNN MobileNetV3-FPN": "MobileNetV3-FPN"}
BIN_LABELS = ["<6", "6~10", "10~14", "14~20", "20 이상"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/minyeop/08_synth_eval_v1")
    a = ap.parse_args()
    run = ROOT / a.run
    for f in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    S = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(run / "per_dot.csv", encoding="utf-8")))
    hit_key = [k for k in rows[0] if k.startswith("검출")][0]
    models = list(S["결과"])
    r_tag = [k for k in S["결과"][models[0]] if k.startswith("R")][0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    # (a) 대비별
    x = np.arange(5)
    for m in models:
        d = S["결과"][m][r_tag]["대비"]
        v = [d[n]["검출률"] * 100 for n in ["<6", "6~10", "10~14", "14~20", "≥20"]]
        lo = [d[n]["구간"][0] * 100 for n in ["<6", "6~10", "10~14", "14~20", "≥20"]]
        hi = [d[n]["구간"][1] * 100 for n in ["<6", "6~10", "10~14", "14~20", "≥20"]]
        ax[0].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - v], marker="o", color=COLORS[m], label=SHORT[m], capsize=3, lw=2)
    n_per = [S["결과"][models[0]][r_tag]["대비"][n]["점 수"] for n in ["<6", "6~10", "10~14", "14~20", "≥20"]]
    ax[0].set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(BIN_LABELS, n_per)])
    ax[0].set_xlabel("합성 점의 대비 (둘레 밝기에서 점 밝기를 뺀 값)")
    ax[0].set_ylabel("검출률 (%)")
    ax[0].set_ylim(-3, 103)
    ax[0].set_title("(a) 대비가 낮을수록 놓친다", fontsize=12)
    ax[0].legend(loc="upper left", frameon=False)
    ax[0].grid(alpha=0.25)

    # (b), (c) 호기별
    def by_machine(cond):
        out = {}
        for m in models:
            out[m] = []
            for mc in "123":
                sub = [r for r in rows if r["모델"] == m and r["호기"] == mc and cond(r)]
                out[m].append((100 * np.mean([int(r[hit_key]) for r in sub]), len(sub)))
        return out

    for k, (cond, title) in enumerate([(lambda r: r["s"] == "1.0" and r["f"] == "1.0", "(b) 실제 세기·크기(s=1.0, f=1.0)"),
                                       (lambda r: float(r["대비"]) >= 14, "(c) 대비 14 이상")], start=1):
        d = by_machine(cond)
        w = 0.26
        for i, m in enumerate(models):
            vals = [v for v, _ in d[m]]
            bars = ax[k].bar(np.arange(3) + (i - 1) * w, vals, w, color=COLORS[m], label=SHORT[m])
            for b, (v, n) in zip(bars, d[m]):
                ax[k].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
        n_text = [d[models[0]][j][1] for j in range(3)]
        ax[k].set_xticks(range(3), [f"{j + 1}호기\n(n={n_text[j]})" for j in range(3)])
        ax[k].set_ylim(0, 112)
        ax[k].set_ylabel("검출률 (%)")
        ax[k].set_title(title, fontsize=12)
        ax[k].grid(axis="y", alpha=0.25)
    ax[2].set_title("(c) 대비 14 이상에서도 MobileNet은 3호기를 못 찾는다", fontsize=12)
    fig.suptitle("합성 점(작대기 안, 시드 42) 검출률: 모델마다 val에서 정한 임계값, 중심 5px 이내", fontsize=13, y=1.02)
    fig.tight_layout()
    (run / "figures").mkdir(exist_ok=True)
    fig.savefig(run / "figures" / "synth_eval.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("저장:", run / "figures" / "synth_eval.png")


if __name__ == "__main__":
    main()
