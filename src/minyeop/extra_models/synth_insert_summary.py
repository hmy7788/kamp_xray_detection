"""합성 점 1차(원래 자리 / 무작위 자리) 6개 모델 결과를 요약하고 그림을 만든다.

  python src/minyeop/extra_models/synth_insert_summary.py --run runs/minyeop/11_synth_insert_6models_v1
입력: insert_dots.py --extra 가 만든 per_dot.csv (점마다 모델별 최고 신뢰도와 임계값 이상 여부, 대비, 자리 종류, 호기)
출력: summary_6models.json, figures/synth_insert_6models.png
기존 06_synth_insert_v1 과 같은 시드이므로 기존 3개 모델의 칸별 검출률이 같아야 한다(재현 확인).
"""
import argparse
import csv
import json
import sys
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
BINS = [(-99, 6), (6, 10), (10, 14), (14, 20), (20, 999)]
BIN_LABELS = ["<6", "6~10", "10~14", "14~20", "20 이상"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/minyeop/11_synth_insert_6models_v1")
    a = ap.parse_args()
    run = ROOT / a.run
    rows = list(csv.DictReader(open(run / "per_dot.csv", encoding="utf-8")))
    for r in rows:
        r["contrast"] = float(r["contrast"])
        r["s"], r["f"] = float(r["s"]), float(r["f"])
    # 재현 확인: 기존 06 결과의 칸별 검출률과 비교
    old = json.loads((ROOT / "runs/minyeop/06_synth_insert_v1/summary.json").read_text(encoding="utf-8"))["검출률"]
    bad = 0
    for m in old:
        for key, v in old[m].items():
            if "|" not in key:
                continue
            site, sk, fk = key.split("|")
            sel = [int(r[m + "|det"]) for r in rows if r["site"] == site and f"s{r['s']}" == sk and f"f{r['f']}" == fk]
            if len(sel) != v["n"] or abs(np.mean(sel) - v["검출률"]) > 1e-9:
                bad += 1
    print("기존 06 결과(3개 모델)와 칸별 검출률이 다른 칸 수:", bad, "(0이어야 함)")

    res = {"재현 확인(다른 칸 수)": bad, "s=1.0,f=1.0": {}, "대비별": {}, "호기별(s=1.0,f=1.0, 무작위 자리)": {}}
    print(f"\n{'모델':30} | s=1.0 f=1.0 원래 자리 | 무작위 자리 | 무작위 자리 호기 1/2/3")
    for m in MODELS:
        out = {}
        for site in ("orig", "rand"):
            sel = [int(r[m + "|det"]) for r in rows if r["site"] == site and r["s"] == 1.0 and r["f"] == 1.0]
            out[site] = {"n": len(sel), "검출률": float(np.mean(sel)), "구간": list(wilson(sum(sel), len(sel)))}
        res["s=1.0,f=1.0"][m] = out
        mach = {}
        for mc in "123":
            sel = [int(r[m + "|det"]) for r in rows if r["site"] == "rand" and r["s"] == 1.0 and r["f"] == 1.0 and r["machine"] == mc]
            mach[mc] = {"n": len(sel), "검출률": float(np.mean(sel)) if sel else None}
        res["호기별(s=1.0,f=1.0, 무작위 자리)"][m] = mach
        res["대비별"][m] = {}
        for site in ("orig", "rand"):
            res["대비별"][m][site] = {}
            for (lo, hi), lab in zip(BINS, BIN_LABELS):
                sel = [int(r[m + "|det"]) for r in rows if r["site"] == site and lo <= r["contrast"] < hi]
                res["대비별"][m][site][lab] = {"n": len(sel), "검출률": float(np.mean(sel)) if sel else None}
        print(f"{m:30} | {out['orig']['검출률'] * 100:6.1f}% | {out['rand']['검출률'] * 100:6.1f}% | " + " / ".join(f"{(mach[mc]['검출률'] or 0) * 100:.0f}%" for mc in '123'))
    (run / "summary_6models.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    for f in ("C:/Windows/Fonts/malgun.ttf",):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    (run / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    x = np.arange(len(BINS))
    for k, (site, ttl) in enumerate((("orig", "(a) 원래 자리(실제 점이 있던 곳)"), ("rand", "(b) 무작위 자리(막대 밖 포함)"))):
        for i, m in enumerate(MODELS):
            v = [(res["대비별"][m][site][lab]["검출률"] or 0) * 100 for lab in BIN_LABELS]
            ax[k].plot(x, v, marker="o", color=COL[i], lw=2, label=SHORT.get(m, m))
        n = [res["대비별"][MODELS[0]][site][lab]["n"] for lab in BIN_LABELS]
        ax[k].set_xticks(x, [f"{l}\n(n={c})" for l, c in zip(BIN_LABELS, n)])
        ax[k].set_xlabel("합성 점의 대비")
        ax[k].set_ylabel("검출률 (%)")
        ax[k].set_ylim(-3, 103)
        ax[k].set_title(ttl, fontsize=12)
        ax[k].grid(alpha=0.3)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    w = 0.38
    xs = np.arange(len(MODELS))
    ax[2].bar(xs - w / 2, [res["s=1.0,f=1.0"][m]["orig"]["검출률"] * 100 for m in MODELS], w, color="#9db7e8", label="원래 자리")
    ax[2].bar(xs + w / 2, [res["s=1.0,f=1.0"][m]["rand"]["검출률"] * 100 for m in MODELS], w, color="#e8912d", label="무작위 자리")
    for i, m in enumerate(MODELS):
        ax[2].text(i + w / 2, res["s=1.0,f=1.0"][m]["rand"]["검출률"] * 100 + 1.5, f"{res['s=1.0,f=1.0'][m]['rand']['검출률'] * 100:.0f}", ha="center", fontsize=9)
    ax[2].set_xticks(xs, [SHORT.get(m, m).replace("-FPN", "") for m in MODELS], rotation=25, ha="right", fontsize=9)
    ax[2].set_ylim(0, 112)
    ax[2].set_title("(c) 실제 세기·크기(s=1.0, f=1.0)", fontsize=12)
    ax[2].legend(frameon=False, fontsize=9, loc="lower left")
    ax[2].grid(axis="y", alpha=0.3)
    fig.suptitle("합성 점 1차(원래 자리 663개 / 무작위 자리 1,107개), 6개 모델 전부, 모델마다 val 임계값", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(run / "figures" / "synth_insert_6models.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("저장:", run)


if __name__ == "__main__":
    main()
