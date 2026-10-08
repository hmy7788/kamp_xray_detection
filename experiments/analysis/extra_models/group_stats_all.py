"""호기·해상도별 통계를 6개 모델 전부에 대해 낸다(group_stats.py 의 group_row 를 그대로 쓴다).

  python src/minyeop/extra_models/group_stats_all.py --out runs/minyeop/12_group_stats_6models_v1

그룹: 전체, 호기, 해상도, 호기 x 해상도. 임계값은 모델마다 val 에서 정한 값(09_extra_models_v1/summary.json)으로 고정.
지표: P/R/F1(IoU 0.5), AP50, mAP50-95, 평가 v2 의 F1(R=2, R=5), 맞힌 검출의 중심 오차 중앙값. --official-only 로 공식 라벨 이미지만 본다.
기존 3개 모델의 값이 07_group_stats_v1 과 같은지 확인한다(같은 코드, 같은 임계값이어야 함).
그림: 호기·해상도별 mAP50-95, F1(IoU 0.5), F1(v2 R=2) 막대 + 그룹별 공식 라벨 박스 비율.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("group_stats", ROOT / "src" / "minyeop" / "group_stats" / "group_stats.py")
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

PRED = {"Faster R-CNN R50-FPN": "runs/minyeop/01_frcnn_r50fpn_min640_v1", "YOLOv3-tiny": "runs/minyeop/02_yolov3tiny_img640_v1",
        "Faster R-CNN MobileNetV3-FPN": "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", "YOLO26n": "runs/minyeop/09_extra_models_v1/yolo26n",
        "RT-DETR-l": "runs/minyeop/09_extra_models_v1/rtdetr_l", "D-FINE-N": "runs/minyeop/09_extra_models_v1/dfine_n"}
SHORT = {"Faster R-CNN R50-FPN": "R50-FPN", "Faster R-CNN MobileNetV3-FPN": "MobileNetV3-FPN"}
COLORS = ["#3b6fd4", "#e8912d", "#4aa564", "#d6604d", "#8e6bbf", "#2aa7b8"]
GROUPS = [("전체", "전체"), ("호기 1 · 316x332", "1호기\n316x332"), ("호기 1 · 352x332", "1호기\n352x332"), ("호기 1 · 412x332", "1호기\n412x332*"),
          ("호기 2 · 316x332", "2호기\n316x332"), ("호기 3 · 576x444", "3호기\n576x444")]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="runs/minyeop/12_group_stats_6models_v1")
    a = ap.parse_args()
    out = ROOT / a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    thr = {m: v["임계값(val, IoU0.5)"] for m, v in json.loads((ROOT / "runs/minyeop/09_extra_models_v1/summary.json").read_text(encoding="utf-8"))["실제 val·test"].items()}
    all_rows = {}
    for official in (False, True):
        rows = []
        for m, run in PRED.items():
            recs = json.loads((ROOT / run / f"preds_{a.split}.json").read_text(encoding="utf-8"))
            if official:
                recs = [r for r in recs if man[r["name"]]["source"] == "official"]
            groups = {"전체": recs}
            for r in recs:
                mi = man[r["name"]]
                res = f"{mi['width']}x{mi['height']}"
                groups.setdefault(f"호기 {mi['machine']}", []).append(r)
                groups.setdefault(f"해상도 {res}", []).append(r)
                groups.setdefault(f"호기 {mi['machine']} · {res}", []).append(r)
            for g in sorted(groups):
                if groups[g]:
                    rows.append({"모델": m, "평가셋": f"{a.split}{'_official' if official else ''}", "그룹": g, **gs.group_row(groups[g], thr[m])})
        tag = f"{a.split}{'_official' if official else ''}"
        with open(out / f"group_stats_{tag}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        all_rows[tag] = rows
    # 기존 3개 모델 재현 확인
    old = list(csv.DictReader(open(ROOT / "runs/minyeop/07_group_stats_v1/group_stats_test.csv", encoding="utf-8")))
    bad = 0
    for o in old:
        n = next(r for r in all_rows["test"] if r["모델"] == o["모델"] and r["그룹"] == o["그룹"])
        for col in ("이미지", "박스", "TP", "FP", "FN"):
            if int(float(o[col])) != int(n[col]):
                bad += 1
        if abs(float(o["mAP50-95"]) - n["mAP50-95"]) > 1e-9:
            bad += 1
    print("기존 07 결과와 불일치 항목 수:", bad, "(0이어야 함)")

    # 표 출력
    rows = all_rows["test"]
    print(f"\n{'그룹':<20}" + "".join(f"{SHORT.get(m, m)[:10]:>12}" for m in PRED) + "   (mAP50-95 / F1(IoU0.5))")
    for g, _ in GROUPS:
        line = f"{g:<20}"
        for m in PRED:
            r = next(x for x in rows if x["모델"] == m and x["그룹"] == g)
            line += f"{r['mAP50-95']:6.3f}/{r['F1']:5.3f}"
        print(line)

    # 그림
    for f in ("C:/Windows/Fonts/malgun.ttf",):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    share = {}
    for r in json.loads((ROOT / PRED["YOLOv3-tiny"] / f"preds_{a.split}.json").read_text(encoding="utf-8")):
        mi = man[r["name"]]
        d = share.setdefault(f"호기 {mi['machine']} · {mi['width']}x{mi['height']}", [0, 0])
        d[1] += len(r["gt"])
        d[0] += len(r["gt"]) if mi["source"] == "official" else 0
    share["전체"] = [sum(v[0] for v in share.values()), sum(v[1] for v in share.values())]
    panels = [("mAP50-95", "mAP50-95 (IoU 0.5~0.95)"), ("F1", "F1 (IoU 0.5), 세로축 0.90부터"), ("F1_v2_R2", "F1 (평가 v2, 중심 2px), 세로축 0.90부터")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    nm = len(PRED)
    wd = 0.8 / nm
    for ax, (col, title) in zip(axes, panels):
        for i, m in enumerate(PRED):
            vals = []
            for g, _ in GROUPS:
                r = next((x for x in rows if x["모델"] == m and x["그룹"] == g), None)
                vals.append(float(r[col]) if r else 0)
            xs = [j + (i - (nm - 1) / 2) * wd for j in range(len(GROUPS))]
            ax.bar(xs, vals, width=wd * 0.95, color=COLORS[i], label=SHORT.get(m, m))
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{lab}\n공식 {100 * share[g][0] / max(share[g][1], 1):.0f}%" for g, lab in GROUPS], fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.0 if col == "mAP50-95" else 0.9, 0.9 if col == "mAP50-95" else 1.0)
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.suptitle(f"호기·해상도별 성능, 6개 모델 ({a.split}, 모델별 val 고정 임계값). x축 아래 '공식 N%' = 그 그룹의 공식 라벨 박스 비율. *412x332는 박스 2개뿐", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "figures" / f"group_chart_{a.split}_6models.png", dpi=140, facecolor="white")
    print("저장:", out)


if __name__ == "__main__":
    main()
