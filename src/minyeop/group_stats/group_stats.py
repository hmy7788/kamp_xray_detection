"""호기·해상도별 검출 통계. 모델 검출 결과(preds_*.json)와 data/manifest.csv 를 이름으로 합쳐 그룹별 지표를 낸다.

  python src/minyeop/group_stats/group_stats.py --split test --out runs/minyeop/07_group_stats_v1

그룹: 호기(1/2/3), 해상도(316x332 등), 호기x해상도, 라벨 출처(공식/팀)와의 교차는 --official-only 로 공식 라벨만 따로 본다.
지표 (모델마다 val 에서 정한 임계값으로 고정, IoU 0.5)
  P / R / F1, TP/FP/FN, AP@0.5, mAP50-95, 재현율의 Wilson 95% 구간
  평가 v2(중심 거리): 같은 임계값에서 R=2px, R=5px 의 F1 과, 맞힌 검출의 중심 오차 중앙값(px)
주의: 해상도는 호기와 겹치지 않는 축이 아니다(1호기에 세 해상도가 섞여 있고 2호기 316x332 는 1호기 316x332 와 같은 해상도).
그룹이 작으면(정답 박스 수 약 30개 미만) 구간이 넓으니 구간을 함께 볼 것.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
import metrics  # noqa: E402

RUNS = {"Faster R-CNN R50-FPN": ("runs/minyeop/01_frcnn_r50fpn_min640_v1", 0.95),
        "YOLOv3-tiny": ("runs/minyeop/02_yolov3tiny_img640_v1", 0.06),
        "Faster R-CNN MobileNetV3-FPN": ("runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", 0.93)}
IOUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def center_errors(recs, thr, r=5.0):
    """중심 거리 매칭(R=r)으로 맞힌 검출의 중심 오차(px) 목록."""
    errs = []
    for rec in recs:
        gt = [((g[0] + g[2]) / 2, (g[1] + g[3]) / 2) for g in rec["gt"]]
        used = set()
        for d in sorted((d for d in rec["dets"] if d[4] >= thr), key=lambda d: -d[4]):
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            order = sorted(range(len(gt)), key=lambda j: np.hypot(gt[j][0] - cx, gt[j][1] - cy))
            for j in order:
                dist = np.hypot(gt[j][0] - cx, gt[j][1] - cy)
                if dist > r:
                    break
                if j not in used:
                    used.add(j)
                    errs.append(dist)
                    break
    return errs


def group_row(recs, thr):
    n_box = sum(len(r["gt"]) for r in recs)
    p = metrics.prf_at(recs, thr)
    p2 = metrics.prf_at(recs, thr, center_r=2.0)
    p5 = metrics.prf_at(recs, thr, center_r=5.0)
    lo, hi = wilson(p["TP"], n_box)
    ce = center_errors(recs, thr)
    return {"이미지": len(recs), "박스": n_box, "TP": p["TP"], "FP": p["FP"], "FN": p["FN"], "P": p["P"], "R": p["R"], "R_lo": lo, "R_hi": hi,
            "F1": p["F1"], "AP50": metrics.ap(recs), "mAP50-95": float(np.mean([metrics.ap(recs, i) for i in IOUS])),
            "F1_v2_R2": p2["F1"], "F1_v2_R5": p5["F1"], "FN_v2_R5": p5["FN"], "FP_v2_R5": p5["FP"],
            "중심오차_중앙값": float(np.median(ce)) if ce else float("nan")}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="runs/minyeop/07_group_stats_v1")
    ap.add_argument("--official-only", action="store_true", help="공식 라벨 이미지만 대상으로 한다")
    a = ap.parse_args()
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.split}{'_official' if a.official_only else ''}"
    rows = []
    for m, (run, thr) in RUNS.items():
        recs = json.loads((ROOT / run / f"preds_{a.split}.json").read_text(encoding="utf-8"))
        if a.official_only:
            recs = [r for r in recs if man[r["name"]]["source"] == "official"]
        groups = {"전체": recs}
        for r in recs:
            mi = man[r["name"]]
            res = f"{mi['width']}x{mi['height']}"
            groups.setdefault(f"호기 {mi['machine']}", []).append(r)
            groups.setdefault(f"해상도 {res}", []).append(r)
            groups.setdefault(f"호기 {mi['machine']} · {res}", []).append(r)
        print(f"\n===== {m} (임계값 {thr}, {a.split}{', 공식 라벨만' if a.official_only else ''}) =====")
        print(f"{'그룹':<22}{'이미지':>5}{'박스':>5} | {'F1':>5} {'R':>5} {'R 95%구간':>13} {'FP':>3}{'FN':>4} | {'AP50':>5} {'mAP50-95':>8} | {'v2 R=2 F1':>9} {'v2 FN/FP':>8} | {'중심오차':>6}")
        for g in sorted(groups):
            if not groups[g]:
                continue
            s = group_row(groups[g], thr)
            rows.append({"모델": m, "평가셋": tag, "그룹": g, **s})
            print(f"{g:<22}{s['이미지']:>5}{s['박스']:>5} | {s['F1']:5.3f} {s['R']:5.3f} {s['R_lo']:5.3f}~{s['R_hi']:5.3f} {s['FP']:>3}{s['FN']:>4} | {s['AP50']:5.3f} {s['mAP50-95']:8.3f} | {s['F1_v2_R2']:9.3f} {s['FN_v2_R5']:>4}/{s['FP_v2_R5']:<3} | {s['중심오차_중앙값']:5.2f}px")
    with open(out / f"group_stats_{tag}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n저장:", out / f"group_stats_{tag}.csv")


if __name__ == "__main__":
    main()
