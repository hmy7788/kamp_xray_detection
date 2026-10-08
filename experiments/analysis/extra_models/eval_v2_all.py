"""평가 v2(중심 거리 기준)를 6개 모델 전부에 대해 R = 1, 1.5, 2, 3, 4, 5 px 로 계산한다.

  python src/minyeop/extra_models/eval_v2_all.py --out runs/minyeop/09_extra_models_v1

규칙(docs/experiments.md "평가 v2"와 같음): 매칭은 신뢰도 순으로 정답 하나에 검출 하나, 검출 중심과 정답 중심의 거리 <= R(px).
임계값은 **R마다 val 에서 F1 최대인 값으로 다시 정한다**(동률이면 가운데 값). test 는 그 임계값으로 한 번 계산한다.
출력: test F1(P=R 이 아닐 수 있어 F1 으로 표기), TP/FP/FN, AP(임계값 무관), 공식 라벨 이미지만의 F1.
기존 3개 모델(R50, YOLOv3-tiny, MobileNetV3)의 R=1~3, 5 행이 docs/experiments.md 의 값과 같은지 함께 확인한다.
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

PRED = {"Faster R-CNN R50-FPN": "runs/minyeop/01_frcnn_r50fpn_min640_v1", "YOLOv3-tiny": "runs/minyeop/02_yolov3tiny_img640_v1",
        "Faster R-CNN MobileNetV3-FPN": "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", "YOLO26n": "runs/minyeop/09_extra_models_v1/yolo26n",
        "RT-DETR-l": "runs/minyeop/09_extra_models_v1/rtdetr_l", "D-FINE-N": "runs/minyeop/09_extra_models_v1/dfine_n"}
RS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
# docs/experiments.md "평가 v2" 의 기존 값 (R, 모델) -> (F1, TP, FP, FN), 재현 확인용
DOC = {(1.0, "Faster R-CNN R50-FPN"): (0.771, 511, 152, 152), (1.0, "YOLOv3-tiny"): (0.719, 477, 186, 186), (1.0, "Faster R-CNN MobileNetV3-FPN"): (0.682, 452, 211, 211),
       (2.0, "Faster R-CNN R50-FPN"): (0.988, 655, 8, 8), (2.0, "YOLOv3-tiny"): (0.982, 651, 12, 12), (2.0, "Faster R-CNN MobileNetV3-FPN"): (0.965, 640, 23, 23),
       (3.0, "Faster R-CNN R50-FPN"): (1.000, 663, 0, 0), (3.0, "YOLOv3-tiny"): (0.998, 662, 1, 1), (3.0, "Faster R-CNN MobileNetV3-FPN"): (0.997, 661, 2, 2),
       (5.0, "Faster R-CNN R50-FPN"): (1.000, 663, 0, 0), (5.0, "YOLOv3-tiny"): (1.000, 663, 0, 0), (5.0, "Faster R-CNN MobileNetV3-FPN"): (0.998, 662, 1, 1)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    a = ap.parse_args()
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    recs = {m: {s: json.loads((ROOT / p / f"preds_{s}.json").read_text(encoding="utf-8")) for s in ("val", "test")} for m, p in PRED.items()}
    res = {}
    mism = []
    print(f"{'R':>4} {'모델':30} {'임계값':>6} {'F1':>6} {'TP/FP/FN':>14} {'AP':>6} {'공식 F1':>7}")
    for R in RS:
        res[str(R)] = {}
        for m in PRED:
            b = metrics.best_f1(recs[m]["val"], center_r=R)
            thr = float(b["thr"])
            te = recs[m]["test"]
            pt = metrics.prf_at(te, thr, center_r=R)
            off = [r for r in te if man[r["name"]]["source"] == "official"]
            row = {"임계값": thr, "F1": pt["F1"], "P": pt["P"], "R": pt["R"], "TP": pt["TP"], "FP": pt["FP"], "FN": pt["FN"],
                   "AP": metrics.ap(te, center_r=R), "공식 라벨 F1": metrics.prf_at(off, thr, center_r=R)["F1"]}
            res[str(R)][m] = row
            tag = ""
            if (R, m) in DOC:
                d = DOC[(R, m)]
                same = (round(row["F1"], 3) == d[0] and (row["TP"], row["FP"], row["FN"]) == d[1:])
                tag = "  (기존 표와 일치)" if same else f"  !! 기존 표 {d}"
                if not same:
                    mism.append((R, m))
            print(f"{R:>4} {m:30} {thr:6.2f} {row['F1']:6.3f} {row['TP']:>5}/{row['FP']:>3}/{row['FN']:>3} {row['AP']:6.3f} {row['공식 라벨 F1']:7.3f}{tag}")
    out = ROOT / a.out / "v2_all_R.json"
    out.write_text(json.dumps({"설명": "평가 v2 중심 거리 R별(임계값은 R마다 val 에서 재선정), test 663개", "R": RS, "결과": res, "기존 표와 불일치": [list(x) for x in mism]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("기존 표와 불일치:", mism if mism else "없음")
    print("저장:", out)


if __name__ == "__main__":
    main()
