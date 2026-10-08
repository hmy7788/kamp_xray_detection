"""4단계: 채점. 모든 모델에 같은 채점 코드(src/kamp_xray/metrics.py)를 쓴다.

    python scripts/evaluate.py                         # 최종 모델: val 로 임계값을 정한 뒤 test 채점
    python scripts/evaluate.py --model yolov3_tiny     # 베이스라인

- val: 신뢰도 임계값을 F1(IoU 0.5) 최대값으로 정한다.  test: outputs/<model>/eval_report_val.json 의 임계값을 쓴다.
- 지표: P/R/F1, TP/FP/FN, AP50, mAP50-95, 중심 거리 1~5px 매칭, 이미지 단위 판정, 출처·호기·해상도·월별 분해
- test 에서는 대회 제출용 예측 결과 파일도 만든다
    test_predictions.csv      : 임계값 이상 검출 상자 (image_id, 파일명, x1, y1, x2, y2, score; 원본 픽셀 좌표)
    test_image_decisions.csv  : 이미지별 최고 신뢰도와 판정 (불합격 = 최고 신뢰도 >= 임계값)
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

from kamp_xray import metrics as M
from kamp_xray.common import OUTPUTS, check_data_version, env_info, load_json, read_manifest, save_json


def write_csvs(out, recs, thr):
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "class", "x1", "y1", "x2", "y2", "score"])
        for r in recs:
            for d in sorted(r["dets"], key=lambda d: -d[4]):
                if d[4] >= thr:
                    w.writerow([r["image_id"], r["name"], "defect", *[round(v, 2) for v in d[:4]], round(d[4], 4)])
    with open(out / "test_image_decisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "n_detections", "max_score", "decision", "n_gt"])
        for r in recs:
            kept = [d for d in r["dets"] if d[4] >= thr]
            mx = max((d[4] for d in r["dets"]), default=0.0)
            w.writerow([r["image_id"], r["name"], len(kept), round(mx, 4), "불합격(이물질)" if mx >= thr else "합격",
                        len(r["gt"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dfine_n", choices=["dfine_n", "yolov3_tiny"])
    ap.add_argument("--split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--out", default=None, help="기본: outputs/<model>")
    a = ap.parse_args()
    for split in (["val", "test"] if a.split == "all" else [a.split]):
        evaluate_split(a.model, split, Path(a.out) if a.out else OUTPUTS / a.model)


def evaluate_split(model, split, out):
    a = SimpleNamespace(model=model, split=split)
    digest = check_data_version()
    preds = load_json(out / f"preds_{a.split}.json")
    recs = preds["records"]
    if a.split == "val":
        thr = M.best_threshold(recs)
        rule = "val F1 최대 (IoU 0.5)"
    else:
        vrep = out / "eval_report_val.json"
        if not vrep.exists():
            raise SystemExit("먼저 --split val 로 임계값을 정하세요.")
        thr = load_json(vrep)["threshold"]
        rule = "val 에서 정한 값을 그대로 적용"
    rep = {"model": a.model, "split": a.split, "threshold_rule": rule, "weights": preds.get("weights"),
           "manifest_sha256": digest, "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": env_info(),
           **M.report(recs, thr, read_manifest())}
    save_json(out / f"eval_report_{a.split}.json", rep)
    if a.split == "test":
        write_csvs(out, recs, thr)
    o = rep["overall"]
    print(f"[{a.model}] {a.split}: 임계값 {thr:.4f} | AP50 {o['ap50']:.4f}  mAP50-95 {o['map50_95']:.4f} | "
          f"P {o['precision']:.4f} R {o['recall']:.4f} F1 {o['f1']:.4f} (TP {o['tp']} FP {o['fp']} FN {o['fn']})")
    for k, v in rep["by_group"]["source"].items():
        print(f"    출처 {k:8}: F1 {v['f1']:.4f} (정답 {v['n_gt']}개)")
    print(f"    중심 거리 3px 기준 F1 {rep['center_distance']['r3px']['f1']:.4f}")
    print("  →", out / f"eval_report_{a.split}.json")


if __name__ == "__main__":
    main()
