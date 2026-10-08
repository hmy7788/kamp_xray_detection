"""3단계: 추론. val 또는 test 전체를 추론해 공통 예측 기록(preds_<split>.json)을 저장한다.

    python scripts/predict.py                          # 최종 모델, 검증용과 평가용 영상 모두
    python scripts/predict.py --model yolov3_tiny      # 베이스라인
    python scripts/predict.py --weights outputs/dfine_n/train/weights/best   # 직접 학습한 가중치

- 기본 가중치: weights/dfine_n/best (Hugging Face 형식 폴더), weights/yolov3_tiny/best.pt
- 신뢰도 0.001 이상 검출을 모두 저장한다. 임계값은 채점(evaluate.py) 단계에서 val로 정한다.
- 결과: outputs/<model>/preds_<split>.json  (형식은 src/kamp_xray/common.py 머리말)
"""
import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

from kamp_xray.common import CONFIGS, OUTPUTS, check_data_version, default_weights, load_yaml, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dfine_n", choices=["dfine_n", "yolov3_tiny"])
    ap.add_argument("--split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None, help="기본: outputs/<model>")
    ap.add_argument("--device", default="0", help="GPU 번호 또는 cpu")
    a = ap.parse_args()
    check_data_version()
    weights = Path(a.weights) if a.weights else default_weights(a.model)
    if not weights.exists():
        raise SystemExit(f"가중치가 없습니다: {weights}\n  weights/README.md 를 보거나 scripts/train.py 로 학습하세요.")
    hyp = load_yaml(CONFIGS / f"{a.model}.yaml")
    out = Path(a.out) if a.out else OUTPUTS / a.model
    w = weights.resolve()
    w_name = w.relative_to(_bootstrap.ROOT).as_posix() if w.is_relative_to(_bootstrap.ROOT) else w.name
    for split in (["val", "test"] if a.split == "all" else [a.split]):
        t0 = time.time()
        if a.model == "dfine_n":
            from kamp_xray import dfine
            recs = dfine.predict(weights, split, a.device, hyp["imgsz"])
        else:
            from kamp_xray import yolov3_tiny
            recs = yolov3_tiny.predict(weights, split, a.device, hyp["img_size"], hyp["nms_iou"], hyp["cfg"])
        save_json(out / f"preds_{split}.json", {"model": a.model, "split": split, "weights": w_name,
                                                "conf_floor": 0.001, "records": recs})
        n_det = sum(len(r["dets"]) for r in recs)
        print(f"[{a.model}] {split} {len(recs)}장 추론, 검출 후보 {n_det}개, {time.time() - t0:.0f}초 → {out / f'preds_{split}.json'}")


if __name__ == "__main__":
    main()
