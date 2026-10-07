"""합성 점 평가를 임계값과 무관하게 비교한다: 6개 모델의 합성 점 AP(중심 5px)와 '오경보 예산이 같을 때'의 검출률.

  python src/minyeop/extra_models/compare_synth.py --out runs/minyeop/09_extra_models_v1

val 에서 F1 최대로 고른 임계값은 모델마다 달라(0.06~0.95) 합성 점 검출률을 그대로 비교하면 '모델 실력'과 '임계값이 보수적인 정도'가 섞인다.
그래서 (1) 임계값 전체를 훑는 AP, (2) 합성 이미지에서 이미지당 평균 오경보(FPPI)가 같아지는 임계값에서의 검출률을 함께 본다.
(2) 의 임계값은 합성 이미지 자체로 고르므로 모델 선정이 아니라 비교 분석용이다. 우리 3개 모델은 여기서 다시 추론한다(시간 수 초).
"""
import argparse
import csv
import json
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FR, YO, MB = (ROOT / "src" / "minyeop" / p for p in ("faster_rcnn", "yolov3_tiny", "faster_rcnn_mobilenet"))
sys.path.insert(0, str(MB))
sys.path.insert(0, str(FR))
import metrics  # noqa: E402

BUDGETS = (0.01, 0.03, 0.1)    # 합성 이미지당 평균 오경보(FPPI) 한도


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def our_preds(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr_r50 = load_module("fr_r50", FR / "predict.py")
    fr_mob = load_module("fr_mob", FR / "predict.py")
    from mobilenet_model import build_model as mob_build  # noqa: E402
    fr_mob.build_model = mob_build
    yo = load_module("yolo_predict", YO / "predict.py")
    from utils import torch_utils  # noqa: E402
    runs = {"Faster R-CNN R50-FPN": ROOT / "runs/minyeop/01_frcnn_r50fpn_min640_v1", "YOLOv3-tiny": ROOT / "runs/minyeop/02_yolov3tiny_img640_v1",
            "Faster R-CNN MobileNetV3-FPN": ROOT / "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1"}
    ycfg = json.loads((runs["YOLOv3-tiny"] / "config.json").read_text(encoding="utf-8"))
    return {"Faster R-CNN R50-FPN": fr_r50.predict_split(runs["Faster R-CNN R50-FPN"], "test", "best", device, 4, data_dir=str(data_root))[0],
            "YOLOv3-tiny": yo.predict_split(runs["YOLOv3-tiny"], "test", "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(data_root))[0],
            "Faster R-CNN MobileNetV3-FPN": fr_mob.predict_split(runs["Faster R-CNN MobileNetV3-FPN"], "test", "best", device, 4, data_dir=str(data_root))[0]}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    a = ap.parse_args()
    out = ROOT / a.out
    droot = ROOT / "data_synth"
    preds = our_preds(droot)
    for m, d in {"YOLO26n": "yolo26n", "RT-DETR-l": "rtdetr_l", "D-FINE-N": "dfine_n"}.items():
        preds[m] = json.loads((out / d / "preds_synth.json").read_text(encoding="utf-8"))
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    grid = np.unique(np.round(np.r_[np.linspace(0.001, 0.1, 120), np.linspace(0.1, 0.999, 360)], 4))
    n_img = len(preds["YOLOv3-tiny"])
    res = {}
    print(f"합성 점 {sum(len(r['gt']) for r in preds['YOLOv3-tiny'])}개, 이미지 {n_img}장 (중심 5px)\n")
    print(f"{'모델':30} {'AP(R=5)':>8} | " + " ".join(f"FPPI≤{b:<5}" for b in BUDGETS) + " (검출률 %, 그때 임계값)")
    for m, recs in preds.items():
        rows = [(t, metrics.prf_at(recs, t, center_r=5.0)) for t in grid]
        ap5 = metrics.ap(recs, center_r=5.0)
        r_b = {}
        for b in BUDGETS:
            ok = [(t, p) for t, p in rows if p["FP"] / n_img <= b]
            t, p = min(ok, key=lambda x: x[0])           # 예산 안에서 가장 낮은 임계값(= 가장 많이 찾는 점)
            r_b[str(b)] = {"임계값": t, "검출률": p["R"], "오경보": p["FP"]}
        res[m] = {"AP(R=5)": ap5, "오경보 예산별": r_b}
        print(f"{m:30} {ap5:8.3f} | " + " ".join(f"{r_b[str(b)]['검출률']*100:5.1f}({r_b[str(b)]['임계값']:.2f})" for b in BUDGETS))
    summ["합성 점: 임계값 무관 비교"] = res
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장:", out / "summary.json")


if __name__ == "__main__":
    main()
