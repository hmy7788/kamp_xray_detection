"""학습된 Faster R-CNN 으로 val 또는 test 를 추론해 검출 결과 기록(preds_<split>.json)을 저장하고 지표를 출력한다.

  python src/minyeop/faster_rcnn/predict.py --name 01_frcnn_r50fpn_min640 --split val
  python src/minyeop/faster_rcnn/predict.py --name 01_frcnn_r50fpn_min640 --split test    # 최종 후보만, 한 번만

- 신뢰도 임계값은 **val 에서 F1 이 최대인 값**을 쓰고 test 에 그대로 적용한다 (test 에서 정하지 않음).
- 기록 형식은 metrics.py 의 설명을 보세요 (다른 모델과 같은 평가 코드로 채점하기 위한 공통 형식).
- 기록에는 정답 박스 좌표(KAMP 라벨)가 들어 있어 Git 에는 올라가지 않는다 (.gitignore 의 preds_*.json).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from dataset import ROOT, YoloDetDataset  # noqa: E402
from model import build_model  # noqa: E402
from train import infer  # noqa: E402


def predict_split(run, split, ckpt_name, device, batch_size, limit=None):
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model = build_model(cfg["min_size"], cfg["max_size"], cfg.get("anchor_sizes_list"), pretrained=False).to(device)
    ck = torch.load(run / "weights" / f"{ckpt_name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    recs, ms = infer(model, YoloDetDataset(split, train=False, limit=limit), device, batch_size, amp=not cfg.get("no_amp", False))
    return recs, ms, ck.get("epoch")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=None, help="직접 지정할 임계값 (기본: val 에서 F1 최대인 값)")
    a = ap.parse_args()

    run = ROOT / "runs" / a.owner / a.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 임계값은 val 에서 정한다
    val_path = run / "preds_val.json"
    if a.split == "val" or not val_path.exists():
        vrecs, vms, ep = predict_split(run, "val", a.ckpt, device, a.batch_size)
        val_path.write_text(json.dumps(vrecs), encoding="utf-8")
    else:
        vrecs, vms, ep = json.loads(val_path.read_text(encoding="utf-8")), None, None
    bf = metrics.best_f1(vrecs)
    thr = a.threshold if a.threshold is not None else bf["thr"]

    if a.split == "val":
        recs, ms = vrecs, vms
    else:
        recs, ms, ep = predict_split(run, "test", a.ckpt, device, a.batch_size)
        (run / "preds_test.json").write_text(json.dumps(recs), encoding="utf-8")

    r = metrics.prf_at(recs, thr)
    print(f"\n=== {a.owner}/{a.name} | {a.ckpt}.pt (epoch {ep}) | {a.split} {len(recs)}장 ===")
    print(f"임계값 {thr} ({'지정값' if a.threshold is not None else 'val 에서 F1 최대'})")
    print(f"AP@0.5 {metrics.ap(recs):.3f} | P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} | TP {r['TP']} FP {r['FP']} FN {r['FN']}")
    if ms is not None:
        print(f"추론 시간 {ms:.1f} ms/장 ({device})")


if __name__ == "__main__":
    main()
