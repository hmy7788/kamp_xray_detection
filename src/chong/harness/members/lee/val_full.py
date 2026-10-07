"""synth_band 설정(6개)마다 원본 val 369장 전체를 하네스 계산기(common/metrics)로 채점한다.

왜: synth_band evaluate 는 원본 val 을 F1·중심 적중만, ONNX CPU 는 앞 65장만 채점했다. 보고서에는 설정별 mAP50 이
    필요하고, 모든 줄이 같은 369장이어야 비교가 된다. 속도는 다시 재지 않는다 (summary.json 의 값 유지).

결과: runs/synth_band/<tag>/val_full.json  (mAP50, mAP50-95, 임계값별 P/R/F1/중심 적중, 사진별 예측)
사용:  KAMP_MEMBER=lee PYTHONUTF8=1 python members/lee/val_full.py [--tags a b ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))
import kx  # noqa: E402
import metrics as M  # noqa: E402
from synth_band import OUT, THRS, _predict, _to_bgr, imread  # noqa: E402


def load_model(summ: dict):
    a = SimpleNamespace(weights=str(kx.ROOT / summ["weights"]) if not Path(summ["weights"]).is_absolute() else summ["weights"],
                        device=summ["device"], threads=summ.get("threads") or 4)
    import torch
    from ultralytics import YOLO
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    if a.weights.endswith(".onnx"):
        import onnxruntime as ort
        _Orig = ort.InferenceSession

        class _Sess(_Orig):
            def __init__(self, path, sess_options=None, providers=None, **kw):
                so = sess_options or ort.SessionOptions()
                so.intra_op_num_threads = a.threads
                so.inter_op_num_threads = 1
                super().__init__(path, so, providers=providers, **kw)
        ort.InferenceSession = _Sess
    return YOLO(a.weights, task="detect"), a.device


def score_tag(tag: str, split_val: list[str], imgs: dict, labs: dict) -> dict:
    summ = kx.read_json(OUT / tag / "summary.json")
    model, device = load_model(summ)
    imgsz = summ["imgsz"]
    gt, pred = {}, {}
    for iid in split_val:
        im = imread(imgs[iid]); h, w = im.shape
        d = _predict(model, _to_bgr(im), imgsz, device)
        pred[iid] = [((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h, float(c)) for x0, y0, x1, y1, c in d.tolist()]
        gt[iid] = kx.read_yolo_labels(labs[iid])
    n_gt = sum(len(v) for v in gt.values())
    out = dict(tag=tag, weights=summ["weights"], imgsz=imgsz, device=device, threads=summ.get("threads"), backend=summ["backend"],
               n_images=len(split_val), n_boxes=n_gt, map50=round(M.map50(gt, pred), 4), map50_95=round(M.map50_95(gt, pred), 4), at={})
    for thr in sorted(set(THRS + [0.05, 0.15, 0.25, 0.5, 0.75])):
        tp, fp, fn, _, _ = M.match(gt, pred, thr, 0.5, "iou")
        p, r, f = M.prf(tp, fp, fn)
        ctp, cfp, cfn, _, _ = M.match(gt, pred, thr, 0.5, "center")
        out["at"][str(thr)] = dict(tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4), f1=round(f, 4),
                                   center_hit_recall=round(ctp / max(1, n_gt), 4))
    out["pred"] = {k: [[round(x, 5) for x in p] for p in v] for k, v in pred.items()}
    out["scored_at"] = kx.now_iso()
    kx.write_json(OUT / tag / "val_full.json", out)
    a42 = out["at"]["0.42"]
    print(f"[{tag}] val {len(split_val)}장: mAP50 {out['map50']:.4f} mAP50-95 {out['map50_95']:.4f} | @0.42 F1 {a42['f1']:.4f} "
          f"TP/FN/FP {a42['tp']}/{a42['fn']}/{a42['fp']} 중심 {a42['center_hit_recall']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", help="비우면 summary.json 이 있는 설정 전부")
    a = ap.parse_args()
    tags = a.tags or sorted(p.parent.name for p in OUT.glob("*/summary.json"))
    split = kx.load_split()
    imgs, labs = kx.path_maps(split["val"])
    for t in tags:
        score_tag(t, split["val"], imgs, labs)


if __name__ == "__main__":
    main()
