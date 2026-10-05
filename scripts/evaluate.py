"""신뢰도 임계값을 val 에서 정해 고정하고 bbox/이미지 단위 P/R/F1 을 직접 계산한다.

왜 필요한가: test.py 의 P/R/F1 은 신뢰도 0.1 한 지점 값이라 모델의 신뢰도 보정이나 평가 해상도에 따라
크게 변한다(docs/experiment_results.md). 여기서는 같은 가중치의 검출 결과(신뢰도 0.001 이상)를 저장해 두고
임계값을 바꿔 가며 지표를 계산한다. 임계값은 val 에서만 정하고 test 에는 그대로 적용한다.

  python scripts/evaluate.py 01_yolov3spp_coco [--img-size 640] [--iou 0.5]

산출물(runs/<name>/): preds_<split>_img<크기>.json(검출 캐시, 로컬 경로 없음),
                              eval_report_img<크기>.json
지표 정의
  - bbox 단위: 검출을 신뢰도 순으로 IoU>=iou 인 미매칭 정답과 짝지음(TP), 짝이 없으면 FP, 못 찾은 정답은 FN.
  - 이미지 단위(정답이 있는 이미지만 존재): 검출률 = 정답 중 1개 이상 찾은 이미지 비율,
    완전 검출률 = 정답을 전부 찾은 이미지 비율. 정상 이미지가 없어 이미지 단위 FP/정밀도는 정의할 수 없다.
  - AP@iou: 신뢰도 0.001 이상 전체 검출로 계산한 AP (test.py 의 mAP 와 비교용).
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "src" / "yolov3"
SPLITS = ("val", "test")
CONF_FLOOR = 0.001


# ---------------------------------------------------------------- 추론 (검출 캐시)
def predict(name, split, img_size, device_arg, list_dir=None):
    """best.pt 로 split 의 모든 이미지를 추론해 [{name, w, h, gt, dets}] 를 반환한다."""
    import torch
    sys.path.insert(0, str(YOLO))
    from models import Darknet  # noqa: E402
    from utils import torch_utils  # noqa: E402
    from utils.datasets import LoadImages  # noqa: E402
    from utils.utils import non_max_suppression, scale_coords  # noqa: E402

    run = ROOT / "runs" / name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    ckpt = run / "weights" / ("best.pt" if (run / "weights" / "best.pt").exists() else "last.pt")
    device = torch_utils.select_device(device=device_arg or cfg.get("device") or "")
    model = Darknet(str(YOLO / cfg["cfg"]), img_size)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.to(device).eval()

    list_file = Path(list_dir or ROOT / "data" / "splits") / f"{split}.txt"
    paths = [l.strip() for l in list_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    with torch.no_grad():
        for p in paths:
            with contextlib.redirect_stdout(io.StringIO()):  # LoadImages 가 이미지 경로를 출력하는 것을 막는다
                _, img, im0, _ = next(iter(LoadImages(p, img_size=img_size)))
            x = torch.from_numpy(img).to(device).float() / 255.0
            pred = model(x.unsqueeze(0))[0]
            det = non_max_suppression(pred, CONF_FLOOR, 0.6, multi_label=False)[0]
            dets = []
            if det is not None:
                det[:, :4] = scale_coords(x.shape[1:], det[:, :4], im0.shape).round()
                dets = [[float(v) for v in d[:5]] for d in det.cpu().numpy()]  # x1 y1 x2 y2 conf
            h, w = im0.shape[:2]
            lab = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
            gt = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _, cx, cy, bw, bh = map(float, line.split()[:5])
                    gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            out.append({"name": Path(p).name, "w": w, "h": h, "gt": gt, "dets": dets})
    return out


# ---------------------------------------------------------------- 매칭 / 지표
def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_image(rec, thr, iou_thr):
    """한 이미지에서 신뢰도>=thr 검출을 정답과 짝짓는다. (신뢰도 내림차순 TP 여부 배열, 신뢰도 배열) 반환."""
    dets = sorted((d for d in rec["dets"] if d[4] >= thr), key=lambda d: -d[4])
    gt = rec["gt"]
    used = set()
    is_tp = []
    if dets and gt:
        ious = iou_matrix([d[:4] for d in dets], gt)
    for i, d in enumerate(dets):
        ok = False
        if gt:
            order = np.argsort(-ious[i])
            for j in order:
                if ious[i, j] < iou_thr:
                    break
                if j not in used:
                    used.add(j)
                    ok = True
                    break
        is_tp.append(ok)
    return np.array(is_tp, bool), np.array([d[4] for d in dets], float), len(used)


def metrics_at(recs, thr, iou_thr):
    tp = fp = fn = n_gt = 0
    found = full = 0
    for r in recs:
        is_tp, _, n_match = match_image(r, thr, iou_thr)
        t = int(is_tp.sum()); f = len(is_tp) - t
        tp += t; fp += f; fn += len(r["gt"]) - n_match; n_gt += len(r["gt"])
        found += n_match >= 1
        full += n_match == len(r["gt"])
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"thr": round(float(thr), 3), "P": p, "R": rc, "F1": f1, "TP": tp, "FP": fp, "FN": fn,
            "image_detect_rate": found / len(recs), "image_full_rate": full / len(recs),
            "fp_per_image": fp / len(recs)}


def average_precision(recs, iou_thr):
    """신뢰도 0.001 이상 전체 검출로 AP 를 계산한다 (VOC 식 전 구간 보간)."""
    confs, tps, n_gt = [], [], 0
    for r in recs:
        is_tp, c, _ = match_image(r, CONF_FLOOR, iou_thr)
        confs.append(c); tps.append(is_tp); n_gt += len(r["gt"])
    conf = np.concatenate(confs) if confs else np.array([])
    tp = np.concatenate(tps) if tps else np.array([], bool)
    if not len(conf) or not n_gt:
        return 0.0
    o = np.argsort(-conf)
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pick_threshold(recs, iou_thr):
    """val 에서 F1 이 최대인 임계값(동률이면 가운데 값)을 고른다."""
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    rows = [metrics_at(recs, t, iou_thr) for t in grid]
    best = max(r["F1"] for r in rows)
    ts = [r["thr"] for r in rows if r["F1"] >= best - 1e-12]
    return ts[len(ts) // 2], rows


def per_unit(recs, thr, iou_thr):
    units = {r["name"]: r["unit"] for r in recs}
    out = {}
    for u in sorted(set(units.values())):
        sub = [r for r in recs if units[r["name"]] == u]
        m = metrics_at(sub, thr, iou_thr)
        out[u] = {k: m[k] for k in ("P", "R", "F1", "TP", "FP", "FN")} | {"images": len(sub)}
    return out


# ---------------------------------------------------------------- main
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--img-size", type=int, default=None, help="기본: 학습 --img-size 의 마지막 값")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="")
    ap.add_argument("--refresh", action="store_true", help="저장된 검출 캐시를 무시하고 다시 추론")
    ap.add_argument("--list-dir", default=None, help="val.txt/test.txt 가 있는 폴더 (기본: 실험의 dataset 에 맞는 data/splits 또는 data/splits_nomark)")
    ap.add_argument("--tag", default="", help="산출물 이름에 붙일 꼬리표 (예: _nomark)")
    a = ap.parse_args()

    run = ROOT / "runs" / a.name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    img = a.img_size or cfg["img_size"][-1]
    if not a.list_dir and cfg.get("dataset") == "nomark":
        a.list_dir = str(ROOT / "data" / "splits_nomark")  # 표시 제거본으로 학습한 실험은 제거본으로 평가
    unit = {Path(r["name"]).stem: r["unit"] for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}  # stem 으로 매핑(제거본은 .png)

    recs = {}
    for sp in SPLITS:
        cache = run / f"preds_{sp}_img{img}{a.tag}.json"
        if cache.exists() and not a.refresh:
            recs[sp] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"[{sp}] 추론 중 ({a.name}, {img}px) ...", flush=True)
            recs[sp] = predict(a.name, sp, img, a.device, a.list_dir)
            cache.write_text(json.dumps(recs[sp]), encoding="utf-8")
        for r in recs[sp]:
            r["unit"] = unit.get(Path(r["name"]).stem, "?")

    thr, sweep = pick_threshold(recs["val"], a.iou)
    rep = {"run": a.name, "eval_img_size": img, "iou": a.iou, "threshold_from": "val", "threshold": thr,
           "n_images": {s: len(recs[s]) for s in SPLITS}, "n_gt": {s: sum(len(r["gt"]) for r in recs[s]) for s in SPLITS}}
    for sp in SPLITS:
        rep[sp] = metrics_at(recs[sp], thr, a.iou) | {"AP": average_precision(recs[sp], a.iou), "per_unit": per_unit(recs[sp], thr, a.iou)}
        rep[f"{sp}_at_0.1"] = metrics_at(recs[sp], 0.1, a.iou)  # test.py 와 같은 신뢰도 0.1 지점 (비교용)
    rep["sensitivity"] = {sp: [metrics_at(recs[sp], t, a.iou) for t in (0.05, 0.1, 0.25, 0.5, 0.75)] for sp in SPLITS}
    (run / f"eval_report_img{img}{a.tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    old = json.loads((run / "metrics.json").read_text(encoding="utf-8")) if (run / "metrics.json").exists() else {}
    print(f"\n=== {a.name} | {img}px | IoU {a.iou} | 임계값 {thr} (val 에서 F1 최대) ===")
    print(f"{'':6s}{'P':>7s}{'R':>7s}{'F1':>7s}{'TP':>5s}{'FP':>5s}{'FN':>5s}{'AP@.5':>8s}{'검출률':>8s}{'완전검출':>9s}")
    for sp in SPLITS:
        m = rep[sp]
        print(f"{sp:6s}{m['P']:7.3f}{m['R']:7.3f}{m['F1']:7.3f}{m['TP']:5d}{m['FP']:5d}{m['FN']:5d}{m['AP']:8.3f}"
              f"{m['image_detect_rate']:8.3f}{m['image_full_rate']:9.3f}")
    print("test 호기별 (임계값 고정):", {u: round(v["F1"], 3) for u, v in rep["test"]["per_unit"].items()})
    print("-- 신뢰도 0.1 고정 (test.py 와 같은 지점)")
    for sp in SPLITS:
        m = rep[f"{sp}_at_0.1"]
        ref = old.get(sp)
        print(f"{sp:6s} P {m['P']:.3f} R {m['R']:.3f} F1 {m['F1']:.3f}" + (f"  | test.py: P {ref['precision']:.3f} R {ref['recall']:.3f} F1 {ref['F1']:.3f} mAP {ref['mAP@0.5']:.3f}" if ref else ""))


if __name__ == "__main__":
    main()
