"""채점 계산. evaluate.py 만 부른다. 단일 클래스 전용.

입력 형식:
  gt:   {image_id: [(cx, cy, w, h), ...]}            정규화
  pred: {image_id: [(cx, cy, w, h, conf), ...]}      정규화, 빈 리스트 가능
"""
from __future__ import annotations

import numpy as np


def _xyxy(b):
    cx, cy, w, h = b[:4]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = _xyxy(a)
    bx0, by0, bx1, by1 = _xyxy(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def center_in(p, g) -> bool:
    gx0, gy0, gx1, gy1 = _xyxy(g)
    return gx0 <= p[0] <= gx1 and gy0 <= p[1] <= gy1


def match(gt: dict, pred: dict, thr_conf: float, iou_thr: float = 0.5, mode: str = "iou"):
    """확신도 thr_conf 이상 예측을 확신도 순으로 정답에 1:1 매칭.
    돌려줌: tp, fp, fn, matched_gt {(image_id, gt_idx)}, fp_list [(image_id, pred)]"""
    tp = fp = 0
    matched = set()
    fps = []
    n_gt = sum(len(v) for v in gt.values())
    for img, preds in pred.items():
        g = gt.get(img, [])
        used = [False] * len(g)
        for p in sorted([p for p in preds if p[4] >= thr_conf], key=lambda x: -x[4]):
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if used[i]:
                    continue
                s = iou(p, gb) if mode == "iou" else (1.0 if center_in(p, gb) else 0.0)
                if s > best:
                    best, bi = s, i
            ok = (best >= iou_thr) if mode == "iou" else (best > 0)
            if ok:
                used[bi] = True
                matched.add((img, bi))
                tp += 1
            else:
                fp += 1
                fps.append((img, p))
    fn = n_gt - tp
    return tp, fp, fn, matched, fps


def average_precision(gt: dict, pred: dict, iou_thr: float) -> float:
    """VOC 전 구간 보간 AP. 모든 예측을 확신도 순으로 훑는다."""
    n_gt = sum(len(v) for v in gt.values())
    if n_gt == 0:
        return 0.0
    allp = [(img, p) for img, ps in pred.items() for p in ps]
    allp.sort(key=lambda x: -x[1][4])
    used = {img: [False] * len(g) for img, g in gt.items()}
    tps = []
    for img, p in allp:
        g = gt.get(img, [])
        best, bi = 0.0, -1
        for i, gb in enumerate(g):
            if used[img][i]:
                continue
            s = iou(p, gb)
            if s > best:
                best, bi = s, i
        if best >= iou_thr:
            used[img][bi] = True
            tps.append(1)
        else:
            tps.append(0)
    if not tps:
        return 0.0
    tps = np.array(tps)
    ctp = np.cumsum(tps)
    cfp = np.cumsum(1 - tps)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map50(gt, pred) -> float:
    return average_precision(gt, pred, 0.5)


def map50_95(gt, pred) -> float:
    return float(np.mean([average_precision(gt, pred, t) for t in np.arange(0.5, 0.96, 0.05)]))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def image_level(gt: dict, pred: dict, thr_conf: float):
    """사진 단위 있음/없음. 주의: 제공 라벨 사진은 전부 이물질이 있어 음성이 없다."""
    tp = fp = fn = tn = 0
    for img in set(gt) | set(pred):
        has_gt = len(gt.get(img, [])) > 0
        has_pred = any(p[4] >= thr_conf for p in pred.get(img, []))
        if has_gt and has_pred:
            tp += 1
        elif has_gt:
            fn += 1
        elif has_pred:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def best_threshold(gt, pred, iou_thr=0.5, grid=None) -> float:
    """상자 F1 이 최대인 확신도 임계값 (검증 예측 전체에서)."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    best, bt = -1.0, 0.5
    for t in grid:
        tp, fp, fn, _, _ = match(gt, pred, float(t), iou_thr)
        f = prf(tp, fp, fn)[2]
        if f > best:
            best, bt = f, float(t)
    return bt


def threshold_sweep(gt, pred, target_recall=0.99, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    n_img = len(set(gt) | set(pred))
    for t in grid:
        tp, fp, fn, _, _ = match(gt, pred, float(t), 0.5)
        p, r, f = prf(tp, fp, fn)
        flagged = sum(1 for img in set(gt) | set(pred) if any(x[4] >= t for x in pred.get(img, [])))
        rows.append(dict(thr=float(t), recall=r, precision=p, f1=f, recheck_rate=flagged / n_img if n_img else 0.0, missed=fn))
    ok = [r for r in rows if r["recall"] >= target_recall]
    safety = max(ok, key=lambda r: r["thr"]) if ok else None
    return rows, {"target_recall": target_recall,
                  "threshold": safety["thr"] if safety else None,
                  "recheck_rate": safety["recheck_rate"] if safety else None,
                  "note": "" if safety else "어느 임계값에서도 목표 재현율에 못 미침"}
