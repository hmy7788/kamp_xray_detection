"""검출 결과 기록으로 AP@0.5 와 고정 임계값의 P/R/F1 을 계산한다 (학습 중 val 로 체크포인트를 고르는 용도).

기록(record) 형식은 모델 공통이다 (다른 모델의 결과와 같은 평가 코드로 채점하기 위한 형식):
  {"name": "<이미지 파일명>", "w": 가로, "h": 세로,
   "gt":   [[x1, y1, x2, y2], ...],
   "dets": [[x1, y1, x2, y2, score], ...]}

매칭: 신뢰도가 높은 검출부터 IoU >= iou_thr 인 아직 짝이 없는 정답과 짝지음(TP), 짝이 없으면 FP, 남은 정답은 FN.
AP: 신뢰도 0.001 이상 전체 검출로 PR 곡선을 만들어 전 구간 보간(VOC 방식)으로 면적을 구한다.
"""
import numpy as np


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_image(rec, thr, iou_thr=0.5):
    """(신뢰도 내림차순 TP 여부 배열, 신뢰도 배열, 짝지어진 정답 수)"""
    dets = sorted((d for d in rec["dets"] if d[4] >= thr), key=lambda d: -d[4])
    gt = rec["gt"]
    used, is_tp = set(), []
    ious = iou_matrix([d[:4] for d in dets], gt) if dets and gt else None
    for i in range(len(dets)):
        ok = False
        if gt:
            for j in np.argsort(-ious[i]):
                if ious[i, j] < iou_thr:
                    break
                if j not in used:
                    used.add(j)
                    ok = True
                    break
        is_tp.append(ok)
    return np.array(is_tp, bool), np.array([d[4] for d in dets], float), len(used)


def prf_at(records, thr, iou_thr=0.5):
    tp = fp = fn = n_gt = 0
    for r in records:
        t, _, n_match = match_image(r, thr, iou_thr)
        k = int(t.sum())
        tp += k; fp += len(t) - k; fn += len(r["gt"]) - n_match; n_gt += len(r["gt"])
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"thr": round(float(thr), 3), "P": p, "R": rc, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


def ap(records, iou_thr=0.5, floor=0.001):
    confs, tps, n_gt = [], [], 0
    for r in records:
        t, c, _ = match_image(r, floor, iou_thr)
        confs.append(c); tps.append(t); n_gt += len(r["gt"])
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


def best_f1(records, iou_thr=0.5):
    """F1 이 최대인 임계값(동률이면 가운데 값)과 그때의 P/R/F1."""
    grid = np.round(np.arange(0.05, 0.99, 0.01), 2)
    rows = [prf_at(records, t, iou_thr) for t in grid]
    top = max(r["F1"] for r in rows)
    cand = [r for r in rows if r["F1"] >= top - 1e-12]
    return cand[len(cand) // 2]
