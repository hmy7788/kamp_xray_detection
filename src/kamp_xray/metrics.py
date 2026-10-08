"""모든 모델이 함께 쓰는 채점 코드. 입력은 common.py의 예측 기록(record) 목록이다.

매칭 (IoU 기준, 기본)
  이미지마다 신뢰도가 높은 검출부터, IoU >= 0.5 이고 아직 짝이 없는 정답 중 IoU가 가장 큰 정답과 짝짓는다(TP).
  짝이 없으면 FP, 끝까지 남은 정답은 FN. 정답 하나에는 검출 하나만 짝지어진다.
매칭 (중심 거리, 보조 지표)
  center_r(px)를 주면 IoU 대신 "검출 중심과 정답 중심의 거리 <= center_r" 로 짝짓는다. 이물질이 약 10px 점이라
  상자 크기가 라벨과 달라도 위치가 맞으면 찾은 것으로 보는 지표다 (보고서 3.2절).
AP
  신뢰도 0.001 이상 검출 전체로 정밀도-재현율 곡선을 만들고 전 구간 보간(VOC 방식)으로 면적을 구한다.
  mAP50-95는 IoU 0.50~0.95(0.05 간격) 10개 AP의 평균.
임계값
  val에서 F1이 최대가 되는 신뢰도(검출 신뢰도 값 중 하나, 동률이면 높은 값)로 정하고, test에는 그 값을 그대로 쓴다.
"""
from collections import defaultdict

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
CONF_FLOOR = 0.001


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_image(rec, thr=CONF_FLOOR, iou_thr=0.5, center_r=None):
    """반환: (신뢰도 내림차순 검출의 TP 여부, 그 신뢰도, 정답별 짝지어진 검출 신뢰도(없으면 None))."""
    dets = sorted((d for d in rec["dets"] if d[4] >= thr), key=lambda d: -d[4])
    gt = rec["gt"]
    gt_conf = [None] * len(gt)
    is_tp = []
    if not dets:
        return np.zeros(0, bool), np.zeros(0), gt_conf
    if not gt:
        return np.zeros(len(dets), bool), np.array([d[4] for d in dets], float), gt_conf
    if center_r is None:
        score, ok_min = iou_matrix([d[:4] for d in dets], gt), iou_thr
    else:
        g = np.asarray(gt, float)
        d = np.asarray([x[:4] for x in dets], float)
        gc = np.stack([(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2], 1)
        dc = np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2], 1)
        score, ok_min = center_r - np.linalg.norm(dc[:, None] - gc[None], axis=2), 0.0
    used = np.zeros(len(gt), bool)
    for i, det in enumerate(dets):
        cand = np.where(~used & (score[i] >= ok_min))[0]
        if len(cand):
            j = cand[np.argmax(score[i, cand])]
            used[j] = True
            gt_conf[j] = det[4]
            is_tp.append(True)
        else:
            is_tp.append(False)
    return np.array(is_tp, bool), np.array([d[4] for d in dets], float), gt_conf


def _scored(records, iou_thr=0.5, center_r=None):
    confs, tps, n_gt = [], [], 0
    for r in records:
        t, c, _ = match_image(r, CONF_FLOOR, iou_thr, center_r)
        confs.append(c); tps.append(t); n_gt += len(r["gt"])
    conf = np.concatenate(confs) if confs else np.zeros(0)
    tp = np.concatenate(tps) if tps else np.zeros(0, bool)
    o = np.argsort(-conf, kind="stable")
    return conf[o], tp[o].astype(float), n_gt


def ap(records, iou_thr=0.5, center_r=None):
    conf, tp, n_gt = _scored(records, iou_thr, center_r)
    if not len(conf) or not n_gt:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map50_95(records):
    return float(np.mean([ap(records, t) for t in IOU_THRS]))


def best_threshold(records, iou_thr=0.5, center_r=None):
    """F1이 최대인 신뢰도 임계값 (검출 신뢰도 값 중 하나, conf >= thr 인 검출을 채택). 동률이면 높은 값."""
    conf, tp, n_gt = _scored(records, iou_thr, center_r)
    if not len(conf):
        return 0.5
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    last = np.r_[conf[1:] != conf[:-1], True]           # 같은 신뢰도의 마지막 위치만 후보
    f1 = 2 * ctp[last] / np.maximum(ctp[last] + cfp[last] + n_gt, 1e-9)
    return float(conf[last][int(np.argmax(f1))])


def prf_at(records, thr, iou_thr=0.5, center_r=None):
    tp = fp = n_gt = 0
    img_pos = img_hit = img_full = empty = empty_fp = 0
    for r in records:
        t, _, gconf = match_image(r, thr, iou_thr, center_r)
        k = int(t.sum())
        tp += k; fp += len(t) - k; n_gt += len(r["gt"])
        if r["gt"]:
            img_pos += 1; img_hit += k > 0; img_full += k == len(r["gt"])
        else:
            empty += 1; empty_fp += len(t) > 0
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
            "n_images": len(records), "n_gt": n_gt, "fppi": round(fp / max(len(records), 1), 4),
            "image_detect_rate": round(img_hit / img_pos, 4) if img_pos else None,
            "image_full_detect_rate": round(img_full / img_pos, 4) if img_pos else None,
            "empty_label_images": empty, "empty_label_images_with_fp": empty_fp}


def image_level(records, thr):
    """현장 판정용 이미지 단위 지표: 영상 내 최고 신뢰도 >= thr 이면 불합격(이물질 있음)으로 판정 (보고서 4.1절)."""
    pos = [max((d[4] for d in r["dets"]), default=0.0) for r in records if r["gt"]]
    neg = [max((d[4] for d in r["dets"]), default=0.0) for r in records if not r["gt"]]
    q = lambda v: [round(float(x), 4) for x in np.quantile(v, [0, 0.05, 0.5, 0.95, 1])] if v else None
    return {"defect_images": len(pos), "defect_images_missed": int(sum(s < thr for s in pos)),
            "empty_images": len(neg), "empty_images_flagged": int(sum(s >= thr for s in neg)),
            "defect_max_conf_quantiles_0_5_50_95_100": q(pos), "empty_max_conf_quantiles_0_5_50_95_100": q(neg)}


def report(records, thr, manifest=None):
    """한 split의 전체 채점 결과 (eval_report_<split>.json의 본문)."""
    out = {"threshold": round(float(thr), 6),
           "overall": {**prf_at(records, thr), "ap50": round(ap(records), 4), "map50_95": round(map50_95(records), 4)},
           "ap_by_iou": {f"{t:.2f}": round(ap(records, t), 4) for t in IOU_THRS},
           "center_distance": {f"r{r}px": prf_at(records, thr, center_r=float(r)) for r in (1, 2, 3, 4, 5)},
           "image_level": image_level(records, thr)}
    if manifest is not None:
        groups = defaultdict(lambda: defaultdict(list))
        for r in records:
            m = manifest[r["image_id"]]
            groups["source"][m["source"]].append(r)
            groups["machine"][m["machine"]].append(r)
            groups["resolution"][f'{m["width"]}x{m["height"]}'].append(r)
            groups["month"][m["month"]].append(r)
        out["by_group"] = {g: {k: {**prf_at(v, thr), "ap50": round(ap(v), 4)} for k, v in sorted(d.items())}
                           for g, d in groups.items()}
    return out
