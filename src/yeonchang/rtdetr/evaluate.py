"""RT-DETR 평가. 신뢰도 임계값은 val에서 정해 고정하고, test는 그 값으로 한 번만 채점한다.

    python src/yeonchang/rtdetr/evaluate.py --exp 01_rtdetrl_img640 --split val    # 임계값 결정
    python src/yeonchang/rtdetr/evaluate.py --exp 01_rtdetrl_img640 --split test   # 최종 1회

매칭: 신뢰도 높은 예측부터 IoU가 가장 큰 미매칭 정답과 짝지음 (IoU >= 0.5). 정답 하나엔 예측 하나만.
AP@0.5는 VOC 전 구간 보간. 보조 지표로 중심 거리 매칭(r px 이내)도 함께 낸다.
결과: eval_report_<split>.json (Git 포함), preds_<split>.json (예측 + 정답별 매칭, Git 제외)
"""
import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np

from common import DATA, RUNS, check_data_version, env_info, load_gt, read_manifest

IOU_THR = 0.5
CENTER_R = 5.0  # px


def iou_matrix(p, g):
    p, g = np.asarray(p, float).reshape(-1, 4), np.asarray(g, float).reshape(-1, 4)
    x1 = np.maximum(p[:, None, 0], g[None, :, 0]); y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2]); y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]); ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ap[:, None] + ag[None, :] - inter + 1e-9)


def center_score(p, g):
    """중심 거리 매칭용 점수: r 이내면 (r - 거리), 아니면 음수."""
    p, g = np.asarray(p, float).reshape(-1, 4), np.asarray(g, float).reshape(-1, 4)
    pc = np.stack([(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2], 1)
    gc = np.stack([(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2], 1)
    d = np.linalg.norm(pc[:, None] - gc[None], axis=2)
    return CENTER_R - d


def match_image(preds, gts, mode):
    """preds: [(conf, x1,y1,x2,y2)], gts: [(idx, x1,y1,x2,y2)].
    반환: 예측별 (conf, is_tp), 정답별 매칭된 예측 conf(없으면 None)."""
    preds = sorted(preds, key=lambda x: -x[0])
    gt_conf = [None] * len(gts)
    out = []
    if not gts:
        return [(c, False) for c, *_ in preds], gt_conf
    if not preds:
        return out, gt_conf
    pb = [x[1:] for x in preds]; gb = [x[1:] for x in gts]
    if mode == "iou":
        s, thr = iou_matrix(pb, gb), IOU_THR
    else:
        s, thr = center_score(pb, gb), 0.0
    used = np.zeros(len(gts), bool)
    for i, (c, *_) in enumerate(preds):
        cand = np.where(~used & (s[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(s[i, cand])]
            used[j] = True; gt_conf[j] = c
            out.append((c, True))
        else:
            out.append((c, False))
    return out, gt_conf


def pr_curve(scored, n_gt):
    """scored: 모든 예측 (conf, is_tp). 임계값별 TP/FP/FN과 AP(VOC 전 구간)를 계산."""
    if not scored:
        return {"ap": 0.0, "thr": np.array([]), "tp": np.array([]), "fp": np.array([])}
    a = np.array(sorted(scored, key=lambda x: -x[0]), dtype=float)
    conf, tp = a[:, 0], a[:, 1]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    # 같은 신뢰도의 마지막 위치만 임계값 후보로 둔다 (conf >= thr 인 예측을 채택)
    last = np.r_[conf[1:] != conf[:-1], True]
    return {"ap": ap, "thr": conf[last], "tp": ctp[last], "fp": cfp[last]}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def evaluate(preds_by_img, gt, thr, mode, subset=None):
    """subset: 포함할 image_id 집합 (None이면 전체). 임계값 thr에서의 지표와 전 구간 AP."""
    ids = [i for i in gt if subset is None or i in subset]
    scored, n_gt = [], 0
    tp = fp = fn = 0
    img_hit = img_full = img_pos = 0
    empty_imgs = empty_fp_imgs = 0
    gt_match = {}
    for i in ids:
        g = gt[i]; n_gt += len(g)
        res, _ = match_image(preds_by_img.get(i, []), g, mode)
        scored += res
        # 임계값 적용 후 다시 매칭 (낮은 신뢰도 예측이 매칭을 가져가지 않도록)
        kept = [p for p in preds_by_img.get(i, []) if p[0] >= thr]
        res_t, gconf = match_image(kept, g, mode)
        t = sum(1 for _, ok in res_t if ok); f = len(res_t) - t
        tp += t; fp += f; fn += len(g) - t
        for (bidx, *_), c in zip(g, gconf):
            gt_match[(i, bidx)] = c
        if g:
            img_pos += 1; img_hit += t > 0; img_full += t == len(g)
        else:
            empty_imgs += 1; empty_fp_imgs += f > 0
    curve = pr_curve(scored, n_gt)
    out = prf(tp, fp, fn)
    out.update({
        "ap50" if mode == "iou" else "ap_center": round(curve["ap"], 4),
        "n_images": len(ids), "n_gt": n_gt,
        "fppi": round(fp / max(len(ids), 1), 4),
        "image_detect_rate": round(img_hit / img_pos, 4) if img_pos else None,
        "image_full_detect_rate": round(img_full / img_pos, 4) if img_pos else None,
        "empty_label_images": empty_imgs,
        "empty_label_images_with_fp": empty_fp_imgs,
    })
    return out, curve, gt_match


def predict(exp_dir, split, manifest, imgsz, device, batch):
    from ultralytics import RTDETR
    model = RTDETR(str(exp_dir / "weights" / "best.pt"))
    ids = [i for i, r in manifest.items() if r["split"] == split]
    preds, t_infer = {}, 0.0
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        paths = [str(DATA / manifest[i]["image_path"]) for i in chunk]
        results = model.predict(paths, imgsz=imgsz, conf=0.001, device=device, verbose=False, max_det=300)
        for i, r in zip(chunk, results):
            b = r.boxes
            preds[i] = [(float(c), *map(float, xy)) for c, xy in zip(b.conf.tolist(), b.xyxy.tolist())]
            t_infer += r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
    return preds, t_infer / max(len(ids), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--imgsz", type=int, default=None, help="기본: 학습 때 imgsz (config.json)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reuse-preds", action="store_true", help="preds_<split>.json이 있으면 추론을 건너뜀")
    args = ap.parse_args()

    exp_dir = RUNS / args.exp
    reuse = args.reuse_preds and (exp_dir / f"preds_{args.split}.json").exists()
    # 저장된 예측을 쓰면 모델 종류와 무관하게 채점한다 (D-FINE은 ../dfine/predict.py가 예측을 만든다)
    if not reuse and not (exp_dir / "weights" / "best.pt").exists():
        sys.exit(f"best.pt가 없습니다: {exp_dir / 'weights'}")
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    if "finished" not in cfg:
        sys.exit("학습이 끝나지 않은 실험입니다. 최종 결과로 쓰지 않습니다 (_incomplete_로 이름을 바꾸세요).")
    imgsz = args.imgsz or cfg["hyp"]["imgsz"]

    if args.split == "test":
        val_rep = exp_dir / "eval_report_val.json"
        if not val_rep.exists():
            sys.exit("먼저 --split val로 임계값을 정하세요.")
        if (exp_dir / "eval_report_test.json").exists():
            sys.exit("test는 이미 한 번 채점했습니다. 다시 보지 않습니다.")
        thr = json.loads(val_rep.read_text(encoding="utf-8"))["threshold"]
    digest = check_data_version()
    manifest = read_manifest()
    gt = load_gt(args.split, manifest)

    pred_file = exp_dir / f"preds_{args.split}.json"
    if reuse:
        saved = json.loads(pred_file.read_text(encoding="utf-8"))
        preds, ms = {k: [tuple(p) for p in v] for k, v in saved["preds"].items()}, saved["ms_per_image"]
    else:
        preds, ms = predict(exp_dir, args.split, manifest, imgsz, args.device, args.batch)

    if args.split == "val":
        # val에서 F1 최대 임계값 (IoU 0.5 기준)
        _, curve, _ = evaluate(preds, gt, 1.1, "iou")
        n_gt = sum(len(g) for g in gt.values())
        f1 = 2 * curve["tp"] / np.maximum(curve["tp"] + curve["fp"] + n_gt, 1e-9)
        thr = float(curve["thr"][int(np.argmax(f1))]) if len(f1) else 0.5

    report = {"exp": args.exp, "split": args.split, "threshold": thr,
              "threshold_rule": "val F1 최대 (IoU 0.5), test에는 val 값을 그대로 적용",
              "imgsz": imgsz, "ms_per_image": round(ms, 2), "manifest_sha256": digest,
              "match": {"iou": IOU_THR, "center_r_px": CENTER_R}, "env": env_info(),
              "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    overall, curve, gt_match = evaluate(preds, gt, thr, "iou")
    report["overall"] = overall
    report["overall_center"] = evaluate(preds, gt, thr, "center")[0]
    report["curve"] = [{"thr": round(float(t), 4), **prf(tp, fp, overall["n_gt"] - tp)}
                       for t, tp, fp in zip(curve["thr"], curve["tp"], curve["fp"])
                       if t >= 0.01][::max(1, len(curve["thr"]) // 200)]
    groups = defaultdict(lambda: defaultdict(set))
    for i in gt:
        r = manifest[i]
        groups["source"][r["source"]].add(i)
        groups["machine"][r["machine"]].add(i)
        groups["resolution"][f'{r["width"]}x{r["height"]}'].add(i)
        groups["month"][r["month"]].add(i)
    report["by"] = {g: {k: evaluate(preds, gt, thr, "iou", s)[0] for k, s in sorted(v.items())}
                    for g, v in groups.items()}
    (exp_dir / f"eval_report_{args.split}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 예측과 정답별 매칭(오류분석용, conditions.csv와 image_id·box_idx로 조인). 정답 좌표가 들어가므로 Git 제외 파일명.
    pred_file.write_text(json.dumps({
        "threshold": thr, "ms_per_image": ms,
        "preds": preds,
        "gt_match": [{"image_id": i, "box_idx": b, "matched_conf": c} for (i, b), c in gt_match.items()],
    }), encoding="utf-8")

    o = overall
    print(f"[{args.split}] thr={thr:.4f}  AP50={o['ap50']}  P={o['precision']} R={o['recall']} F1={o['f1']}  "
          f"TP/FP/FN={o['tp']}/{o['fp']}/{o['fn']}  FPPI={o['fppi']}  {ms:.1f} ms/img")
    for k, v in report["by"]["source"].items():
        print(f"  source={k}: F1={v['f1']} R={v['recall']} P={v['precision']} (n_gt={v['n_gt']})")
    print(f"  center<= {CENTER_R}px: F1={report['overall_center']['f1']}")


if __name__ == "__main__":
    main()
