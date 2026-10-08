"""중심 거리 매칭 민감도 분석. 저장된 예측(preds_<split>.json)만 다시 집계하고 모델은 돌리지 않는다.

    python src/yeonchang/rtdetr/center_sensitivity.py --exp 02_rtdetrl_img640_e30

매칭: 예측 중심이 정답 중심에서 r px 이내면 TP (신뢰도 순 탐욕 매칭, 정답 하나에 예측 하나).
임계값: 각 기준마다 val F1 최대로 다시 정해 test에 적용. 비교용으로 IoU 기준 임계값(eval_report_val.json)도 적용.
결과: runs/yeonchang/<exp>/eval_center_sensitivity.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

import evaluate as E
from common import RUNS, read_manifest, load_gt

RADII = [2.0, 3.0, 5.0]


def best_threshold(preds, gt, mode):
    _, curve, _ = E.evaluate(preds, gt, 1.1, mode)
    n_gt = sum(len(g) for g in gt.values())
    f1 = 2 * curve["tp"] / np.maximum(curve["tp"] + curve["fp"] + n_gt, 1e-9)
    return float(curve["thr"][int(np.argmax(f1))])


def report(preds, gt, thr, mode, manifest):
    overall = E.evaluate(preds, gt, thr, mode)[0]
    overall["ap"] = overall.pop("ap50" if mode == "iou" else "ap_center")
    groups = defaultdict(lambda: defaultdict(set))
    for i in gt:
        r = manifest[i]
        groups["source"][r["source"]].add(i)
        groups["machine"][r["machine"]].add(i)
        groups["resolution"][f'{r["width"]}x{r["height"]}'].add(i)
        groups["month"][r["month"]].add(i)
    by = {}
    for g, v in groups.items():
        by[g] = {}
        for k, s in sorted(v.items()):
            o = E.evaluate(preds, gt, thr, mode, s)[0]
            o["ap"] = o.pop("ap50" if mode == "iou" else "ap_center")
            by[g][k] = o
    return {"threshold": thr, "overall": overall, "by": by}


def min_gt_distance(gt):
    """같은 이미지 안 정답 중심 사이 최소 거리(px). r보다 작으면 매칭이 모호해질 수 있다."""
    best = np.inf
    for boxes in gt.values():
        c = np.array([((b[1] + b[3]) / 2, (b[2] + b[4]) / 2) for b in boxes])
        if len(c) > 1:
            d = np.linalg.norm(c[:, None] - c[None], axis=2)
            best = min(best, d[np.triu_indices(len(c), 1)].min())
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    exp_dir = RUNS / args.exp
    manifest = read_manifest()
    iou_thr = json.loads((exp_dir / "eval_report_val.json").read_text(encoding="utf-8"))["threshold"]

    data = {}
    for split in ("val", "test"):
        saved = json.loads((exp_dir / f"preds_{split}.json").read_text(encoding="utf-8"))
        data[split] = ({k: [tuple(p) for p in v] for k, v in saved["preds"].items()}, load_gt(split, manifest))

    out = {"exp": args.exp, "match_rule": "예측 중심이 정답 중심에서 r px 이내 (신뢰도 순 탐욕, 1:1)",
           "threshold_rule": "기준마다 val F1 최대, test에 그대로 적용",
           "min_gt_center_distance_px": {s: round(min_gt_distance(data[s][1]), 2) for s in data},
           "criteria": {}}
    criteria = [("iou0.5", "iou", None)] + [(f"center_r{int(r)}", "center", r) for r in RADII]
    for name, mode, r in criteria:
        if r is not None:
            E.CENTER_R = r
        thr = best_threshold(*data["val"], mode)
        out["criteria"][name] = {
            "val": report(*data["val"], thr, mode, manifest),
            "test": report(*data["test"], thr, mode, manifest),
            "test_at_iou_threshold": E.evaluate(*data["test"], iou_thr, mode)[0],
        }
    (exp_dir / "eval_center_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("min GT center distance (px):", out["min_gt_center_distance_px"])
    for name, c in out["criteria"].items():
        for split in ("val", "test"):
            o = c[split]["overall"]
            print(f"{name:10} {split:4} thr={c[split]['threshold']:.4f} AP={o['ap']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} "
                  f"F1={o['f1']:.4f} TP/FP/FN={o['tp']}/{o['fp']}/{o['fn']} FPPI={o['fppi']:.4f} "
                  f"img_det={o['image_detect_rate']} img_full={o['image_full_detect_rate']} emptyFP={o['empty_label_images_with_fp']}/{o['empty_label_images']}")
        t = c["test_at_iou_threshold"]
        print(f"{'':10} test @iou-thr {iou_thr:.4f}: F1={t['f1']:.4f} TP/FP/FN={t['tp']}/{t['fp']}/{t['fn']}")
    for g in ("source", "machine", "resolution", "month"):
        print(f"[test F1 by {g}]")
        keys = out["criteria"]["iou0.5"]["test"]["by"][g].keys()
        for k in keys:
            row = "  ".join(f"{n}={c['test']['by'][g][k]['f1']:.3f}({c['test']['by'][g][k]['fp']}/{c['test']['by'][g][k]['fn']})"
                            for n, c in out["criteria"].items())
            print(f"  {k:9} {row}")


if __name__ == "__main__":
    main()
