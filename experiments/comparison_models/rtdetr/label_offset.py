"""라벨 박스 중심이 실제 점(결함) 중심에서 얼마나 벗어났는지 측정. 모델은 돌리지 않는다(예측은 저장된 파일만 읽음).

실제 점 중심: 박스+search px 안에서 black top-hat(7px) 최대값의 절반 이상인 연결 영역의 밝기 가중 중심.
    python src/yeonchang/rtdetr/label_offset.py --exps 02_rtdetrl_img640_e30 03_dfinen_img640_e50

결과: runs/yeonchang/label_offset.json, (--show) runs/yeonchang/<첫 exp>/result/label_offset/ 실패 사례 확대 그림
"""
import argparse
import json

import cv2
import numpy as np

import evaluate as E
from common import DATA, RUNS, load_gt, read_manifest
from erase_defects import imread, imwrite


def dot_center(g, box, search=4, se=7, k=3.0):
    """g: 회색조. 반환 (cx, cy, peak/sd) 또는 None."""
    h, w = g.shape
    _, x1, y1, x2, y2 = box
    X1, Y1 = max(0, int(np.floor(x1 - search))), max(0, int(np.floor(y1 - search)))
    X2, Y2 = min(w, int(np.ceil(x2 + search))), min(h, int(np.ceil(y2 + search)))
    pad = 10
    PX1, PY1, PX2, PY2 = max(0, X1 - pad), max(0, Y1 - pad), min(w, X2 + pad), min(h, Y2 + pad)
    patch = g[PY1:PY2, PX1:PX2]
    th = cv2.morphologyEx(patch, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))).astype(np.float32)
    inner = np.zeros_like(th, bool)
    inner[Y1 - PY1:Y2 - PY1, X1 - PX1:X2 - PX1] = True
    sd = float(th[~inner].std()) + 1e-6
    local = np.where(inner, th, 0)
    py, px = np.unravel_index(int(np.argmax(local)), local.shape)
    peak = float(local[py, px])
    if peak < k * sd:
        return None
    _, lab = cv2.connectedComponents((local >= 0.5 * peak).astype(np.uint8), connectivity=8)
    m = lab == lab[py, px]
    ys, xs = np.nonzero(m)
    wgt = local[m]
    return float((xs * wgt).sum() / wgt.sum()) + PX1, float((ys * wgt).sum() / wgt.sum()) + PY1, peak / sd


def stats(rows):
    if not rows:
        return {}
    d = np.array([r["dist"] for r in rows]); dx = np.array([r["dx"] for r in rows]); dy = np.array([r["dy"] for r in rows])
    return {"n": len(rows), "dist_median": round(float(np.median(d)), 2), "dist_mean": round(float(d.mean()), 2),
            "dist_p90": round(float(np.percentile(d, 90)), 2), "ge_1px": round(float((d >= 1).mean()), 3),
            "ge_2px": round(float((d >= 2).mean()), 3), "ge_3px": round(float((d >= 3).mean()), 3),
            "dx_mean": round(float(dx.mean()), 2), "dy_mean": round(float(dy.mean()), 2),
            "side_median": round(float(np.median([r["side"] for r in rows])), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["02_rtdetrl_img640_e30", "03_dfinen_img640_e50"])
    ap.add_argument("--show", action="store_true", help="test 실패 사례 확대 그림 저장")
    args = ap.parse_args()
    manifest = read_manifest()

    rows, missing = [], {"official": 0, "team": 0}
    for split in ("train", "val", "test"):
        gt = load_gt(split, manifest)
        for i, boxes in gt.items():
            if not boxes:
                continue
            g = imread(DATA / manifest[i]["image_path"])[..., 0]
            src = manifest[i]["source"]
            for b in boxes:
                c = dot_center(g, b)
                if c is None:
                    missing[src] += 1
                    continue
                bx, by = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
                rows.append({"split": split, "image_id": i, "box_idx": b[0], "source": src, "machine": manifest[i]["machine"],
                             "dx": bx - c[0], "dy": by - c[1], "dist": float(np.hypot(bx - c[0], by - c[1])),
                             "side": float(max(b[3] - b[1], b[4] - b[2])), "dot": (c[0], c[1]), "snr": c[2]})

    out = {"method": "라벨 박스 중심 - 실제 점 중심(black top-hat 7px, 반치폭 영역의 가중 중심). dx>0이면 라벨이 점보다 오른쪽",
           "dot_not_found": missing, "by_source": {}, "by_source_machine": {}}
    for s in ("official", "team"):
        out["by_source"][s] = stats([r for r in rows if r["source"] == s])
        for m in ("1", "2", "3"):
            out["by_source_machine"][f"{s}_m{m}"] = stats([r for r in rows if r["source"] == s and r["machine"] == m])

    # test 실패 사례: 예측 중심이 라벨 중심과 실제 점 중 어디에 가까운가
    dot_of = {(r["image_id"], r["box_idx"]): r for r in rows if r["split"] == "test"}
    gt_te = load_gt("test", manifest)
    out["test_failures"] = {}
    for exp in args.exps:
        d = json.loads((RUNS / exp / "preds_test.json").read_text(encoding="utf-8"))
        thr = json.loads((RUNS / exp / "eval_report_val.json").read_text(encoding="utf-8"))["threshold"]
        cases = []
        for i, boxes in gt_te.items():
            kept = [tuple(p) for p in d["preds"].get(i, []) if p[0] >= thr]
            _, gconf = E.match_image(kept, boxes, "iou")
            for b, c in zip(boxes, gconf):
                if c is not None:
                    continue
                r = dot_of.get((i, b[0]))
                gc = np.array([(b[1] + b[3]) / 2, (b[2] + b[4]) / 2])
                near = sorted(kept, key=lambda p: np.hypot((p[1] + p[3]) / 2 - gc[0], (p[2] + p[4]) / 2 - gc[1]))
                if not near or r is None:
                    cases.append({"image_id": i, "box_idx": b[0], "source": manifest[i]["source"], "note": "예측 없음" if not near else "점 못 찾음"})
                    continue
                p = near[0]; pc = np.array([(p[1] + p[3]) / 2, (p[2] + p[4]) / 2]); dc = np.array(r["dot"])
                iou_label = float(E.iou_matrix([p[1:]], [b[1:]])[0, 0])
                # 라벨 박스를 크기 그대로 실제 점 중심으로 옮겼을 때의 IoU
                hw, hh = (b[3] - b[1]) / 2, (b[4] - b[2]) / 2
                moved = [dc[0] - hw, dc[1] - hh, dc[0] + hw, dc[1] + hh]
                cases.append({"image_id": i, "box_idx": b[0], "source": manifest[i]["source"],
                              "label_box_wh": [round(b[3] - b[1], 1), round(b[4] - b[2], 1)],
                              "pred_box_wh": [round(p[3] - p[1], 1), round(p[4] - p[2], 1)],
                              "label_to_dot_px": round(r["dist"], 2), "pred_to_dot_px": round(float(np.linalg.norm(pc - dc)), 2),
                              "pred_to_label_px": round(float(np.linalg.norm(pc - gc)), 2),
                              "iou_pred_vs_label": round(iou_label, 3),
                              "iou_pred_vs_label_moved_to_dot": round(float(E.iou_matrix([p[1:]], [moved])[0, 0]), 3)})
        out["test_failures"][exp] = cases
    (RUNS / "label_offset.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("dot not found:", missing)
    for k, v in out["by_source"].items():
        print(f"[{k}] {v}")
    for k, v in out["by_source_machine"].items():
        print(f"  {k}: n={v.get('n')} median {v.get('dist_median')}px  >=2px {v.get('ge_2px')}  dx {v.get('dx_mean')} dy {v.get('dy_mean')}  side {v.get('side_median')}")
    for exp, cases in out["test_failures"].items():
        print(f"== {exp} test FN cases")
        for c in cases:
            print("  ", c)

    if args.show:
        exp = args.exps[0]
        d = json.loads((RUNS / exp / "preds_test.json").read_text(encoding="utf-8"))
        thr = json.loads((RUNS / exp / "eval_report_val.json").read_text(encoding="utf-8"))["threshold"]
        out_dir = RUNS / exp / "result" / "label_offset"
        for n, c in enumerate(out["test_failures"][exp], 1):
            if "label_to_dot_px" not in c:
                continue
            i, b = c["image_id"], gt_te[c["image_id"]][c["box_idx"]]
            img = imread(DATA / manifest[i]["image_path"])
            r = dot_of[(i, b[0])]
            cx, cy, R, Z = int(r["dot"][0]), int(r["dot"][1]), 14, 16
            y1, x1 = max(0, cy - R), max(0, cx - R)
            crop = cv2.resize(img[y1:cy + R, x1:cx + R], None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
            def rect(bb, color):
                cv2.rectangle(crop, (int((bb[0] - x1) * Z), int((bb[1] - y1) * Z)), (int((bb[2] - x1) * Z), int((bb[3] - y1) * Z)), color, 2)
            rect(b[1:], (0, 200, 0))
            kept = [p for p in d["preds"].get(i, []) if p[0] >= thr]
            for p in kept:
                rect(p[1:], (0, 0, 255))
            cv2.drawMarker(crop, (int((r["dot"][0] - x1) * Z), int((r["dot"][1] - y1) * Z)), (255, 80, 0), cv2.MARKER_CROSS, 24, 2)
            bar = np.full((30, crop.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f"{i} #{b[0]} green=label red=pred blue+=dot", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            imwrite(out_dir / f"{n:02d}_{i}_{b[0]}.png", np.vstack([bar, crop]))
        print("saved:", out_dir)


if __name__ == "__main__":
    main()
