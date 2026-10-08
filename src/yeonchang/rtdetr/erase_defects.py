"""결함을 지운 이미지에서도 모델이 반응하는지 보는 지름길 검증.

정답 박스(+pad px)를 cv2.inpaint로 주변에서 보간해 지운 뒤 같은 모델·임계값으로 추론한다.
지운 자리에서도 임계값 이상 검출이 나오면 모델이 결함이 아닌 다른 단서(표시 제거 흔적 등)를 본다는 뜻이다.

    python src/yeonchang/rtdetr/erase_defects.py --exp 02_rtdetrl_img640_e30

    python src/yeonchang/rtdetr/erase_defects.py --exp 02_rtdetrl_img640_e30 --fill mean_noise

결과: runs/yeonchang/<exp>/result/erase_<fill>/ (비교 그림, Git 제외), eval_erase_test_<fill>.json (요약)
"""
import argparse
import json
import random

import cv2
import numpy as np

from common import DATA, RUNS, load_gt, read_manifest


def imread(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return img


def imwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, img)[1].tofile(str(path))


def _rect(box, pad, h, w):
    _, x1, y1, x2, y2 = box
    return (max(0, int(np.floor(y1 - pad))), min(h, int(np.ceil(y2 + pad))),
            max(0, int(np.floor(x1 - pad))), min(w, int(np.ceil(x2 + pad))))


def dot_mask(img, boxes, k=3.0, se=7, grow=1, search_pad=2, max_area=40, half=0.5):
    """박스 안에서 결함 '점'만 찾는다. black top-hat(closing - 원본, 구조 요소 se px)으로 se보다 작은 어두운 점만 남기고,
    박스 안 최대값(점의 가장 어두운 곳)의 half배 이상인 픽셀 중 최대값 픽셀과 이어진 영역을 점으로 본다(반치폭).
    최대값이 주변 고리의 표준편차 k배보다 작으면 점이 없다고 보고, max_area보다 크면 점이 아니라고 보고 지우지 않는다.
    반환: 마스크(255=점), 박스별 점 픽셀 수(0=못 찾음, -n=너무 큼)."""
    g = img[..., 0]
    h, w = g.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
    resid = (cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel).astype(np.float32) - g.astype(np.float32))  # 작은 어두운 점일수록 양수
    mask = np.zeros((h, w), np.uint8)
    sizes = []
    for b in boxes:
        y1, y2, x1, x2 = _rect(b, search_pad, h, w)
        Y1, Y2, X1, X2 = _rect(b, search_pad + 8, h, w)
        ring = np.ones((Y2 - Y1, X2 - X1), bool)
        ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
        sd = float(resid[Y1:Y2, X1:X2][ring].std()) + 1e-6
        local = resid[y1:y2, x1:x2]
        py, px = np.unravel_index(int(np.argmax(local)), local.shape)
        peak = float(local[py, px])
        if peak < k * sd:
            sizes.append(0)
            continue
        cand = (local >= max(half * peak, k * sd)).astype(np.uint8)
        _, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        j = lab[py, px]
        area = int(stats[j, cv2.CC_STAT_AREA])
        if area > max_area:
            sizes.append(-area)
            continue
        sizes.append(area)
        mask[y1:y2, x1:x2] |= (lab == j).astype(np.uint8) * 255
    if grow:
        mask = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return mask, sizes


def erase(img, boxes, pad, radius, fill="inpaint", ring=3, rng=None):
    """fill: inpaint(TELEA 보간) / mean(주변 고리 평균) / mean_noise(평균 + 고리와 같은 표준편차의 노이즈)
    / dot(박스가 아니라 점 픽셀만 찾아 inpaint)."""
    h, w = img.shape[:2]
    if fill == "dot":
        mask, _ = dot_mask(img, boxes)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA), mask
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        y1, y2, x1, x2 = _rect(b, pad, h, w)
        mask[y1:y2, x1:x2] = 255
    if fill == "inpaint":
        return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA), mask
    rng = rng or np.random.default_rng(0)
    if fill in ("inpaint_texture", "mean_texture"):
        # 질감 세기: 원본 - 흐린 원본(고주파)의 표준편차. 경계의 밝기 차이는 빠진다.
        g = img[..., 0].astype(np.float32)
        resid = g - cv2.GaussianBlur(g, (0, 0), 1.5)
        base = erase(img, boxes, pad, radius, "inpaint" if fill == "inpaint_texture" else "mean")[0][..., 0].astype(np.float32)
        out = base.copy()
        for b in boxes:
            y1, y2, x1, x2 = _rect(b, pad, h, w)
            Y1, Y2, X1, X2 = _rect(b, pad + 6, h, w)
            sd = float(resid[Y1:Y2, X1:X2][mask[Y1:Y2, X1:X2] == 0].std())
            out[y1:y2, x1:x2] += rng.normal(0, sd, (y2 - y1, x2 - x1))
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
        return np.repeat(out[..., None], 3, axis=2), mask
    out = img.copy()
    for b in boxes:
        y1, y2, x1, x2 = _rect(b, pad, h, w)
        Y1, Y2, X1, X2 = _rect(b, pad + ring, h, w)
        region = img[Y1:Y2, X1:X2, 0].astype(np.float32)
        ring_px = region[mask[Y1:Y2, X1:X2] == 0]  # 지울 영역(다른 결함 포함)을 뺀 고리
        mu, sd = float(ring_px.mean()), float(ring_px.std())
        patch = np.full((y2 - y1, x2 - x1), mu, np.float32)
        if fill == "mean_noise":
            patch += rng.normal(0, sd, patch.shape)
        out[y1:y2, x1:x2] = np.clip(np.round(patch), 0, 255).astype(np.uint8)[..., None]
    return out, mask


def predict(model, imgs, imgsz, device):
    out = []
    for k in range(0, len(imgs), 8):
        for r in model.predict(imgs[k:k + 8], imgsz=imgsz, conf=0.001, device=device, verbose=False):
            out.append([(float(c), *map(float, xy)) for c, xy in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist())])
    return out


def hits(preds, boxes, thr, r=5.0):
    """정답 중심 r px 이내에 신뢰도 thr 이상 예측이 있는 정답 수, 정답과 무관한 thr 이상 예측 수."""
    gc = np.array([((b[1] + b[3]) / 2, (b[2] + b[4]) / 2) for b in boxes]).reshape(-1, 2)
    kept = [p for p in preds if p[0] >= thr]
    pc = np.array([((p[1] + p[3]) / 2, (p[2] + p[4]) / 2) for p in kept]).reshape(-1, 2)
    if len(gc) == 0 or len(pc) == 0:
        return 0, len(kept)
    d = np.linalg.norm(pc[:, None] - gc[None], axis=2)
    return int((d.min(0) <= r).sum()), int((d.min(1) > r).sum())


def draw(img, boxes, preds, thr, scale, title, erased_boxes=()):
    """초록: 정답, 파랑: 지운 정답 자리, 빨강: 임계값 이상 예측, 노랑: 0.1~임계값 예측."""
    v = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for bs, color in ((boxes, (0, 200, 0)), (erased_boxes, (255, 80, 0))):
        for _, x1, y1, x2, y2 in bs:
            cv2.rectangle(v, (int(x1 * scale) - 3, int(y1 * scale) - 3), (int(x2 * scale) + 3, int(y2 * scale) + 3), color, 1)
    for c, x1, y1, x2, y2 in sorted(preds, key=lambda p: p[0]):
        if c < 0.1:
            continue
        color = (0, 0, 255) if c >= thr else (0, 220, 255)
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(v, p1, p2, color, 2 if c >= thr else 1)
        cv2.putText(v, f"{c:.2f}", (p1[0], p2[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    bar = np.full((28, v.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-show", type=int, default=10)
    ap.add_argument("--pad", type=float, default=3.0, help="정답 박스 바깥으로 더 지울 px")
    ap.add_argument("--radius", type=int, default=5, help="cv2.inpaint 반경")
    ap.add_argument("--fill", choices=["inpaint", "mean", "mean_noise", "inpaint_texture", "mean_texture", "dot"], default="inpaint")
    ap.add_argument("--device", default="0")
    ap.add_argument("--one", action="store_true",
                    help="이미지마다 점 1개만(무작위, seed 0) 지우고 나머지는 남긴다. 점 인식 vs 위치 암기 구분용")
    args = ap.parse_args()
    if args.one:
        return main_one(args)

    from ultralytics import RTDETR
    exp_dir = RUNS / args.exp
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    thr = json.loads((exp_dir / "eval_report_val.json").read_text(encoding="utf-8"))["threshold"]
    model = RTDETR(str(exp_dir / "weights" / "best.pt"))
    manifest = read_manifest()
    gt = load_gt("test", manifest)
    ids = sorted(i for i in gt if gt[i])  # 결함이 있는 test 이미지

    # 보여줄 10장: 호기별로 고르게 무작위 (seed 0)
    rng = random.Random(0)
    by_m = {}
    for i in ids:
        by_m.setdefault(manifest[i]["machine"], []).append(i)
    show = []
    while len(show) < args.n_show:
        for m in sorted(by_m):
            if len(show) < args.n_show and by_m[m]:
                show.append(by_m[m].pop(rng.randrange(len(by_m[m]))))

    orig = [imread(DATA / manifest[i]["image_path"]) for i in ids]
    noise_rng = np.random.default_rng(0)
    erased = [erase(img, gt[i], args.pad, args.radius, args.fill, rng=noise_rng)[0] for img, i in zip(orig, ids)]
    p_orig = predict(model, orig, cfg["hyp"]["imgsz"], args.device)
    p_erased = predict(model, erased, cfg["hyp"]["imgsz"], args.device)

    fill_desc = {"inpaint": f"cv2.inpaint TELEA r={args.radius}", "mean": "주변 3px 고리 평균",
                 "mean_noise": "주변 3px 고리 평균 + 고리 표준편차의 가우시안 노이즈 (seed 0)",
                 "inpaint_texture": "inpaint + 주변 고주파 잔차 표준편차의 가우시안 노이즈 (seed 0)",
                 "mean_texture": "주변 평균 + 주변 고주파 잔차 표준편차의 가우시안 노이즈 (seed 0)",
                 "dot": "점만: black top-hat(7px)의 박스 내 최대값 절반 이상 연결 영역(40px 이하) + 1px, inpaint TELEA r=3"}[args.fill]
    summary = {"exp": args.exp, "threshold": thr, "pad_px": args.pad, "fill": args.fill, "fill_desc": fill_desc,
               "n_images": len(ids), "n_gt": sum(len(gt[i]) for i in ids), "shown": show, "per_image": {}}
    tot = {"orig_hit": 0, "erased_hit": 0, "orig_other": 0, "erased_other": 0}
    max_conf_at_gt = []
    for i, po, pe in zip(ids, p_orig, p_erased):
        ho, oo = hits(po, gt[i], thr)
        he, oe = hits(pe, gt[i], thr)
        tot["orig_hit"] += ho; tot["erased_hit"] += he; tot["orig_other"] += oo; tot["erased_other"] += oe
        for b in gt[i]:  # 지운 자리(정답 중심 5px 이내)에서 나온 최고 신뢰도
            c = np.array([(b[1] + b[3]) / 2, (b[2] + b[4]) / 2])
            near = [p[0] for p in pe if np.hypot((p[1] + p[3]) / 2 - c[0], (p[2] + p[4]) / 2 - c[1]) <= 5]
            max_conf_at_gt.append(max(near, default=0.0))
        summary["per_image"][i] = {"n_gt": len(gt[i]), "orig_hit": ho, "erased_hit": he, "erased_other": oe}
    mc = np.array(max_conf_at_gt)
    summary["total"] = tot
    summary["erased_max_conf_at_gt"] = {"mean": round(float(mc.mean()), 4), "median": round(float(np.median(mc)), 4),
                                        "ge_0.1": int((mc >= 0.1).sum()), "ge_0.5": int((mc >= 0.5).sum()),
                                        "ge_thr": int((mc >= thr).sum())}
    (exp_dir / f"eval_erase_test_{args.fill}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_dir = exp_dir / "result" / f"erase_{args.fill}"
    idx = {i: k for k, i in enumerate(ids)}
    for n, i in enumerate(show, 1):
        k = idx[i]
        scale = 2 if orig[k].shape[1] < 500 else 1.5
        panels = [draw(orig[k], gt[i], p_orig[k], thr, scale, "original + pred"),
                  draw(erased[k], [], [], thr, scale, "erased (input)"),
                  draw(erased[k], gt[i], p_erased[k], thr, scale, "erased + pred")]
        imwrite(out_dir / f"{n:02d}_{i}.png", np.hstack([np.hstack([p, np.full((p.shape[0], 6, 3), 255, np.uint8)]) for p in panels]))

    print(f"threshold {thr:.4f}, test images with defects {len(ids)}, GT {summary['n_gt']}")
    print(f"GT hit (pred >= thr within 5px): original {tot['orig_hit']} / erased {tot['erased_hit']}")
    print(f"preds >= thr not near any GT: original {tot['orig_other']} / erased {tot['erased_other']}")
    print("erased: max conf near GT", summary["erased_max_conf_at_gt"])
    for n, i in enumerate(show, 1):
        s = summary["per_image"][i]
        print(f"  {n:02d} {i}  machine {manifest[i]['machine']}  GT {s['n_gt']}  hit orig {s['orig_hit']} -> erased {s['erased_hit']}  other {s['erased_other']}")
    print("saved:", out_dir)


def near_conf(preds, box, r=5.0):
    c = ((box[1] + box[3]) / 2, (box[2] + box[4]) / 2)
    return max((p[0] for p in preds if np.hypot((p[1] + p[3]) / 2 - c[0], (p[2] + p[4]) / 2 - c[1]) <= r), default=0.0)


def main_one(args):
    """점 1개만 지우기. 지운 자리에서 검출되면 위치 암기, 안 되고 남은 점은 검출되면 점 인식."""
    from ultralytics import RTDETR
    exp_dir = RUNS / args.exp
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    thr = json.loads((exp_dir / "eval_report_val.json").read_text(encoding="utf-8"))["threshold"]
    model = RTDETR(str(exp_dir / "weights" / "best.pt"))
    manifest = read_manifest()
    gt = load_gt("test", manifest)
    ids = sorted(i for i in gt if gt[i])

    pick = random.Random(0)
    chosen = {i: pick.randrange(len(gt[i])) for i in ids}  # 지울 정답의 순번
    orig = [imread(DATA / manifest[i]["image_path"]) for i in ids]
    noise_rng = np.random.default_rng(0)
    erased = [erase(img, [gt[i][chosen[i]]], args.pad, args.radius, args.fill, rng=noise_rng)[0] for img, i in zip(orig, ids)]
    p_orig = predict(model, orig, cfg["hyp"]["imgsz"], args.device)
    p_erased = predict(model, erased, cfg["hyp"]["imgsz"], args.device)

    rows = []
    for i, po, pe in zip(ids, p_orig, p_erased):
        e = gt[i][chosen[i]]
        kept = [b for k, b in enumerate(gt[i]) if k != chosen[i]]
        _, other = hits(pe, gt[i], thr)  # 어떤 정답(지운 자리 포함)과도 5px 밖인 임계값 이상 예측
        rows.append({"image_id": i, "machine": manifest[i]["machine"], "source": manifest[i]["source"], "n_gt": len(gt[i]),
                     "erased_box_idx": e[0],
                     "erased_conf_orig": round(near_conf(po, e), 4), "erased_conf_after": round(near_conf(pe, e), 4),
                     "kept_n": len(kept), "kept_hit_orig": hits(po, kept, thr)[0] if kept else 0,
                     "kept_hit_after": hits(pe, kept, thr)[0] if kept else 0,
                     "kept_conf_after": [round(near_conf(pe, b), 4) for b in kept], "other_after": other})

    def agg(rs):
        ca = np.array([r["erased_conf_after"] for r in rs])
        return {"n_images": len(rs),
                "erased_detected_orig": sum(r["erased_conf_orig"] >= thr for r in rs),
                "erased_detected_after": int((ca >= thr).sum()),
                "erased_conf_after_median": round(float(np.median(ca)), 4), "erased_conf_after_max": round(float(ca.max()), 4),
                "erased_conf_after_ge_0.1": int((ca >= 0.1).sum()), "erased_conf_after_ge_0.5": int((ca >= 0.5).sum()),
                "kept_total": sum(r["kept_n"] for r in rs), "kept_hit_orig": sum(r["kept_hit_orig"] for r in rs),
                "kept_hit_after": sum(r["kept_hit_after"] for r in rs), "other_after": sum(r["other_after"] for r in rs)}

    summary = {"exp": args.exp, "threshold": thr, "fill": args.fill, "pad_px": args.pad,
               "design": "test 결함 이미지마다 정답 1개(무작위, seed 0)만 지우고 나머지는 둠. 판정은 정답 중심 5px 이내 예측 신뢰도",
               "all": agg(rows), "by_n_gt": {str(n): agg([r for r in rows if r["n_gt"] == n]) for n in sorted({r["n_gt"] for r in rows})},
               "by_machine": {m: agg([r for r in rows if r["machine"] == m]) for m in sorted({r["machine"] for r in rows})},
               "per_image": rows}
    (exp_dir / f"eval_erase_one_test_{args.fill}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int), encoding="utf-8")

    # 보여줄 10장: 남은 점과 비교할 수 있게 결함이 여러 개인 이미지에서 호기별로 고르게 (seed 0)
    rng = random.Random(0)
    by_m = {}
    for i in ids:
        if len(gt[i]) > 1:
            by_m.setdefault(manifest[i]["machine"], []).append(i)
    show = []
    while len(show) < min(args.n_show, sum(len(v) for v in by_m.values())):
        for m in sorted(by_m):
            if len(show) < args.n_show and by_m[m]:
                show.append(by_m[m].pop(rng.randrange(len(by_m[m]))))
    out_dir = exp_dir / "result" / f"erase_one_{args.fill}"
    idx = {i: k for k, i in enumerate(ids)}
    for n, i in enumerate(show, 1):
        k = idx[i]
        e = gt[i][chosen[i]]
        kept = [b for j, b in enumerate(gt[i]) if j != chosen[i]]
        scale = 2 if orig[k].shape[1] < 500 else 1.5
        panels = [draw(orig[k], gt[i], p_orig[k], thr, scale, "original + pred"),
                  draw(erased[k], [], [], thr, scale, "1 dot erased (input)"),
                  draw(erased[k], kept, p_erased[k], thr, scale, "erased + pred (blue = erased spot)", erased_boxes=[e])]
        imwrite(out_dir / f"{n:02d}_{i}.png", np.hstack([np.hstack([p, np.full((p.shape[0], 6, 3), 255, np.uint8)]) for p in panels]))

    a = summary["all"]
    print(f"threshold {thr:.4f}, fill {args.fill}, images {a['n_images']}")
    for name, s in [("ALL", a)] + [(f"n_gt={k}", v) for k, v in summary["by_n_gt"].items()] + [(f"machine {k}", v) for k, v in summary["by_machine"].items()]:
        print(f"[{name}] images {s['n_images']} | erased spot detected: before {s['erased_detected_orig']} -> after {s['erased_detected_after']} "
              f"(conf median {s['erased_conf_after_median']}, max {s['erased_conf_after_max']}, >=0.5: {s['erased_conf_after_ge_0.5']}) | "
              f"kept dots detected: {s['kept_hit_orig']} -> {s['kept_hit_after']} / {s['kept_total']} | other FP after {s['other_after']}")
    for n, i in enumerate(show, 1):
        r = rows[idx[i]]
        print(f"  {n:02d} {i} m{r['machine']} n_gt {r['n_gt']}: erased #{r['erased_box_idx']} conf {r['erased_conf_orig']} -> {r['erased_conf_after']}, kept conf {r['kept_conf_after']}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
