"""합성 정상 사진 검사: 학습이 끝난 모델이 "이물질을 보고" 찾는지, "제품의 그 자리"를 외웠는지 가른다.

사용: python members/lee/synth_normal_check.py runs/<exp_id> [--seed 42] [--thr 0.5]

하는 일 (학습 없음, 검증(val) 사진만 사용, 확정 데이터는 읽기만 한다):
  1. 합성 정상(synth_normal): val 사진의 라벨 상자 자리를 주변 회색으로 메워 "이물질 없는 같은 제품" 사진을 만든다.
     여기서 무엇이든 찾으면 오경보다. 찾은 상자가 지운 자리에 있으면 "자리(또는 메운 흔적)를 보고 찾은 것"이다.
  2. 대조(control): 같은 val 사진에서 라벨과 멀리 떨어진 제품 안의 임의 자리를 같은 방식으로 메운다 (라벨은 그대로).
     모델이 이 메운 자리를 찾으면 "메운 흔적"이 단서라는 뜻이다. 안 찾으면 1 의 오경보는 자리 외우기 쪽이다.
  3. 원본(original): 손대지 않은 val 사진. 기준점.
결과: runs/<exp_id>/synth_normal.json, synth_normal.md. 사진은 runs/<exp_id>/work/synth/ (git 제외).

메우는 방식은 "바로 옆의 같은 크기 조각을 복사해 덮기"다 (fill_mask 참고). 2026-10-06 두 번 고침.
라벨 txt 만 읽는다. 장비 색상 박스 좌표는 쓰지 않는다 (CLAUDE.md 규칙 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import kx  # noqa: E402

MARGIN = 2      # 라벨 상자보다 이만큼 넓게 메운다 (px)
MIN_DIST = 30   # 대조용 임의 자리와 라벨 중심의 최소 거리 (px)


def fill_mask(im: np.ndarray, mask: np.ndarray, avoid: np.ndarray | None = None) -> np.ndarray:
    """mask 가 True 인 자리를 '바로 옆의 같은 크기 조각'으로 덮는다 (질감까지 그대로 옮겨 메운 티가 안 나게).
    mask 의 덩어리마다 상하좌우 네 방향으로 한 칸(덩어리 크기+2px) 옮긴 자리를 후보로 두고,
    테두리 3px 띠가 원래 테두리와 가장 비슷한 후보를 고른다. 후보는 mask 와 avoid(라벨 자리) 를 밟으면 안 된다.
    (이전 두 버전: 주변 평균 채우기는 점이 번지거나 매끈하게 뭉개져 티가 났다.)"""
    import cv2
    out = im.copy()
    h, w = im.shape
    forbid = mask if avoid is None else (mask | avoid)
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for c in range(1, n):
        ys, xs = np.where(comp == c)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        ring = np.zeros((h, w), bool)
        ring[max(0, y0 - 3):min(h, y1 + 3), max(0, x0 - 3):min(w, x1 + 3)] = True
        ring[y0:y1, x0:x1] = False
        ring &= ~forbid
        best = None
        for dy, dx in ((0, bw + 2), (0, -(bw + 2)), (bh + 2, 0), (-(bh + 2), 0)):
            sy0, sx0 = y0 + dy, x0 + dx
            if sy0 - 3 < 0 or sx0 - 3 < 0 or sy0 + bh + 3 > h or sx0 + bw + 3 > w:
                continue
            if forbid[sy0:sy0 + bh, sx0:sx0 + bw].any():
                continue
            ry, rx = np.where(ring)
            sry, srx = ry + dy, rx + dx
            ok = (sry >= 0) & (sry < h) & (srx >= 0) & (srx < w)
            if ok.sum() == 0:
                continue
            score = np.abs(im[ry[ok], rx[ok]].astype(np.float32) - im[sry[ok], srx[ok]].astype(np.float32)).mean()
            if best is None or score < best[0]:
                best = (score, dy, dx)
        if best is None:
            continue  # 네 방향 모두 막히면 그대로 둔다 (거의 없음)
        _, dy, dx = best
        sel = comp[y0:y1, x0:x1] == c
        out[y0:y1, x0:x1][sel] = im[y0 + dy:y1 + dy, x0 + dx:x1 + dx][sel]
    return out


def box_px(b, w, h):
    cx, cy, bw, bh = b
    return cx * w, cy * h, max(bw * w, 4.0), max(bh * h, 4.0)


def mask_box(h, w, cx, cy, bw, bh, margin=MARGIN) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x0, x1 = int(round(cx - bw / 2 - margin)), int(round(cx + bw / 2 + margin)) + 1
    y0, y1 = int(round(cy - bh / 2 - margin)), int(round(cy + bh / 2 + margin)) + 1
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def random_spot(im: np.ndarray, boxes_px, rng, size) -> tuple[float, float] | None:
    """제품(어두운 영역) 안, 모든 라벨에서 MIN_DIST 이상 떨어진 임의 자리."""
    h, w = im.shape
    thr = np.percentile(im, 40)
    for _ in range(200):
        x, y = rng.uniform(size, w - size), rng.uniform(size, h - size)
        yi, xi = int(y), int(x)
        if im[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4].mean() >= thr:
            continue
        if all(np.hypot(x - bx, y - by) >= MIN_DIST for bx, by, _, _ in boxes_px):
            return x, y
    return None


def build(ids, imgs, labs, out: Path, seed: int) -> dict:
    """세 묶음의 사진을 만든다. 돌려주는 값: {set: {image_id: {"path", "removed": [(x,y,s)], "spot": (x,y,s)|None, "gt": [...]}}}"""
    rng = np.random.default_rng(seed)
    sets = {"original": {}, "synth_normal": {}, "control": {}}
    for d in sets:
        (out / d).mkdir(parents=True, exist_ok=True)
    for iid in ids:
        gt = kx.read_yolo_labels(labs[iid])
        if not gt:
            continue  # 원래 빈 사진은 이미 진짜 "이물질 없음" 사진이라 합성 대상이 아니다
        im = np.array(Image.open(imgs[iid]).convert("L"))
        h, w = im.shape
        bpx = [box_px(b, w, h) for b in gt]
        sets["original"][iid] = {"path": str(imgs[iid]), "gt": bpx}

        m = np.zeros((h, w), bool)
        for cx, cy, bw, bh in bpx:
            m |= mask_box(h, w, cx, cy, bw, bh)
        p = out / "synth_normal" / f"{iid}.png"
        Image.fromarray(fill_mask(im, m)).save(p)
        sets["synth_normal"][iid] = {"path": str(p), "removed": bpx, "gt": bpx}

        size = float(np.mean([max(bw, bh) for _, _, bw, bh in bpx]))
        spot = random_spot(im, bpx, rng, size)
        if spot is not None:
            p = out / "control" / f"{iid}.png"
            Image.fromarray(fill_mask(im, mask_box(h, w, spot[0], spot[1], size, size), avoid=m)).save(p)
            sets["control"][iid] = {"path": str(p), "spot": (spot[0], spot[1], size), "gt": bpx}
    return sets


def predict(model, items: dict, imgsz: int, device=None, chunk: int = 8) -> dict:
    """image_id -> [(x_px, y_px, w_px, h_px, conf)]"""
    ids = list(items)
    paths = [items[i]["path"] for i in ids]
    out = {}

    def chunked(chunk=chunk):  # ultralytics 8.4 는 목록 전체를 한 묶음으로 올리므로 끊어서 넘긴다
        for s in range(0, len(paths), chunk):
            yield from model.predict(paths[s:s + chunk], imgsz=imgsz, conf=0.001, max_det=50, verbose=False, stream=True, device=device, half=True)

    for iid, res in zip(ids, chunked()):
        b = res.boxes
        rows = []
        if b is not None and len(b):
            for (x, y, w, h), c in zip(b.xywh.cpu().numpy(), b.conf.cpu().numpy()):
                if w > 0 and h > 0:
                    rows.append((float(x), float(y), float(w), float(h), float(c)))
        out[iid] = rows
    return out


def near(px, py, targets, factor=1.5) -> bool:
    return any(np.hypot(px - tx, py - ty) <= factor * max(ts, 6.0) for tx, ty, ts in targets)


def summarize(sets: dict, preds: dict, thr: float) -> dict:
    r = {"threshold": thr}
    # 원본: 중심 적중 재현율, 사진당 검출 수
    o = sets["original"]
    hit = tot = 0
    for iid, it in o.items():
        det = [d for d in preds["original"][iid] if d[4] >= thr]
        for gx, gy, gw, gh in it["gt"]:
            tot += 1
            hit += any(abs(dx - gx) <= gw / 2 + 2 and abs(dy - gy) <= gh / 2 + 2 for dx, dy, _, _, _ in det)
    r["original"] = {"n_images": len(o), "center_hit_recall": hit / max(tot, 1),
                     "images_with_det": sum(any(d[4] >= thr for d in preds["original"][i]) for i in o)}
    # 합성 정상: 오경보 사진 비율, 검출 상자 수, 그중 지운 자리에 있는 비율
    s = sets["synth_normal"]
    n_img_fp = n_box = n_box_at_removed = 0
    max_conf = []
    for iid, it in s.items():
        det = [d for d in preds["synth_normal"][iid] if d[4] >= thr]
        allc = [d[4] for d in preds["synth_normal"][iid]]
        max_conf.append(max(allc) if allc else 0.0)
        n_img_fp += bool(det)
        n_box += len(det)
        n_box_at_removed += sum(near(d[0], d[1], [(x, y, max(w, h)) for x, y, w, h in it["removed"]]) for d in det)
    r["synth_normal"] = {"n_images": len(s), "fp_image_rate": n_img_fp / max(len(s), 1), "n_fp_boxes": n_box,
                         "fp_boxes_at_removed_spot": n_box_at_removed,
                         "frac_fp_at_removed_spot": n_box_at_removed / max(n_box, 1),
                         "max_conf_p50": float(np.median(max_conf)) if max_conf else 0.0,
                         "max_conf_p90": float(np.percentile(max_conf, 90)) if max_conf else 0.0}
    # 대조: 메운 임의 자리를 찾은 사진 수, 라벨 재현율(메움이 다른 곳 성능을 해쳤는지)
    c = sets["control"]
    n_spot = hit = tot = 0
    for iid, it in c.items():
        det = [d for d in preds["control"][iid] if d[4] >= thr]
        n_spot += any(near(d[0], d[1], [it["spot"]]) for d in det)
        for gx, gy, gw, gh in it["gt"]:
            tot += 1
            hit += any(abs(dx - gx) <= gw / 2 + 2 and abs(dy - gy) <= gh / 2 + 2 for dx, dy, _, _, _ in det)
    r["control"] = {"n_images": len(c), "spot_detected_rate": n_spot / max(len(c), 1), "center_hit_recall": hit / max(tot, 1)}
    # 임계값별: 합성 정상 오경보 사진 비율 vs 원본 재현율
    sweep = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        fp = np.mean([any(d[4] >= t for d in preds["synth_normal"][i]) for i in s]) if s else 0.0
        h = n = 0
        for iid, it in o.items():
            det = [d for d in preds["original"][iid] if d[4] >= t]
            for gx, gy, gw, gh in it["gt"]:
                n += 1
                h += any(abs(dx - gx) <= gw / 2 + 2 and abs(dy - gy) <= gh / 2 + 2 for dx, dy, _, _, _ in det)
        sweep.append({"thr": float(t), "synth_fp_image_rate": float(fp), "orig_center_hit_recall": h / max(n, 1)})
    r["sweep"] = sweep
    return r


def write_md(rd: Path, r: dict, exp: str) -> None:
    o, s, c = r["original"], r["synth_normal"], r["control"]
    verdict = []
    if s["fp_image_rate"] < 0.05:
        verdict.append("합성 정상 사진에서 거의 아무것도 찾지 않는다. 모델은 이물질 자체를 보고 찾는다.")
    else:
        verdict.append(f"합성 정상 사진의 {s['fp_image_rate']:.0%} 에서 무언가를 찾는다. "
                       f"그 상자의 {s['frac_fp_at_removed_spot']:.0%} 가 지운 자리에 있다.")
        if c["spot_detected_rate"] < 0.05:
            verdict.append("대조 사진의 임의 메운 자리는 찾지 않으므로, 메운 흔적이 아니라 '제품의 그 자리'(또는 색상 박스를 지운 테두리 흔적)를 보고 찾는 것이다.")
        else:
            verdict.append(f"대조 사진의 임의 메운 자리도 {c['spot_detected_rate']:.0%} 찾는다. 메운 흔적 자체가 단서다. 메우는 방식을 바꿔 다시 봐야 한다.")
    lines = [f"# 합성 정상 사진 검사: {exp}", "",
             f"임계값 {r['threshold']} (metrics.json 의 값). 검증(val) 사진 중 라벨이 있는 {o['n_images']}장 사용.", "",
             "| 묶음 | 뜻 | 결과 |", "|---|---|---|",
             f"| original | 손대지 않은 val 사진 | 중심 적중 재현율 {o['center_hit_recall']:.3f}, 검출 있는 사진 {o['images_with_det']}/{o['n_images']} |",
             f"| synth_normal | 라벨 자리를 메운 사진 (이물질 없음) | 오경보 사진 {s['fp_image_rate']:.1%}, 오경보 상자 {s['n_fp_boxes']}개, 그중 지운 자리 {s['fp_boxes_at_removed_spot']}개 ({s['frac_fp_at_removed_spot']:.0%}), 최대 확신도 중앙값 {s['max_conf_p50']:.2f} / 90% {s['max_conf_p90']:.2f} |",
             f"| control | 라벨은 두고 임의 자리를 메운 사진 | 메운 자리 검출 {c['spot_detected_rate']:.1%}, 라벨 재현율 {c['center_hit_recall']:.3f} |",
             "", "## 판정", ""] + [f"- {v}" for v in verdict] + [
             "", "## 임계값별 (합성 정상 오경보 사진 비율 / 원본 재현율)", "", "| thr | synth FP 사진 | 원본 재현율 |", "|---|---|---|"]
    for row in r["sweep"]:
        lines.append(f"| {row['thr']:.2f} | {row['synth_fp_image_rate']:.3f} | {row['orig_center_hit_recall']:.3f} |")
    lines += ["", "주의: 합성 정상은 같은 시편에서 점만 뺀 사진이지 진짜 정상 제품이 아니다. 오경보율은 '합성 정상 기준'으로 적는다.",
              "지운 자리 주변에는 데이터 계보상 장비 색상 박스를 메운 테두리 흔적이 남아 있을 수 있어, 지운 자리의 오경보는 '자리 외우기'와 '테두리 흔적' 둘 다일 수 있다."]
    (rd / "synth_normal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--device", default=None, help="예: cpu")
    ap.add_argument("--chunk", type=int, default=8, help="한 번에 추론하는 사진 수 (GPU 메모리가 부족하면 줄인다)")
    a = ap.parse_args()
    rd = Path(a.run_dir)
    cfg = kx.read_yaml(rd / "config.yaml")
    seed = a.seed if a.seed is not None else int(cfg["seeds"][0])
    imgsz = int(cfg["model"].get("imgsz", 640))
    thr = a.thr
    if thr is None:
        thr = float(kx.read_json(rd / "metrics.json")["threshold"]) if (rd / "metrics.json").exists() else 0.5
    wdir = rd / "work" / f"seed{seed}" / "train" / "weights"
    weights = wdir / "best.pt" if (wdir / "best.pt").exists() else wdir / "last.pt"
    if not weights.exists():
        print(f"가중치가 없다: {wdir}")
        return 1

    split = kx.load_split()
    imgs, labs = kx.path_maps(split["val"])
    out = rd / "work" / "synth"
    sets = build(split["val"], imgs, labs, out, seed)
    print({k: len(v) for k, v in sets.items()})

    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = {name: predict(model, items, imgsz, a.device, a.chunk) for name, items in sets.items()}
    r = summarize(sets, preds, thr)
    r.update({"exp_id": cfg["exp_id"], "seed": seed, "weights": str(weights), "imgsz": imgsz, "evaluated_at": kx.now_iso()})
    kx.write_json(rd / "synth_normal.json", r)
    # 예측 원본도 남긴다 (사진별, 픽셀 좌표)
    kx.write_json(out / "preds.json", {k: {i: v for i, v in d.items()} for k, d in preds.items()})
    write_md(rd, r, cfg["exp_id"])
    print(json.dumps({k: r[k] for k in ("original", "synth_normal", "control")}, ensure_ascii=False, indent=1))
    print(f"저장: {rd / 'synth_normal.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
