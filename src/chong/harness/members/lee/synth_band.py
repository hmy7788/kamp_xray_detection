"""검은 띠 안 합성 이물질 검사 (미탐 조건 + 경량화 비교용 평가 자료).

왜: zip(10/6) 의 합성 762곳은 대부분 밝은 제품 몸통에 있어 데이터에 없는 조건이었다. 실제 이물질은 전부
    띠(어두운 막대) 안에 있다. 그래서 띠 안에만 점을 넣고, 점의 진하기를 단계별로 낮춰 "어디서부터 못 잡는가"를 본다.
    학습에는 쓰지 않는다. 평가만 한다.

build    : val 사진에서 띠를 찾아 띠 안 자리를 고르고(기존 라벨에서 떨어진 곳), 호기별 신호 2개 × 진하기 5단계를 샘플로 만든다.
           runs/synth_band/samples.jsonl, backgrounds.json, qa.png, placement_*.png
evaluate : 모델 하나로 (1) 원본 val 전체 채점(중심 적중·IoU0.5 F1, 임계값 0.42/0.10) (2) 샘플마다 한 곳씩 넣어 예측
           (3) 한 장 속도. runs/synth_band/<tag>/predictions.json, summary.json
report   : 여러 tag 를 한 표로. runs/synth_band/compare.md

사용 (저장소 루트, KAMP_MEMBER=lee PYTHONUTF8=1):
  python members/lee/synth_band.py build
  python members/lee/synth_band.py evaluate --tag lee001_gpu1024 --weights runs/lee_001/work/seed42/train/weights/best.pt --imgsz 1024
  python members/lee/synth_band.py evaluate --tag lee001_gpu640  --weights ... --imgsz 640
  python members/lee/synth_band.py evaluate --tag lee001_onnx640_t4 --weights <onnx> --imgsz 640 --device cpu --threads 4
  python members/lee/synth_band.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import kx  # noqa: E402

OUT = kx.RUNS / "synth_band"
# 점 신호 템플릿(.npz, 다른 PC 의 05_probe_map_val 에서 val 공식 라벨로 추출). 기본은 이 결과 폴더의 사본, KAMP_TEMPLATE_SRC 로 바꿀 수 있다
TEMPLATE_SRC = Path(os.environ.get("KAMP_TEMPLATE_SRC", str(kx.RUNS / "synth_band" / "templates")))
SCALES = [1.0, 0.7, 0.5, 0.35, 0.25]
THRS = [0.42, 0.10]
PER_MACHINE = 12
STEP = 10          # 띠 안 자리 간격 (px)
LABEL_MARGIN = 12  # 기존 라벨 중심과 최소 거리
HALF_W, HALF_H = 6, 5  # 적중 판정 상자 (zip 과 같음: 신호 지지 영역 +3px 정도)


def imread(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def product_mask(im: np.ndarray):
    rim = np.concatenate([im[:8].ravel(), im[-8:].ravel(), im[:, :8].ravel(), im[:, -8:].ravel()])
    level = float(np.median(rim))
    binary = (im < level - 12).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < im.size * 0.08:
        return None
    return labels == idx


def find_bands(im: np.ndarray, prod: np.ndarray):
    """제품 안에서 몸통보다 뚜렷이 어두운 길쭉한 덩어리 = 띠. (x, y, w, h, mask, median) 목록."""
    body = float(np.median(im[prod]))
    dark = ((im < body - 14) & prod).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    bands = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < 250:
            continue
        long_side, short_side = max(w, h), min(w, h)
        if long_side < 40 or long_side / max(short_side, 1) < 2.5 or short_side < 9:
            continue
        # 제품 가장자리 그림자(얇고 긴 테두리) 제외: 채움 비율
        if area / (w * h) < 0.55:
            continue
        m = labels == i
        bands.append(dict(x=x, y=y, w=w, h=h, area=area, mask=m, median=float(np.median(im[m]))))
    bands.sort(key=lambda b: b["y"])
    return bands


def band_positions(band: dict, labels_xy: list[tuple[int, int]], im: np.ndarray):
    """띠 마스크를 안쪽으로 깎은 뒤 가로 방향으로 STEP 간격 자리. 기존 라벨과 가까운 자리는 뺀다.

    한 열에서 세로로 가장 어두운 곳을 자리로 잡는다. 막대 두세 개가 한 덩어리로 붙은 경우(2호기 위쪽 막대)
    어두운 골이 여러 개면 9px 이상 떨어진 골마다 하나씩 잡아 밝은 틈에 자리가 생기지 않게 한다."""
    m = band["mask"].astype(np.uint8)
    inner = cv2.erode(m, np.ones((2 * HALF_H + 3, 2 * HALF_W + 3), np.uint8)).astype(bool)
    ys, xs = np.where(inner)
    if len(xs) == 0:
        return []
    prof = cv2.blur(im, (3, 5)).astype(float)
    out = []
    x0, x1 = xs.min(), xs.max()
    for x in range(int(x0) + 2, int(x1) - 1, STEP):
        col = np.sort(ys[xs == x])
        if len(col) < 3:
            continue
        vals = prof[col, x]
        order = col[np.argsort(vals)]
        best = float(vals.min())
        picked = []
        for y in order:
            if prof[y, x] > best + 6:
                break
            if all(abs(int(y) - p) >= 9 for p in picked):
                picked.append(int(y))
        for y in picked:
            if any(abs(x - lx) < LABEL_MARGIN and abs(y - ly) < LABEL_MARGIN for lx, ly in labels_xy):
                continue
            rel = (x - x0) / max(1, (x1 - x0))
            out.append((int(x), int(y), round(float(rel), 3)))
    return out


def inject(im: np.ndarray, signal: np.ndarray, x: int, y: int, scale: float) -> np.ndarray:
    h, w = signal.shape
    oy, ox = h // 2, w // 2
    y0, x0 = y - oy, x - ox
    if y0 < 0 or x0 < 0 or y0 + h > im.shape[0] or x0 + w > im.shape[1]:
        raise ValueError("signal out of bounds")
    out = im.astype(np.float32)
    out[y0:y0 + h, x0:x0 + w] += signal * scale
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def ring_mean(im: np.ndarray, x: int, y: int, side: int = 10, pad: int = 3) -> float:
    h, w = im.shape
    r = side // 2
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    X0, X1, Y0, Y1 = max(0, x0 - pad), min(w, x1 + pad), max(0, y0 - pad), min(h, y1 + pad)
    outer = im[Y0:Y1, X0:X1].astype(float)
    m = np.ones_like(outer, bool)
    m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    return float(outer[m].mean())


# ----------------------------------------------------------------------------- build
def build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "templates").mkdir(exist_ok=True)
    templates = {}
    for p in sorted(TEMPLATE_SRC.glob("*.npz")):
        d = np.load(p)
        np.savez(OUT / "templates" / p.name, signal=d["signal"], sham=d["sham"])
        templates[p.stem] = d["signal"].astype(np.float32)
    split = kx.load_split()
    man = kx.load_manifest().set_index("image_id")
    imgs, labs = kx.path_maps(split["val"])
    # 배경 후보: 라벨 있는 val 사진, 묶음별 1장, 해시 순서로 호기별 PER_MACHINE 장
    cands = {"1": [], "2": [], "3": []}
    seen_burst = set()
    for iid in sorted(split["val"], key=lambda s: hashlib.sha1(s.encode()).hexdigest()):
        r = man.loc[iid]
        if r.burst_id in seen_burst:
            continue
        boxes = kx.read_yolo_labels(labs[iid])
        if not boxes:
            continue
        cands[str(r.machine)].append(iid)
        seen_burst.add(r.burst_id)
    samples, backgrounds, qa_rows = [], [], []
    for mc in ["1", "2", "3"]:
        tids = [t for t in templates if t.startswith(f"m{mc}_")]
        n_ok = 0
        for iid in cands[mc]:
            if n_ok >= PER_MACHINE:
                break
            im = imread(imgs[iid])
            prod = product_mask(im)
            if prod is None:
                continue
            bands = find_bands(im, prod)
            if not bands:
                continue
            h, w = im.shape
            lxy = [(int(round(cx * w)), int(round(cy * h))) for cx, cy, _, _ in kx.read_yolo_labels(labs[iid])]
            pos_all = []
            for bi, b in enumerate(bands):
                for x, y, rel in band_positions(b, lxy, im):
                    if ring_mean(im, x, y) > b["median"] + 8:  # 띠 가운데가 아니라 밝은 틈에 걸린 자리 제외
                        continue
                    pos_all.append((bi, x, y, rel, b["median"]))
            if len(pos_all) < 3:
                continue
            n_ok += 1
            backgrounds.append(dict(image_id=iid, machine=mc, n_bands=len(bands), n_positions=len(pos_all),
                                    band_boxes=[[b["x"], b["y"], b["w"], b["h"]] for b in bands], labels_xy=lxy))
            for bi, x, y, rel, bmed in pos_all:
                for tid in tids:
                    for sc in SCALES:
                        samples.append(dict(id=f"{iid}|b{bi}|{x},{y}|{tid}|{sc}", background_id=iid, machine=mc, band=bi,
                                            x=x, y=y, rel_pos=rel, band_median=round(bmed, 1), ring_mean=round(ring_mean(im, x, y), 1),
                                            template=tid, scale=sc,
                                            target_box=[x - HALF_W, y - HALF_H, x + HALF_W, y + HALF_H]))
            if len(qa_rows) < 12 and n_ok <= 4:
                qa_rows.append((iid, mc, pos_all[len(pos_all) // 2], tids[0]))
    (OUT / "samples.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8")
    kx.write_json(OUT / "backgrounds.json", backgrounds)
    n_pos = sum(b["n_positions"] for b in backgrounds)
    info = dict(n_backgrounds=len(backgrounds), n_positions=n_pos, n_samples=len(samples), scales=SCALES,
                templates={t: {"shape": list(templates[t].shape), "min": float(templates[t].min())} for t in templates},
                per_machine={mc: sum(1 for b in backgrounds if b["machine"] == mc) for mc in "123"},
                samples_sha256=kx.sha256_file(OUT / "samples.jsonl"), manifest_sha256=kx.manifest_sha(), built_at=kx.now_iso())
    kx.write_json(OUT / "build_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=1))

    # QA 그림: 배경 3장 배치도 + 조각 (넣기 전 / 1.0 / 0.5 / 0.25)
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 12)
    UP, R = 5, 18
    panels = []
    for iid, mc, (bi, x, y, rel, bmed), tid in qa_rows:
        im = imread(imgs[iid])
        row = []
        for sc, title in [(None, "넣기 전"), (1.0, "진하기 1.0"), (0.5, "진하기 0.5"), (0.25, "진하기 0.25")]:
            a2 = im if sc is None else inject(im, templates[tid], x, y, sc)
            c = a2[max(0, y - R):y + R, max(0, x - R):x + R]
            pil = Image.fromarray(c).resize((c.shape[1] * UP, c.shape[0] * UP), Image.NEAREST).convert("RGB")
            d = ImageDraw.Draw(pil)
            cx, cy = pil.width // 2, pil.height // 2
            d.line([cx - 8, cy, cx - 3, cy], fill=(80, 170, 255)); d.line([cx + 3, cy, cx + 8, cy], fill=(80, 170, 255))
            d.line([cx, cy - 8, cx, cy - 3], fill=(80, 170, 255)); d.line([cx, cy + 3, cx, cy + 8], fill=(80, 170, 255))
            lab = Image.new("RGB", (pil.width, pil.height + 18), (28, 32, 40)); lab.paste(pil, (0, 18))
            ImageDraw.Draw(lab).text((2, 1), f"{mc}호기 {title}", fill=(230, 235, 245), font=font)
            row.append(lab)
        W = sum(p.width for p in row) + 6 * (len(row) + 1); H = row[0].height + 22
        strip = Image.new("RGB", (W, H), (20, 23, 30))
        xx = 6
        for p in row:
            strip.paste(p, (xx, 0)); xx += p.width + 6
        ImageDraw.Draw(strip).text((6, H - 20), f"{iid}  띠 {bi}  자리 ({x},{y})  띠 밝기 {bmed:.0f}  신호 {tid}", fill=(160, 170, 190), font=font)
        panels.append(strip)
    W = max(p.width for p in panels); H = sum(p.height for p in panels)
    qa = Image.new("RGB", (W, H), (20, 23, 30)); yy = 0
    for p in panels:
        qa.paste(p, (0, yy)); yy += p.height
    qa.save(OUT / "qa.png")
    # 배치도: 호기별 첫 배경
    for mc in "123":
        b = next(b for b in backgrounds if b["machine"] == mc)
        im = imread(imgs[b["image_id"]])
        pil = Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST).convert("RGB")
        d = ImageDraw.Draw(pil)
        for bx, by, bw, bh in b["band_boxes"]:
            d.rectangle([bx * 2, by * 2, (bx + bw) * 2, (by + bh) * 2], outline=(80, 170, 255), width=1)
        for s in samples:
            if s["background_id"] == b["image_id"] and s["scale"] == 1.0 and s["template"].endswith("t1"):
                d.ellipse([s["x"] * 2 - 3, s["y"] * 2 - 3, s["x"] * 2 + 3, s["y"] * 2 + 3], outline=(255, 160, 40), width=2)
        for lx, ly in b["labels_xy"]:
            d.rectangle([lx * 2 - 10, ly * 2 - 10, lx * 2 + 10, ly * 2 + 10], outline=(60, 220, 90), width=2)
        pil.save(OUT / f"placement_m{mc}.png")
    print("QA:", OUT / "qa.png")


# ----------------------------------------------------------------------------- evaluate
def _to_bgr(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def _predict(model, im_bgr, imgsz, device):
    r = model.predict(im_bgr, imgsz=imgsz, conf=0.001, max_det=50, device=device, verbose=False)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], axis=1)


def _hit_at(dets: np.ndarray, box) -> float:
    if len(dets) == 0:
        return 0.0
    cx = (dets[:, 0] + dets[:, 2]) / 2; cy = (dets[:, 1] + dets[:, 3]) / 2
    m = (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
    return float(dets[m, 4].max()) if m.any() else 0.0


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _score_val(all_dets: dict, gts: dict, thr: float) -> dict:
    tp = fp = fn = 0; center_hit = 0; n_gt = 0
    for iid, gt in gts.items():
        d = all_dets[iid]; d = d[d[:, 4] >= thr]
        d = d[np.argsort(-d[:, 4])]
        used = set()
        for g in gt:
            n_gt += 1
            if any(g[0] <= (x[0] + x[2]) / 2 <= g[2] and g[1] <= (x[1] + x[3]) / 2 <= g[3] for x in d):
                center_hit += 1
        for x in d:
            best, bi = 0.0, -1
            for j, g in enumerate(gt):
                if j in used:
                    continue
                v = _iou(x, g)
                if v > best:
                    best, bi = v, j
            if best >= 0.5:
                tp += 1; used.add(bi)
            else:
                fp += 1
        fn += len(gt) - len(used)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return dict(threshold=thr, tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / max(1e-9, p + r), 4),
                center_recall=round(center_hit / max(1, n_gt), 4), n_gt=n_gt)


def evaluate(a):
    if a.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
        os.environ.setdefault("ORT_NUM_THREADS", str(a.threads))
    import torch
    from ultralytics import YOLO
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    if str(a.weights).endswith(".onnx"):
        # ultralytics 는 SessionOptions 를 안 넘기므로(기본 = 물리 코어 수) 스레드 수가 실제로 적용되게 세션 생성을 감싼다
        import onnxruntime as ort
        _Orig = ort.InferenceSession

        class _Sess(_Orig):
            def __init__(self, path, sess_options=None, providers=None, **kw):
                so = sess_options or ort.SessionOptions()
                so.intra_op_num_threads = a.threads
                so.inter_op_num_threads = 1
                super().__init__(path, so, providers=providers, **kw)
        ort.InferenceSession = _Sess
    model = YOLO(str(a.weights), task="detect")
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(l) for l in (OUT / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    templates = {p.stem: np.load(p)["signal"].astype(np.float32) for p in (OUT / "templates").glob("*.npz")}
    split = kx.load_split()
    imgs, labs = kx.path_maps(split["val"])

    # 1) 원본 val 전체 + 속도 (batch 1)
    all_dets, gts, times = {}, {}, []
    for i, iid in enumerate(split["val"]):
        im = imread(imgs[iid]); h, w = im.shape
        bgr = _to_bgr(im)
        t0 = time.perf_counter()
        d = _predict(model, bgr, a.imgsz, a.device)
        if a.device != "cpu":
            torch.cuda.synchronize()
        if i >= 5:
            times.append((time.perf_counter() - t0) * 1000)
        all_dets[iid] = d
        gts[iid] = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h] for cx, cy, bw, bh in kx.read_yolo_labels(labs[iid])]
        if a.device == "cpu" and i >= a.cpu_images + 4:
            break  # CPU 는 느려서 원본 채점은 앞 n 장만 (속도용). 전체 채점은 GPU 설정에서.
    val_scores = [_score_val(all_dets, {k: gts[k] for k in all_dets}, t) for t in THRS]
    ts = sorted(times)
    speed = dict(n=len(ts), ms_mean=round(sum(ts) / len(ts), 2), ms_p50=round(ts[len(ts) // 2], 2), ms_p95=round(ts[int(len(ts) * 0.95)], 2),
                 fps=round(1000 / (sum(ts) / len(ts)), 1))
    if a.speed_only:  # 속도만 다시 재서 기존 summary.json 의 speed 만 바꾼다 (다른 작업과 겹쳐 잰 값 교정용)
        summ = kx.read_json(out / "summary.json"); summ["speed"] = speed; summ["speed_measured_at"] = kx.now_iso()
        kx.write_json(out / "summary.json", summ); print(f"[{a.tag}] speed 갱신", speed); return
    print(f"[{a.tag}] val({len(all_dets)}장) " + " | ".join(f"thr{s['threshold']}: F1 {s['f1']} center {s['center_recall']}" for s in val_scores), "| speed", speed, flush=True)

    # 2) 합성: 배경마다 원본 예측 1회(before), 샘플마다 넣고 예측
    before = {}
    bg_cache = {}
    results = []
    t_start = time.perf_counter()
    for k, s in enumerate(samples):
        iid = s["background_id"]
        if iid not in bg_cache:
            bg_cache[iid] = imread(imgs[iid])
            before[iid] = all_dets.get(iid)
            if before[iid] is None:
                before[iid] = _predict(model, _to_bgr(bg_cache[iid]), a.imgsz, a.device)
        im = bg_cache[iid]
        after = inject(im, templates[s["template"]], s["x"], s["y"], s["scale"])
        d = _predict(model, _to_bgr(after), a.imgsz, a.device)
        results.append({**s, "before_conf": _hit_at(before[iid], s["target_box"]), "after_conf": _hit_at(d, s["target_box"])})
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(samples)}  {(time.perf_counter() - t_start) / 60:.1f}분", flush=True)
    kx.write_json(out / "predictions.json", results)

    # 3) 요약
    import pandas as pd
    df = pd.DataFrame(results)
    df["pos_bin"] = pd.cut(df.rel_pos, [-0.01, 0.2, 0.5, 0.8, 1.01], labels=["왼쪽끝", "왼쪽중간", "오른쪽중간", "오른쪽끝"])
    summ = dict(tag=a.tag, weights=str(a.weights), imgsz=a.imgsz, device=a.device, threads=a.threads if a.device == "cpu" else None,
                backend="onnx" if str(a.weights).endswith(".onnx") else "torch",
                env=dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cpu=platform.processor(), torch=torch.__version__),
                val=val_scores, speed=speed, n_samples=len(df),
                before_any=int((df.before_conf >= 0.10).sum()), finished_at=kx.now_iso())
    for thr in THRS:
        hit = (df.after_conf >= thr) & (df.before_conf < thr)
        key = f"hit@{thr}"
        summ[key] = dict(overall=round(float(hit.mean()), 4),
                         by_scale={str(s): round(float(hit[df.scale == s].mean()), 4) for s in SCALES},
                         by_machine={m: round(float(hit[df.machine == m].mean()), 4) for m in "123"},
                         by_scale_machine={f"{s}|{m}": round(float(hit[(df.scale == s) & (df.machine == m)].mean()), 4) for s in SCALES for m in "123"},
                         by_pos={str(p): round(float(hit[df.pos_bin == p].mean()), 4) for p in df.pos_bin.cat.categories},
                         by_template={t: round(float(hit[df.template == t].mean()), 4) for t in sorted(df.template.unique())})
    summ["conf_median_by_scale"] = {str(s): round(float(df[df.scale == s].after_conf.median()), 3) for s in SCALES}
    kx.write_json(out / "summary.json", summ)
    print(json.dumps({k: summ[k] for k in ("hit@0.42", "hit@0.1", "conf_median_by_scale")}, ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------------- report
def report(a):
    rows = []
    for p in sorted(OUT.glob("*/summary.json")):
        rows.append(kx.read_json(p))
    L = ["# 띠 안 합성 이물질 + 경량화 비교 (val 369장, 합성 {}개)".format(rows[0]["n_samples"] if rows else 0), "",
         "검출 = 넣은 뒤 확신도가 임계값 이상이고 넣기 전에는 미만. 속도는 한 장씩(batch 1), 전처리·추론·후처리 포함, 처음 5장 제외.", "",
         "주의: ONNX CPU 줄의 원본 val F1 은 앞 65장만 채점한 값이라 GPU 줄(369장)과 직접 비교하지 않는다. 합성 검출률은 전부 5,930개 동일 조건.",
         "CPU 속도는 하이브리드 CPU(P코어 6 + E코어 4)에서 E코어 배정 시 3~4배 느려져, P코어에 고정해 다시 잰 값이다(라벨 PC i7-13세대 노트북).", "",
         "## 원본 val 성능과 속도", "",
         "| 설정 | 입력 | 실행 | F1@0.42 (IoU0.5) | 중심 재현율@0.42 | F1@0.10 | 중심 재현율@0.10 | 한 장 p50 ms | p95 ms | FPS |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v42, v10 = r["val"][0], r["val"][1]
        run = f"{r['backend']} {r['device']}" + (f" {r['threads']}스레드" if r.get("threads") else "")
        L.append(f"| {r['tag']} | {r['imgsz']} | {run} | {v42['f1']:.4f} | {v42['center_recall']:.4f} | {v10['f1']:.4f} | {v10['center_recall']:.4f} | {r['speed']['ms_p50']} | {r['speed']['ms_p95']} | {r['speed']['fps']} |")
    for thr in ("0.42", "0.1"):
        key = f"hit@{thr}"
        L += ["", f"## 띠 안 합성 검출률 (임계값 {thr}) — 진하기별", "", "| 설정 | " + " | ".join(f"진하기 {s}" for s in SCALES) + " | 전체 |", "|---|" + "---|" * (len(SCALES) + 1)]
        for r in rows:
            L.append(f"| {r['tag']} | " + " | ".join(f"{r[key]['by_scale'][str(s)]*100:.0f}%" for s in SCALES) + f" | {r[key]['overall']*100:.0f}% |")
        L += ["", f"### 호기별 (임계값 {thr})", "", "| 설정 | 1호기 | 2호기 | 3호기 |", "|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['tag']} | " + " | ".join(f"{r[key]['by_machine'][m]*100:.0f}%" for m in "123") + " |")
        L += ["", f"### 띠 안 위치별 (임계값 {thr}, 실제 이물질은 왼쪽 끝)", "", "| 설정 | 왼쪽끝 | 왼쪽중간 | 오른쪽중간 | 오른쪽끝 |", "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['tag']} | " + " | ".join(f"{r[key]['by_pos'][p]*100:.0f}%" for p in ["왼쪽끝", "왼쪽중간", "오른쪽중간", "오른쪽끝"]) + " |")
    L += ["", "## 진하기별 확신도 중앙값", "", "| 설정 | " + " | ".join(f"진하기 {s}" for s in SCALES) + " |", "|---|" + "---|" * len(SCALES)]
    for r in rows:
        L.append(f"| {r['tag']} | " + " | ".join(f"{r['conf_median_by_scale'][str(s)]:.2f}" for s in SCALES) + " |")
    (OUT / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("evaluate")
    e.add_argument("--tag", required=True); e.add_argument("--weights", type=Path, required=True)
    e.add_argument("--imgsz", type=int, default=1024); e.add_argument("--device", default="0")
    e.add_argument("--threads", type=int, default=4); e.add_argument("--cpu-images", type=int, default=60)
    e.add_argument("--speed-only", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
