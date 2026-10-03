"""이미지에 찍힌 장비 표시(색 박스)를 지운다.

배경: 원본 X선 이미지에는 장비가 그린 색 사각형 표시(빨강/파랑/분홍/노랑 등)가 들어 있고, 모델이 결함이 아니라
이 표시를 보고 맞히는 문제가 확인되었다 (docs/experiment_results.md, docs/dataset.md 8절).

방법: X선 영상은 거의 무채색이므로 채널 간 차이(max-min)가 임계값을 넘는 픽셀을 표시로 보고,
2px 팽창한 마스크를 cv2.inpaint(Telea)로 주변 픽셀로 메운다.

사용
  python scripts/remove_marks.py preview [--n-per-unit 6]   # 육안 확인용 HTML 미리보기 (data/preview_nomark/)
  python scripts/remove_marks.py compare [--n-quant 200]    # Telea vs Navier-Stokes 비교 (정량 + 확대 그림)
  python scripts/remove_marks.py build [--method ns --noise 0.75]   # n500 전체에서 표시 제거 -> data/nomark/
"""
import argparse
import csv
import html
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "subsets" / "n500"
COLOR_THR = 50   # 채널 차이 임계값 (X선 영상은 무채색)
DILATE = 2       # 마스크 팽창(px): 선 가장자리까지 덮기
INPAINT_R = 3


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    buf.tofile(str(path))
    return ok


def mark_mask(bgr):
    """표시 픽셀 마스크(bool). BGR 입력."""
    c = bgr.astype(np.int16)
    return (c.max(2) - c.min(2)) > COLOR_THR


METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def hp_std(gray, region):
    """고주파(노이즈 질감) 표준편차: 이미지 - 가우시안 블러(sigma 1.5) 의 region 내부 표준편차."""
    g = gray.astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 1.5))[region].std())


def add_matched_noise(clean, md, rng, scale=1.0):
    """메운 영역(md)의 질감이 주변(링)과 같아지도록 가우시안 노이즈를 더한다 (Telea/NS 가 만든 매끈함 보정)."""
    gray = clean.mean(2)
    ring = ndi.binary_dilation(md, iterations=10) & ~ndi.binary_dilation(md, iterations=3)
    add = np.sqrt(max(hp_std(gray, ring) ** 2 - hp_std(gray, md) ** 2, 0.0))
    n = rng.normal(0, add * scale, md.shape)[..., None] * md[..., None]
    return np.clip(clean.astype(np.float32) + n, 0, 255).astype(np.uint8)


def remove_marks(bgr, method="telea", noise=False, seed=0):
    """표시를 지운 이미지와 (표시 마스크, 표시 연결 성분 bbox 목록)을 반환한다.
    method: 'telea' 또는 'ns'(Navier-Stokes). noise: 메운 자리에 더할 노이즈 배율(True=1.0, 0/False=없음)."""
    m = mark_mask(bgr)
    md = ndi.binary_dilation(m, iterations=DILATE)
    clean = cv2.inpaint(bgr, md.astype(np.uint8) * 255, INPAINT_R, METHODS[method])
    if noise:
        clean = add_matched_noise(clean, md, np.random.default_rng(seed), scale=float(noise))
    lbl, _ = ndi.label(md)
    boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndi.find_objects(lbl)]
    return clean, m, boxes


def gt_boxes(stem, w, h):
    out = []
    for line in (SRC / "labels" / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split()[:5])
            out.append((cx * w, cy * h, bw * w, bh * h))
    return out


def draw_gt(img, gts, off=(0, 0), scale=1):
    g = img.copy()
    for cx, cy, bw, bh in gts:
        p1 = (int((cx - bw / 2 - off[0]) * scale), int((cy - bh / 2 - off[1]) * scale))
        p2 = (int((cx + bw / 2 - off[0]) * scale), int((cy + bh / 2 - off[1]) * scale))
        cv2.rectangle(g, p1, p2, (0, 255, 0), 1)  # 초록: 정답(GT) 박스
    return g


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (255, 255, 255), -1)
    cv2.putText(out, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def preview(n_per_unit, seed):
    out = ROOT / "data" / "preview_nomark"
    (out / "img").mkdir(parents=True, exist_ok=True)
    units = {r["name"]: r["unit"] for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}

    # 전체 400장 통계 (마스크 면적/성분 수) -> 이상치 찾기
    stats = []
    for name in sorted(units):
        bgr = imread(SRC / "images" / name)
        clean, m, boxes = remove_marks(bgr)
        stats.append({"name": name, "unit": units[name], "mask_px": int(m.sum()), "n_marks": len(boxes)})
    areas = np.array([s["mask_px"] for s in stats])
    p99 = float(np.percentile(areas, 99))

    rng = random.Random(seed)
    picks = []
    for u in ("1호기", "2호기", "3호기"):
        cand = [s for s in stats if s["unit"] == u]
        rnd = rng.sample(cand, min(n_per_unit, len(cand)))
        big = sorted(cand, key=lambda s: -s["mask_px"])[:2]       # 마스크 면적이 가장 큰 것 (이상치 확인)
        many = sorted([s for s in cand if s["n_marks"] >= 3], key=lambda s: -s["mask_px"])[:1]
        for s, why in [(x, "무작위") for x in rnd] + [(x, "마스크 면적 최대") for x in big] + [(x, "표시 3개") for x in many]:
            if s["name"] not in {p[0]["name"] for p in picks}:
                picks.append((s, why))

    cards = []
    for s, why in picks:
        name = s["name"]; stem = Path(name).stem
        bgr = imread(SRC / "images" / name); h, w = bgr.shape[:2]
        clean, m, boxes = remove_marks(bgr)
        gts = gt_boxes(stem, w, h)
        full = np.hstack([label(bgr, "원본"), label(clean, "지운 결과"), label(draw_gt(clean, gts), "지운 결과 + 정답(초록)")])
        imwrite(out / "img" / f"{stem}_full.png", full)
        crops = []
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            pad = 10; X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
            sc = 8
            up = lambda a: cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            o, c = up(bgr[Y0:Y1, X0:X1]), up(clean[Y0:Y1, X0:X1])
            g = draw_gt(c, gts, (X0, Y0), sc)
            tile = np.hstack([label(o, "원본"), label(c, "지운 결과"), label(g, "정답(초록)")])
            fn = f"{stem}_crop{i}.png"; imwrite(out / "img" / fn, tile); crops.append(fn)
        cards.append((s, why, f"{stem}_full.png", crops, len(gts)))

    flagged = [s for s in stats if s["mask_px"] > p99]
    rows = "".join(
        f"<section><h3>{html.escape(s['name'])} <small>{s['unit']} · 표시 {s['n_marks']}개 · 마스크 {s['mask_px']}px · 정답 {ng}개 · 선택 이유: {why}</small></h3>"
        f"<img class='full' src='img/{html.escape(full)}'>"
        + "".join(f"<img class='crop' src='img/{html.escape(c)}'>" for c in crops) + "</section>"
        for s, why, full, crops, ng in cards)
    doc = f"""<!doctype html><meta charset="utf-8"><title>표시 제거 미리보기</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;margin:16px;background:#fafafa}}
section{{margin:0 0 28px;padding:10px;background:#fff;border:1px solid #ddd}}
h3{{margin:0 0 8px}} small{{color:#555;font-weight:normal}}
img{{display:block;margin:6px 0;image-rendering:pixelated;max-width:100%}} .crop{{border:1px solid #ccc}}</style>
<h2>장비 표시(색 박스) 제거 미리보기</h2>
<p>방법: 색 픽셀(채널 차이 &gt; {COLOR_THR}) 마스크를 {DILATE}px 팽창해 Telea 인페인팅. 각 이미지마다 위는 전체(원본 | 지운 결과 | 지운 결과+정답), 아래는 표시 하나씩 8배 확대(원본 | 지운 결과 | 정답).<br>
<b>확인할 것</b>: ① 색 선이 남아 있지 않은가 ② 초록 정답 박스 안의 작은 어두운 점(결함)이 지운 결과에도 남아 있는가 ③ 메운 자리가 부자연스러운 얼룩인가.</p>
<p>전체 400장 마스크 면적: 중앙값 {int(np.median(areas))}px, 99% {int(p99)}px, 최대 {int(areas.max())}px. 면적이 큰 이미지 상위는 아래에 "마스크 면적 최대"로 포함했습니다. 표시가 3개인 이미지 {sum(1 for s in stats if s['n_marks']>=3)}장.</p>
{rows}"""
    (out / "index.html").write_text(doc, encoding="utf-8")
    print(f"미리보기 {len(cards)}장 -> {out / 'index.html'}")
    print(f"마스크 면적 중앙값 {int(np.median(areas))}px, 99% {int(p99)}px, 최대 {int(areas.max())}px | 99% 초과 {len(flagged)}장")
    return out / "index.html"


def compare(n_quant, seed):
    import math
    out = ROOT / "data" / "preview_nomark"
    (out / "img").mkdir(parents=True, exist_ok=True)
    units = {r["name"]: r["unit"] for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    rng = random.Random(seed)

    # ---- 시험 세트: 표시 없는 위치에 표시 모양 마스크를 옮겨 놓는다 (정답 = 그 위치의 실제 픽셀)
    items = []
    for name in rng.sample(sorted(units), n_quant):
        bgr = imread(SRC / "images" / name)
        md0 = ndi.binary_dilation(mark_mask(bgr), iterations=DILATE)
        guard = ndi.binary_dilation(md0, iterations=INPAINT_R + 3)
        for _ in range(30):
            ang, d = rng.uniform(0, 2 * math.pi), rng.uniform(25, 50)
            sh = ndi.shift(md0.astype(np.uint8), (int(d * math.sin(ang)), int(d * math.cos(ang))), order=0).astype(bool)
            if sh.sum() == md0.sum() and not (sh & guard).any():
                break
        else:
            continue
        gt = bgr.mean(2)
        fills = {m: cv2.inpaint(bgr, sh.astype(np.uint8) * 255, INPAINT_R, f) for m, f in METHODS.items()}
        items.append((sh, gt, fills))

    def score(method, scale):
        mae, ratio = [], []
        for sh, gt, fills in items:
            f = fills[method]
            if scale:
                f = add_matched_noise(f, sh, np.random.default_rng(0), scale)
            mae.append(float(np.abs(f.mean(2) - gt)[sh].mean()))
            ratio.append(hp_std(f.mean(2), sh) / max(hp_std(gt, sh), 1e-6))
        return float(np.mean(mae)), float(np.median(mae)), float(np.mean(ratio))

    # 노이즈 배율: 방식별로 질감비가 1.0 에 가장 가까워지는 값을 격자 탐색
    grid = np.round(np.arange(0.2, 1.21, 0.05), 2)
    best_scale = {m: min(grid, key=lambda g: abs(score(m, g)[2] - 1.0)) for m in METHODS}
    variants = [("telea", "Telea", 0.0), ("ns", "Navier-Stokes", 0.0),
                ("telea", f"Telea+노이즈(x{best_scale['telea']})", best_scale["telea"]),
                ("ns", f"NS+노이즈(x{best_scale['ns']})", best_scale["ns"])]
    rows = {t: score(m, sc) for m, t, sc in variants}
    print(f"정량 비교 (표시 없는 위치 {len(items)}곳, 메운 픽셀 vs 실제 픽셀)")
    for t, (mean, med, ratio) in rows.items():
        print(f"  {t:22s} MAE 평균 {mean:.2f} / 중앙값 {med:.2f} | 질감비 {ratio:.2f}")
    table = "".join(f"<tr><td>{html.escape(t)}</td><td>{m:.2f}</td><td>{md:.2f}</td><td>{r:.2f}</td></tr>" for t, (m, md, r) in rows.items())

    # ---- 정성: 표시별 확대 비교
    names = ["001_20200623_003520(5).jpg"] + [n for u in ("1호기", "2호기", "3호기")
                                           for n in rng.sample([k for k in sorted(units) if units[k] == u], 2)]
    blocks = []
    for name in names:
        stem = Path(name).stem
        bgr = imread(SRC / "images" / name); h, w = bgr.shape[:2]
        outs = {t: remove_marks(bgr, m_, sc)[0] for m_, t, sc in variants}
        boxes = remove_marks(bgr)[2]
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            pad = 10; X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
            up = lambda a: cv2.resize(a, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            diff = np.abs(outs["Telea"].astype(int) - outs["Navier-Stokes"].astype(int)).mean(2)[Y0:Y1, X0:X1]
            dimg = cv2.cvtColor(np.clip(diff * 10, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            panels = [label(up(bgr[Y0:Y1, X0:X1]), "원본")] + [label(up(outs[t][Y0:Y1, X0:X1]), t) for _, t, _ in variants] + [label(up(dimg), "|Telea-NS| x10")]
            fn = f"cmp_{stem}_{i}.png"; imwrite(out / "img" / fn, np.hstack(panels))
            blocks.append((name, units[name], i, fn))
    cards = "".join(f"<section><h3>{html.escape(n)} <small>{u} · 표시 {i + 1}</small></h3><img src='img/{html.escape(fn)}'></section>" for n, u, i, fn in blocks)
    doc = f"""<!doctype html><meta charset="utf-8"><title>인페인팅 방식 비교</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;margin:16px;background:#fafafa}}section{{margin:0 0 20px;padding:10px;background:#fff;border:1px solid #ddd}}
h3{{margin:0 0 6px}}small{{color:#555;font-weight:normal}}img{{display:block;image-rendering:pixelated;max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:4px 10px;text-align:right}}</style>
<h2>인페인팅 방식 비교: Telea vs Navier-Stokes, 각각 노이즈 보정 포함</h2>
<p><b>정량</b> (표시 없는 위치 {len(items)}곳에 표시 모양 마스크를 옮겨 메운 뒤 실제 픽셀과 비교. MAE는 0~255 밝기 차이로 낮을수록 좋고, 질감비는 메운 영역의 고주파 표준편차 / 실제 값으로 1에 가까울수록 질감이 보존됨. 노이즈 배율은 질감비가 1.0에 가장 가깝도록 방식별로 탐색)</p>
<table><tr><th>방식</th><th>MAE 평균</th><th>MAE 중앙값</th><th>질감비</th></tr>{table}</table>
<p><b>정성</b>: 왼쪽부터 원본 | Telea | Navier-Stokes | Telea+노이즈 | NS+노이즈 | 두 기본 방식의 차이(x10). 첫 이미지는 표시가 막대 가장자리에 걸린 경우.</p>{cards}"""
    (out / "compare.html").write_text(doc, encoding="utf-8")
    print("->", out / "compare.html")


def build(method, noise):
    """n500 전체에서 표시를 지워 data/nomark/{images,labels} 를 만든다 (PNG 무손실, 라벨은 그대로 복사)."""
    import shutil
    import zlib
    out = ROOT / "data" / "nomark"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    residual, cover, n = [], [], 0
    for img in sorted((SRC / "images").iterdir()):
        bgr = imread(img)
        h, w = bgr.shape[:2]
        clean, m, boxes = remove_marks(bgr, method, noise, seed=zlib.crc32(img.stem.encode()))
        imwrite(out / "images" / f"{img.stem}.png", clean)
        shutil.copy2(SRC / "labels" / f"{img.stem}.txt", out / "labels" / f"{img.stem}.txt")
        residual.append(int(mark_mask(clean).sum()))              # 지운 뒤에도 남은 색 픽셀 수
        md = ndi.binary_dilation(m, iterations=DILATE)
        for cx, cy, bw, bh in gt_boxes(img.stem, w, h):           # 정답 박스 중 마스크(지운 자리)와 겹친 면적 비율
            x0, y0, x1, y1 = [int(round(v)) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)]
            reg = md[max(y0, 0):y1, max(x0, 0):x1]
            cover.append(float(reg.mean()) if reg.size else 0.0)
        n += 1
    cover = np.array(cover)
    print(f"{n}장 -> {out}  (방식 {method}, 노이즈 x{noise})")
    print(f"지운 뒤 남은 색 픽셀: 최대 {max(residual)}px, 남은 이미지 {sum(r > 0 for r in residual)}장")
    print(f"정답 박스 {len(cover)}개 중 지운 자리와 겹치는 박스 {int((cover > 0).sum())}개 | 겹침 비율 평균 {cover.mean():.3f}, 최대 {cover.max():.3f}")
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--n-per-unit", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("compare")
    c.add_argument("--n-quant", type=int, default=200)
    c.add_argument("--seed", type=int, default=42)
    b = sub.add_parser("build")
    b.add_argument("--method", default="ns", choices=list(METHODS))
    b.add_argument("--noise", type=float, default=0.75)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.method, a.noise)
    elif a.cmd == "preview":
        preview(a.n_per_unit, a.seed)
    elif a.cmd == "compare":
        compare(a.n_quant, a.seed)


if __name__ == "__main__":
    main()
