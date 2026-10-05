"""preds_<split>.json 의 검출 결과를 이미지에 그려 runs/<owner>/<name>/figures/<split>/ 에 저장한다.

  python src/minyeop/faster_rcnn/visualize.py --name 01_frcnn_r50fpn_min640_v1 --split val

정답은 초록, 예측은 빨강(신뢰도 표기). 만드는 그림:
  samples_ok.png     정상 검출 예시 (호기·라벨 출처를 섞어서)
  errors.png         FP/FN 이 있는 이미지의 결함 주변 확대 (왼쪽: 전체, 오른쪽: 확대)
  empty_labels.png   정답이 없는 이미지(빈 라벨)에서 모델이 무엇을 찾았는지
  confidence_hist.png 신뢰도 분포 (정답과 짝지어진 검출 / 아닌 검출)
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from dataset import DATA, ROOT  # noqa: E402

GT, PR = (40, 200, 60), (235, 50, 50)


def font(size=13):
    """한글이 나오는 글꼴 (PIL 기본 글꼴은 한글이 깨진다). 없으면 기본 글꼴로 대체한다."""
    for f in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_img(split, name):
    return Image.open(DATA / split / "images" / name).convert("RGB")


def draw(img, rec, thr, scale=2, crop=None):
    """crop=(x1,y1,x2,y2) 이면 그 영역만 잘라 scale 배로 확대한다."""
    if crop:
        img = img.crop(crop)
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    tr = lambda b: [(b[0] - ox) * scale, (b[1] - oy) * scale, (b[2] - ox) * scale, (b[3] - oy) * scale]
    for g in rec["gt"]:
        d.rectangle(tr(g), outline=GT, width=2)
    for p in rec["dets"]:
        if p[4] >= thr:
            b = tr(p)
            d.rectangle(b, outline=PR, width=2)
            d.text((b[0], max(b[1] - 11, 0)), f"{p[4]:.2f}", fill=PR)
    return img


def grid(tiles, cols, title=None, pad=6, caption_h=16):
    w = max(t[0].width for t in tiles); h = max(t[0].height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    top = 24 if title else 0
    canvas = Image.new("RGB", (cols * (w + pad) + pad, top + rows * (h + caption_h + pad) + pad), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((pad, 3), title, fill=(255, 255, 255), font=font(14))
    for i, (im, cap) in enumerate(tiles):
        x, y = pad + (i % cols) * (w + pad), top + pad + (i // cols) * (h + caption_h + pad)
        canvas.paste(im, (x, y + caption_h))
        d.text((x, y + 1), cap, fill=(220, 220, 220), font=font(12))
    return canvas


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--threshold", type=float, default=None, help="기본: val 에서 F1 최대인 값")
    ap.add_argument("--n-ok", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = ROOT / "runs" / a.owner / a.name
    recs = json.loads((run / f"preds_{a.split}.json").read_text(encoding="utf-8"))
    thr = a.threshold if a.threshold is not None else metrics.best_f1(
        json.loads((run / "preds_val.json").read_text(encoding="utf-8")))["thr"]
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8"))}
    out = run / "figures" / a.split; out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)

    def tag(r):
        m = man[r["name"]]
        return f"{m['machine']}호기 {'공식' if m['source'] == 'official' else '팀'}라벨"

    def n_err(r):
        t, _, nm = metrics.match_image(r, thr)
        return int((~t).sum()), len(r["gt"]) - nm  # (FP, FN)

    ok = [r for r in recs if r["gt"] and n_err(r) == (0, 0)]
    bad = [r for r in recs if n_err(r) != (0, 0)]
    empty = [r for r in recs if not r["gt"]]

    # 1) 정상 검출 예시: (호기, 출처) 조합이 고르게 섞이도록 뽑는다
    groups = {}
    for r in ok:
        groups.setdefault(tag(r), []).append(r)
    pick = []
    while len(pick) < min(a.n_ok, len(ok)):
        for k in sorted(groups):
            if groups[k] and len(pick) < a.n_ok:
                pick.append(groups[k].pop(rnd.randrange(len(groups[k]))))
    tiles = [(draw(load_img(a.split, r["name"]), r, thr, 1), f"{tag(r)} | {r['name'][:22]}") for r in pick]
    grid(tiles, 3, f"정상 검출 예시 ({a.split}, 임계값 {thr}) — 초록=정답, 빨강=예측(신뢰도)").save(out / "samples_ok.png")

    # 2) 오류: 전체 + 오류 주변 확대
    tiles = []
    for r in bad:
        img = load_img(a.split, r["name"])
        t, _, _ = metrics.match_image(r, thr)
        dets = sorted([d for d in r["dets"] if d[4] >= thr], key=lambda d: -d[4])
        ious = metrics.iou_matrix([d[:4] for d in dets], r["gt"]) if dets and r["gt"] else None
        # 확대 대상: 짝이 안 지어진 정답
        focus = [g for j, g in enumerate(r["gt"]) if ious is None or ious[:, j].max() < 0.5] or r["gt"][:1]
        g = focus[0]; cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        x1, y1 = int(max(cx - 30, 0)), int(max(cy - 30, 0))
        x2, y2 = int(min(x1 + 60, img.width)), int(min(y1 + 60, img.height))
        fp, fn = n_err(r)
        tiles.append((draw(img, r, thr, 1), f"{tag(r)} | FP {fp} FN {fn} | {r['name'][:20]}"))
        tiles.append((draw(img, r, thr, 6, (x1, y1, x2, y2)), "확대: 정답 %dx%dpx" % (g[2] - g[0], g[3] - g[1])))
    if tiles:
        grid(tiles, 4, f"오류 이미지 {len(bad)}장 ({a.split}, 임계값 {thr}) — 초록=정답, 빨강=예측").save(out / "errors.png")

    # 3) 빈 라벨 이미지
    if empty:
        sel = empty[:: max(len(empty) // 8, 1)][:8]
        tiles = [(draw(load_img(a.split, r["name"]), r, thr, 1),
                  f"{tag(r)} | 검출 {sum(d[4] >= thr for d in r['dets'])}개 | {r['name'][:18]}") for r in sel]
        n_fp = sum(any(d[4] >= thr for d in r["dets"]) for r in empty)
        grid(tiles, 3, f"빈 라벨 이미지 {len(empty)}장 중 {n_fp}장에서 오검출 (임계값 {thr})").save(out / "empty_labels.png")

    # 4) 신뢰도 분포
    tp_c, other_c = [], []
    for r in recs:
        t, c, _ = metrics.match_image(r, 0.05)
        tp_c += c[t].tolist(); other_c += c[~t].tolist()
    W, H, bins = 640, 260, 20
    hist = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(hist)
    d.text((10, 6), f"신뢰도 분포 (0.05 이상 검출) — 파랑: 정답과 짝지어진 {len(tp_c)}개, 주황: 아닌 {len(other_c)}개 (로그 스케일)", fill=(0, 0, 0), font=font(12))
    import math
    cnt = lambda xs: [sum(1 for x in xs if i / bins <= x < (i + 1) / bins + (1e-9 if i == bins - 1 else 0)) for i in range(bins)]
    ct, co = cnt(tp_c), cnt(other_c); mx = max(max(ct), max(co), 1)
    bw = (W - 60) / bins
    for i in range(bins):
        for k, (c, col) in enumerate(((ct[i], (50, 110, 220)), (co[i], (240, 150, 40)))):
            hgt = int((H - 60) * math.log10(c + 1) / math.log10(mx + 1))
            x = 40 + i * bw + k * bw / 2
            d.rectangle([x, H - 30 - hgt, x + bw / 2 - 1, H - 30], fill=col)
        if i % 4 == 0:
            d.text((40 + i * bw, H - 24), f"{i / bins:.1f}", fill=(0, 0, 0))
    d.line([(40, H - 30), (W - 10, H - 30)], fill=(0, 0, 0))
    hist.save(out / "confidence_hist.png")

    print(f"저장: {out}")
    for f in sorted(out.glob("*.png")):
        print(" ", f.name, Image.open(f).size)
    print(f"정상 {len(ok)}장 / 오류 이미지 {len(bad)}장 / 빈 라벨 {len(empty)}장 | 임계값 {thr}")


if __name__ == "__main__":
    main()
