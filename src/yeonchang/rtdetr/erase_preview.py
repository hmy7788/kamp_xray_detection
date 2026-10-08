"""결함 지우기 방식별 미리보기 (모델은 돌리지 않음). 사람이 지운 티를 눈으로 비교하는 용도.

    python src/yeonchang/rtdetr/erase_preview.py --exp 02_rtdetrl_img640_e30

결과: runs/yeonchang/<exp>/result/erase_preview/ (Git 제외). erase_defects.py와 같은 10장(seed 0)을 쓴다.
"""
import argparse
import random

import cv2
import numpy as np

from common import DATA, RUNS, load_gt, read_manifest
from erase_defects import dot_mask, erase, imread, imwrite

FILLS = ["inpaint", "mean", "inpaint_texture", "mean_texture"]


def overlay(img, mask):
    v = img.copy()
    v[mask > 0] = (0.4 * v[mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    return v


def label(img, text):
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def hcat(imgs, gap=6):
    h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        i = np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)])
        out += [i, np.full((h, gap, 3), 255, np.uint8)]
    return np.hstack(out[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-show", type=int, default=10)
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("--crop", type=int, default=24, help="확대 영역 반경(px)")
    ap.add_argument("--zoom", type=int, default=6)
    ap.add_argument("--mode", choices=["box", "dot"], default="dot",
                    help="box: 박스 영역을 지우는 4방식 비교 / dot: 점만 찾아 지움(찾은 점 표시 포함)")
    args = ap.parse_args()

    manifest = read_manifest()
    gt = load_gt("test", manifest)
    ids = sorted(i for i in gt if gt[i])
    rng = random.Random(0)  # erase_defects.py와 같은 선택
    by_m = {}
    for i in ids:
        by_m.setdefault(manifest[i]["machine"], []).append(i)
    show = []
    while len(show) < args.n_show:
        for m in sorted(by_m):
            if len(show) < args.n_show and by_m[m]:
                show.append(by_m[m].pop(rng.randrange(len(by_m[m]))))

    out_dir = RUNS / args.exp / "result" / ("erase_preview" if args.mode == "box" else "erase_preview_dot")
    for n, i in enumerate(show, 1):
        img = imread(DATA / manifest[i]["image_path"])
        if args.mode == "dot":
            mask, sizes = dot_mask(img, gt[i])
            versions = [("original", img), ("detected dot (red)", overlay(img, mask)),
                        ("dot erased", erase(img, gt[i], 0, 3, "dot")[0])]
            print(f"   dot pixels per defect (before 1px grow): {sizes}")
        else:
            versions = [("original", img)] + [(f, erase(img, gt[i], args.pad, 5, f, rng=np.random.default_rng(0))[0]) for f in FILLS]
        scale = 2 if img.shape[1] < 500 else 1.5
        full = hcat([label(cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), name) for name, v in versions])
        rows = [full]
        h, w = img.shape[:2]
        for b in gt[i]:  # 결함마다 확대 (표시 없이)
            cx, cy = int((b[1] + b[3]) / 2), int((b[2] + b[4]) / 2)
            y1, y2, x1, x2 = max(0, cy - args.crop), min(h, cy + args.crop), max(0, cx - args.crop), min(w, cx + args.crop)
            crops = [label(cv2.resize(v[y1:y2, x1:x2], None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST),
                           f"{name} zoom x{args.zoom} @({cx},{cy})") for name, v in versions]
            rows.append(hcat(crops))
        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 255, np.uint8)]) for r in rows]
        imwrite(out_dir / f"{n:02d}_{i}.png", np.vstack([np.vstack([r, np.full((10, wmax, 3), 255, np.uint8)]) for r in rows]))
        print(f"{n:02d} {i}  machine {manifest[i]['machine']}  defects {len(gt[i])}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
