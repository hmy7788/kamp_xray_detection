"""표시를 지운 자리의 흔적(고스트 윤곽)을 모델이 결함의 단서로 쓰는지 검증한다.

방법: 표시를 지운 val/test 이미지(data/nomark) 위에서, 결함이 없는 임의의 위치에 원래 표시와 같은 모양의 색 선을
다시 그린 뒤 **실제 제거와 같은 절차**(Navier-Stokes + 노이즈)로 지운다. 이렇게 만든 "가짜 흔적" 자리와, 아무것도
하지 않은 같은 크기의 "대조" 자리에서 모델이 얼마나 반응(신뢰도>=임계값인 검출의 중심이 자리 안)하는지 비교한다.
  - 가짜 흔적 반응률이 대조보다 뚜렷하게 높으면 모델이 지운 흔적을 결함 단서로 쓰는 것이다.
  - 비슷하면 이 검사로는 그런 지름길이 드러나지 않는다 (다른 단서가 있을 가능성은 배제하지 못함).

  python scripts/check_ring_cue.py 03_yolov3tiny_nomark [--device cpu] [--seed 0]
임계값은 실험의 eval_report_img<크기>.json(val에서 정한 값)을 쓴다.
"""
import argparse
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E  # noqa: E402
import remove_marks as R  # noqa: E402

ROOT = R.ROOT


def pick_shift(md0, guard, gts, w, h, taken, rng):
    """원본 표시 마스크를 옮겨 놓을 위치를 고른다 (경계 안, 실제 표시/정답/다른 자리와 겹치지 않게)."""
    for _ in range(80):
        ang, d = rng.uniform(0, 2 * math.pi), rng.uniform(25, 70)
        dy, dx = int(d * math.sin(ang)), int(d * math.cos(ang))
        sh = ndi.shift(md0.astype(np.uint8), (dy, dx), order=0).astype(bool)
        if sh.sum() != md0.sum() or (sh & guard).any() or (sh & taken).any():
            continue
        lbl, _ = ndi.label(sh)
        boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndi.find_objects(lbl)]
        if any(b[0] - 6 <= cx <= b[2] + 6 and b[1] - 6 <= cy <= b[3] + 6 for b in boxes for cx, cy, _, _ in gts):
            continue
        return (dy, dx), sh, boxes
    return None


def hit(dets, box, thr, pad=4):
    x0, y0, x1, y1 = box
    return any(d[4] >= thr and x0 - pad <= (d[0] + d[2]) / 2 <= x1 + pad and y0 - pad <= (d[1] + d[3]) / 2 <= y1 + pad for d in dets)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = ROOT / "outputs" / "runs" / a.name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    img = a.img_size or cfg["img_size"][-1]
    rep = json.loads((run / f"eval_report_img{img}.json").read_text(encoding="utf-8"))
    thr = rep["threshold"]
    list_dir = ROOT / "data" / ("splits_nomark" if cfg.get("dataset") == "nomark" else "splits")
    rng = random.Random(a.seed)

    tmp = Path(tempfile.mkdtemp(prefix="ringcue_"))
    (tmp / "images").mkdir(); (tmp / "labels").mkdir()
    sites = {}  # 이미지 파일명 -> (fake_boxes, control_boxes)
    for sp in ("val", "test"):
        lines = []
        for l in (list_dir / f"{sp}.txt").read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            p = Path(l)
            clean = R.imread(p)                                     # 표시를 지운 이미지(제거본)
            h, w = clean.shape[:2]
            orig = R.imread(next((ROOT / "data" / "subsets" / "n500" / "images").glob(p.stem + ".*")))
            m = R.mark_mask(orig)                                   # 원본의 색 선 마스크
            md0 = ndi.binary_dilation(m, iterations=R.DILATE)
            guard = ndi.binary_dilation(md0, iterations=R.INPAINT_R + 6)
            gts = R.gt_boxes(p.stem, w, h)
            fake = pick_shift(md0, guard, gts, w, h, np.zeros_like(md0), rng)
            if fake is None:
                continue
            ctrl = pick_shift(md0, guard, gts, w, h, ndi.binary_dilation(fake[1], iterations=8), rng)
            if ctrl is None:
                continue
            dy, dx = fake[0]
            line = ndi.shift(m.astype(np.uint8), (dy, dx), order=0).astype(bool)   # 옮긴 색 선
            painted = clean.copy()
            painted[line] = (0, 0, 255)                              # BGR 빨강으로 다시 그림
            fk, _, _ = R.remove_marks(painted, "ns", 0.75, seed=rng.randrange(1 << 30))  # 실제와 같은 절차로 지움
            dst = tmp / "images" / f"{p.stem}.png"
            R.imwrite(dst, fk)
            shutil.copy2(ROOT / "data" / "nomark" / "labels" / f"{p.stem}.txt", tmp / "labels" / f"{p.stem}.txt")
            lines.append(dst.as_posix())
            sites[p.stem] = (fake[2], ctrl[2], sp)
        (tmp / f"{sp}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = {"fake": [0, 0], "control": [0, 0]}                      # [반응한 자리, 전체 자리]
    fake_recs, base_recs = {}, {}
    for sp in ("val", "test"):
        fake_recs.update({Path(r["name"]).stem: r for r in E.predict(a.name, sp, img, a.device, str(tmp))})
        base_recs.update({Path(r["name"]).stem: r for r in json.loads((run / f"preds_{sp}_img{img}.json").read_text(encoding="utf-8"))})
    for stem, (fboxes, cboxes, sp) in sites.items():
        for b in fboxes:
            stats["fake"][1] += 1; stats["fake"][0] += hit(fake_recs[stem]["dets"], b, thr)
        for b in cboxes:                                              # 대조: 아무것도 하지 않은 원 이미지(제거본)에서의 같은 크기 자리
            stats["control"][1] += 1; stats["control"][0] += hit(base_recs[stem]["dets"], b, thr)
    f, c = stats["fake"], stats["control"]
    print(f"\n=== {a.name} | 임계값 {thr} | 이미지 {len(sites)}장 ===")
    print(f"가짜 흔적 자리   : {f[0]}/{f[1]} = {100 * f[0] / f[1]:.1f}% 에서 모델이 반응")
    print(f"대조(아무것 안 함): {c[0]}/{c[1]} = {100 * c[0] / c[1]:.1f}% 에서 모델이 반응")
    # 실제 결함 검출이 가짜 흔적 때문에 달라지는지 (전체 F1)
    rec_f = [fake_recs[s] for s in sites]; rec_b = [base_recs[s] for s in sites]
    mf, mb = E.metrics_at(rec_f, thr, 0.5), E.metrics_at(rec_b, thr, 0.5)
    print(f"같은 이미지의 전체 F1: 가짜 흔적 없음 {mb['F1']:.3f} (FP {mb['FP']}) -> 가짜 흔적 있음 {mf['F1']:.3f} (FP {mf['FP']})")
    out = {"run": a.name, "threshold": thr, "images": len(sites), "fake_sites": f, "control_sites": c,
           "f1_without_fake": mb["F1"], "f1_with_fake": mf["F1"], "fp_without_fake": mb["FP"], "fp_with_fake": mf["FP"]}
    (run / f"ring_cue_check_img{img}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
