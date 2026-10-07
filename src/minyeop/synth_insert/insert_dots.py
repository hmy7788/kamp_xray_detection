"""점 합성 검출 시험: 옅고 작은 점을 넣어 세 모델이 어디까지 찾는지 본다 (미탐지 조건 분석).

설계
  바탕     : test 이미지의 결함 점을 평균 보간(13x13)으로 지운 이미지 (가짜 정상, 오경보가 거의 없던 쪽)
  점 은행  : val 이미지의 실제 점. 점 주변 9x9 조각을 둘레 밝기로 나눈 투과율 T (0~1)로 저장한다. 학습·test 와 겹치지 않는 val 에서만 가져온다.
  합성     : 바탕 밝기에 T^s 를 곱한다 (X선은 겹친 물체의 투과율이 곱해지므로 밝은 곳에서 더 많이 어두워진다). s 가 작을수록 옅다.
             크기 배율 f 는 조각을 f 배로 다시 그린다. 같은 호기의 점을 쓰고 좌우·상하 뒤집기를 무작위로 한다.
  자리 두 가지
     orig : 원래 점이 있던 자리 (실제와 같은 맥락)
     rand : 제품 안의 무작위 자리 (가장자리·다른 점과 거리를 둠). orig 와 rand 의 검출률이 같으면 '자리'를 지름길로 쓰지 않는다는 뜻이다.
  판정     : 각 모델의 val 임계값 이상 검출의 중심이 넣은 점 중심에서 5px 이내이면 검출. 그 밖의 검출은 오경보로 센다.
같은 자리·같은 점 조각을 모든 (s, f) 칸에서 똑같이 써서 칸끼리 비교가 되게 한다.
원본 data/ 는 바꾸지 않고, 합성 이미지는 --work-dir 에만 만든다 (Git 제외). 결과 요약과 점별 기록만 --out 에 저장한다.

  python src/minyeop/synth_insert/insert_dots.py --work-dir <임시 폴더> --out runs/minyeop/06_synth_insert_v1
"""
import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FR, YO, MB, DR = (ROOT / "src" / "minyeop" / p for p in ("faster_rcnn", "yolov3_tiny", "faster_rcnn_mobilenet", "dot_removal"))
sys.path.insert(0, str(MB))
sys.path.insert(0, str(FR))
sys.path.insert(0, str(DR))

PATCH = 9


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def build_bank(rd, man):
    """val 의 실제 점 -> 투과율 조각 목록 {호기: [T(9x9)]}."""
    bank = {"1": [], "2": [], "3": []}
    val_root = ROOT / "data" / "val"
    for lab in sorted((val_root / "labels").glob("*.txt")):
        rows = [l.split() for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        name = lab.stem + ".png"
        im = np.array(Image.open(val_root / "images" / name).convert("L")).astype(np.float32)
        h, w = im.shape
        blur = cv2.GaussianBlur(im, (0, 0), 1.0)
        for r in rows:
            cx, cy, bw, bh = (float(v) for v in r[1:5])
            g = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            x, y = rd.locate_dot(blur, g)
            if min(x, y) < 10 or x > w - 11 or y > h - 11:
                continue
            yy, xx = np.mgrid[y - 7:y + 8, x - 7:x + 8]
            ring = im[y - 7:y + 8, x - 7:x + 8][np.maximum(abs(xx - x), abs(yy - y)) >= 5]
            B = float(np.median(ring))
            T = np.clip(im[y - 4:y + 5, x - 4:x + 5] / max(B, 1.0), 0.0, 1.0)
            if T.min() < 0.95:                                       # 뚜렷한 점만 쓴다
                bank[man[name]["machine"]].append(T.astype(np.float32))
    return bank


def product_masks(base):
    """제품 영역(어두운 쪽 가장 큰 덩어리)과 가장자리까지 거리."""
    blur = cv2.GaussianBlur(base, (0, 0), 3)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n < 2:
        return None, None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    return m, cv2.distanceTransform(m, cv2.DIST_L2, 3)


def choose_sites(rng, dist, spots, shape, n=3, min_edge=10, min_gap=25):
    h, w = shape
    cand = np.argwhere(dist >= min_edge)
    out = []
    if len(cand) == 0:
        return out
    for _ in range(n):
        for _try in range(300):
            y, x = cand[rng.integers(len(cand))]
            if min(x, y) < 12 or x > w - 13 or y > h - 13:
                continue
            if all(np.hypot(x - sx, y - sy) >= min_gap for sx, sy in list(spots) + out):
                out.append((int(x), int(y)))
                break
    return out


def make_patch(T, f, flip):
    if flip & 1:
        T = T[:, ::-1]
    if flip & 2:
        T = T[::-1, :]
    if flip & 4:
        T = T.T
    k = max(3, int(round(PATCH * f)) // 2 * 2 + 1)
    return np.clip(cv2.resize(np.ascontiguousarray(T), (k, k), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC), 0, 1)


def insert(img, x, y, T, s):
    k = T.shape[0] // 2
    y0, y1, x0, x1 = y - k, y + k + 1, x - k, x + k + 1
    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
        return False
    img[y0:y1, x0:x1] *= np.power(T, s)
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default="runs/minyeop/06_synth_insert_v1")
    ap.add_argument("--strengths", default="0.15,0.3,0.5,1.0")
    ap.add_argument("--scales", default="0.67,1.0,1.5")
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--r", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    S = [float(v) for v in a.strengths.split(",")]
    F = [float(v) for v in a.scales.split(",")]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    rd = load_module("remove_dot", DR / "remove_dot.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr_r50 = load_module("fr_r50", FR / "predict.py")
    fr_mob = load_module("fr_mob", FR / "predict.py")
    from mobilenet_model import build_model as mob_build  # noqa: E402
    fr_mob.build_model = mob_build
    yo = load_module("yolo_predict", YO / "predict.py")
    from utils import torch_utils  # noqa: E402
    runs = {"Faster R-CNN R50-FPN": (ROOT / "runs/minyeop/01_frcnn_r50fpn_min640_v1", 0.95),
            "YOLOv3-tiny": (ROOT / "runs/minyeop/02_yolov3tiny_img640_v1", 0.06),
            "Faster R-CNN MobileNetV3-FPN": (ROOT / "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", 0.93)}
    ycfg = json.loads((runs["YOLOv3-tiny"][0] / "config.json").read_text(encoding="utf-8"))
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}

    # 1) 점 은행 (val)
    bank = build_bank(rd, man)
    print("점 은행(val 실제 점):", {k: len(v) for k, v in bank.items()}, flush=True)

    # 2) 바탕: test 의 점을 평균 보간으로 지운 이미지 + 자리 고정
    gt = {r["name"]: r["gt"] for r in json.loads((runs["YOLOv3-tiny"][0] / "preds_test.json").read_text(encoding="utf-8"))}
    names = sorted(gt)
    rng = np.random.default_rng(a.seed)
    base, sites = {}, {}
    for n in names:
        src = ROOT / "data/test/images" / n
        if not gt[n]:
            base[n] = np.array(Image.open(src).convert("L"))
            continue
        im, spots, _, _ = rd.remove_dots(src, gt[n], 6, rng, 0.0, "mean")
        base[n] = im
        mask, dist = product_masks(im)
        mach = man[n]["machine"]
        pool = bank[mach]
        sl = [("orig", x, y) for x, y in spots]
        if dist is not None:
            sl += [("rand", x, y) for x, y in choose_sites(rng, dist, spots, im.shape, a.n_rand)]
        sites[n] = [{"site": t, "x": int(x), "y": int(y), "bank": int(rng.integers(len(pool))), "flip": int(rng.integers(8)),
                     "edge": float(dist[y, x]) if dist is not None else -1.0} for t, x, y in sl]
    n_orig = sum(1 for v in sites.values() for s in v if s["site"] == "orig")
    n_rand = sum(1 for v in sites.values() for s in v if s["site"] == "rand")
    print(f"바탕 {len(names)}장 | 넣을 점 자리: 원래 자리 {n_orig}개, 무작위 자리 {n_rand}개 (칸마다 같은 자리·같은 점 조각 사용)", flush=True)

    # 3) 칸 (s, f) 마다 합성 -> 세 모델 추론 -> 검출 판정
    rows = []
    fa_rows = []
    for s in S:
        for f in F:
            tag = f"s{s}_f{f}"
            work = Path(a.work_dir) / "synth" / tag
            if work.exists():
                shutil.rmtree(work)
            (work / "test" / "images").mkdir(parents=True)
            shutil.copytree(ROOT / "data/test/labels", work / "test" / "labels")
            meta = {}
            for n in names:
                img = base[n].astype(np.float32)
                placed = []
                for st in sites.get(n, []):
                    T = make_patch(bank[man[n]["machine"]][st["bank"]], f, st["flip"])
                    if insert(img, st["x"], st["y"], T, s):
                        placed.append(st)
                im8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
                Image.fromarray(im8, "L").save(work / "test" / "images" / n)
                blur = cv2.GaussianBlur(im8.astype(np.float32), (0, 0), 1.0)
                meta[n] = [(st, rd.contrast_at(im8.astype(np.float32), blur, st["x"], st["y"])[0]) for st in placed]
            recs = {}
            recs["Faster R-CNN R50-FPN"] = fr_r50.predict_split(runs["Faster R-CNN R50-FPN"][0], "test", "best", device, 4, data_dir=str(work))[0]
            recs["Faster R-CNN MobileNetV3-FPN"] = fr_mob.predict_split(runs["Faster R-CNN MobileNetV3-FPN"][0], "test", "best", device, 4, data_dir=str(work))[0]
            recs["YOLOv3-tiny"] = yo.predict_split(runs["YOLOv3-tiny"][0], "test", "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(work))[0]
            rm = {m: {r["name"]: r for r in rl} for m, rl in recs.items()}
            for n in names:
                for st, con in meta.get(n, []):
                    row = {"s": s, "f": f, "site": st["site"], "name": n, "machine": man[n]["machine"], "x": st["x"], "y": st["y"], "contrast": round(con, 2), "edge": round(st["edge"], 1)}
                    for m, (_, thr) in runs.items():
                        sc = [d[4] for d in rm[m][n]["dets"] if np.hypot(center(d)[0] - st["x"], center(d)[1] - st["y"]) <= a.r]
                        mxs = max(sc, default=0.0)
                        row[m] = round(mxs, 4)
                        row[m + "|det"] = int(mxs >= thr)
                    rows.append(row)
                for m, (_, thr) in runs.items():
                    pts = [(st["x"], st["y"]) for st, _ in meta.get(n, [])]
                    fa = sum(1 for d in rm[m][n]["dets"] if d[4] >= thr and all(np.hypot(center(d)[0] - px, center(d)[1] - py) > a.r for px, py in pts))
                    fa_rows.append({"s": s, "f": f, "model": m, "name": n, "fa": fa})
            line = []
            for m in runs:
                for site in ("orig", "rand"):
                    sel = [r for r in rows if r["s"] == s and r["f"] == f and r["site"] == site]
                    line.append(f"{m.split()[0] if 'R-CNN' not in m else ('R50' if 'R50' in m else 'MOB')}-{site} {100 * np.mean([r[m + '|det'] for r in sel]):5.1f}%")
            print(f"[s={s:<4} f={f:<4}] " + " | ".join(line), flush=True)

    # 4) 저장
    keys = list(rows[0])
    with open(out / "per_dot.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "false_alarm_per_image.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["s", "f", "model", "name", "fa"])
        w.writeheader()
        w.writerows(fa_rows)
    summary = {"설정": {"강도 s": S, "크기 배율 f": F, "무작위 자리 수/이미지": a.n_rand, "판정 반경(px)": a.r, "점 은행": {k: len(v) for k, v in bank.items()},
                      "넣은 자리": {"원래 자리": n_orig, "무작위 자리": n_rand}}, "검출률": {}}
    for m, (_, thr) in runs.items():
        summary["검출률"][m] = {"임계값": thr}
        for site in ("orig", "rand"):
            for s in S:
                for f in F:
                    sel = [r[m + "|det"] for r in rows if r["s"] == s and r["f"] == f and r["site"] == site]
                    lo, hi = wilson(sum(sel), len(sel))
                    summary["검출률"][m][f"{site}|s{s}|f{f}"] = {"n": len(sel), "검출률": float(np.mean(sel)), "구간": [lo, hi]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
