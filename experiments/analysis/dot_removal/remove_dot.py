"""test 이미지의 결함(어두운 점)을 보간으로 지운 뒤 두 모델에 다시 넣어, 점이 없어도 같은 자리에서 "결함"을 찾는지 본다.

  python src/minyeop/dot_removal/remove_dot.py --work-dir <임시 폴더> --out runs/minyeop/03_dot_removal_v1

목적: 모델이 결함 자체(어두운 점)를 보는지, 아니면 위치·주변(막대 끝, 지운 표시의 흔적 등)을 지름길로 쓰는지 가려낸다.
  - 점을 지웠는데도 같은 자리에서 높은 신뢰도로 검출 -> 지름길(주변 맥락) 의심
  - 점을 지우자 검출이 사라짐 -> 점 자체를 본다
지우는 방법(이전에 검증한 방식): 정답 중심 근처에서 가장 어두운 점을 찾아 그 주변 정사각형을 Navier-Stokes 보간으로 메우고,
  주변 링의 노이즈 표준편차 x0.75 의 가우시안 노이즈를 더해 메운 자리가 매끈해 보이지 않게 한다.
원본 data/ 는 바꾸지 않는다. 지운 이미지는 --work-dir 에만 만들고 Git 에 올리지 않는다 (결과 요약과 그림만 --out 에 저장).
주의: 보간 자국이 새 단서가 될 수 있다. 그래서 지운 이미지에서도 점의 대비가 얼마나 줄었는지 함께 잰다.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
FR, YO = ROOT / "src" / "minyeop" / "faster_rcnn", ROOT / "src" / "minyeop" / "yolov3_tiny"
sys.path.insert(0, str(FR))
import metrics  # noqa: E402
from visualize import font  # noqa: E402


def load_module(name, path):  # 두 predict.py 의 이름이 같아서 파일 경로로 따로 불러온다 (faster_rcnn 쪽을 먼저 불러야 train 모듈이 겹치지 않음)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def center(g):
    return (g[0] + g[2]) / 2, (g[1] + g[3]) / 2


def locate_dot(blur, g, win=5):
    """정답 중심 ±win px 안에서 가장 어두운 점의 (x, y)."""
    cx, cy = center(g)
    x0, y0 = int(round(cx)), int(round(cy))
    ys, xs = max(y0 - win, 0), max(x0 - win, 0)
    sub = blur[ys:y0 + win + 1, xs:x0 + win + 1]
    j = np.unravel_index(np.argmin(sub), sub.shape)
    return xs + int(j[1]), ys + int(j[0])


def contrast_at(im, blur, x, y):
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    d = np.maximum(abs(xx - x), abs(yy - y))
    ring = im[(d > 6) & (d <= 9)]
    return float(np.median(ring) - blur[y, x]), float(ring.std())


def remove_dots(img_path, gts, half, rng, noise_gain=0.75, method="ns"):
    im8 = np.array(Image.open(img_path).convert("L"))
    im = im8.astype(np.float32)
    blur = cv2.GaussianBlur(im, (0, 0), 1.0)
    spots = [locate_dot(blur, g) for g in gts]
    before = [contrast_at(im, blur, x, y) for x, y in spots]
    mask = np.zeros_like(im8)
    for x, y in spots:
        mask[max(y - half, 0):y + half + 1, max(x - half, 0):x + half + 1] = 255
    if method == "mean":  # 평균 보간: 지운 자리를 바로 바깥 3px 링의 평균 밝기 한 값으로 채운다 (데이터 전처리의 "주변 회색 메움"과 같은 계열이지만 같지는 않다: 전처리는 색 픽셀을 5x5 평균으로 반복해 채워 한 값이 아니다)
        out = im.copy()
        H, W = im.shape
        for x, y in spots:
            y0, y1, x0, x1 = max(y - half, 0), min(y + half + 1, H), max(x - half, 0), min(x + half + 1, W)
            Y0, Y1, X0, X1 = max(y - half - 3, 0), min(y + half + 4, H), max(x - half - 3, 0), min(x + half + 4, W)
            ring = np.ones((Y1 - Y0, X1 - X0), bool)
            ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
            ring &= mask[Y0:Y1, X0:X1] == 0  # 다른 점의 제거 영역은 평균에서 뺀다
            out[y0:y1, x0:x1] = im[Y0:Y1, X0:X1][ring].mean()
    else:  # ns: Navier-Stokes 보간
        out = cv2.inpaint(im8, mask, 3, cv2.INPAINT_NS).astype(np.float32)
    for (x, y), (_, sigma) in zip(spots, before):  # 메운 자리에 주변과 같은 세기의 노이즈를 더한다
        sl = (slice(max(y - half, 0), y + half + 1), slice(max(x - half, 0), x + half + 1))
        out[sl] += rng.normal(0, sigma * noise_gain, out[sl].shape)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    blur2 = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.0)
    after = [contrast_at(out.astype(np.float32), blur2, x, y)[0] for x, y in spots]
    return out, spots, [b[0] for b in before], after


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, help="지운 이미지를 둘 임시 폴더 (Git 에 올리지 않음)")
    ap.add_argument("--out", default="runs/minyeop/03_dot_removal_v1")
    ap.add_argument("--half", type=int, default=4, help="점 주변을 지울 정사각형의 반변 (4 -> 9x9px)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["ns", "mean"], default="ns", help="ns: Navier-Stokes 보간, mean: 주변 평균 보간")
    ap.add_argument("--noise-gain", type=float, default=0.75, help="메운 자리에 더할 노이즈 세기(주변 링 표준편차의 배수). 0이면 노이즈 없음")
    ap.add_argument("--r", type=float, default=5.0, help="지운 자리에서 검출을 '같은 자리'로 볼 중심 거리(px)")
    a = ap.parse_args()

    work = Path(a.work_dir) / "dotless"
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    # 1) 지운 test 이미지 만들기
    src = ROOT / "data" / "test"
    if work.exists():
        shutil.rmtree(work)
    (work / "test" / "images").mkdir(parents=True)
    shutil.copytree(src / "labels", work / "test" / "labels")
    gt = {r["name"]: r["gt"] for r in json.loads((ROOT / "runs/minyeop/01_frcnn_r50fpn_min640_v1/preds_test.json").read_text(encoding="utf-8"))}
    rng = np.random.default_rng(a.seed)
    c_before, c_after, spots_all, cb_by, ca_by = [], [], {}, {}, {}
    for p in sorted((src / "images").glob("*.png")):
        g = gt[p.name]
        if g:
            im, spots, cb, ca = remove_dots(p, g, a.half, rng, a.noise_gain, a.method)
            c_before += cb; c_after += ca; spots_all[p.name] = spots; cb_by[p.name] = cb; ca_by[p.name] = ca
            Image.fromarray(im, "L").save(work / "test" / "images" / p.name)
        else:
            shutil.copy(p, work / "test" / "images" / p.name)
    print(f"지운 점 {len(c_before)}개 | 점의 대비 중앙값: 지우기 전 {np.median(c_before):.1f} -> 후 {np.median(c_after):.1f} (0~255 밝기 차)")

    # 2) 두 모델로 다시 추론 (임계값은 원래 val 에서 정한 값 그대로)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr = load_module("frcnn_predict", FR / "predict.py")
    yo = load_module("yolo_predict", YO / "predict.py")
    models = {
        "Faster R-CNN": dict(run=ROOT / "runs/minyeop/01_frcnn_r50fpn_min640_v1", thr=0.95),
        "YOLOv3-tiny": dict(run=ROOT / "runs/minyeop/02_yolov3tiny_img640_v1", thr=0.06),
    }
    recs_new = {}
    recs_new["Faster R-CNN"], _, _ = fr.predict_split(models["Faster R-CNN"]["run"], "test", "best", device, 4, data_dir=str(work))
    ycfg = json.loads((models["YOLOv3-tiny"]["run"] / "config.json").read_text(encoding="utf-8"))
    from utils import torch_utils  # noqa: E402  (YOLO 쪽 utils)
    recs_new["YOLOv3-tiny"], _, _ = yo.predict_split(models["YOLOv3-tiny"]["run"], "test", "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(work))

    # 3) 지표: 원본 vs 지운 뒤
    perspot_models = {}
    summary = {"설정": {"half": a.half, "방법": a.method, "노이즈 배율": a.noise_gain, "같은 자리 판정 반경(px)": a.r, "지운 점 수": len(c_before),
                      "점 대비 중앙값(전→후)": [float(np.median(c_before)), float(np.median(c_after))]}, "모델": {}}
    for m, info in models.items():
        thr = info["thr"]
        orig = json.loads((info["run"] / "preds_test.json").read_text(encoding="utf-8"))
        new = recs_new[m]
        def stats(recs):
            tp = n_gt = 0
            near_conf = []
            per = {}
            n_det = 0
            for r in recs:
                n_det += sum(d[4] >= thr for d in r["dets"])
                for gi, gb in enumerate(r["gt"]):
                    n_gt += 1
                    cs = [d[4] for d in r["dets"] if np.hypot(center(d)[0] - center(gb)[0], center(d)[1] - center(gb)[1]) <= a.r]
                    near_conf.append(max(cs, default=0.0))
                    per[f"{r['name']}|{gi}"] = near_conf[-1]
            tp = int(sum(c >= thr for c in near_conf))
            return {"per": per, "정답 자리 검출 수": tp, "정답 수": n_gt, "정답 자리 검출률": tp / n_gt, "정답 자리 최고 신뢰도 중앙값": float(np.median(near_conf)),
                    "전체 검출 수(임계값 이상)": n_det, "near_conf": near_conf}
        so, sn = stats(orig), stats(new)
        perspot_models[m] = {"원본": so.pop("per"), "점 지운 뒤": sn.pop("per")}
        for s in (so, sn):
            s["near_conf_분위"] = {"10%": float(np.percentile(s["near_conf"], 10)), "50%": float(np.percentile(s["near_conf"], 50)), "90%": float(np.percentile(s["near_conf"], 90))}
            del s["near_conf"]
        summary["모델"][m] = {"임계값": thr, "원본": so, "점 지운 뒤": sn}
        print(f"\n[{m}] 임계값 {thr}")
        print(f"  원본      : 정답 자리 검출 {so['정답 자리 검출 수']}/{so['정답 수']} ({so['정답 자리 검출률']:.3f}), 신뢰도 중앙값 {so['정답 자리 최고 신뢰도 중앙값']:.3f}, 전체 검출 {so['전체 검출 수(임계값 이상)']}")
        print(f"  점 지운 뒤: 정답 자리 검출 {sn['정답 자리 검출 수']}/{sn['정답 수']} ({sn['정답 자리 검출률']:.3f}), 신뢰도 중앙값 {sn['정답 자리 최고 신뢰도 중앙값']:.3f}, 전체 검출 {sn['전체 검출 수(임계값 이상)']}")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 지점별 기록 (어디서 검출이 남았는지 분석용): 점 대비(전/후), 두 모델의 정답 자리 최고 신뢰도(전/후)
    spot_rows = []
    for nm_, cbs in cb_by.items():
        for gi in range(len(cbs)):
            k = f"{nm_}|{gi}"
            spot_rows.append({"name": nm_, "idx": gi, "대비_전": cbs[gi], "대비_후": ca_by[nm_][gi],
                              **{f"{m}|{w}": perspot_models[m][w][k] for m in models for w in ("원본", "점 지운 뒤")}})
    (out / "per_spot.json").write_text(json.dumps(spot_rows, ensure_ascii=False), encoding="utf-8")

    # 4) 그림: 호기별로 예시 (원본 / 지운 뒤 + R-CNN / 지운 뒤 + YOLO)
    man = {}
    import csv
    for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")):
        man[r["image_id"] + ".png"] = r["machine"]
    pick, seen = [], {}
    for name in sorted(spots_all):
        mc = man[name]
        if seen.get(mc, 0) < 2:
            seen[mc] = seen.get(mc, 0) + 1
            pick.append(name)
    S, W = 8, 44
    rowsimg = []
    for name in pick:
        panels = []
        x, y = spots_all[name][0]
        x1 = int(min(max(x - W // 2, 0), 10 ** 6)); y1 = int(max(y - W // 2, 0))
        for title, imgp, m in (("원본 (정답 초록)", ROOT / "data/test/images" / name, None), ("점 지운 뒤 + Faster R-CNN", work / "test/images" / name, "Faster R-CNN"),
                               ("점 지운 뒤 + YOLOv3-tiny", work / "test/images" / name, "YOLOv3-tiny")):
            im = Image.open(imgp).convert("RGB")
            x1c = min(x1, im.width - W); y1c = min(y1, im.height - W)
            crop = im.crop((x1c, y1c, x1c + W, y1c + W)).resize((W * S, W * S), Image.NEAREST)
            d = ImageDraw.Draw(crop)
            tr = lambda b: [(b[0] - x1c) * S, (b[1] - y1c) * S, (b[2] - x1c) * S, (b[3] - y1c) * S]
            for g in gt[name]:
                d.rectangle(tr(g), outline=(40, 200, 60), width=2)
            if m:
                rec = next(r for r in recs_new[m] if r["name"] == name)
                for dd in rec["dets"]:
                    if dd[4] >= models[m]["thr"]:
                        d.rectangle(tr(dd[:4]), outline=(235, 50, 50), width=2)
                        d.text((tr(dd)[0], max(tr(dd)[1] - 12, 0)), f"{dd[4]:.2f}", fill=(235, 50, 50), font=font(12))
            panels.append((title, crop))
        rowsimg.append((f"{man[name]}호기 | {name[:-4]}", panels))
    pw = W * S
    canvas = Image.new("RGB", (3 * (pw + 6) + 6, 30 + len(rowsimg) * (pw + 24)), (30, 30, 30))
    dd = ImageDraw.Draw(canvas)
    dd.text((6, 6), "test 점 지우기 실험: 점을 지워도 같은 자리에서 검출이 남는가 (초록=정답 위치, 빨강=검출·신뢰도)", fill=(255, 255, 255), font=font(14))
    for i, (cap, panels) in enumerate(rowsimg):
        yy = 30 + i * (pw + 24)
        dd.text((6, yy), cap, fill=(220, 220, 220), font=font(12))
        for j, (t, crop) in enumerate(panels):
            canvas.paste(crop, (6 + j * (pw + 6), yy + 18))
            dd.text((10 + j * (pw + 6), yy + 20), t, fill=(255, 255, 0), font=font(12))
    canvas.save(out / "figures" / "before_after.png")
    print("\n저장:", out)


if __name__ == "__main__":
    main()
