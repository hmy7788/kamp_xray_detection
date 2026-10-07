"""발표자료용 예시 사진·그래프를 만든다 (ppt/figures/ 에 저장).

  python ppt/make_example_figures.py            # PYTHONUTF8=1
만드는 것
  fig5a_preprocess.png     전처리 전후: 원본 BMP(장비의 색 윤곽) -> 색 표시 제거 후 (호기마다 1장, 확대)
  fig5b_detection_examples.png  실제 test 이미지에서 6개 모델의 검출 (호기마다 1장 + IoU 미달 예시 1장)
  fig5c_fake_examples.png  가짜 정상 예시: 원본 / 평균 보간으로 점을 지움 / NS+노이즈로 점을 지움, 모델별 반응
  fig5d_fn_by_model.png    모델별 못 찾은 개수: IoU 0.5 / 중심 5px / 중심 2px
  fig5e_select_ci.png      val mAP50-95 와 95% 신뢰구간(모델 선정)
  fig3n_synth_dots.png     합성 점 조건별 확대 (data_synth/test/preview_dots.png 복사)
  fig6a_label_whole.png    라벨 예시: 이미지 전체에 박스(공식 라벨 / 팀 라벨, 1·3호기)
  fig6b_label_zoom.png     박스 확대: 공식 라벨은 크기가 제각각, 팀 라벨은 고정 크기
데이터 이미지가 그려진 그림이라 저장소가 비공개인 동안에만 Git 에 둔다. 원본 BMP 가 있는 PC 에서만 fig5a 를 만들 수 있다.
fig5c 의 점 제거는 이미지마다 새 난수(시드 0)로 만든 예시라 robust_extra.py 의 집계 이미지와 노이즈 값이 다르다(예시 용도).
"""
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ppt" / "figures"
RUN = ROOT / "runs" / "minyeop"
EX = RUN / "09_extra_models_v1"
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn_mobilenet"))
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "dot_removal"))
import metrics  # noqa: E402

MODELS = ["R50-FPN", "YOLOv3-tiny", "MobileNetV3", "YOLO26n", "RT-DETR-l", "D-FINE-N"]
KEY = {"R50-FPN": "Faster R-CNN R50-FPN", "YOLOv3-tiny": "YOLOv3-tiny", "MobileNetV3": "Faster R-CNN MobileNetV3-FPN",
       "YOLO26n": "YOLO26n", "RT-DETR-l": "RT-DETR-l", "D-FINE-N": "D-FINE-N"}
PRED = {"R50-FPN": RUN / "01_frcnn_r50fpn_min640_v1/preds_test.json", "YOLOv3-tiny": RUN / "02_yolov3tiny_img640_v1/preds_test.json",
        "MobileNetV3": RUN / "04_frcnn_mobv3_min640_anc16_v1/preds_test.json", "YOLO26n": EX / "yolo26n/preds_test.json",
        "RT-DETR-l": EX / "rtdetr_l/preds_test.json", "D-FINE-N": EX / "dfine_n/preds_test.json"}
COL = {"R50-FPN": (59, 111, 212), "YOLOv3-tiny": (232, 145, 45), "MobileNetV3": (74, 165, 100),
       "YOLO26n": (214, 96, 77), "RT-DETR-l": (142, 107, 191), "D-FINE-N": (42, 167, 184)}
GREEN = (40, 200, 60)
FONT = "C:/Windows/Fonts/malgun.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


MAN = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
THR = {m: load_json(EX / "summary.json")["실제 val·test"][KEY[m]]["임계값(val, IoU0.5)"] for m in MODELS}
RECS = {m: {r["name"]: r for r in load_json(PRED[m])} for m in MODELS}
NAMES = sorted(RECS["YOLOv3-tiny"])


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def dets_at(m, name):
    return [d for d in RECS[m][name]["dets"] if d[4] >= THR[m]]


def crop_tile(img, cx, cy, half, scale):
    """회색조 PIL 이미지를 (cx, cy) 중심 2*half 정사각형으로 잘라 scale 배 확대(NEAREST)."""
    x0 = int(round(min(max(cx - half, 0), img.width - 2 * half)))
    y0 = int(round(min(max(cy - half, 0), img.height - 2 * half)))
    t = img.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).convert("RGB").resize((2 * half * scale, 2 * half * scale), Image.NEAREST)
    return t, x0, y0


def box_on(draw, b, x0, y0, scale, color, width=2):
    draw.rectangle([(b[0] - x0) * scale, (b[1] - y0) * scale, (b[2] - x0) * scale, (b[3] - y0) * scale], outline=color, width=width)


# ------------------------------------------------------------------ fig5a 전처리 전후
def fig5a():
    raw_root = ROOT / "4. X-ray 검사장비 AI 데이터셋" / "dataset" / "test1" / "yolov3"
    index = defaultdict(list)
    for dp, _dn, fn in os.walk(raw_root):
        for f in fn:
            if f.lower().endswith(".bmp"):
                index[f[:-4]].append(Path(dp) / f)
    rows = []
    for mc in ("1", "3"):
        for n in NAMES:
            iid = n[:-4]
            if MAN[iid]["machine"] == mc and len(RECS["YOLOv3-tiny"][n]["gt"]) >= 1 and iid in index and "__m" not in iid and MAN[iid]["source"] == "official":
                rows.append((mc, n))
                break
    half, scale = 22, 7
    tiles = []
    for mc, n in rows:
        iid = n[:-4]
        proc = Image.open(ROOT / "data" / MAN[iid]["image_path"]).convert("L")
        cands = [p for p in index[iid] if Image.open(p).size == proc.size]
        raw = Image.open(cands[0]).convert("RGB")
        g = RECS["YOLOv3-tiny"][n]["gt"][0]
        cx, cy = center(g)
        x0 = int(round(min(max(cx - half, 0), raw.width - 2 * half)))
        y0 = int(round(min(max(cy - half, 0), raw.height - 2 * half)))
        a = raw.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize((2 * half * scale,) * 2, Image.NEAREST)
        b = proc.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).convert("RGB").resize((2 * half * scale,) * 2, Image.NEAREST)
        whole = raw.copy()
        d = ImageDraw.Draw(whole)
        d.rectangle([x0 - 2, y0 - 2, x0 + 2 * half + 2, y0 + 2 * half + 2], outline=(255, 220, 0), width=2)
        s = 2 * half * scale / whole.height
        whole = whole.resize((int(whole.width * s), 2 * half * scale), Image.LANCZOS)
        tiles.append((mc, iid, whole, a, b))
    W = max(t[2].width for t in tiles) + 2 * 2 * half * scale + 4 * 12
    H = 40 + len(tiles) * (2 * half * scale + 40)
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), "전처리: 장비가 그린 색 윤곽(원본 BMP)을 주변 회색 5x5 평균으로 메운다. 점(가운데)은 그대로 남는다", fill=(255, 255, 255), font=font(18))
    y = 40
    for mc, iid, whole, a, b in tiles:
        x = 12
        for img, cap in ((whole, f"{mc}호기 원본 (노랑 = 확대 영역)"), (a, "원본 확대: 색 윤곽"), (b, "전처리 후 (data/의 PNG)")):
            d.text((x, y), cap, fill=(255, 255, 0), font=font(15))
            canvas.paste(img, (x, y + 24))
            x += img.width + 12
        y += 2 * half * scale + 40
    canvas.save(OUT / "fig5a_preprocess.png")
    print("저장 fig5a_preprocess.png")


# ------------------------------------------------------------------ fig5b 검출 예시
def fig5b():
    sel = []
    for mc in ("1", "2", "3"):
        for n in NAMES:
            iid = n[:-4]
            gt = RECS["YOLOv3-tiny"][n]["gt"]
            if MAN[iid]["machine"] != mc or len(gt) != 1:
                continue
            ok = True
            for m in MODELS:
                ds = dets_at(m, n)
                if not any(np.hypot(center(d)[0] - center(gt[0])[0], center(d)[1] - center(gt[0])[1]) <= 2 and metrics.iou_matrix([d[:4]], gt)[0, 0] >= 0.5 for d in ds):
                    ok = False
                    break
            if ok:
                sel.append((f"{mc}호기: 6개 모델이 모두 찾음", n))
                break
    for n in NAMES:                      # IoU 미달 예: 공식 라벨, 모든 모델이 중심 3px 이내로 찾았지만 3개 이상이 IoU 0.5 미달
        iid = n[:-4]
        gt = RECS["YOLOv3-tiny"][n]["gt"]
        if MAN[iid]["source"] != "official" or len(gt) < 1:
            continue
        low, found = 0, 0
        for m in MODELS:
            ds = [d for d in dets_at(m, n) if np.hypot(center(d)[0] - center(gt[0])[0], center(d)[1] - center(gt[0])[1]) <= 3]
            if not ds:
                continue
            found += 1
            if max(metrics.iou_matrix([d[:4]], gt[:1])[0, 0] for d in ds) < 0.5:
                low += 1
        if found >= 5 and low >= 3:
            sel.append(("IoU 미달 예: 점은 찾았지만 정답 박스 크기와 달라 IoU 0.5 미만", n))
            break
    half, scale = 20, 6
    tw = 2 * half * scale
    canvas = Image.new("RGB", (len(MODELS) * (tw + 8) + 8, 70 + len(sel) * (tw + 40)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), "실제 test 이미지의 검출 예시 (초록 = 정답, 색 = 모델의 검출과 신뢰도)", fill=(255, 255, 255), font=font(18))
    for j, m in enumerate(MODELS):
        d.text((10 + j * (tw + 8), 38), m, fill=COL[m], font=font(16))
    y = 70
    for cap, n in sel:
        d.text((10, y), cap + "  |  " + n[:-4][:24], fill=(255, 255, 0), font=font(14))
        img = Image.open(ROOT / "data" / MAN[n[:-4]]["image_path"]).convert("L")
        gt = RECS["YOLOv3-tiny"][n]["gt"][0]
        for j, m in enumerate(MODELS):
            t, x0, y0 = crop_tile(img, *center(gt), half, scale)
            td = ImageDraw.Draw(t)
            box_on(td, gt, x0, y0, scale, GREEN, 2)
            for dd in dets_at(m, n):
                if abs(center(dd)[0] - center(gt)[0]) < half and abs(center(dd)[1] - center(gt)[1]) < half:
                    box_on(td, dd, x0, y0, scale, COL[m], 2)
                    tx, ty = (dd[0] - x0) * scale, max((dd[1] - y0) * scale - 18, 0)
                    td.rectangle([tx, ty, tx + 38, ty + 17], fill=(20, 20, 20))
                    td.text((tx + 2, ty), f"{dd[4]:.2f}", fill=(255, 255, 255), font=font(14))
            canvas.paste(t, (8 + j * (tw + 8), y + 22))
        y += tw + 40
    canvas.save(OUT / "fig5b_detection_examples.png")
    print("저장 fig5b_detection_examples.png", [c for c, _ in sel])


# ------------------------------------------------------------------ fig5c 가짜 정상 예시
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fig5c():
    rd = load_module("remove_dot", ROOT / "src" / "minyeop" / "dot_removal" / "remove_dot.py")
    px = load_module("predict_extra", ROOT / "src" / "minyeop" / "extra_models" / "predict_extra.py")
    FR, YO = ROOT / "src" / "minyeop" / "faster_rcnn", ROOT / "src" / "minyeop" / "yolov3_tiny"
    fr_r50 = load_module("fr_r50", FR / "predict.py")
    fr_mob = load_module("fr_mob", FR / "predict.py")
    from mobilenet_model import build_model as mob_build
    fr_mob.build_model = mob_build
    yo = load_module("yolo_predict", YO / "predict.py")
    from utils import torch_utils
    ycfg = load_json(RUN / "02_yolov3tiny_img640_v1/config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    picks = []
    for mc in ("3", "1"):
        for n in NAMES:
            if MAN[n[:-4]]["machine"] == mc and len(RECS["YOLOv3-tiny"][n]["gt"]) == 1 and MAN[n[:-4]]["source"] == "official":
                picks.append(n)
                break
    versions = [("원본", None, 0.0), ("평균 보간으로 점을 지움", "mean", 0.0), ("NS 보간 + 노이즈로 점을 지움", "ns", 0.75)]
    half, scale = 24, 6
    tw = 2 * half * scale
    canvas = Image.new("RGB", (len(versions) * (tw + 10) + 10, 80 + len(picks) * (tw + 70)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), "가짜 정상 예시: 점을 지운 이미지에서 모델이 반응하는가 (모델마다 val 임계값 이상의 검출을 색 상자로 표시)", fill=(255, 255, 255), font=font(18))
    x = 10
    for mdl in MODELS:
        d.text((x, 40), mdl, fill=COL[mdl], font=font(15))
        x += 150
    tmp = Path(tempfile.mkdtemp())
    try:
        for vi, (vt, method, noise) in enumerate(versions):
            work = tmp / f"v{vi}"
            (work / "test" / "images").mkdir(parents=True)
            (work / "test" / "labels").mkdir(parents=True)
            for n in picks:
                src = ROOT / "data" / "test" / "images" / n
                gt = RECS["YOLOv3-tiny"][n]["gt"]
                if method is None:
                    shutil.copy(src, work / "test" / "images" / n)
                else:
                    im, *_ = rd.remove_dots(src, gt, 6, np.random.default_rng(0), noise, method)
                    Image.fromarray(im, "L").save(work / "test" / "images" / n)
                shutil.copy(ROOT / "data" / "test" / "labels" / (n[:-4] + ".txt"), work / "test" / "labels" / (n[:-4] + ".txt"))
            res = {}
            res["R50-FPN"] = fr_r50.predict_split(RUN / "01_frcnn_r50fpn_min640_v1", "test", "best", device, 4, data_dir=str(work))[0]
            res["MobileNetV3"] = fr_mob.predict_split(RUN / "04_frcnn_mobv3_min640_anc16_v1", "test", "best", device, 4, data_dir=str(work))[0]
            res["YOLOv3-tiny"] = yo.predict_split(RUN / "02_yolov3tiny_img640_v1", "test", "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(work))[0]
            paths = sorted((work / "test" / "images").glob("*.png"))
            imgs = px.load_images(paths)
            for mdl, key in (("YOLO26n", "yolo26n"), ("RT-DETR-l", "rtdetr_l"), ("D-FINE-N", "dfine_n")):
                spec = px.MODELS[key]
                dets = (px.run_ultra if spec["kind"] == "ultra" else px.run_dfine)(spec, imgs, "cuda:0" if torch.cuda.is_available() else "cpu", 8)
                res[mdl] = [{"name": p.name, "dets": dd} for p, dd in zip(paths, dets)]
            for ri, n in enumerate(picks):
                gt = RECS["YOLOv3-tiny"][n]["gt"][0]
                img = Image.open(work / "test" / "images" / n).convert("L")
                t, x0, y0 = crop_tile(img, *center(gt), half, scale)
                td = ImageDraw.Draw(t)
                box_on(td, gt, x0, y0, scale, GREEN, 1)
                fired = []
                for mdl in MODELS:
                    rec = next(r for r in res[mdl] if r["name"] == n)
                    ds = [dd for dd in rec["dets"] if dd[4] >= THR[mdl] and abs(center(dd)[0] - center(gt)[0]) < half and abs(center(dd)[1] - center(gt)[1]) < half]
                    if ds:
                        fired.append(mdl)
                        for dd in ds:
                            box_on(td, dd, x0, y0, scale, COL[mdl], 2)
                ox, oy = 10 + vi * (tw + 10), 80 + ri * (tw + 70)
                d.text((ox, oy), f"{MAN[n[:-4]]['machine']}호기 | {vt}", fill=(255, 255, 0), font=font(15))
                canvas.paste(t, (ox, oy + 24))
                label = "6개 모델 모두" if len(fired) == len(MODELS) else (", ".join(fired) if fired else "없음")
                words, lines, cur = label.split(", "), [], ""
                for w_ in words:
                    if len(cur) + len(w_) > 30 and cur:
                        lines.append(cur)
                        cur = ""
                    cur += (", " if cur else "") + w_
                lines.append(cur)
                for li, ln in enumerate(lines[:3]):
                    d.text((ox, oy + tw + 28 + li * 17), ("반응: " if li == 0 else "        ") + ln, fill=(255, 255, 255), font=font(13))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    canvas.save(OUT / "fig5c_fake_examples.png")
    print("저장 fig5c_fake_examples.png")


# ------------------------------------------------------------------ 그래프
def setup_font():
    if Path(FONT).exists():
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def fig5d():
    setup_font()
    res = {}
    for m in MODELS:
        fn = {None: 0, 5.0: 0, 2.0: 0}
        for n in NAMES:
            for cr in fn:
                t, c, nm = metrics.match_image(RECS[m][n], THR[m], 0.5, cr)
                fn[cr] += len(RECS[m][n]["gt"]) - nm
        res[m] = fn
    x = np.arange(len(MODELS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    series = [(None, "IoU 0.5 기준", "#c0504d"), (2.0, "중심 2px 기준", "#e8a33d"), (5.0, "중심 5px 기준", "#2aa7b8")]
    for k, (cr, lab, color) in enumerate(series):
        vals = [res[m][cr] for m in MODELS]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=color, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(v), ha="center", fontsize=10)
    ax.set_xticks(x, MODELS, fontsize=11)
    ax.set_ylabel("못 찾은 정답 수 (정답 663개 중)")
    ax.set_title("매칭 기준에 따라 못 찾은 개수가 달라진다: 중심 5px로 보면 거의 0", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(res[m][2.0] for m in MODELS) + 4)
    fig.tight_layout()
    fig.savefig(OUT / "fig5d_fn_by_model.png", dpi=150, facecolor="white")
    print("저장 fig5d_fn_by_model.png", {m: {str(k): v for k, v in r.items()} for m, r in res.items()})


def fig5e():
    setup_font()
    B = load_json(EX / "bootstrap_val_map.json")["결과"]
    colors = {"R50": "#3b6fd4", "YOLOv3-tiny": "#e8912d", "MobileNetV3": "#4aa564", "YOLO26n": "#d6604d", "RT-DETR-l": "#8e6bbf", "D-FINE-N": "#2aa7b8"}
    kmap = {"R50": "R50", "YOLOv3-tiny": "YOLOv3-tiny", "MobileNetV3": "MobileNetV3", "YOLO26n": "YOLO26n", "RT-DETR-l": "RT-DETR-l", "D-FINE-N": "D-FINE-N"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, sname in zip(axes, ("전체 val", "공식 라벨 이미지만")):
        d = B[sname]["모델"]
        order = sorted(d, key=lambda k: d[k]["mAP50-95"])
        for i, m in enumerate(order):
            v = d[m]
            ax.plot(v["mAP50-95"], i, "o", color=colors[m], markersize=12, zorder=3)
            ax.plot(v["구간"], [i, i], color="black", lw=2)
            ax.plot([v["구간"][0]] * 2, [i - 0.15, i + 0.15], color="black", lw=2)
            ax.plot([v["구간"][1]] * 2, [i - 0.15, i + 0.15], color="black", lw=2)
            ax.text(v["구간"][1] + 0.004, i, f"{v['mAP50-95']:.3f}", va="center", fontsize=10)
        ax.set_yticks(range(len(order)), order, fontsize=11)
        lo = min(d[m]["구간"][0] for m in d)
        ax.set_xlim(lo - 0.03, max(d[m]["구간"][1] for m in d) + 0.05)
        ax.set_xlabel("val mAP50-95 (검은 선 = 95% 신뢰구간)")
        n_img = B[sname]["이미지"]
        ax.set_title(f"{sname} ({n_img}장)", fontsize=12)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("RT-DETR-l과 D-FINE-N이 동률 선두 (전체 val). 공식 라벨만 보면 구간이 넓어 R50도 동률", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig5e_select_ci.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("저장 fig5e_select_ci.png")


# ------------------------------------------------------------------ fig6 라벨 예시 (공식 라벨 / 팀 라벨)
def _label_boxes(row):
    """manifest 한 행의 라벨을 픽셀 xyxy 박스 목록으로."""
    w, h = int(row["width"]), int(row["height"])
    out = []
    for ln in (ROOT / "data" / row["label_path"]).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            _, cx, cy, bw, bh = (float(v) for v in ln.split()[:5])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def fig6a():
    """호기(1, 3) x 라벨 출처(공식, 팀)의 이미지 전체와 박스."""
    rows = list(MAN.values())
    picks = {}
    for mc in ("1", "3"):
        for src in ("official", "team"):
            for r in sorted(rows, key=lambda x: x["image_id"]):
                if r["machine"] == mc and r["source"] == src and r["split"] == "test" and len(_label_boxes(r)) == 3 and (mc != "1" or r["width"] == "352"):
                    picks[(mc, src)] = r
                    break
    H = 360
    cells = []
    for (mc, src), r in picks.items():
        img = Image.open(ROOT / "data" / r["image_path"]).convert("RGB")
        s = H / img.height
        big = img.resize((int(img.width * s), H), Image.LANCZOS)
        d = ImageDraw.Draw(big)
        color = (60, 220, 90) if src == "official" else (240, 60, 60)
        for b in _label_boxes(r):
            d.rectangle([b[0] * s - 1, b[1] * s - 1, b[2] * s + 1, b[3] * s + 1], outline=color, width=2)
        cells.append(((mc, src), big, len(_label_boxes(r))))
    cw = max(c[1].width for c in cells)
    canvas = Image.new("RGB", (2 * (cw + 14) + 14, 40 + 2 * (H + 34)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((14, 8), "라벨 예시: 이미지 전체에 박스를 그린 모습 (test 이미지)", fill=(255, 255, 255), font=font(18))
    for (mc, src), big, n in cells:
        col = 0 if src == "official" else 1
        row = 0 if mc == "1" else 1
        x = 14 + col * (cw + 14)
        y = 40 + row * (H + 34)
        name = "공식 라벨 (사람이 그림)" if src == "official" else "팀 라벨 (클릭 → 고정 크기 박스)"
        d.text((x, y), f"{mc}호기 | {name} | 박스 {n}개", fill=(255, 255, 0), font=font(15))
        canvas.paste(big, (x, y + 26))
    canvas.save(OUT / "fig6a_label_whole.png")
    print("저장 fig6a_label_whole.png")


def fig6b():
    """박스 확대: 공식(크기 제각각) 4개, 팀(고정 크기) 4개."""
    rows = [r for r in MAN.values()]
    boxes = {"official": [], "team": []}
    for r in sorted(rows, key=lambda x: x["image_id"]):
        w, h = int(r["width"]), int(r["height"])
        for b in _label_boxes(r):
            side = max(b[2] - b[0], b[3] - b[1])
            cx, cy = center(b)
            if 24 <= cx <= w - 24 and 24 <= cy <= h - 24:
                boxes[r["source"]].append((side, r["machine"], r, b))

    def pick(src, machine, target):
        c = [x for x in boxes[src] if x[1] == machine]
        return min(c, key=lambda x: (abs(x[0] - target), x[2]["image_id"]))

    off = [pick("official", "1", 6), pick("official", "2", 11), pick("official", "3", 14), pick("official", "3", 21)]
    def pick_nth(src, machine, target, n):
        c = sorted((x for x in boxes[src] if x[1] == machine and abs(x[0] - target) < 0.3), key=lambda x: x[2]["image_id"])
        return c[min(n, len(c) - 1)]

    team = [pick("team", "1", 10), pick("team", "2", 10), pick("team", "3", 13.4), pick_nth("team", "3", 13.4, 400)]
    half, scale = 20, 7
    tw = 2 * half * scale
    canvas = Image.new("RGB", (4 * (tw + 10) + 10, 50 + 2 * (tw + 62)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), "박스 확대(40x40px 를 7배): 공식 라벨은 크기가 제각각, 팀 라벨은 고정 크기", fill=(255, 255, 255), font=font(18))
    for ri, (title, group, color) in enumerate((("공식 라벨", off, (60, 220, 90)), ("팀 라벨", team, (240, 60, 60)))):
        y = 50 + ri * (tw + 62)
        for ci, (side, mc, r, b) in enumerate(group):
            img = Image.open(ROOT / "data" / r["image_path"]).convert("L")
            cx, cy = center(b)
            t_, x0, y0 = crop_tile(img, cx, cy, half, scale)
            td = ImageDraw.Draw(t_)
            box_on(td, b, x0, y0, scale, color, 2)
            x = 10 + ci * (tw + 10)
            d.text((x, y), f"{title} | {mc}호기 | 한 변 {side:.1f}px", fill=(255, 255, 0), font=font(15))
            canvas.paste(t_, (x, y + 26))
            d.text((x, y + tw + 30), r["image_id"][:26], fill=(180, 180, 180), font=font(12))
    canvas.save(OUT / "fig6b_label_zoom.png")
    print("저장 fig6b_label_zoom.png", [(round(x[0], 1), x[1]) for x in off], [(round(x[0], 1), x[1]) for x in team])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    which = set(sys.argv[1:]) or {"a", "b", "c", "d", "e", "n", "f", "g"}
    if "a" in which:
        fig5a()
    if "b" in which:
        fig5b()
    if "c" in which:
        fig5c()
    if "d" in which:
        fig5d()
    if "e" in which:
        fig5e()
    if "f" in which:
        fig6a()
    if "g" in which:
        fig6b()
    if "n" in which:
        src = ROOT / "data_synth" / "test" / "preview_dots.png"
        if src.exists():
            shutil.copy(src, OUT / "fig3n_synth_dots.png")
            print("복사 fig3n_synth_dots.png")
