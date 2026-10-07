"""호기·해상도별 결과 그림: ① 그룹별 지표 막대 그래프, ② 그룹마다 같은 이미지에 세 모델의 검출을 그린 예시.

  python src/minyeop/group_stats/figures.py --split test

입력: group_stats.py 가 만든 runs/minyeop/07_group_stats_v1/group_stats_<split>.csv 와 각 모델의 preds_<split>.json
출력: runs/minyeop/07_group_stats_v1/figures/{group_chart,group_samples}_<split>.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
from visualize import draw, font, load_img  # noqa: E402

RUNS = {"Faster R-CNN R50-FPN": ("runs/minyeop/01_frcnn_r50fpn_min640_v1", 0.95, "R50"),
        "YOLOv3-tiny": ("runs/minyeop/02_yolov3tiny_img640_v1", 0.06, "YOLO"),
        "Faster R-CNN MobileNetV3-FPN": ("runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", 0.93, "MobileNet")}
GROUPS = [("전체", "전체"), ("호기 1 · 316x332", "1호기\n316x332"), ("호기 1 · 352x332", "1호기\n352x332"), ("호기 1 · 412x332", "1호기\n412x332*"),
          ("호기 2 · 316x332", "2호기\n316x332"), ("호기 3 · 576x444", "3호기\n576x444")]
COLORS = ["#3b6fd4", "#e8912d", "#4aa564"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="runs/minyeop/07_group_stats_v1")
    a = ap.parse_args()
    out = ROOT / a.out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("C:/Windows/Fonts/malgun.ttf",):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    rows = list(csv.DictReader(open(ROOT / a.out / f"group_stats_{a.split}.csv", encoding="utf-8")))
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    recs = {m: json.loads((ROOT / run / f"preds_{a.split}.json").read_text(encoding="utf-8")) for m, (run, _, _) in RUNS.items()}

    # 그룹별 공식 라벨 박스 비율 (교란 요인이므로 그래프에 병기)
    share = {}
    for r in recs["YOLOv3-tiny"]:
        mi = man[r["name"]]
        key = f"호기 {mi['machine']} · {mi['width']}x{mi['height']}"
        d = share.setdefault(key, [0, 0])
        d[1] += len(r["gt"])
        d[0] += len(r["gt"]) if mi["source"] == "official" else 0
    tot = [sum(v[0] for v in share.values()), sum(v[1] for v in share.values())]
    share["전체"] = tot

    # ① 막대 그래프
    panels = [("mAP50-95", "mAP50-95 (IoU 0.5~0.95)"), ("F1", "F1 (IoU 0.5) · 세로축 0.90부터"), ("F1_v2_R2", "F1 (평가 v2, 중심 2px) · 세로축 0.90부터")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (col, title) in zip(axes, panels):
        for i, m in enumerate(RUNS):
            vals = []
            for g, _ in GROUPS:
                r = next((x for x in rows if x["모델"] == m and x["그룹"] == g), None)
                vals.append(float(r[col]) if r else 0)
            xs = [j + (i - 1) * 0.26 for j in range(len(GROUPS))]
            ax.bar(xs, vals, width=0.25, color=COLORS[i], label=RUNS[m][2])
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{lab}\n공식 {100 * share[g][0] / max(share[g][1], 1):.0f}%" for g, lab in GROUPS], fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        lo = 0.0 if col == "mAP50-95" else 0.9
        ax.set_ylim(lo, 1.0 if col != "mAP50-95" else 0.9)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"호기·해상도별 성능 ({a.split}, 모델별 val 고정 임계값). x축 아래 '공식 N%' = 그 그룹의 공식 라벨 박스 비율(라벨 크기가 제각각이라 IoU 점수가 낮아짐). *412x332는 박스 2개뿐", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"group_chart_{a.split}.png", dpi=110)
    plt.close(fig)

    # ② 그룹마다 같은 이미지, 모델 3개 검출
    picks, best = {}, {}
    for r in sorted(recs["YOLOv3-tiny"], key=lambda x: x["name"]):
        mi = man[r["name"]]
        key = f"호기 {mi['machine']} · {mi['width']}x{mi['height']}"
        if len(r["gt"]) > best.get(key, 0):      # 그룹마다 정답이 가장 많은 이미지(같으면 이름순 첫 번째)를 고른다
            best[key] = len(r["gt"])
            picks[key] = r["name"]
    order = [g for g, _ in GROUPS if g in picks]
    tiles = []
    for g in order:
        n = picks[g]
        row = []
        for m, (run, thr, short) in RUNS.items():
            rec = next(x for x in recs[m] if x["name"] == n)
            row.append((short, draw(load_img(a.split, n), rec, thr, 1)))
        tiles.append((g, n, row))
    pad, cap = 8, 20
    colw = [max(t[2][j][1].width for t in tiles) for j in range(3)]
    rowh = [max(im.height for _, im in t[2]) for t in tiles]
    canvas = Image.new("RGB", (sum(colw) + pad * 4, 30 + sum(rowh) + (cap + pad) * len(tiles)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 7), f"그룹별 예시 ({a.split}): 같은 이미지에 세 모델의 검출 (초록 = 정답, 빨강 = 검출·신뢰도, 모델별 val 임계값)", fill=(255, 255, 255), font=font(14))
    y = 30
    for (g, n, row), rh in zip(tiles, rowh):
        x = pad
        for j, (short, im) in enumerate(row):
            d.text((x, y), f"{g} | {short} | {n[:-4][:24]}", fill=(255, 255, 0), font=font(12))
            canvas.paste(im, (x, y + cap))
            x += colw[j] + pad
        y += rh + cap + pad
    canvas.save(out / f"group_samples_{a.split}.png")
    print("저장:", out / f"group_chart_{a.split}.png", "|", out / f"group_samples_{a.split}.png", canvas.size)
    print("선택한 이미지:", picks)


if __name__ == "__main__":
    main()
