"""내가 라벨링할 사진을 한 폴더에 모은다 (Label Studio 에 올릴 용도).

python data/labeling/export_for_labeling.py --member kim [--version v1_fill]

만드는 것 (git 에 올라가지 않는다):
  data/labeling/work/<member>/images/NNNN__<image_id>.png   라벨링할 사진 (묶음 순서대로 번호)
  data/labeling/work/<member>/reference/                    공식 라벨 예시 12장 (초록 네모 = 정답 크기 참고)
  data/labeling/work/<member>/label_config.xml              Label Studio 라벨 설정
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "common"))
import kx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--version", default="v1_fill", help="박스를 지운 데이터 버전")
    a = ap.parse_args()

    asg = HERE / "assignments.csv"
    if not asg.exists():
        sys.exit("assignments.csv 가 없다. python data/labeling/make_assignments.py 먼저.")
    rows = [r for r in csv.DictReader(asg.open(encoding="utf-8")) if r["assignee"] in (a.member, "all")]
    if not rows:
        sys.exit(f"{a.member} 에게 배정된 사진이 없다. assignments.csv 의 assignee 값을 확인.")

    vdir = kx.version_dir(a.version)
    work = HERE / "work" / a.member
    img_out = work / "images"
    if img_out.exists():
        shutil.rmtree(img_out)
    img_out.mkdir(parents=True)

    rows.sort(key=lambda r: (r["burst_id"], r["image_id"]))
    missing = 0
    for i, r in enumerate(rows, 1):
        src = vdir / "images" / f"{r['image_id']}.png"
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, img_out / f"{i:04d}__{r['image_id']}.png")

    # 정답 예시: 공식 라벨이 있는 사진에 초록 네모를 그려 둔다 (크기 감 잡기용)
    ref = work / "reference"
    if ref.exists():
        shutil.rmtree(ref)
    ref.mkdir()
    df = kx.load_manifest()
    df = df[df["labeled"].astype(int) == 1]
    picked = []
    for m in sorted(df["machine"].astype(str).unique()):
        picked += list(df[df["machine"].astype(str) == m].sort_values("image_id")["image_id"].head(4))
    for iid in picked:
        img = cv2.imread(str(vdir / "images" / f"{iid}.png"))
        if img is None:
            continue
        h, w = img.shape[:2]
        for cx, cy, bw, bh in kx.read_yolo_labels(vdir / "labels" / f"{iid}.txt"):
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (40, 220, 40), 1)
        big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(ref / f"{iid}.png"), big)

    shutil.copy2(HERE / "label_config.xml", work / "label_config.xml")
    print(f"{a.member}: 사진 {len(rows) - missing}장 → {img_out}" + (f" (없는 사진 {missing})" if missing else ""))
    print(f"정답 예시 {len(picked)}장 → {ref}")
    print(f"Label Studio 설정 → {work / 'label_config.xml'}")


if __name__ == "__main__":
    main()
