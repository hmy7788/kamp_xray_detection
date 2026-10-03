"""n500 이미지별 메타데이터(data/manifest.csv)를 만든다.

파일명 접두(001/002)는 호기가 아니므로, 원본 bmp 폴더(호기별/NgImage)와 내용(MD5)을 대조해
실제 호기를 찾는다. 원본 폴더가 있는 PC에서만 실행 가능하며, 결과 csv 는 저장소에 보관한다.
컬럼: name, unit(1호기/2호기/3호기), raw_folder, width, height, n_box
실행: python scripts/build_manifest.py
"""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "4. X-ray 검사장비 AI 데이터셋" / "dataset" / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"
SRC = ROOT / "data" / "subsets" / "n500"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    idx = defaultdict(dict)  # 파일 stem -> {md5: (호기, NgImage 폴더)}
    for p in RAW.rglob("*.bmp"):
        idx[p.stem][md5(p)] = (p.parts[-3][:3], p.parent.name)
    rows = []
    for img in sorted((SRC / "images").iterdir()):
        unit, folder = idx[img.stem][md5(img)]  # 내용이 일치하는 원본이 없으면 KeyError (의도된 실패)
        n_box = len([l for l in (SRC / "labels" / f"{img.stem}.txt").read_text().splitlines() if l.strip()])
        w, h = Image.open(img).size
        rows.append([img.name, unit, folder, w, h, n_box])
    with open(ROOT / "data" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "unit", "raw_folder", "width", "height", "n_box"])
        wr.writerows(rows)
    print(f"{len(rows)}장 기록: data/manifest.csv")


if __name__ == "__main__":
    main()
